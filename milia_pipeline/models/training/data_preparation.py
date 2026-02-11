"""
Data Preparation Module

Task-specific data preparation for graph ML tasks:
- Graph classification/regression
- Node classification/regression  
- Edge classification/regression
- Link prediction

DYNAMIC: Supports all task types via dispatcher pattern
PRODUCTION-READY: Comprehensive validation, clear error messages
FUTURE-PROOF: Extensible for new task types via registry pattern

Author: milia Team
Version: 1.0.0
"""

import logging
import copy
from typing import Tuple, List, Optional, Any, Dict, Union

import torch

# =============================================================================
# CONDITIONAL IMPORTS WITH FALLBACK
# =============================================================================

# Import DataCompatibilityError with fallback
try:
    from milia_pipeline.exceptions import DataCompatibilityError
except ImportError:
    class DataCompatibilityError(Exception):
        """Fallback exception for data compatibility issues."""
        def __init__(self, message: str, model_name: Optional[str] = None,
                     missing_features: Optional[List[str]] = None,
                     incompatibility_reason: Optional[str] = None, **kwargs):
            self.model_name = model_name
            self.missing_features = missing_features or []
            self.incompatibility_reason = incompatibility_reason
            # Store any additional kwargs (e.g., task_type)
            for key, value in kwargs.items():
                setattr(self, key, value)
            super().__init__(message)

# Import TargetSelectionConfig with fallback
try:
    from milia_pipeline.models.factory.target_selection_config import TargetSelectionConfig
    TARGET_SELECTION_AVAILABLE = True
except ImportError:
    TARGET_SELECTION_AVAILABLE = False
    TargetSelectionConfig = None


logger = logging.getLogger(__name__)


# =============================================================================
# DISCRETIZE TARGETS IMPORT HELPER
# =============================================================================

def _get_discretize_targets_class(task_type: str):
    """
    Lazy import of DiscretizeTargets with fallback.
    
    DYNAMIC: Tries multiple import paths
    PRODUCTION-READY: Clear error messages on failure
    FUTURE-PROOF: Supports custom_transforms and graph_transforms registry
    
    Args:
        task_type: Task type for error context
        
    Returns:
        DiscretizeTargets class
        
    Raises:
        DataCompatibilityError: If DiscretizeTargets is not available
    """
    # Primary import path: custom_transforms module
    try:
        from milia_pipeline.transformations.custom_transforms import DiscretizeTargets
        return DiscretizeTargets
    except ImportError:
        pass
    
    # Fallback: try to get from graph_transforms registry
    try:
        from milia_pipeline.transformations.graph_transforms import get_transform_class
        DiscretizeTargets = get_transform_class('DiscretizeTargets')
        return DiscretizeTargets
    except (ImportError, KeyError):
        pass
    
    # Neither path worked - raise error
    logger.error(
        f"{task_type}: DiscretizeTargets transform not available. "
        f"Cannot convert float targets to class indices. "
        f"Please ensure custom_transforms module is properly installed."
    )
    raise DataCompatibilityError(
        "DiscretizeTargets transform required but not available. "
        "Float targets cannot be converted to class indices for classification.",
        task_type=task_type,
    )


# =============================================================================
# TASK DATA PREPARER CLASS
# =============================================================================

class TaskDataPreparer:
    """
    Task-specific data preparation for graph ML tasks.
    
    Provides centralized, DYNAMIC data preparation with:
    - Task-aware target extraction (node, edge, graph level)
    - Automatic discretization for classification tasks
    - RandomLinkSplit for link prediction
    - Target source inference via TargetSelectionConfig
    
    DYNAMIC: Supports all task types via dispatcher pattern
    PRODUCTION-READY: Comprehensive validation, clear error messages
    FUTURE-PROOF: Extensible for new task types via registry pattern
    
    Supported task types:
    - graph_regression: No transform needed (float targets)
    - graph_classification: May apply DiscretizeTargets for float targets
    - node_regression: Extracts node-level targets from configured source
    - node_classification: Extracts + discretizes node-level targets
    - edge_regression: Extracts edge-level targets from configured source
    - edge_classification: Extracts + discretizes edge-level targets
    - link_prediction: Applies RandomLinkSplit for edge_label generation
    
    Usage:
        >>> from milia_pipeline.models.training import TaskDataPreparer
        >>> train, val, test, num_classes = TaskDataPreparer.prepare_for_task(
        ...     train_data, val_data, test_data,
        ...     task_type='graph_classification',
        ...     target_selection_config=config
        ... )
    """
    
    # Registry of supported task types and their handlers
    # FUTURE-PROOF: New task types can be added here without modifying dispatch logic
    _task_handlers: Dict[str, str] = {
        'graph_regression': '_prepare_graph_regression',
        'graph_classification': '_prepare_graph_classification',
        'node_regression': '_prepare_node_regression',
        'node_classification': '_prepare_node_classification',
        'edge_regression': '_prepare_edge_regression',
        'edge_classification': '_prepare_edge_classification',
        'link_prediction': '_prepare_link_prediction',
    }
    
    @classmethod
    def prepare_for_task(
        cls,
        train_data,
        val_data,
        test_data,
        task_type: str,
        logger: Optional[logging.Logger] = None,
        target_selection_config: Optional[Any] = None,
    ) -> Tuple[Any, Any, Any, Optional[int]]:
        """
        Prepare split data for specific task types by applying appropriate transforms.
        
        This method applies task-specific transformations AFTER dataset splitting
        and BEFORE DataLoader creation, following PyG conventions.
        
        DYNAMIC: Uses target_selection_config for automatic level/source inference
        PRODUCTION-READY: Handles all task types with clear error messages
        FUTURE-PROOF: Extensible for new task types and target sources
        
        Args:
            train_data: Training subset from DataSplitter
            val_data: Validation subset from DataSplitter
            test_data: Test subset from DataSplitter  
            task_type: Task type string from config
            logger: Logger instance (uses module logger if None)
            target_selection_config: Optional TargetSelectionConfig for specifying
                target source (x, y, edge_attr) and indices. If None, defaults to y.
            
        Returns:
            Tuple of (train_data, val_data, test_data, num_classes) where:
            - train_data, val_data, test_data: potentially transformed data
            - num_classes: Number of classes for classification (from discretization n_bins),
                           or None for non-classification tasks
            
        Raises:
            DataCompatibilityError: If data is incompatible with task_type
            
        Example:
            >>> train, val, test, num_classes = TaskDataPreparer.prepare_for_task(
            ...     train_data, val_data, test_data,
            ...     task_type='graph_classification'
            ... )
            >>> print(f"Number of classes: {num_classes}")
        """
        # Use module logger if none provided
        log = logger or globals()['logger']
        
        task_lower = task_type.lower()
        
        # =========================================================================
        # Resolve target selection config if provided
        # =========================================================================
        if target_selection_config is not None and TARGET_SELECTION_AVAILABLE:
            # Get sample data for shape validation
            sample = train_data[0] if hasattr(train_data, '__getitem__') and len(train_data) > 0 else None
            target_selection_config.resolve_for_task(task_type, sample)
        
        # =========================================================================
        # Dispatch to task-specific handler via registry
        # =========================================================================
        if task_lower in cls._task_handlers:
            handler_name = cls._task_handlers[task_lower]
            handler = getattr(cls, handler_name)
            return handler(
                train_data, val_data, test_data, 
                log, target_selection_config
            )
        
        # Unknown task type: log warning and return unchanged
        log.warning(
            f"Unknown task type '{task_type}'. No data preparation applied. "
            f"Supported types: {', '.join(sorted(cls._task_handlers.keys()))}"
        )
        return train_data, val_data, test_data, None
    
    @classmethod
    def list_supported_tasks(cls) -> List[str]:
        """
        List all supported task types.
        
        Returns:
            Sorted list of supported task type names
            
        Example:
            >>> TaskDataPreparer.list_supported_tasks()
            ['edge_classification', 'edge_regression', 'graph_classification', ...]
        """
        return sorted(cls._task_handlers.keys())
    
    # =========================================================================
    # TASK-SPECIFIC HANDLERS
    # =========================================================================
    
    @classmethod
    def _prepare_graph_regression(
        cls,
        train_data,
        val_data,
        test_data,
        logger: logging.Logger,
        target_selection_config: Optional[Any] = None,
    ) -> Tuple[Any, Any, Any, Optional[int]]:
        """
        Prepare data for graph regression task.
        
        Graph regression uses float targets directly - no transform needed.
        
        Args:
            train_data: Training subset
            val_data: Validation subset
            test_data: Test subset
            logger: Logger instance
            target_selection_config: Optional target selection config (unused)
            
        Returns:
            Tuple of (train_data, val_data, test_data, None)
        """
        logger.debug("Task 'graph_regression' uses standard graph-level data (no transform needed)")
        return train_data, val_data, test_data, None
    
    @classmethod
    def _prepare_graph_classification(
        cls,
        train_data,
        val_data,
        test_data,
        logger: logging.Logger,
        target_selection_config: Optional[Any] = None,
    ) -> Tuple[Any, Any, Any, Optional[int]]:
        """
        Prepare data for graph classification task.
        
        Graph classification requires integer class indices as targets. This method:
        1. Checks if targets are already integers (classification-ready)
        2. If float targets detected, dynamically applies DiscretizeTargets transform
        3. Fits discretization on train_data, applies consistently to all splits
        
        DYNAMIC: Auto-detects target dtype and applies transform only when needed
        PRODUCTION-READY: Validates data, provides clear error messages
        FUTURE-PROOF: Uses registry pattern for DiscretizeTargets import
        
        Args:
            train_data: Training subset from DataSplitter
            val_data: Validation subset from DataSplitter  
            test_data: Test subset from DataSplitter
            logger: Logger instance
            target_selection_config: Optional target selection config (unused for graph-level)
            
        Returns:
            Tuple of (train_data, val_data, test_data, num_classes) where:
            - train_data, val_data, test_data: potentially transformed data
            - num_classes: Number of classes (from discretization n_bins or counted from int targets),
                           or None if data validation fails
        """
        # Get sample to check target dtype
        sample = train_data[0] if hasattr(train_data, '__getitem__') and len(train_data) > 0 else None
        
        if sample is None:
            logger.warning("graph_classification: Cannot validate data - empty dataset")
            return train_data, val_data, test_data, None
        
        # Check if y attribute exists
        if not hasattr(sample, 'y') or sample.y is None:
            logger.warning(
                "graph_classification: No 'y' attribute found in data. "
                "Model will not have targets for training."
            )
            return train_data, val_data, test_data, None
        
        target = sample.y
        
        # Check if targets are already integer (classification-ready)
        if target.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            # Count unique classes from training data
            all_classes = set()
            for data in train_data:
                if hasattr(data, 'y') and data.y is not None:
                    if data.y.dim() == 0:
                        all_classes.add(data.y.item())
                    else:
                        all_classes.update(data.y.flatten().tolist())
            num_classes = len(all_classes) if all_classes else None
            logger.info(f"graph_classification: Targets are already integer class indices (num_classes={num_classes})")
            return train_data, val_data, test_data, num_classes
        
        # Check if data was already discretized (metadata flag)
        if hasattr(sample, 'targets_discretized') and sample.targets_discretized:
            logger.info("graph_classification: Data already marked as discretized (no transform needed)")
            return train_data, val_data, test_data, None
        
        # Float targets detected - need to apply DiscretizeTargets
        logger.info(
            f"graph_classification: Float targets detected (dtype={target.dtype}). "
            f"Applying DiscretizeTargets transform to convert to class indices."
        )
        
        # Get DiscretizeTargets class (with fallback)
        DiscretizeTargets = _get_discretize_targets_class('graph_classification')
        
        # Create DiscretizeTargets transform with default settings
        # Uses quantile strategy for balanced class distribution
        n_bins = 10  # Default: 10 classes
        discretize_transform = DiscretizeTargets(
            attrs=['y'],
            n_bins=n_bins,
            strategy='quantile',  # Balanced class distribution
            target_level='graph',  # CRITICAL: Explicit graph-level to prevent misclassification
                                   # when y.size(0) == num_nodes by coincidence.
                                   # Without this, auto-detection heuristic may incorrectly treat
                                   # graph-level targets as node-level, causing batch size mismatch:
                                   # "Expected input batch_size (8) to match target batch_size (29)"
        )
        
        # Fit on training data to learn bin edges (CRITICAL for consistency)
        logger.info("graph_classification: Fitting DiscretizeTargets on training data...")
        discretize_transform.fit(train_data)
        
        if not discretize_transform.is_fitted():
            logger.error("graph_classification: Failed to fit DiscretizeTargets")
            raise DataCompatibilityError(
                "Failed to fit DiscretizeTargets on training data.",
                task_type="graph_classification",
            )
        
        logger.info(
            f"graph_classification: DiscretizeTargets fitted with global bin edges. "
            f"All splits will use consistent discretization."
        )
        
        # Apply transform to all splits using the fitted bin edges
        train_data = cls._apply_discretize_to_subset(train_data, discretize_transform, logger, "train")
        val_data = cls._apply_discretize_to_subset(val_data, discretize_transform, logger, "val")
        test_data = cls._apply_discretize_to_subset(test_data, discretize_transform, logger, "test")
        
        # Log class distribution info
        logger.info(f"graph_classification: Data discretized into {n_bins} classes (num_classes={n_bins})")
        
        return train_data, val_data, test_data, n_bins
    
    @classmethod
    def _prepare_node_regression(
        cls,
        train_data,
        val_data,
        test_data,
        logger: logging.Logger,
        target_selection_config: Optional[Any] = None,
    ) -> Tuple[Any, Any, Any, Optional[int]]:
        """
        Prepare data for node regression task.
        
        Node regression may need extraction from x based on target_selection.
        
        Args:
            train_data: Training subset
            val_data: Validation subset
            test_data: Test subset
            logger: Logger instance
            target_selection_config: Optional target selection config
            
        Returns:
            Tuple of (train_data, val_data, test_data, None)
        """
        train_data, val_data, test_data = cls._prepare_node_level_data(
            train_data, val_data, test_data, 'node_regression', logger, target_selection_config
        )
        return train_data, val_data, test_data, None
    
    @classmethod
    def _prepare_node_classification(
        cls,
        train_data,
        val_data,
        test_data,
        logger: logging.Logger,
        target_selection_config: Optional[Any] = None,
    ) -> Tuple[Any, Any, Any, Optional[int]]:
        """
        Prepare data for node classification task.
        
        Node classification requires integer class indices as targets. This method:
        1. First extracts node-level targets using _prepare_node_level_data logic
        2. Checks if targets are already integers (classification-ready)
        3. If float targets detected, dynamically applies DiscretizeTargets transform
        4. Fits discretization on train_data, applies consistently to all splits
        
        DYNAMIC: Uses target_selection_config for automatic source inference
        PRODUCTION-READY: Handles all target types with clear error messages
        FUTURE-PROOF: Extensible for new target sources and discretization strategies
        
        Args:
            train_data: Training subset from DataSplitter
            val_data: Validation subset from DataSplitter  
            test_data: Test subset from DataSplitter
            logger: Logger instance
            target_selection_config: Optional TargetSelectionConfig for specifying
                target source (x, y, etc.) and indices
            
        Returns:
            Tuple of (train_data, val_data, test_data, num_classes)
        """
        # First, ensure node-level targets are properly extracted
        # This handles extraction from x or other sources if y is not node-level
        train_data, val_data, test_data = cls._prepare_node_level_data(
            train_data, val_data, test_data, 'node_classification', logger, target_selection_config
        )
        
        # Get sample to check target dtype
        sample = train_data[0] if hasattr(train_data, '__getitem__') and len(train_data) > 0 else None
        
        if sample is None:
            logger.warning("node_classification: Cannot validate data - empty dataset")
            return train_data, val_data, test_data, None
        
        # Check if y attribute exists
        if not hasattr(sample, 'y') or sample.y is None:
            logger.warning(
                "node_classification: No 'y' attribute found in data. "
                "Model will not have targets for training."
            )
            return train_data, val_data, test_data, None
        
        target = sample.y
        
        # Check if targets are already integer (classification-ready)
        if target.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            # Count unique classes from training data
            all_classes = set()
            for data in train_data:
                if hasattr(data, 'y') and data.y is not None:
                    all_classes.update(data.y.flatten().tolist())
            num_classes = len(all_classes) if all_classes else None
            logger.info(f"node_classification: Targets are already integer class indices (num_classes={num_classes})")
            return train_data, val_data, test_data, num_classes
        
        # Check if data was already discretized (metadata flag)
        if hasattr(sample, 'y_discretized') and sample.y_discretized:
            logger.info("node_classification: Data already marked as discretized (no transform needed)")
            return train_data, val_data, test_data, None
        
        # Float targets detected - need to apply DiscretizeTargets
        logger.info(
            f"node_classification: Float targets detected (dtype={target.dtype}). "
            f"Applying DiscretizeTargets transform to convert to class indices."
        )
        
        # Get DiscretizeTargets class (with fallback)
        DiscretizeTargets = _get_discretize_targets_class('node_classification')
        
        # Create DiscretizeTargets transform with explicit node-level target
        # Uses quantile strategy for balanced class distribution
        n_bins = 10  # Default: 10 classes
        discretize_transform = DiscretizeTargets(
            attrs=['y'],
            n_bins=n_bins,
            strategy='quantile',  # Balanced class distribution
            target_level='node',  # CRITICAL: Explicit node-level to prevent misclassification
                                  # This ensures proper handling regardless of tensor dimensions.
        )
        
        # Fit on training data to learn bin edges (CRITICAL for consistency)
        logger.info("node_classification: Fitting DiscretizeTargets on training data...")
        discretize_transform.fit(train_data)
        
        if not discretize_transform.is_fitted():
            logger.error("node_classification: Failed to fit DiscretizeTargets")
            raise DataCompatibilityError(
                "Failed to fit DiscretizeTargets on training data.",
                task_type="node_classification",
            )
        
        logger.info(
            f"node_classification: DiscretizeTargets fitted with global bin edges. "
            f"All splits will use consistent discretization."
        )
        
        # Apply transform to all splits using the fitted bin edges
        train_data = cls._apply_discretize_to_subset(train_data, discretize_transform, logger, "train")
        val_data = cls._apply_discretize_to_subset(val_data, discretize_transform, logger, "val")
        test_data = cls._apply_discretize_to_subset(test_data, discretize_transform, logger, "test")
        
        # Log class distribution info
        logger.info(f"node_classification: Data discretized into {n_bins} classes (num_classes={n_bins})")
        
        return train_data, val_data, test_data, n_bins
    
    @classmethod
    def _prepare_edge_regression(
        cls,
        train_data,
        val_data,
        test_data,
        logger: logging.Logger,
        target_selection_config: Optional[Any] = None,
    ) -> Tuple[Any, Any, Any, Optional[int]]:
        """
        Prepare data for edge regression task.
        
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
            test_data: Test data subset
            logger: Logger instance
            target_selection_config: Optional TargetSelectionConfig for specifying
                target source (edge_attr, edge_y, etc.) and indices
            
        Returns:
            Tuple of (train_data, val_data, test_data, None)
            
        Raises:
            DataCompatibilityError: If no valid target source found
        """
        sample = train_data[0] if hasattr(train_data, '__getitem__') and len(train_data) > 0 else None
        
        if sample is None:
            raise DataCompatibilityError(
                "edge_regression: Cannot validate data - empty dataset",
                task_type="edge_regression",
            )
        
        # =========================================================================
        # CASE 1: edge_y or edge_value already exists
        # =========================================================================
        has_edge_value = hasattr(sample, 'edge_value') and sample.edge_value is not None
        has_edge_y = hasattr(sample, 'edge_y') and sample.edge_y is not None
        
        if has_edge_value or has_edge_y:
            attr_name = 'edge_value' if has_edge_value else 'edge_y'
            logger.info(f"edge_regression: Using existing '{attr_name}' attribute")
            return train_data, val_data, test_data, None
        
        # =========================================================================
        # CASE 2: Extract from configured source (edge_attr by default)
        # =========================================================================
        # Determine source attribute from target_selection_config
        if (target_selection_config is not None and 
            TARGET_SELECTION_AVAILABLE and
            hasattr(target_selection_config, 'resolved_source_attr') and
            target_selection_config.resolved_source_attr is not None):
            source_attr = target_selection_config.resolved_source_attr
        else:
            # Default for edge-level: extract from edge_attr
            source_attr = 'edge_attr'
        
        # Check if source exists
        if not hasattr(sample, source_attr) or getattr(sample, source_attr) is None:
            raise DataCompatibilityError(
                f"edge_regression task requires edge-level targets. "
                f"Neither 'edge_y' nor 'edge_value' found, and source '{source_attr}' is not available. "
                f"Options:\n"
                f"  1. Add edge_value during data processing (e.g., bond lengths)\n"
                f"  2. Configure target_selection.target_source to specify which attribute contains targets\n"
                f"  3. Use a different task type (e.g., graph_regression)",
                task_type="edge_regression",
            )
        
        source_tensor = getattr(sample, source_attr)
        
        # Get number of edges for validation
        num_edges = sample.edge_index.size(1) if hasattr(sample, 'edge_index') and sample.edge_index is not None else None
        
        if num_edges is not None and source_tensor.size(0) != num_edges:
            raise DataCompatibilityError(
                f"edge_regression: Source '{source_attr}' has shape {list(source_tensor.shape)}, "
                f"expected first dim = {num_edges} (number of edges)",
                task_type="edge_regression",
            )
        
        # =========================================================================
        # RESOLVE target selection indices against SOURCE tensor (not edge_y)
        # =========================================================================
        # CRITICAL: Must resolve indices before extraction so that:
        # 1. Invalid indices are caught early with clear error messages
        # 2. resolved_indices is populated for model_factory to skip re-resolution
        # The total_count should be the source tensor's column count, not edge_y's
        # =========================================================================
        indices = None
        if target_selection_config is not None:
            if hasattr(target_selection_config, 'resolved_indices') and target_selection_config.resolved_indices is not None:
                # Already resolved - use existing indices
                indices = target_selection_config.resolved_indices
            elif hasattr(target_selection_config, 'indices') and target_selection_config.indices is not None:
                # Not yet resolved - resolve against source tensor dimensions
                source_total_count = source_tensor.size(-1) if source_tensor.dim() > 1 else source_tensor.size(0)
                target_selection_config.resolve(available_names=None, total_count=source_total_count)
                indices = target_selection_config.resolved_indices
                logger.debug(
                    f"edge_regression: Resolved target indices against source '{source_attr}' "
                    f"(total_count={source_total_count}): {indices}"
                )
        
        # =========================================================================
        # Extract targets from source and assign to edge_y
        # =========================================================================
        logger.info(
            f"edge_regression: Extracting edge-level targets from '{source_attr}' "
            f"(indices={indices})"
        )
        
        train_data = cls._extract_targets_from_source(
            train_data, source_attr, indices, 'edge_y', logger, "train"
        )
        val_data = cls._extract_targets_from_source(
            val_data, source_attr, indices, 'edge_y', logger, "val"
        )
        test_data = cls._extract_targets_from_source(
            test_data, source_attr, indices, 'edge_y', logger, "test"
        )
        
        return train_data, val_data, test_data, None
    
    @classmethod
    def _prepare_edge_classification(
        cls,
        train_data,
        val_data,
        test_data,
        logger: logging.Logger,
        target_selection_config: Optional[Any] = None,
    ) -> Tuple[Any, Any, Any, Optional[int]]:
        """
        Prepare data for edge classification task.
        
        Edge classification requires integer class indices as targets. This method:
        1. First extracts edge-level targets using similar logic to edge_regression
        2. Checks if targets are already integers (classification-ready)
        3. If float targets detected, dynamically applies DiscretizeTargets transform
        4. Fits discretization on train_data, applies consistently to all splits
        
        DYNAMIC: Uses target_selection_config for automatic source inference
        PRODUCTION-READY: Handles all target types with clear error messages
        FUTURE-PROOF: Extensible for new target sources and discretization strategies
        
        Args:
            train_data: Training subset from DataSplitter
            val_data: Validation subset from DataSplitter  
            test_data: Test subset from DataSplitter
            logger: Logger instance
            target_selection_config: Optional TargetSelectionConfig for specifying
                target source (edge_attr, edge_y, etc.) and indices
            
        Returns:
            Tuple of (train_data, val_data, test_data, num_classes)
        """
        # First, ensure edge-level targets are properly extracted
        # Reuse edge_regression logic for extraction, then apply discretization
        train_data, val_data, test_data, _ = cls._prepare_edge_regression(
            train_data, val_data, test_data, logger, target_selection_config
        )
        
        # Get sample to check target dtype
        sample = train_data[0] if hasattr(train_data, '__getitem__') and len(train_data) > 0 else None
        
        if sample is None:
            logger.warning("edge_classification: Cannot validate data - empty dataset")
            return train_data, val_data, test_data, None
        
        # Determine which edge target attribute exists (edge_y, edge_value, or edge_label)
        target = None
        target_attr = None
        for attr in ['edge_y', 'edge_value', 'edge_label']:
            if hasattr(sample, attr) and getattr(sample, attr) is not None:
                target = getattr(sample, attr)
                target_attr = attr
                break
        
        if target is None:
            logger.warning(
                "edge_classification: No edge target attribute found (edge_y, edge_value, edge_label). "
                "Model will not have targets for training."
            )
            return train_data, val_data, test_data, None
        
        # Check if targets are already integer (classification-ready)
        if target.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            # Count unique classes from training data
            all_classes = set()
            for data in train_data:
                edge_target = None
                for attr in ['edge_y', 'edge_value', 'edge_label']:
                    if hasattr(data, attr) and getattr(data, attr) is not None:
                        edge_target = getattr(data, attr)
                        break
                if edge_target is not None:
                    all_classes.update(edge_target.flatten().tolist())
            num_classes = len(all_classes) if all_classes else None
            logger.info(f"edge_classification: Targets are already integer class indices (num_classes={num_classes})")
            return train_data, val_data, test_data, num_classes
        
        # Check if data was already discretized (metadata flag)
        if hasattr(sample, f'{target_attr}_discretized') and getattr(sample, f'{target_attr}_discretized'):
            logger.info("edge_classification: Data already marked as discretized (no transform needed)")
            return train_data, val_data, test_data, None
        
        # Float targets detected - need to apply DiscretizeTargets
        logger.info(
            f"edge_classification: Float targets detected in '{target_attr}' (dtype={target.dtype}). "
            f"Applying DiscretizeTargets transform to convert to class indices."
        )
        
        # Get DiscretizeTargets class (with fallback)
        DiscretizeTargets = _get_discretize_targets_class('edge_classification')
        
        # Create DiscretizeTargets transform with explicit edge-level target
        # Uses quantile strategy for balanced class distribution
        n_bins = 10  # Default: 10 classes
        discretize_transform = DiscretizeTargets(
            attrs=[target_attr],  # Use the detected edge target attribute
            n_bins=n_bins,
            strategy='quantile',  # Balanced class distribution
            target_level='edge',  # CRITICAL: Explicit edge-level to prevent misclassification
                                  # This ensures proper handling regardless of tensor dimensions.
        )
        
        # Fit on training data to learn bin edges (CRITICAL for consistency)
        logger.info("edge_classification: Fitting DiscretizeTargets on training data...")
        discretize_transform.fit(train_data)
        
        if not discretize_transform.is_fitted():
            logger.error("edge_classification: Failed to fit DiscretizeTargets")
            raise DataCompatibilityError(
                "Failed to fit DiscretizeTargets on training data.",
                task_type="edge_classification",
            )
        
        logger.info(
            f"edge_classification: DiscretizeTargets fitted with global bin edges. "
            f"All splits will use consistent discretization."
        )
        
        # Apply transform to all splits using the fitted bin edges
        train_data = cls._apply_discretize_to_subset(train_data, discretize_transform, logger, "train")
        val_data = cls._apply_discretize_to_subset(val_data, discretize_transform, logger, "val")
        test_data = cls._apply_discretize_to_subset(test_data, discretize_transform, logger, "test")
        
        # Log class distribution info
        logger.info(f"edge_classification: Data discretized into {n_bins} classes (num_classes={n_bins})")
        
        return train_data, val_data, test_data, n_bins
    
    @classmethod
    def _prepare_link_prediction(
        cls,
        train_data,
        val_data,
        test_data,
        logger: logging.Logger,
        target_selection_config: Optional[Any] = None,
    ) -> Tuple[Any, Any, Any, Optional[int]]:
        """
        Prepare data for link prediction task.
        
        Link prediction requires edge_label attribute. This method:
        1. Checks if edge_label already exists in data
        2. If not, applies RandomLinkSplit to each graph to generate edge_label
        
        Note: For molecular datasets (multiple small graphs), link prediction
        predicts edges within each molecule. RandomLinkSplit is applied per-graph.
        
        Args:
            train_data: Training subset
            val_data: Validation subset
            test_data: Test subset
            logger: Logger instance
            target_selection_config: Optional target selection config (unused)
            
        Returns:
            Tuple of (train_data, val_data, test_data, None)
        """
        from torch_geometric.transforms import RandomLinkSplit
        
        # Check if edge_label already exists in sample
        sample = train_data[0] if hasattr(train_data, '__getitem__') and len(train_data) > 0 else None
        
        if sample is not None and hasattr(sample, 'edge_label') and sample.edge_label is not None:
            logger.info("link_prediction: edge_label already exists in data")
            return train_data, val_data, test_data, None
        
        # edge_label does not exist - need to apply RandomLinkSplit
        logger.info(
            "link_prediction: edge_label not found. Applying RandomLinkSplit transform. "
            "Note: For molecular datasets, this predicts bond existence within molecules."
        )
        
        # Create transform for link prediction
        # For molecular graphs, we split edges WITHIN each graph
        link_split_transform = RandomLinkSplit(
            num_val=0.0,  # We already have val split at graph level
            num_test=0.0,  # We already have test split at graph level
            is_undirected=True,  # Molecular bonds are typically undirected
            add_negative_train_samples=True,
            neg_sampling_ratio=1.0,
        )
        
        # Apply transform to each graph in each split
        train_data = cls._apply_transform_to_subset(train_data, link_split_transform, logger, "train")
        val_data = cls._apply_transform_to_subset(val_data, link_split_transform, logger, "val")
        test_data = cls._apply_transform_to_subset(test_data, link_split_transform, logger, "test")
        
        return train_data, val_data, test_data, None
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    @classmethod
    def _prepare_node_level_data(
        cls,
        train_data,
        val_data,
        test_data,
        task_type: str,
        logger: logging.Logger,
        target_selection_config: Optional[Any] = None,
    ) -> Tuple[Any, Any, Any]:
        """
        Prepare data for node-level tasks (node_regression, node_classification).
        
        DYNAMIC: Auto-extracts targets from configured source attribute
        PRODUCTION-READY: Validates shapes, provides clear errors
        FUTURE-PROOF: Works with any source attribute (x, y, edge_attr, custom)
        
        Logic:
        1. Check if y already has correct shape (num_nodes) -> use as-is
        2. If not, check configured source (x by default for node tasks)
        3. Extract targets from source and assign to y
        
        Args:
            train_data: Training data subset
            val_data: Validation data subset
            test_data: Test data subset
            task_type: Task type string
            logger: Logger instance
            target_selection_config: Optional TargetSelectionConfig for specifying
                target source (x, y, edge_attr) and indices
            
        Returns:
            Tuple of (train_data, val_data, test_data) with y properly set
            
        Raises:
            DataCompatibilityError: If no valid target source found
        """
        sample = train_data[0] if hasattr(train_data, '__getitem__') and len(train_data) > 0 else None
        
        if sample is None:
            raise DataCompatibilityError(
                f"{task_type}: Cannot validate data - empty dataset",
                task_type=task_type,
            )
        
        num_nodes = sample.num_nodes if hasattr(sample, 'num_nodes') else None
        if num_nodes is None and hasattr(sample, 'x') and sample.x is not None:
            num_nodes = sample.x.size(0)
        
        if num_nodes is None:
            raise DataCompatibilityError(
                f"{task_type}: Cannot determine number of nodes. "
                f"Data must have 'num_nodes' attribute or 'x' with shape [num_nodes, ...]",
                task_type=task_type,
            )
        
        # =========================================================================
        # CASE 1: y already has correct node-level shape
        # =========================================================================
        if hasattr(sample, 'y') and sample.y is not None:
            y = sample.y
            if y.dim() >= 1 and y.size(0) == num_nodes:
                logger.info(
                    f"{task_type}: y already has node-level shape {list(y.shape)}, "
                    f"num_nodes={num_nodes}. Using as-is."
                )
                return train_data, val_data, test_data
        
        # =========================================================================
        # CASE 2: y is graph-level or missing - extract from configured source
        # =========================================================================
        # Determine source attribute from target_selection_config
        if (target_selection_config is not None and 
            TARGET_SELECTION_AVAILABLE and
            hasattr(target_selection_config, 'resolved_source_attr') and
            target_selection_config.resolved_source_attr is not None):
            source_attr = target_selection_config.resolved_source_attr
        else:
            # Default for node-level: extract from x
            source_attr = 'x'
        
        # Check if source exists
        if not hasattr(sample, source_attr) or getattr(sample, source_attr) is None:
            raise DataCompatibilityError(
                f"{task_type} task requires node-level targets. "
                f"y.shape={list(sample.y.shape) if hasattr(sample, 'y') and sample.y is not None else 'None'} "
                f"is not node-level (expected first dim = {num_nodes}). "
                f"Source '{source_attr}' also not available. "
                f"Configure target_selection.indices to specify which columns of x to use as targets, "
                f"or ensure your data has node-level targets in y.",
                task_type=task_type,
            )
        
        source_tensor = getattr(sample, source_attr)
        
        # Validate source has correct first dimension
        if source_tensor.size(0) != num_nodes:
            raise DataCompatibilityError(
                f"{task_type}: Source '{source_attr}' has shape {list(source_tensor.shape)}, "
                f"expected first dim = {num_nodes}",
                task_type=task_type,
            )
        
        # =========================================================================
        # RESOLVE target selection indices against SOURCE tensor (not y)
        # =========================================================================
        # CRITICAL: Must resolve indices before extraction so that:
        # 1. Invalid indices are caught early with clear error messages
        # 2. resolved_indices is populated for model_factory to skip re-resolution
        # The total_count should be the source tensor's column count, not y's
        # =========================================================================
        indices = None
        if target_selection_config is not None:
            if hasattr(target_selection_config, 'resolved_indices') and target_selection_config.resolved_indices is not None:
                # Already resolved - use existing indices
                indices = target_selection_config.resolved_indices
            elif hasattr(target_selection_config, 'indices') and target_selection_config.indices is not None:
                # Not yet resolved - resolve against source tensor dimensions
                source_total_count = source_tensor.size(-1) if source_tensor.dim() > 1 else source_tensor.size(0)
                target_selection_config.resolve(available_names=None, total_count=source_total_count)
                indices = target_selection_config.resolved_indices
                logger.debug(
                    f"{task_type}: Resolved target indices against source '{source_attr}' "
                    f"(total_count={source_total_count}): {indices}"
                )
        
        # =========================================================================
        # Extract targets from source and assign to y
        # =========================================================================
        logger.info(
            f"{task_type}: Extracting node-level targets from '{source_attr}' "
            f"(indices={indices})"
        )
        
        train_data = cls._extract_targets_from_source(
            train_data, source_attr, indices, 'y', logger, "train"
        )
        val_data = cls._extract_targets_from_source(
            val_data, source_attr, indices, 'y', logger, "val"
        )
        test_data = cls._extract_targets_from_source(
            test_data, source_attr, indices, 'y', logger, "test"
        )
        
        return train_data, val_data, test_data
    
    @classmethod
    def _extract_targets_from_source(
        cls,
        data_subset,
        source_attr: str,
        indices: Optional[List[int]],
        target_attr: str,
        logger: logging.Logger,
        split_name: str,
    ) -> List:
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
            logger: Logger instance
            split_name: Name of split for logging
            
        Returns:
            List of modified Data objects with target attribute set
        """
        modified_data = []
        
        for i, data in enumerate(data_subset):
            # Create a shallow copy to avoid modifying original data
            data_copy = copy.copy(data)
            
            source_tensor = getattr(data_copy, source_attr, None)
            if source_tensor is None:
                logger.warning(
                    f"Graph {i} in {split_name} missing '{source_attr}', skipping target extraction"
                )
                modified_data.append(data_copy)
                continue
            
            # Validate source tensor size matches edge_index for edge-level attributes
            # This is critical for edge_classification to work correctly
            if source_attr.startswith('edge_') and hasattr(data_copy, 'edge_index'):
                num_edges = data_copy.edge_index.size(1)
                source_edges = source_tensor.size(0)
                if source_edges != num_edges:
                    logger.warning(
                        f"Graph {i} in {split_name}: {source_attr} has {source_edges} rows "
                        f"but edge_index has {num_edges} edges. This may cause batch size "
                        f"mismatches during training. Ensure edge attributes match edge_index."
                    )
            
            # Extract targets from source
            if indices is not None:
                # Extract specific columns
                if source_tensor.dim() == 1:
                    # 1D tensor - single index only makes sense
                    if len(indices) == 1:
                        target_tensor = source_tensor[indices[0]].unsqueeze(-1)
                    else:
                        logger.warning(
                            f"Graph {i} in {split_name}: source is 1D but multiple indices specified. "
                            f"Using first index only."
                        )
                        target_tensor = source_tensor[indices[0]].unsqueeze(-1)
                else:
                    # 2D+ tensor - extract columns
                    target_tensor = source_tensor[:, indices]
            else:
                # Use all columns
                target_tensor = source_tensor
            
            # Set target attribute
            setattr(data_copy, target_attr, target_tensor)
            modified_data.append(data_copy)
        
        logger.debug(f"Extracted targets to '{target_attr}' for {len(modified_data)} graphs in {split_name} split")
        return modified_data
    
    @classmethod
    def _apply_discretize_to_subset(
        cls,
        subset,
        discretize_transform,
        logger: logging.Logger,
        split_name: str,
    ) -> List:
        """
        Apply DiscretizeTargets transform to each Data object in a subset.
        
        Uses the pre-fitted discretizer to ensure consistent bin edges across all splits.
        
        Args:
            subset: Dataset subset (list-like of Data objects)
            discretize_transform: Fitted DiscretizeTargets transform
            logger: Logger instance
            split_name: Name of split for logging
            
        Returns:
            List of transformed Data objects with integer class targets
        """
        transformed = []
        
        for i, data in enumerate(subset):
            try:
                result = discretize_transform(data)
                transformed.append(result)
            except Exception as e:
                logger.warning(
                    f"DiscretizeTargets failed for graph {i} in {split_name} split: {e}. "
                    f"Keeping original data."
                )
                transformed.append(data)
        
        logger.debug(f"Applied DiscretizeTargets to {len(transformed)} graphs in {split_name} split")
        return transformed
    
    @classmethod
    def _apply_transform_to_subset(
        cls,
        subset,
        transform,
        logger: logging.Logger,
        split_name: str,
    ) -> List:
        """
        Apply a transform to each Data object in a subset.
        
        For link prediction, RandomLinkSplit returns a tuple (train, val, test) for each graph.
        Since we've already split at graph level, we take only the train portion.
        
        Args:
            subset: Dataset subset (list-like of Data objects)
            transform: PyG transform to apply
            logger: Logger instance
            split_name: Name of split for logging
            
        Returns:
            List of transformed Data objects
        """
        transformed = []
        
        for i, data in enumerate(subset):
            try:
                # RandomLinkSplit returns (train, val, test) tuple
                # We take the first element since we already have graph-level splits
                result = transform(data)
                if isinstance(result, tuple):
                    transformed.append(result[0])  # Take train split
                else:
                    transformed.append(result)
            except Exception as e:
                logger.warning(
                    f"Transform failed for graph {i} in {split_name} split: {e}. "
                    f"Keeping original data."
                )
                transformed.append(data)
        
        logger.debug(f"Applied transform to {len(transformed)} graphs in {split_name} split")
        return transformed


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def prepare_data_for_task(
    train_data,
    val_data,
    test_data,
    task_type: str,
    logger: Optional[logging.Logger] = None,
    target_selection_config: Optional[Any] = None,
) -> Tuple[Any, Any, Any, Optional[int]]:
    """
    Convenience function for task-specific data preparation.
    
    Delegates to TaskDataPreparer.prepare_for_task() for:
    - DYNAMIC: Supports 7 task types via registry pattern
    - PRODUCTION-READY: Automatic target extraction, discretization, validation
    - FUTURE-PROOF: New task types auto-available when registered
    
    Args:
        train_data: Training subset from DataSplitter
        val_data: Validation subset from DataSplitter
        test_data: Test subset from DataSplitter  
        task_type: Task type string from config
        logger: Logger instance (uses module logger if None)
        target_selection_config: Optional TargetSelectionConfig for specifying
            target source (x, y, edge_attr) and indices
        
    Returns:
        Tuple of (train_data, val_data, test_data, num_classes) where:
        - train_data, val_data, test_data: potentially transformed data
        - num_classes: Number of classes for classification, or None
    
    Example:
        >>> from milia_pipeline.models.training import prepare_data_for_task
        >>> train, val, test, num_classes = prepare_data_for_task(
        ...     train_data, val_data, test_data,
        ...     task_type='graph_classification'
        ... )
    """
    return TaskDataPreparer.prepare_for_task(
        train_data, val_data, test_data,
        task_type, logger, target_selection_config
    )


def list_supported_tasks() -> List[str]:
    """
    List all supported task types.
    
    Returns:
        Sorted list of supported task type names
        
    Example:
        >>> from milia_pipeline.models.training import list_supported_tasks
        >>> tasks = list_supported_tasks()
        >>> print(tasks)
        ['edge_classification', 'edge_regression', 'graph_classification', ...]
    """
    return TaskDataPreparer.list_supported_tasks()


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info(
    f"data_preparation module loaded - "
    f"{len(TaskDataPreparer._task_handlers)} task types supported"
)
