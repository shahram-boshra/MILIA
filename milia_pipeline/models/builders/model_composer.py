"""
Model Composer

Compose multiple models into ensembles or stacked architectures.
Supports parallel ensembles, sequential stacking, and hierarchical composition.

Pydantic V2 Migration (Phase 34):
    - Migrated from @dataclass to Pydantic BaseModel (mutable)
    - Uses ConfigDict(arbitrary_types_allowed=True) for nn.Module field in ModelSpec
    - Uses Field(default_factory=list) for mutable defaults
    - Preserves custom to_dict() methods for nested serialization
    - NON-BREAKING: Same constructor API and attribute access preserved

Author: milia Team
Version: 1.1.0
"""

import logging
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

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


class ModelSpec(BaseModel):
    """
    Specification for a model in ensemble.

    Pydantic V2 Migration (Phase 34):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses ConfigDict(arbitrary_types_allowed=True) for nn.Module field
        - Preserves custom to_dict() for backward compatibility
        - NON-BREAKING: Same constructor API and attribute access

    Attributes:
        model: The actual model instance
        weight: Weight for weighted fusion
        name: Model name (optional)
        level: Hierarchical level (for hierarchical strategy)
    """

    # Allow arbitrary types for nn.Module field
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: nn.Module
    weight: float = 1.0
    name: str | None = None
    level: int = 0  # For hierarchical composition

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name or self.model.__class__.__name__,
            "weight": self.weight,
            "model_class": self.model.__class__.__name__,
            "level": self.level,
        }


class EnsembleConfig(BaseModel):
    """
    Configuration for model ensemble.

    Pydantic V2 Migration (Phase 34):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=list) for mutable defaults
        - Uses ConfigDict(arbitrary_types_allowed=True) for nested ModelSpec with nn.Module
        - Preserves custom to_dict() for nested serialization
        - NON-BREAKING: Same constructor API and attribute access

    Attributes:
        name: Ensemble name
        task_type: Task type
        models: List of model specifications
        strategy: Composition strategy (parallel, sequential, hierarchical)
        fusion: Fusion method (mean, weighted, attention, voting)
    """

    # Allow arbitrary types for ModelSpec containing nn.Module
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    task_type: str
    models: list[ModelSpec] = Field(default_factory=list)
    strategy: str = "parallel"
    fusion: str = "mean"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "task_type": self.task_type,
            "num_models": len(self.models),
            "models": [m.to_dict() for m in self.models],
            "strategy": self.strategy,
            "fusion": self.fusion,
        }


# =============================================================================
# EXCEPTIONS
# =============================================================================


class CompositionError(ModelError):
    """Exception raised for model composition errors."""

    def __init__(
        self,
        message: str,
        strategy: str | None = None,
        num_models: int | None = None,
        details: str | None = None,
    ):
        self.strategy = strategy
        self.num_models = num_models
        super().__init__(message, details=details)

    def __str__(self) -> str:
        msg = self.message
        if self.strategy:
            msg += f" [Strategy: {self.strategy}]"
        if self.num_models is not None:
            msg += f" [Models: {self.num_models}]"
        if self.details:
            msg += f". {self.details}"
        return msg


# =============================================================================
# MODEL COMPOSER
# =============================================================================


class ModelComposer:
    """
    Compose multiple models into ensembles or stacked architectures.

    Strategies:
    - parallel: Multiple models run in parallel, predictions aggregated
    - sequential: Output of model N → input of model N+1
    - hierarchical: Different models at different processing levels

    Fusion Methods:
    - mean: Simple average of predictions
    - weighted: Weighted average using model weights
    - attention: Learned attention weights
    - voting: Majority voting (classification only)

    Usage:
        >>> composer = ModelComposer(task_type="graph_regression")
        >>> composer.add_model(model1, weight=0.6)
        >>> composer.add_model(model2, weight=0.4)
        >>> composer.set_strategy("parallel")
        >>> composer.set_fusion("weighted")
        >>> ensemble = composer.build()
    """

    def __init__(self, task_type: str, name: str = "Ensemble"):
        """
        Initialize model composer.

        Args:
            task_type: Task type (e.g., "graph_regression", "node_classification")
            name: Ensemble name
        """
        self.task_type = task_type
        self.name = name
        self.models: list[ModelSpec] = []
        self.strategy = "parallel"
        self.fusion = "mean"

        logger.debug(f"Initialized ModelComposer: task={task_type}, name={name}")

    # =========================================================================
    # MODEL MANAGEMENT
    # =========================================================================

    def add_model(
        self, model: nn.Module, weight: float = 1.0, name: str | None = None, level: int = 0
    ) -> "ModelComposer":
        """
        Add a model to the ensemble.

        Args:
            model: Model instance
            weight: Weight for weighted fusion (must be positive)
            name: Optional model name for identification
            level: Hierarchical level (0=first, 1=second, etc.) for hierarchical strategy

        Returns:
            Self for method chaining

        Raises:
            TypeError: If model is not nn.Module
            ValueError: If weight is not positive

        Example:
            >>> composer.add_model(gcn_model, weight=0.6, name="GCN")
            >>> composer.add_model(gat_model, weight=0.4, name="GAT")
            >>> # For hierarchical
            >>> composer.add_model(encoder, level=0, name="Encoder")
            >>> composer.add_model(decoder, level=1, name="Decoder")
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be nn.Module instance, got {type(model).__name__}")

        if weight <= 0:
            raise ValueError(f"weight must be positive, got {weight}")

        if level < 0:
            raise ValueError(f"level must be non-negative, got {level}")

        spec = ModelSpec(model=model, weight=weight, name=name, level=level)
        self.models.append(spec)

        model_name = name or model.__class__.__name__
        logger.debug(f"Added model '{model_name}' with weight={weight}, level={level}")
        return self

    def remove_model(self, index: int) -> "ModelComposer":
        """
        Remove a model from the ensemble.

        Args:
            index: Index of model to remove (0-based)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If index is out of range

        Example:
            >>> composer.remove_model(0)  # Remove first model
        """
        if index < 0 or index >= len(self.models):
            raise ValueError(f"Invalid index: {index}. Valid range: 0-{len(self.models) - 1}")

        removed = self.models.pop(index)
        logger.debug(f"Removed model '{removed.name or 'unnamed'}' at index {index}")
        return self

    def clear_models(self) -> "ModelComposer":
        """
        Clear all models from the ensemble.

        Returns:
            Self for method chaining

        Example:
            >>> composer.clear_models()  # Start fresh
        """
        num_models = len(self.models)
        self.models.clear()
        logger.debug(f"Cleared all {num_models} models")
        return self

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def set_strategy(self, strategy: str) -> "ModelComposer":
        """
        Set composition strategy.

        Args:
            strategy: One of "parallel", "sequential", or "hierarchical"
                - parallel: All models run on same input, outputs aggregated
                - sequential: Models chained, output of M[i] → input of M[i+1]
                - hierarchical: Models organized by level (0, 1, 2, ...)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If strategy is not valid

        Example:
            >>> composer.set_strategy("parallel")
            >>> composer.set_strategy("hierarchical")
        """
        valid_strategies = ["parallel", "sequential", "hierarchical"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: '{strategy}'. Must be one of {valid_strategies}")

        self.strategy = strategy
        logger.debug(f"Set strategy to '{strategy}'")
        return self

    def set_fusion(self, method: str) -> "ModelComposer":
        """
        Set fusion method for aggregating predictions.

        Args:
            method: One of "mean", "weighted", "attention", or "voting"
                - mean: Simple arithmetic mean
                - weighted: Weighted average using model weights
                - attention: Learned attention-based weighting
                - voting: Majority vote (classification tasks only)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If method is not valid

        Example:
            >>> composer.set_fusion("weighted")
            >>> composer.set_fusion("attention")
        """
        valid_methods = ["mean", "weighted", "attention", "voting"]
        if method not in valid_methods:
            raise ValueError(f"Invalid fusion method: '{method}'. Must be one of {valid_methods}")

        self.fusion = method
        logger.debug(f"Set fusion method to '{method}'")
        return self

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_composition(self) -> dict[str, Any]:
        """
        Validate the composition configuration.

        Checks:
        - At least one model exists
        - Strategy-specific requirements met
        - Fusion method compatible with task type
        - Weight normalization for weighted fusion
        - Hierarchical level consistency

        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str],
                'suggestions': List[str]
            }

        Example:
            >>> result = composer.validate_composition()
            >>> if not result['valid']:
            ...     for error in result['errors']:
            ...         print(f"Error: {error}")
        """
        errors = []
        warnings = []
        suggestions = []

        # Check if models exist
        if not self.models:
            errors.append("No models added to ensemble")
            suggestions.append("Add at least one model using add_model()")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "suggestions": suggestions,
            }

        # Check strategy-specific requirements
        if self.strategy == "sequential":
            if len(self.models) < 2:
                errors.append(
                    "Sequential strategy requires at least 2 models, "
                    f"but only {len(self.models)} provided"
                )
                suggestions.append("Add more models or change strategy to 'parallel'")

        elif self.strategy == "parallel":
            if len(self.models) < 2:
                warnings.append(
                    "Parallel ensemble with single model is redundant. "
                    "Consider using the model directly."
                )
                suggestions.append("Add more models to benefit from ensemble")

        elif self.strategy == "hierarchical":
            # Check hierarchical level consistency
            levels = [spec.level for spec in self.models]
            unique_levels = sorted(set(levels))

            if len(unique_levels) < 2:
                warnings.append(
                    "Hierarchical strategy with only one level is equivalent to parallel. "
                    f"Found levels: {unique_levels}"
                )
                suggestions.append("Assign different levels to models or use 'parallel' strategy")

            # Check for missing levels
            expected_levels = list(range(max(levels) + 1))
            missing_levels = set(expected_levels) - set(levels)
            if missing_levels:
                warnings.append(
                    f"Hierarchical levels have gaps: missing levels {sorted(missing_levels)}"
                )
                suggestions.append("Assign models to missing levels or adjust level numbers")

            # Check each level has at least one model
            for level in unique_levels:
                models_at_level = [s for s in self.models if s.level == level]
                if len(models_at_level) == 0:
                    errors.append(f"No models assigned to level {level}")

        # Check fusion method compatibility
        if self.fusion == "voting":
            if "classification" not in self.task_type.lower():
                warnings.append(
                    f"Voting fusion is designed for classification tasks, "
                    f"but task type is '{self.task_type}'"
                )
                suggestions.append("Use 'mean' or 'weighted' fusion for regression tasks")

        # Check weight normalization for weighted fusion
        if self.fusion == "weighted":
            total_weight = sum(spec.weight for spec in self.models)
            if abs(total_weight - 1.0) > 0.01:
                warnings.append(
                    f"Weights sum to {total_weight:.3f}, not 1.0. "
                    f"Weights will be normalized automatically."
                )
                suggestions.append("For manual control, ensure weights sum to 1.0")

        # Check attention fusion requirements
        if self.fusion == "attention" and self.strategy == "sequential":
            warnings.append(
                "Attention fusion with sequential strategy may not be meaningful. "
                "Consider using 'parallel' strategy instead."
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
        Build the composed model.

        Creates the appropriate ensemble architecture based on strategy:
        - parallel: ParallelEnsemble
        - sequential: SequentialStack
        - hierarchical: HierarchicalComposition

        Returns:
            Composed model as nn.Module

        Raises:
            CompositionError: If composition validation fails

        Example:
            >>> ensemble = composer.build()
            >>> output = ensemble(x, edge_index, batch=batch)
        """
        # Validate composition
        validation = self.validate_composition()
        if not validation["valid"]:
            error_msg = "Composition validation failed:\n"
            for error in validation["errors"]:
                error_msg += f"  - {error}\n"

            if validation["suggestions"]:
                error_msg += "\nSuggestions:\n"
                for suggestion in validation["suggestions"]:
                    error_msg += f"  - {suggestion}\n"

            raise CompositionError(
                message=error_msg, strategy=self.strategy, num_models=len(self.models)
            )

        # Log warnings
        for warning in validation["warnings"]:
            logger.warning(f"Composition warning: {warning}")

        # Normalize weights for weighted fusion
        if self.fusion == "weighted":
            total_weight = sum(spec.weight for spec in self.models)
            if abs(total_weight - 1.0) > 0.01:
                logger.info(f"Normalizing weights (sum={total_weight:.3f}) to sum to 1.0")
                for spec in self.models:
                    spec.weight /= total_weight

        # Build composed model based on strategy
        if self.strategy == "parallel":
            composed = ParallelEnsemble(
                models=[spec.model for spec in self.models],
                weights=[spec.weight for spec in self.models],
                fusion=self.fusion,
                task_type=self.task_type,
                name=self.name,
            )

        elif self.strategy == "sequential":
            composed = SequentialStack(
                models=[spec.model for spec in self.models],
                task_type=self.task_type,
                name=self.name,
            )

        elif self.strategy == "hierarchical":
            # Group models by level
            levels_dict = {}
            for spec in self.models:
                if spec.level not in levels_dict:
                    levels_dict[spec.level] = []
                levels_dict[spec.level].append(spec)

            composed = HierarchicalComposition(
                levels_dict=levels_dict,
                fusion=self.fusion,
                task_type=self.task_type,
                name=self.name,
            )

        else:
            raise CompositionError(
                f"Unknown strategy: {self.strategy}",
                strategy=self.strategy,
                num_models=len(self.models),
            )

        logger.info(
            f"Built {self.strategy} ensemble '{self.name}' "
            f"with {len(self.models)} models using {self.fusion} fusion"
        )

        return composed

    # =========================================================================
    # CONFIG IMPORT/EXPORT
    # =========================================================================

    def to_config(self) -> EnsembleConfig:
        """
        Export ensemble configuration.

        Note: Model instances are not serializable, so the exported config
        contains model specifications but not the actual models.

        Returns:
            EnsembleConfig instance

        Example:
            >>> config = composer.to_config()
            >>> print(config.to_dict())
        """
        return EnsembleConfig(
            name=self.name,
            task_type=self.task_type,
            models=deepcopy(self.models),
            strategy=self.strategy,
            fusion=self.fusion,
        )

    @classmethod
    def from_config(cls, config: EnsembleConfig | dict[str, Any]) -> "ModelComposer":
        """
        Create composer from configuration.

        Note: Models themselves cannot be serialized, so this only
        recreates the composer structure. Models must be added separately
        using add_model().

        Args:
            config: EnsembleConfig instance or dictionary

        Returns:
            ModelComposer instance

        Example:
            >>> config = composer.to_config()
            >>> new_composer = ModelComposer.from_config(config)
            >>> # Add models manually
            >>> new_composer.add_model(model1, weight=0.5)
        """
        if isinstance(config, dict):
            name = config.get("name", "Ensemble")
            task_type = config.get("task_type", "graph_regression")
            strategy = config.get("strategy", "parallel")
            fusion = config.get("fusion", "mean")
        else:
            name = config.name
            task_type = config.task_type
            strategy = config.strategy
            fusion = config.fusion

        composer = cls(task_type=task_type, name=name)
        composer.set_strategy(strategy)
        composer.set_fusion(fusion)

        logger.debug(f"Created composer from config: {name}")
        return composer

    # =========================================================================
    # UTILITY
    # =========================================================================

    def __len__(self) -> int:
        """Return number of models in ensemble."""
        return len(self.models)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ModelComposer(name='{self.name}', "
            f"task='{self.task_type}', "
            f"strategy='{self.strategy}', "
            f"fusion='{self.fusion}', "
            f"models={len(self.models)})"
        )

    def summary(self) -> str:
        """
        Get detailed summary of the composition.

        Returns:
            Formatted summary string

        Example:
            >>> print(composer.summary())
        """
        lines = [
            "ModelComposer Summary",
            "=" * 60,
            f"Name: {self.name}",
            f"Task Type: {self.task_type}",
            f"Strategy: {self.strategy}",
            f"Fusion: {self.fusion}",
            f"Number of Models: {len(self.models)}",
            "",
            "Models:",
        ]

        for i, spec in enumerate(self.models):
            model_name = spec.name or spec.model.__class__.__name__
            lines.append(f"  [{i}] {model_name}: weight={spec.weight:.3f}, level={spec.level}")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# COMPOSED MODEL CLASSES
# =============================================================================


class ParallelEnsemble(nn.Module):
    """
    Parallel ensemble - multiple models run in parallel on same input.

    All models receive the same input and their predictions are aggregated
    according to the specified fusion method.

    Architecture:
        Input → [Model1, Model2, ..., ModelN] → Fusion → Output
    """

    def __init__(
        self,
        models: list[nn.Module],
        weights: list[float],
        fusion: str,
        task_type: str,
        name: str = "ParallelEnsemble",
    ):
        """
        Initialize parallel ensemble.

        Args:
            models: List of model instances
            weights: List of model weights (should be normalized)
            fusion: Fusion method ('mean', 'weighted', 'attention', 'voting')
            task_type: Task type
            name: Ensemble name
        """
        super().__init__()
        self.name = name
        self.fusion = fusion
        self.task_type = task_type

        # Store models as ModuleList for proper parameter registration
        self.models = nn.ModuleList(models)

        # Store weights as buffer (not trainable by default)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

        # For attention fusion, create attention mechanism
        if fusion == "attention":
            self.attention = nn.Sequential(
                nn.Linear(len(models), len(models)),
                nn.Tanh(),
                nn.Linear(len(models), len(models)),
                nn.Softmax(dim=-1),
            )

        logger.debug(f"Initialized ParallelEnsemble: {len(models)} models, fusion={fusion}")

    def _is_graph_level_task(self) -> bool:
        """
        Determine if the current task requires graph-level predictions.

        Graph-level tasks need global pooling to aggregate node features
        into a single graph representation. This method dynamically checks
        the task_type to support current and future task types.

        Returns:
            True if task_type indicates graph-level prediction needed

        Supports:
            - graph_classification
            - graph_regression
            - Any future task type containing 'graph_' prefix
        """
        if not hasattr(self, "task_type") or self.task_type is None:
            return False

        task_lower = self.task_type.lower()

        # Dynamic detection: any task starting with 'graph_' is graph-level
        # This future-proofs against new task types
        return task_lower.startswith("graph_")

    def _apply_global_pooling(
        self, x: torch.Tensor, batch: torch.Tensor | None, pooling_method: str = "mean"
    ) -> torch.Tensor:
        """
        Apply global pooling to convert node-level to graph-level predictions.

        Uses PyG's native pooling functions for reliability, performance,
        and CUDA compatibility.

        Args:
            x: Node-level features [num_nodes, out_channels]
            batch: Batch assignment tensor [num_nodes] mapping nodes to graphs.
                   If None, assumes single graph.
            pooling_method: Pooling method - 'mean', 'max', or 'add'

        Returns:
            Graph-level features [num_graphs, out_channels]
        """
        from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

        if batch is None:
            # Single graph case - pool all nodes into one representation
            if pooling_method == "mean":
                return x.mean(dim=0, keepdim=True)
            elif pooling_method == "max":
                return x.max(dim=0, keepdim=True)[0]
            elif pooling_method == "add":
                return x.sum(dim=0, keepdim=True)
            else:
                return x.mean(dim=0, keepdim=True)

        # Multiple graphs - use PyG's native pooling
        if pooling_method == "mean":
            return global_mean_pool(x, batch)
        elif pooling_method == "max":
            return global_max_pool(x, batch)
        elif pooling_method == "add":
            return global_add_pool(x, batch)
        else:
            logger.warning(f"Unknown pooling method '{pooling_method}', using 'mean'")
            return global_mean_pool(x, batch)

    def _get_innermost_model(self, model: nn.Module) -> nn.Module:
        """
        Recursively unwrap model wrappers to find the innermost PyG model.

        DYNAMIC: Handles arbitrary nesting of wrappers
        PRODUCTION-READY: Prevents infinite loops with max depth
        FUTURE-PROOF: Works with any wrapper pattern using 'model' attribute

        Args:
            model: The model to unwrap (may be wrapped multiple times)

        Returns:
            The innermost model (actual PyG model like GCN, GAT, etc.)
        """
        current = model
        max_depth = 10  # Prevent infinite loops
        depth = 0

        while depth < max_depth:
            if hasattr(current, "model") and current.model is not current:
                current = current.model
                depth += 1
            else:
                break

        return current

    def _model_supports_edge_attr(self, model: nn.Module) -> bool:
        """
        Check if a model supports multi-dimensional edge attributes (edge_attr).

        PyG BasicGNN models (GCN, GAT, GraphSAGE, GIN, etc.) have class-level
        attributes `supports_edge_attr` and `supports_edge_weight` that indicate
        their edge feature capabilities:

        - GCN: supports_edge_attr=False, supports_edge_weight=True (1D only)
        - GAT: supports_edge_attr=True, supports_edge_weight=False (multi-dim OK)
        - GraphSAGE: supports_edge_attr=False, supports_edge_weight=False
        - GIN: supports_edge_attr=False, supports_edge_weight=False
        - PNA: supports_edge_attr=True, supports_edge_weight=False

        DYNAMIC: Checks actual class attribute from PyG model at runtime
        PRODUCTION-READY: Recursively unwraps wrappers to find actual PyG model
        FUTURE-PROOF: Works with any PyG model following the BasicGNN convention

        Evidence from PyG source (basic_gnn.py):
            class GCN(BasicGNN):
                supports_edge_attr: Final[bool] = False
            class GAT(BasicGNN):
                supports_edge_attr: Final[bool] = True

        Args:
            model: The model to check (may be wrapped)

        Returns:
            True if model supports multi-dimensional edge_attr, False if not,
            True as default if attribute not found (safer to try than fail)
        """
        # Recursively unwrap to find the actual PyG model
        inner_model = self._get_innermost_model(model)

        # Check class attribute (PyG BasicGNN convention)
        # Also check instance attribute for custom models
        supports = getattr(inner_model, "supports_edge_attr", None)

        # Also check the class itself (not just instance)
        if supports is None:
            supports = getattr(type(inner_model), "supports_edge_attr", None)

        if supports is not None:
            return bool(supports)

        # If attribute not found, be conservative:
        # - If model name suggests it doesn't support edge_attr, return False
        # - Otherwise return True (safer to try)
        model_name = type(inner_model).__name__.upper()

        # Models known to NOT support multi-dimensional edge_attr
        # (they use edge_weight which is 1D, causing dimension mismatch with 2D edge_attr)
        non_edge_attr_models = {"GCN", "GRAPHSAGE", "SAGE", "GIN"}

        if model_name in non_edge_attr_models:
            logger.debug(
                f"Model {type(inner_model).__name__} lacks supports_edge_attr attribute, "
                f"but is known to not support multi-dimensional edge_attr"
            )
            return False

        # Default to True for unknown models (let them handle it)
        return True

    def _call_model_with_signature(
        self,
        model: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        edge_attr: torch.Tensor | None,
        batch: torch.Tensor | None,
        edge_label_index: torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Call a model with appropriate arguments based on its forward signature.

        This method enables heterogeneous ensembles containing both standard GNN
        models (GCN, GAT, GraphSAGE) and 3D equivariant models (SchNet, DimeNet).

        DYNAMIC: Introspects model.forward() signature at runtime
        PRODUCTION-READY: Handles wrapper models (GraphLevelModelWrapper)
        FUTURE-PROOF: Works with any model forward signature

        Args:
            model: The model to call
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)
            batch: Batch assignment [num_nodes] (optional)
            edge_label_index: Edge pairs for link prediction (optional)
            **kwargs: Additional arguments (may contain z, pos for 3D models)

        Returns:
            Model output tensor

        Supports:
            - Standard GNN: forward(x, edge_index, batch)
            - Edge feature GNN: forward(x, edge_index, edge_attr, batch)
            - 3D models: forward(z, pos, batch)
            - Wrapped models: Unwraps to check inner model signature
        """
        import inspect

        # Get the actual model (unwrap wrapper if needed) for signature introspection
        inner_model = self._get_innermost_model(model)

        # Log entry with edge_attr status (DEBUG level for production)
        logger.debug(
            f"[DIAGNOSTIC] _call_model_with_signature: model={type(inner_model).__name__}, "
            f"edge_attr={'None' if edge_attr is None else f'shape={list(edge_attr.shape)}'}"
        )

        # ================================================================
        # EDGE FEATURE COMPATIBILITY CHECK
        # ================================================================
        # PyG BasicGNN models have class attributes indicating edge support:
        # - supports_edge_weight (bool): 1D scalar edge weights
        # - supports_edge_attr (bool): Multi-dimensional edge features
        #
        # GCN: supports_edge_weight=True, supports_edge_attr=False
        # GAT: supports_edge_weight=False, supports_edge_attr=True
        # GraphSAGE: supports_edge_weight=False, supports_edge_attr=False
        #
        # Passing edge_attr to models that don't support it causes:
        # "Tensors must have same number of dimensions: got 1 and 2"
        # when add_self_loops tries to concatenate incompatible tensors.
        #
        # DYNAMIC: Checks model's actual supports_edge_attr attribute recursively
        # PRODUCTION-READY: Prevents runtime dimension mismatch errors
        # FUTURE-PROOF: Works with any PyG model following this convention
        # ================================================================

        # Check if edge_attr is multi-dimensional (2D tensor with features)
        edge_attr_is_multidim = (
            edge_attr is not None and edge_attr.dim() >= 2 and edge_attr.size(-1) > 1
        )

        # Check model compatibility with multi-dimensional edge_attr
        if edge_attr_is_multidim:
            model_supports = self._model_supports_edge_attr(model)
            if not model_supports:
                logger.info(
                    f"Model {type(inner_model).__name__} does not support multi-dimensional "
                    f"edge_attr (shape={list(edge_attr.shape)}). Skipping edge_attr to avoid "
                    f"dimension mismatch error in add_self_loops."
                )
                edge_attr = None  # Clear edge_attr for this model

        # Get forward signature parameters
        try:
            sig = inspect.signature(inner_model.forward)
            param_names = [name for name in sig.parameters.keys() if name != "self"]
        except Exception:
            param_names = []

        # Check if this is a 3D model (uses z, pos instead of x, edge_index)
        is_3d_model = "z" in param_names and "pos" in param_names

        if is_3d_model:
            # 3D model signature: forward(z, pos, batch)
            z = kwargs.get("z")
            pos = kwargs.get("pos")

            if z is None or pos is None:
                raise ValueError(
                    f"3D model {type(inner_model).__name__} requires 'z' and 'pos' "
                    f"in kwargs, but got z={z is not None}, pos={pos is not None}. "
                    f"Ensure batch data contains atomic numbers (z) and positions (pos)."
                )

            # Call with 3D signature
            return model(z=z, pos=pos, batch=batch)

        # ================================================================
        # STANDARD GNN SIGNATURE - TRY MULTIPLE STRATEGIES
        # ================================================================
        # Try strategies in order, catching both TypeError (signature mismatch)
        # and RuntimeError (dimension mismatch from add_self_loops).
        # On RuntimeError with edge_attr, retry without edge_attr.
        # ================================================================

        # Strategy 1: Full signature with edge_label_index (edge-level tasks)
        if edge_label_index is not None:
            try:
                if edge_attr is not None and batch is not None:
                    result = model(
                        x,
                        edge_index,
                        edge_attr=edge_attr,
                        batch=batch,
                        edge_label_index=edge_label_index,
                    )
                elif batch is not None:
                    result = model(x, edge_index, batch=batch, edge_label_index=edge_label_index)
                else:
                    result = model(x, edge_index, edge_label_index=edge_label_index)
                # Handle tuple output from variational autoencoders (Fix 30)
                if isinstance(result, tuple):
                    result = result[0]
                return result
            except TypeError:
                pass  # Fall through to next strategy
            except RuntimeError as e:
                # Handle dimension mismatch from add_self_loops
                if "same number of dimensions" in str(e) and edge_attr is not None:
                    logger.warning(
                        f"RuntimeError with edge_attr (likely add_self_loops dimension mismatch): {e}. "
                        f"Retrying without edge_attr."
                    )
                    edge_attr = None  # Clear edge_attr and retry
                else:
                    raise  # Re-raise other RuntimeErrors

        # =================================================================
        # FIX 30: AUTOENCODER ENCODE() METHOD SUPPORT
        # =================================================================
        # GAE/VGAE models have an encode() method that should be used instead
        # of direct forward() calls. The reason:
        # - GAE.forward() returns self.encoder(*args) directly
        # - VGAE.forward() also returns self.encoder(*args) directly (tuple!)
        # - VGAE.encode() performs reparameterization and returns a tensor
        #
        # For ensembles containing autoencoder models, we need to call
        # encode() to get proper tensor outputs instead of tuples.
        #
        # DYNAMIC: Uses hasattr() to detect encode() at runtime
        # PRODUCTION-READY: Works with GAE (tensor) and VGAE (tuple)
        # FUTURE-PROOF: Works with any autoencoder following PyG pattern
        # =================================================================
        use_encode_method = hasattr(model, "encode") and callable(model.encode)

        # Strategy 2: Standard signature with optional edge_attr
        # CRITICAL: PyG BasicGNN.forward() signature is:
        #   forward(x, edge_index, edge_weight=None, edge_attr=None, batch=None, ...)
        # We MUST use keyword arguments for edge_attr and batch to avoid
        # positional argument mismatch (edge_attr going to edge_weight position)
        try:
            # Log which Strategy 2 variant is used (DEBUG level for production)
            if edge_attr is not None and batch is not None:
                logger.debug(
                    "[DIAGNOSTIC] Strategy 2a: model(x, edge_index, edge_attr=edge_attr, batch=batch)"
                )
                if use_encode_method:
                    result = model.encode(x, edge_index, edge_attr=edge_attr, batch=batch)
                else:
                    result = model(x, edge_index, edge_attr=edge_attr, batch=batch)
            elif batch is not None:
                logger.debug("[DIAGNOSTIC] Strategy 2b: model(x, edge_index, batch=batch)")
                if use_encode_method:
                    result = model.encode(x, edge_index, batch=batch)
                else:
                    result = model(x, edge_index, batch=batch)
            elif edge_attr is not None:
                logger.debug("[DIAGNOSTIC] Strategy 2c: model(x, edge_index, edge_attr=edge_attr)")
                if use_encode_method:
                    result = model.encode(x, edge_index, edge_attr=edge_attr)
                else:
                    result = model(x, edge_index, edge_attr=edge_attr)
            else:
                logger.debug("[DIAGNOSTIC] Strategy 2d: model(x, edge_index)")
                if use_encode_method:
                    result = model.encode(x, edge_index)
                else:
                    result = model(x, edge_index)
            # Handle tuple output from variational autoencoders (Fix 30)
            if isinstance(result, tuple):
                logger.debug("[DIAGNOSTIC] Strategy 2: Got tuple output, using first element (mu)")
                result = result[0]
            return result
        except TypeError as te:
            logger.debug(f"[DIAGNOSTIC] Strategy 2 TypeError: {te}")
            pass  # Fall through to next strategy
        except RuntimeError as e:
            # Log RuntimeError details (DEBUG for traceback, WARNING for recovery message)
            import traceback

            logger.debug(f"[DIAGNOSTIC] Strategy 2 RuntimeError: {e}")
            logger.debug(f"[DIAGNOSTIC] edge_attr is None: {edge_attr is None}")
            logger.debug(f"[DIAGNOSTIC] Full traceback:\n{traceback.format_exc()}")
            # Handle dimension mismatch from add_self_loops
            if "same number of dimensions" in str(e) and edge_attr is not None:
                logger.warning(
                    f"RuntimeError with edge_attr (likely add_self_loops dimension mismatch): {e}. "
                    f"Retrying without edge_attr."
                )
                edge_attr = None  # Clear edge_attr and retry without it
            else:
                raise  # Re-raise other RuntimeErrors

        # Strategy 3: Simple signature without edge_attr
        try:
            if batch is not None:
                if use_encode_method:
                    result = model.encode(x, edge_index, batch=batch)
                else:
                    result = model(x, edge_index, batch=batch)
            else:
                if use_encode_method:
                    result = model.encode(x, edge_index)
                else:
                    result = model(x, edge_index)
            # Handle tuple output from variational autoencoders (Fix 30)
            if isinstance(result, tuple):
                result = result[0]
            return result
        except TypeError:
            pass  # Fall through to final fallback

        # Strategy 4: Minimal signature
        try:
            if use_encode_method:
                result = model.encode(x, edge_index)
            else:
                result = model(x, edge_index)
            # Handle tuple output (Fix 30)
            if isinstance(result, tuple):
                result = result[0]
            return result
        except TypeError:
            # Final fallback
            if use_encode_method:
                result = model.encode(x)
            else:
                result = model(x)
            # Handle tuple output (Fix 30)
            if isinstance(result, tuple):
                result = result[0]
            return result

    # -------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        edge_label_index: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through parallel ensemble.

        Supports three task categories:
        - Node-level tasks (node_classification, node_regression): Direct output
        - Graph-level tasks (graph_classification, graph_regression): Global pooling applied
        - Edge-level tasks (link_prediction, edge_regression): Edge decoder applied

        For graph-level tasks, automatically applies global pooling to convert
        node-level model outputs to graph-level predictions before fusion.

        For edge-level tasks, passes edge_label_index to wrapped models
        (EdgeLevelModelWrapper) which compute edge scores from node embeddings.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges] (graph structure)
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)
            batch: Batch assignment [num_nodes] (optional, required for
                   graph-level tasks with multiple graphs)
            edge_label_index: Edge pairs to predict [2, num_edge_pairs] (optional).
                             Required for link_prediction tasks. If None for
                             edge_regression, uses edge_index.
            **kwargs: Additional arguments passed to models
                - pooling_method: str, pooling for graph-level tasks
                                  ('mean', 'max', 'add'). Default: 'mean'

        Returns:
            Aggregated predictions:
            - Node-level tasks: [num_nodes, out_channels]
            - Graph-level tasks: [num_graphs, out_channels]
            - Edge-level tasks: [num_edges] or [num_edge_pairs]
        """
        # ================================================================
        # DATABATCH DETECTION AND EXTRACTION (Fix 26)
        # ================================================================
        # When trainer calls model(batch), the DataBatch object is passed
        # as the first positional argument 'x'. We need to detect this and
        # extract the actual tensors from the batch object.
        #
        # DYNAMIC: Checks if x is a Data/Batch object at runtime
        # PRODUCTION-READY: Handles both direct tensor calls and batch object calls
        # FUTURE-PROOF: Works with any PyG Data/Batch object structure
        # ================================================================
        if hasattr(x, "x") and hasattr(x, "edge_index"):
            # x is actually a DataBatch object, extract tensors
            data_batch = x
            x = data_batch.x
            edge_index = getattr(data_batch, "edge_index", edge_index)
            edge_attr = getattr(data_batch, "edge_attr", edge_attr)
            batch = getattr(data_batch, "batch", batch)
            edge_label_index = getattr(data_batch, "edge_label_index", edge_label_index)

            # Extract 3D model data (z, pos) if present and not already in kwargs
            if "z" not in kwargs and hasattr(data_batch, "z"):
                kwargs["z"] = data_batch.z
            if "pos" not in kwargs and hasattr(data_batch, "pos"):
                kwargs["pos"] = data_batch.pos

            logger.debug(
                f"[ParallelEnsemble] Extracted tensors from DataBatch: "
                f"x={x.shape if x is not None else None}, "
                f"edge_index={edge_index.shape if edge_index is not None else None}, "
                f"batch={batch.shape if batch is not None else None}"
            )

        # Determine if graph-level output is needed
        is_graph_task = self._is_graph_level_task()

        # Extract pooling method from kwargs (don't pass to _call_model_with_signature)
        pooling_method = kwargs.pop("pooling_method", "mean")

        # Calculate number of graphs for shape validation
        num_graphs = 1
        if batch is not None and batch.numel() > 0:
            num_graphs = int(batch.max().item()) + 1

        # Collect predictions from all models
        predictions = []

        for i, model in enumerate(self.models):
            # Log which model is being called and edge_attr status (DEBUG level for production)
            inner = self._get_innermost_model(model)
            logger.debug(
                f"[DIAGNOSTIC] Calling model {i}: {type(inner).__name__}, edge_attr={'None' if edge_attr is None else f'shape={list(edge_attr.shape)}'}"
            )

            try:
                # Use signature-aware model calling for heterogeneous ensembles
                # This handles both standard GNN and 3D models (SchNet, DimeNet)
                pred = self._call_model_with_signature(
                    model=model,
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch,
                    edge_label_index=edge_label_index,
                    **kwargs,
                )
                logger.debug(f"[DIAGNOSTIC] Model {i} ({type(inner).__name__}) forward SUCCEEDED")
            except Exception as e:
                # Log full error with traceback (ERROR for message, DEBUG for traceback)
                import traceback

                logger.error(f"Model {i} ({type(inner).__name__}) forward FAILED: {e}")
                logger.debug(f"[DIAGNOSTIC] Full traceback:\n{traceback.format_exc()}")
                raise
                raise

            # For graph-level tasks: apply pooling if output is node-level
            if is_graph_task and pred.size(0) > num_graphs:
                pred = self._apply_global_pooling(pred, batch, pooling_method)

            predictions.append(pred)
            # -------

        # ================================================================
        # OUTPUT DIMENSION ALIGNMENT (Fix 26b)
        # ================================================================
        # In heterogeneous ensembles, different models may have different
        # native output dimensions:
        # - GCN/GAT: Output channels can be configured via hidden_channels
        # - SchNet/DimeNet: Often hardcoded to output 1 (single property prediction)
        #
        # We need to align output dimensions before stacking. Strategy:
        # 1. Use the MAXIMUM output dimension as reference (preserves more info)
        # 2. Project smaller outputs UP to match the reference dimension
        # 3. Create projection layers dynamically (lazy initialization)
        #
        # DYNAMIC: Creates projections at runtime based on actual output shapes
        # PRODUCTION-READY: Handles any combination of model output shapes
        # FUTURE-PROOF: Works with new models without code changes
        # ================================================================
        if len(predictions) > 1:
            # Find the maximum output dimension (last dim)
            out_dims = [pred.size(-1) for pred in predictions]
            max_out_dim = max(out_dims)

            # Create or reuse projection layers for dimension alignment
            if not hasattr(self, "_output_projections"):
                self._output_projections = nn.ModuleDict()

            aligned_predictions = []
            for i, pred in enumerate(predictions):
                current_dim = pred.size(-1)
                if current_dim != max_out_dim:
                    # Need to project this output to match max dimension
                    proj_key = f"proj_{i}_{current_dim}_to_{max_out_dim}"

                    if proj_key not in self._output_projections:
                        # Create projection layer (lazy initialization)
                        projection = nn.Linear(current_dim, max_out_dim)
                        # Move to same device as prediction
                        projection = projection.to(pred.device)
                        self._output_projections[proj_key] = projection
                        logger.info(
                            f"[ParallelEnsemble] Created output projection for model {i}: "
                            f"{current_dim} -> {max_out_dim}"
                        )

                    # Apply projection
                    pred = self._output_projections[proj_key](pred)
                    logger.debug(
                        f"[ParallelEnsemble] Projected model {i} output: "
                        f"{current_dim} -> {max_out_dim}"
                    )

                aligned_predictions.append(pred)

            predictions = aligned_predictions

        # Stack predictions (now all same dimension)
        predictions = torch.stack(predictions, dim=0)

        # Aggregate based on fusion method
        if self.fusion == "mean":
            output = predictions.mean(dim=0)

        elif self.fusion == "weighted":
            # Handle both 2D and 3D tensors
            if predictions.dim() == 3:
                weights = self.weights.view(-1, 1, 1)
            else:
                weights = self.weights.view(-1, 1)
            output = (predictions * weights).sum(dim=0)

        elif self.fusion == "attention":
            if predictions.dim() < 3:
                predictions = predictions.unsqueeze(-1)

            batch_size = predictions.size(1)

            preds_reshaped = predictions.permute(1, 0, 2)
            preds_mean = preds_reshaped.mean(dim=2)
            attention_weights = self.attention(preds_mean)
            attention_weights = attention_weights.unsqueeze(2)
            output = (preds_reshaped * attention_weights).sum(dim=1)

        elif self.fusion == "voting":
            if predictions.dim() < 3:
                raise ValueError("Voting requires 3D predictions")

            # ================================================================
            # VOTING FUSION: Differentiable during training, hard during inference
            # ================================================================
            # Hard voting (argmax + mode) is NOT differentiable - it breaks
            # gradient flow because argmax returns indices without gradients.
            #
            # Solution:
            # - Training (self.training=True): Use SOFT voting (average logits/probs)
            #   This is differentiable and allows backpropagation through all models
            # - Inference (self.training=False): Use HARD voting (majority vote)
            #   This gives discrete class predictions typical of voting ensembles
            #
            # This approach follows scikit-learn's VotingClassifier pattern where
            # soft voting averages probabilities and hard voting uses majority rule.
            # ================================================================

            if self.training:
                # SOFT VOTING (training): Average predictions across models
                # predictions shape: [num_models, batch_size, num_classes]
                # Apply softmax to get proper probability distributions per model
                # Then average across models - fully differentiable
                probs = torch.softmax(predictions, dim=2)  # Normalize per model
                output = probs.mean(dim=0)  # Average across models [batch_size, num_classes]
            else:
                # HARD VOTING (inference): Majority vote from predicted classes
                # Each model votes for its predicted class (argmax)
                # Final prediction is the class with most votes (mode)
                votes = torch.argmax(predictions, dim=2)  # [num_models, batch_size]
                output, _ = torch.mode(votes, dim=0)  # [batch_size]
                num_classes = predictions.size(2)
                output = torch.nn.functional.one_hot(output, num_classes=num_classes).float()

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion}")

        return output

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.name}(models={len(self.models)}, fusion='{self.fusion}')"


class SequentialStack(nn.Module):
    """
    Sequential stack - models connected in series.

    Output of model N becomes input of model N+1, creating a pipeline.
    For graph-level tasks, automatically applies global pooling to the
    final model's output if it produces node-level predictions.

    Supports heterogeneous model ensembles containing both standard GNN
    models (GCN, GAT, GraphSAGE) and 3D equivariant models (SchNet, DimeNet).

    Architecture:
        Input → Model1 → Model2 → ... → ModelN → [GlobalPooling] → Output

    For heterogeneous ensembles (mixing 2D and 3D models):
        - 2D models (GCN, GAT): Use x, edge_index, batch - process embeddings sequentially
        - 3D models (SchNet, DimeNet): Use z, pos, batch - receive ORIGINAL atomic data

    This design allows 3D models to operate on the original molecular structure
    rather than transformed embeddings, which is semantically correct since 3D
    models require atomic numbers (z) and 3D coordinates (pos).
    """

    def __init__(
        self, models: list[nn.Module], task_type: str | None = None, name: str = "SequentialStack"
    ):
        """
        Initialize sequential stack.

        Args:
            models: List of model instances (in order)
            task_type: Task type (e.g., "graph_regression", "node_classification").
                       Used to determine if global pooling should be applied to
                       the final output for graph-level tasks.
            name: Stack name
        """
        super().__init__()
        self.name = name
        self.task_type = task_type
        self.models = nn.ModuleList(models)

        logger.debug(f"Initialized SequentialStack: {len(models)} models, task_type={task_type}")

    def _is_graph_level_task(self) -> bool:
        """
        Determine if the current task requires graph-level predictions.

        Graph-level tasks need global pooling to aggregate node features
        into a single graph representation.

        Returns:
            True if task_type indicates graph-level prediction needed
        """
        if not hasattr(self, "task_type") or self.task_type is None:
            return False

        task_lower = self.task_type.lower()
        return task_lower.startswith("graph_")

    def _apply_global_pooling(
        self, x: torch.Tensor, batch: torch.Tensor | None, pooling_method: str = "mean"
    ) -> torch.Tensor:
        """
        Apply global pooling to convert node-level to graph-level predictions.

        Uses PyG's native pooling functions for reliability and CUDA compatibility.

        Args:
            x: Node-level features [num_nodes, out_channels]
            batch: Batch assignment tensor [num_nodes] mapping nodes to graphs.
                   If None, assumes single graph.
            pooling_method: Pooling method - 'mean', 'max', or 'add'

        Returns:
            Graph-level features [num_graphs, out_channels]
        """
        from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

        if batch is None:
            # Single graph case - pool all nodes into one representation
            if pooling_method == "mean":
                return x.mean(dim=0, keepdim=True)
            elif pooling_method == "max":
                return x.max(dim=0, keepdim=True)[0]
            elif pooling_method == "add":
                return x.sum(dim=0, keepdim=True)
            else:
                return x.mean(dim=0, keepdim=True)

        # Multiple graphs - use PyG's native pooling
        if pooling_method == "mean":
            return global_mean_pool(x, batch)
        elif pooling_method == "max":
            return global_max_pool(x, batch)
        elif pooling_method == "add":
            return global_add_pool(x, batch)
        else:
            logger.warning(f"Unknown pooling method '{pooling_method}', using 'mean'")
            return global_mean_pool(x, batch)

    def _get_innermost_model(self, model: nn.Module) -> nn.Module:
        """
        Recursively unwrap model wrappers to find the innermost PyG model.

        DYNAMIC: Handles arbitrary nesting of wrappers
        PRODUCTION-READY: Prevents infinite loops with max depth
        FUTURE-PROOF: Works with any wrapper pattern using 'model' attribute

        Args:
            model: The model to unwrap (may be wrapped multiple times)

        Returns:
            The innermost model (actual PyG model like GCN, GAT, SchNet, etc.)
        """
        current = model
        max_depth = 10  # Prevent infinite loops
        depth = 0

        while depth < max_depth:
            if hasattr(current, "model") and current.model is not current:
                current = current.model
                depth += 1
            else:
                break

        return current

    def _model_supports_edge_attr(self, model: nn.Module) -> bool:
        """
        Check if a model supports multi-dimensional edge attributes (edge_attr).

        PyG BasicGNN models (GCN, GAT, GraphSAGE, GIN, etc.) have class-level
        attributes `supports_edge_attr` and `supports_edge_weight` that indicate
        their edge feature capabilities:

        - GCN: supports_edge_attr=False, supports_edge_weight=True (1D only)
        - GAT: supports_edge_attr=True, supports_edge_weight=False (multi-dim OK)
        - GraphSAGE: supports_edge_attr=False, supports_edge_weight=False
        - GIN: supports_edge_attr=False, supports_edge_weight=False
        - PNA: supports_edge_attr=True, supports_edge_weight=False

        DYNAMIC: Checks actual class attribute from PyG model at runtime
        PRODUCTION-READY: Recursively unwraps wrappers to find actual PyG model
        FUTURE-PROOF: Works with any PyG model following the BasicGNN convention

        Args:
            model: The model to check (may be wrapped)

        Returns:
            True if model supports multi-dimensional edge_attr, False if not,
            True as default if attribute not found (safer to try than fail)
        """
        # Recursively unwrap to find the actual PyG model
        inner_model = self._get_innermost_model(model)

        # Check class attribute (PyG BasicGNN convention)
        # Also check instance attribute for custom models
        supports = getattr(inner_model, "supports_edge_attr", None)

        # Also check the class itself (not just instance)
        if supports is None:
            supports = getattr(type(inner_model), "supports_edge_attr", None)

        if supports is not None:
            return bool(supports)

        # If attribute not found, be conservative:
        # - If model name suggests it doesn't support edge_attr, return False
        # - Otherwise return True (safer to try)
        model_name = type(inner_model).__name__.upper()

        # Models known to NOT support multi-dimensional edge_attr
        non_edge_attr_models = {"GCN", "GRAPHSAGE", "SAGE", "GIN"}

        if model_name in non_edge_attr_models:
            logger.debug(
                f"Model {type(inner_model).__name__} lacks supports_edge_attr attribute, "
                f"but is known to not support multi-dimensional edge_attr"
            )
            return False

        # Default to True for unknown models (let them handle it)
        return True

    def _is_3d_model(self, model: nn.Module) -> bool:
        """
        Check if a model is a 3D equivariant model (uses z, pos instead of x, edge_index).

        3D models like SchNet, DimeNet, DimeNetPlusPlus use atomic numbers (z) and
        3D coordinates (pos) rather than node features (x) and edge structure.

        DYNAMIC: Introspects model.forward() signature at runtime
        PRODUCTION-READY: Handles wrapped models by unwrapping first
        FUTURE-PROOF: Works with any model by checking parameter names

        Args:
            model: The model to check (may be wrapped)

        Returns:
            True if model uses 3D signature (z, pos, batch), False otherwise
        """
        import inspect

        inner_model = self._get_innermost_model(model)

        try:
            sig = inspect.signature(inner_model.forward)
            param_names = [name for name in sig.parameters.keys() if name != "self"]
        except Exception:
            param_names = []

        # 3D models have 'z' and 'pos' parameters
        return "z" in param_names and "pos" in param_names

    def _call_model_with_signature(
        self,
        model: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        edge_attr: torch.Tensor | None,
        batch: torch.Tensor | None,
        edge_label_index: torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Call a model with appropriate arguments based on its forward signature.

        This method enables heterogeneous sequential stacks containing both standard
        GNN models (GCN, GAT, GraphSAGE) and 3D equivariant models (SchNet, DimeNet).

        DYNAMIC: Introspects model.forward() signature at runtime
        PRODUCTION-READY: Handles wrapper models (GraphLevelModelWrapper)
        FUTURE-PROOF: Works with any model forward signature

        Args:
            model: The model to call
            x: Node features [num_nodes, in_channels] (for 2D models)
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)
            batch: Batch assignment [num_nodes] (optional)
            edge_label_index: Edge pairs for link prediction (optional)
            **kwargs: Additional arguments (may contain z, pos for 3D models)

        Returns:
            Model output tensor

        Supports:
            - Standard GNN: forward(x, edge_index, batch)
            - Edge feature GNN: forward(x, edge_index, edge_attr, batch)
            - 3D models: forward(z, pos, batch)
            - Wrapped models: Unwraps to check inner model signature
        """
        import inspect

        # Get the actual model (unwrap wrapper if needed) for signature introspection
        inner_model = self._get_innermost_model(model)

        # Log entry with edge_attr status (DEBUG level for production)
        logger.debug(
            f"[SequentialStack] _call_model_with_signature: model={type(inner_model).__name__}, "
            f"edge_attr={'None' if edge_attr is None else f'shape={list(edge_attr.shape)}'}"
        )

        # ================================================================
        # EDGE FEATURE COMPATIBILITY CHECK
        # ================================================================
        # Check if edge_attr is multi-dimensional (2D tensor with features)
        edge_attr_is_multidim = (
            edge_attr is not None and edge_attr.dim() >= 2 and edge_attr.size(-1) > 1
        )

        # Check model compatibility with multi-dimensional edge_attr
        if edge_attr_is_multidim:
            model_supports = self._model_supports_edge_attr(model)
            if not model_supports:
                logger.info(
                    f"Model {type(inner_model).__name__} does not support multi-dimensional "
                    f"edge_attr (shape={list(edge_attr.shape)}). Skipping edge_attr to avoid "
                    f"dimension mismatch error in add_self_loops."
                )
                edge_attr = None  # Clear edge_attr for this model

        # Get forward signature parameters
        try:
            sig = inspect.signature(inner_model.forward)
            param_names = [name for name in sig.parameters.keys() if name != "self"]
        except Exception:
            param_names = []

        # Check if this is a 3D model (uses z, pos instead of x, edge_index)
        is_3d_model = "z" in param_names and "pos" in param_names

        if is_3d_model:
            # 3D model signature: forward(z, pos, batch)
            # 3D models need ORIGINAL atomic data, not transformed embeddings
            z = kwargs.get("z")
            pos = kwargs.get("pos")

            if z is None or pos is None:
                raise ValueError(
                    f"3D model {type(inner_model).__name__} requires 'z' and 'pos' "
                    f"in kwargs, but got z={z is not None}, pos={pos is not None}. "
                    f"Ensure batch data contains atomic numbers (z) and positions (pos)."
                )

            logger.debug(
                f"[SequentialStack] Calling 3D model {type(inner_model).__name__} with z, pos, batch"
            )
            # Call with 3D signature
            return model(z=z, pos=pos, batch=batch)

        # ================================================================
        # STANDARD GNN SIGNATURE - TRY MULTIPLE STRATEGIES
        # ================================================================

        # Strategy 1: Full signature with edge_label_index (edge-level tasks)
        if edge_label_index is not None:
            try:
                if edge_attr is not None and batch is not None:
                    return model(
                        x,
                        edge_index,
                        edge_attr=edge_attr,
                        batch=batch,
                        edge_label_index=edge_label_index,
                    )
                elif batch is not None:
                    return model(x, edge_index, batch=batch, edge_label_index=edge_label_index)
                else:
                    return model(x, edge_index, edge_label_index=edge_label_index)
            except TypeError:
                pass  # Fall through to next strategy
            except RuntimeError as e:
                # Handle dimension mismatch from add_self_loops
                if "same number of dimensions" in str(e) and edge_attr is not None:
                    logger.warning(
                        f"RuntimeError with edge_attr (likely add_self_loops dimension mismatch): {e}. "
                        f"Retrying without edge_attr."
                    )
                    edge_attr = None  # Clear edge_attr and retry
                else:
                    raise  # Re-raise other RuntimeErrors

        # =================================================================
        # FIX 30: AUTOENCODER ENCODE() METHOD SUPPORT
        # =================================================================
        # GAE/VGAE models have an encode() method that should be used instead
        # of direct forward() calls. This ensures proper reparameterization
        # for variational autoencoders that return (mu, logstd) tuples.
        #
        # DYNAMIC: Uses hasattr() to detect encode() at runtime
        # PRODUCTION-READY: Works with GAE (tensor) and VGAE (tuple)
        # FUTURE-PROOF: Works with any autoencoder following PyG pattern
        # =================================================================
        use_encode_method = hasattr(model, "encode") and callable(model.encode)

        # Strategy 2: Standard signature with optional edge_attr
        try:
            if edge_attr is not None and batch is not None:
                logger.debug(
                    "[SequentialStack] Strategy 2a: model(x, edge_index, edge_attr=edge_attr, batch=batch)"
                )
                if use_encode_method:
                    result = model.encode(x, edge_index, edge_attr=edge_attr, batch=batch)
                else:
                    result = model(x, edge_index, edge_attr=edge_attr, batch=batch)
            elif batch is not None:
                logger.debug("[SequentialStack] Strategy 2b: model(x, edge_index, batch=batch)")
                if use_encode_method:
                    result = model.encode(x, edge_index, batch=batch)
                else:
                    result = model(x, edge_index, batch=batch)
            elif edge_attr is not None:
                logger.debug(
                    "[SequentialStack] Strategy 2c: model(x, edge_index, edge_attr=edge_attr)"
                )
                if use_encode_method:
                    result = model.encode(x, edge_index, edge_attr=edge_attr)
                else:
                    result = model(x, edge_index, edge_attr=edge_attr)
            else:
                logger.debug("[SequentialStack] Strategy 2d: model(x, edge_index)")
                if use_encode_method:
                    result = model.encode(x, edge_index)
                else:
                    result = model(x, edge_index)
            # Handle tuple output from variational autoencoders (Fix 30)
            if isinstance(result, tuple):
                logger.debug(
                    "[SequentialStack] Strategy 2: Got tuple output, using first element (mu)"
                )
                result = result[0]
            return result
        except TypeError as te:
            logger.debug(f"[SequentialStack] Strategy 2 TypeError: {te}")
            pass  # Fall through to next strategy
        except RuntimeError as e:
            # Handle dimension mismatch from add_self_loops
            if "same number of dimensions" in str(e) and edge_attr is not None:
                logger.warning(
                    f"RuntimeError with edge_attr (likely add_self_loops dimension mismatch): {e}. "
                    f"Retrying without edge_attr."
                )
                edge_attr = None  # Clear edge_attr and retry without it
            else:
                raise  # Re-raise other RuntimeErrors

        # Strategy 3: Simple signature without edge_attr
        try:
            if batch is not None:
                if use_encode_method:
                    result = model.encode(x, edge_index, batch=batch)
                else:
                    result = model(x, edge_index, batch=batch)
            else:
                if use_encode_method:
                    result = model.encode(x, edge_index)
                else:
                    result = model(x, edge_index)
            # Handle tuple output from variational autoencoders (Fix 30)
            if isinstance(result, tuple):
                result = result[0]
            return result
        except TypeError:
            pass  # Fall through to final fallback

        # Strategy 4: Minimal signature
        try:
            if use_encode_method:
                result = model.encode(x, edge_index)
            else:
                result = model(x, edge_index)
            # Handle tuple output (Fix 30)
            if isinstance(result, tuple):
                result = result[0]
            return result
        except TypeError:
            # Final fallback
            if use_encode_method:
                result = model.encode(x)
            else:
                result = model(x)
            # Handle tuple output (Fix 30)
            if isinstance(result, tuple):
                result = result[0]
            return result

    # -------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        edge_label_index: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through sequential stack with heterogeneous model support.

        Each model processes based on its signature type:
        - 2D GNN models (GCN, GAT, etc.): Receive current embeddings and graph structure
        - 3D models (SchNet, DimeNet): Receive ORIGINAL atomic numbers (z) and positions (pos)

        This design handles the semantic difference between 2D and 3D models:
        - 2D models transform node features while preserving graph structure
        - 3D models require original atomic data (cannot use transformed embeddings)

        For graph-level tasks, automatically applies global pooling to the final output.
        For edge-level tasks, passes edge_label_index to the final model.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)
            batch: Batch assignment [num_nodes] (optional, required for
                   graph-level tasks with multiple graphs)
            edge_label_index: Edge pairs to predict [2, num_edge_pairs] (optional).
                             For edge-level tasks, passed to the final model.
            **kwargs: Additional arguments passed to models
                - pooling_method: str, pooling for graph-level tasks
                                  ('mean', 'max', 'add'). Default: 'mean'
                - z: torch.Tensor, atomic numbers for 3D models
                - pos: torch.Tensor, 3D coordinates for 3D models

        Returns:
            Final predictions:
            - Node-level tasks: [num_nodes, out_channels]
            - Graph-level tasks: [num_graphs, out_channels]
            - Edge-level tasks: [num_edges] or [num_edge_pairs]
        """
        # ================================================================
        # DATABATCH DETECTION AND EXTRACTION (Fix 26)
        # ================================================================
        # When trainer calls model(batch), the DataBatch object is passed
        # as the first positional argument 'x'. We need to detect this and
        # extract the actual tensors from the batch object.
        #
        # DYNAMIC: Checks if x is a Data/Batch object at runtime
        # PRODUCTION-READY: Handles both direct tensor calls and batch object calls
        # FUTURE-PROOF: Works with any PyG Data/Batch object structure
        # ================================================================
        if hasattr(x, "x") and hasattr(x, "edge_index"):
            # x is actually a DataBatch object, extract tensors
            data_batch = x
            x = data_batch.x
            edge_index = getattr(data_batch, "edge_index", edge_index)
            edge_attr = getattr(data_batch, "edge_attr", edge_attr)
            batch = getattr(data_batch, "batch", batch)
            edge_label_index = getattr(data_batch, "edge_label_index", edge_label_index)

            # Extract 3D model data (z, pos) if present and not already in kwargs
            if "z" not in kwargs and hasattr(data_batch, "z"):
                kwargs["z"] = data_batch.z
            if "pos" not in kwargs and hasattr(data_batch, "pos"):
                kwargs["pos"] = data_batch.pos

            logger.debug(
                f"[SequentialStack] Extracted tensors from DataBatch: "
                f"x={x.shape if x is not None else None}, "
                f"edge_index={edge_index.shape if edge_index is not None else None}, "
                f"batch={batch.shape if batch is not None else None}"
            )

        # Determine if graph-level output is needed
        is_graph_task = self._is_graph_level_task()

        # Extract pooling method from kwargs (don't pass to _call_model_with_signature)
        pooling_method = kwargs.pop("pooling_method", "mean")

        # Calculate number of graphs for shape validation
        num_graphs = 1
        if batch is not None and batch.numel() > 0:
            num_graphs = int(batch.max().item()) + 1

        current = x
        num_models = len(self.models)

        for i, model in enumerate(self.models):
            is_last_model = i == num_models - 1
            inner_model = self._get_innermost_model(model)

            logger.debug(
                f"[SequentialStack] Processing model {i}: {type(inner_model).__name__}, "
                f"is_3d={self._is_3d_model(model)}"
            )

            try:
                # Use signature-aware model calling for heterogeneous ensembles
                # This handles both standard GNN (x, edge_index) and 3D models (z, pos)

                # Prepare edge_label_index for edge-level tasks (only for last model)
                current_edge_label_index = edge_label_index if is_last_model else None

                # Call model with appropriate signature
                current = self._call_model_with_signature(
                    model=model,
                    x=current,  # 2D models use current embeddings; 3D models use z from kwargs
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch,
                    edge_label_index=current_edge_label_index,
                    **kwargs,
                )

                logger.debug(
                    f"[SequentialStack] Model {i} ({type(inner_model).__name__}) "
                    f"output shape: {current.shape}"
                )

            except Exception as e:
                # Log full error with traceback
                import traceback

                logger.error(
                    f"[SequentialStack] Model {i} ({type(inner_model).__name__}) forward FAILED: {e}"
                )
                logger.debug(f"[SequentialStack] Full traceback:\n{traceback.format_exc()}")
                raise

        # For graph-level tasks: apply pooling if final output is node-level
        if is_graph_task and current.size(0) > num_graphs:
            current = self._apply_global_pooling(current, batch, pooling_method)

        return current
        # -------

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.name}(models={len(self.models)})"


class HierarchicalComposition(nn.Module):
    """
    Hierarchical composition - models organized by processing level.

    Different models operate at different levels of the hierarchy.
    Each level can have multiple models in parallel, and levels are
    connected sequentially.

    Architecture:
        Input → Level0[Model1, Model2, ...] → Fusion →
                Level1[Model3, Model4, ...] → Fusion →
                ... → Output
    """

    def __init__(
        self,
        levels_dict: dict[int, list[ModelSpec]],
        fusion: str,
        task_type: str,
        name: str = "HierarchicalComposition",
    ):
        """
        Initialize hierarchical composition.

        Args:
            levels_dict: Dictionary mapping level → list of ModelSpecs
            fusion: Fusion method for combining models within each level
            task_type: Task type
            name: Composition name
        """
        super().__init__()
        self.name = name
        self.fusion = fusion
        self.task_type = task_type

        # Sort levels
        self.levels = sorted(levels_dict.keys())

        # Create parallel ensemble for each level
        self.level_ensembles = nn.ModuleList()

        for level in self.levels:
            specs = levels_dict[level]
            models = [spec.model for spec in specs]
            weights = [spec.weight for spec in specs]

            # Normalize weights for this level
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            # Create parallel ensemble for this level
            ensemble = ParallelEnsemble(
                models=models,
                weights=weights,
                fusion=fusion,
                task_type=task_type,
                name=f"{name}_Level{level}",
            )

            self.level_ensembles.append(ensemble)

        logger.debug(
            f"Initialized HierarchicalComposition: {len(self.levels)} levels, fusion={fusion}"
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        edge_label_index: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through hierarchical composition.

        Processes input through each level sequentially, where each level
        may contain multiple models in parallel.

        For edge-level tasks, edge_label_index is passed only to the LAST level,
        as intermediate levels produce node embeddings.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)
            batch: Batch assignment [num_nodes] (optional)
            edge_label_index: Edge pairs to predict [2, num_edge_pairs] (optional).
                             For edge-level tasks, passed to the last level.
            **kwargs: Additional arguments passed to models

        Returns:
            Final predictions from last level:
            - Node-level tasks: [num_nodes, out_channels]
            - Graph-level tasks: [num_graphs, out_channels]
            - Edge-level tasks: [num_edges] or [num_edge_pairs]
        """
        # ================================================================
        # DATABATCH DETECTION AND EXTRACTION (Fix 26)
        # ================================================================
        # When trainer calls model(batch), the DataBatch object is passed
        # as the first positional argument 'x'. We need to detect this and
        # extract the actual tensors from the batch object.
        #
        # DYNAMIC: Checks if x is a Data/Batch object at runtime
        # PRODUCTION-READY: Handles both direct tensor calls and batch object calls
        # FUTURE-PROOF: Works with any PyG Data/Batch object structure
        # ================================================================
        if hasattr(x, "x") and hasattr(x, "edge_index"):
            # x is actually a DataBatch object, extract tensors
            data_batch = x
            x = data_batch.x
            edge_index = getattr(data_batch, "edge_index", edge_index)
            edge_attr = getattr(data_batch, "edge_attr", edge_attr)
            batch = getattr(data_batch, "batch", batch)
            edge_label_index = getattr(data_batch, "edge_label_index", edge_label_index)

            # Extract 3D model data (z, pos) if present and not already in kwargs
            if "z" not in kwargs and hasattr(data_batch, "z"):
                kwargs["z"] = data_batch.z
            if "pos" not in kwargs and hasattr(data_batch, "pos"):
                kwargs["pos"] = data_batch.pos

            logger.debug(
                f"[HierarchicalComposition] Extracted tensors from DataBatch: "
                f"x={x.shape if x is not None else None}, "
                f"edge_index={edge_index.shape if edge_index is not None else None}, "
                f"batch={batch.shape if batch is not None else None}"
            )

        current = x
        num_levels = len(self.level_ensembles)

        # Process through each level sequentially
        for level_idx, ensemble in enumerate(self.level_ensembles):
            is_last_level = level_idx == num_levels - 1

            # Pass edge_label_index only to the last level for edge-level tasks
            if is_last_level and edge_label_index is not None:
                current = ensemble(
                    current,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch,
                    edge_label_index=edge_label_index,
                    **kwargs,
                )
            else:
                current = ensemble(
                    current, edge_index=edge_index, edge_attr=edge_attr, batch=batch, **kwargs
                )

        return current

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.name}(levels={len(self.levels)}, fusion='{self.fusion}')"


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info("model_composer module loaded")
