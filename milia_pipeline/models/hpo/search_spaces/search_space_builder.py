"""
Search Space Builder Module

Provides utilities for building, validating, and manipulating hyperparameter
search spaces for HPO optimization.

This module offers:
- SearchSpaceBuilder class for constructing search spaces programmatically
- **DYNAMIC** search spaces for ANY PyG model via runtime introspection
- Validation utilities for search space configurations
- Integration with loss function and scheduler registries
- Conversion between dict and SearchSpaceParamConfig formats

**Phase 6 Migration (2025-12-08)**:
- `for_model()` now uses dynamic PyG introspection from `pyg_introspector`
- `list_available_models()` returns ALL dynamically discovered PyG models
- Hardcoded model-specific methods (_build_gcn_space, etc.) deprecated as fallback
- No more static 7-model limit - supports ALL PyG models automatically

**Pydantic V2 Migration (Phase 37)**:
- Removed unused `from dataclasses import dataclass, field` import (dead code)
- No dataclasses exist in this module - SearchSpaceBuilder is a regular class
- Uses SearchSpaceParamConfig from param_types.py (already Pydantic BaseModel)
- NON-BREAKING: No API changes; dead code removal only

Location: milia_pipeline/models/hpo/search_spaces/search_space_builder.py
Pattern: Follows registry patterns from loss_functions.py, schedulers.py, optimizers.py
Dependencies: param_types.py, hpo_config.py, exceptions.py, pyg_introspector.py

Author: Milia Team
Version: 2.1.0
"""

import logging
from copy import deepcopy
from typing import Any

from .param_types import ParamType, SearchSpaceParamConfig

try:
    from milia_pipeline.exceptions import ConfigurationError, SearchSpaceError
except ImportError:

    class SearchSpaceError(Exception):
        """Exception for search space errors."""

        def __init__(
            self,
            message: str,
            parameter_name: str | None = None,
            parameter_config: dict[str, Any] | None = None,
            **kwargs,
        ):
            super().__init__(message)
            self.parameter_name = parameter_name
            self.parameter_config = parameter_config

    class ConfigurationError(Exception):
        """Exception for configuration errors."""

        pass


# Phase 6: Import dynamic introspection for search space generation
try:
    from milia_pipeline.models.registry.pyg_introspector import (
        PyGModelIntrospector,
        get_introspector,
    )

    _INTROSPECTOR_AVAILABLE = True
except ImportError:
    get_introspector = None
    PyGModelIntrospector = None
    _INTROSPECTOR_AVAILABLE = False


logger = logging.getLogger(__name__)


# =============================================================================
# SEARCH SPACE BUILDER CLASS
# =============================================================================


class SearchSpaceBuilder:
    """
    Builder class for constructing hyperparameter search spaces.

    Provides a fluent interface for building search spaces programmatically,
    with validation, predefined templates, and integration with existing
    registries (loss functions, schedulers, optimizers).

    Pattern: Follows builder pattern similar to model builders

    Attributes:
        _search_space: Internal dict storing search space configuration
        _categories: Set of valid category names

    Usage:
        >>> # Fluent builder pattern
        >>> builder = SearchSpaceBuilder()
        >>> search_space = (
        ...     builder
        ...     .add_int("hidden_channels", 32, 256, step=32, category="hyperparameters")
        ...     .add_loguniform("lr", 1e-5, 1e-2, category="optimizer")
        ...     .add_categorical("activation", ["relu", "gelu", "elu"], category="hyperparameters")
        ...     .build()
        ... )
        >>>
        >>> # From predefined template
        >>> search_space = SearchSpaceBuilder.for_model("GAT")
        >>>
        >>> # Merge multiple spaces
        >>> combined = SearchSpaceBuilder.merge(space1, space2)
    """

    # Valid category names for organizing parameters
    VALID_CATEGORIES = frozenset(
        [
            "hyperparameters",
            "model",
            "optimizer",
            "scheduler",
            "loss",
            "training",
            "architecture",
        ]
    )

    def __init__(self):
        """Initialize empty search space builder."""
        self._search_space: dict[str, dict[str, SearchSpaceParamConfig]] = {}
        self._frozen = False

    def _ensure_not_frozen(self) -> None:
        """Raise error if builder has been frozen (build() called)."""
        if self._frozen:
            raise SearchSpaceError(
                "Cannot modify search space after build() has been called. "
                "Create a new SearchSpaceBuilder instance."
            )

    def _ensure_category(self, category: str) -> None:
        """Ensure category exists in search space."""
        if category not in self._search_space:
            self._search_space[category] = {}

    def _validate_category(self, category: str) -> None:
        """Validate category name."""
        if category not in self.VALID_CATEGORIES:
            logger.warning(
                f"Non-standard category '{category}'. "
                f"Standard categories: {sorted(self.VALID_CATEGORIES)}"
            )

    # =========================================================================
    # FLUENT BUILDER METHODS
    # =========================================================================

    def add_int(
        self,
        name: str,
        low: int,
        high: int,
        step: int | None = None,
        category: str = "hyperparameters",
    ) -> "SearchSpaceBuilder":
        """
        Add integer parameter to search space.

        Args:
            name: Parameter name
            low: Lower bound (inclusive)
            high: Upper bound (inclusive)
            step: Step size (optional)
            category: Category to add parameter to

        Returns:
            self for method chaining

        Example:
            >>> builder.add_int("hidden_channels", 32, 256, step=32)
            >>> builder.add_int("num_layers", 2, 6)
        """
        self._ensure_not_frozen()
        self._validate_category(category)
        self._ensure_category(category)

        config = SearchSpaceParamConfig(
            type=ParamType.INT, low=float(low), high=float(high), step=step
        )

        self._search_space[category][name] = config
        logger.debug(f"Added int param '{category}.{name}': [{low}, {high}], step={step}")

        return self

    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
        category: str = "hyperparameters",
    ) -> "SearchSpaceBuilder":
        """
        Add float parameter to search space.

        Args:
            name: Parameter name
            low: Lower bound
            high: Upper bound
            log: Whether to use log scale
            category: Category to add parameter to

        Returns:
            self for method chaining

        Example:
            >>> builder.add_float("dropout", 0.0, 0.5)
            >>> builder.add_float("temperature", 0.01, 1.0, log=True)
        """
        self._ensure_not_frozen()
        self._validate_category(category)
        self._ensure_category(category)

        config = SearchSpaceParamConfig(type=ParamType.FLOAT, low=low, high=high, log=log)

        self._search_space[category][name] = config
        logger.debug(f"Added float param '{category}.{name}': [{low}, {high}], log={log}")

        return self

    def add_loguniform(
        self, name: str, low: float, high: float, category: str = "optimizer"
    ) -> "SearchSpaceBuilder":
        """
        Add log-uniform parameter to search space.

        Useful for parameters that span multiple orders of magnitude,
        such as learning rates.

        Args:
            name: Parameter name
            low: Lower bound (must be positive)
            high: Upper bound
            category: Category to add parameter to

        Returns:
            self for method chaining

        Example:
            >>> builder.add_loguniform("lr", 1e-5, 1e-2)
            >>> builder.add_loguniform("weight_decay", 1e-6, 1e-3)
        """
        self._ensure_not_frozen()
        self._validate_category(category)
        self._ensure_category(category)

        if low <= 0:
            raise SearchSpaceError(
                "Log-uniform parameters require positive bounds",
                parameter_name=name,
                parameter_config={"low": low, "high": high},
            )

        config = SearchSpaceParamConfig(type=ParamType.LOGUNIFORM, low=low, high=high)

        self._search_space[category][name] = config
        logger.debug(f"Added loguniform param '{category}.{name}': [{low}, {high}]")

        return self

    def add_categorical(
        self, name: str, choices: list[Any], category: str = "hyperparameters"
    ) -> "SearchSpaceBuilder":
        """
        Add categorical parameter to search space.

        Args:
            name: Parameter name
            choices: List of possible values
            category: Category to add parameter to

        Returns:
            self for method chaining

        Example:
            >>> builder.add_categorical("activation", ["relu", "gelu", "elu"])
            >>> builder.add_categorical("aggregation", ["mean", "sum", "max"])
        """
        self._ensure_not_frozen()
        self._validate_category(category)
        self._ensure_category(category)

        if not choices:
            raise SearchSpaceError(
                "Categorical parameter requires at least one choice", parameter_name=name
            )

        config = SearchSpaceParamConfig(type=ParamType.CATEGORICAL, choices=list(choices))

        self._search_space[category][name] = config
        logger.debug(f"Added categorical param '{category}.{name}': {choices}")

        return self

    def add_uniform(
        self, name: str, low: float, high: float, category: str = "hyperparameters"
    ) -> "SearchSpaceBuilder":
        """
        Add uniform parameter to search space.

        Alias for add_float without log scale.

        Args:
            name: Parameter name
            low: Lower bound
            high: Upper bound
            category: Category to add parameter to

        Returns:
            self for method chaining
        """
        self._ensure_not_frozen()
        self._validate_category(category)
        self._ensure_category(category)

        config = SearchSpaceParamConfig(type=ParamType.UNIFORM, low=low, high=high)

        self._search_space[category][name] = config
        logger.debug(f"Added uniform param '{category}.{name}': [{low}, {high}]")

        return self

    def add_discrete_uniform(
        self, name: str, low: float, high: float, step: float, category: str = "hyperparameters"
    ) -> "SearchSpaceBuilder":
        """
        Add discrete uniform parameter to search space.

        Args:
            name: Parameter name
            low: Lower bound
            high: Upper bound
            step: Step size between values
            category: Category to add parameter to

        Returns:
            self for method chaining

        Example:
            >>> builder.add_discrete_uniform("batch_size", 16, 128, step=16)
        """
        self._ensure_not_frozen()
        self._validate_category(category)
        self._ensure_category(category)

        config = SearchSpaceParamConfig(
            type=ParamType.DISCRETE_UNIFORM, low=low, high=high, step=int(step)
        )

        self._search_space[category][name] = config
        logger.debug(
            f"Added discrete_uniform param '{category}.{name}': [{low}, {high}], step={step}"
        )

        return self

    def add_param(
        self,
        name: str,
        config: SearchSpaceParamConfig | dict[str, Any],
        category: str = "hyperparameters",
    ) -> "SearchSpaceBuilder":
        """
        Add parameter from config object or dict.

        Args:
            name: Parameter name
            config: SearchSpaceParamConfig or dict representation
            category: Category to add parameter to

        Returns:
            self for method chaining

        Example:
            >>> builder.add_param("lr", {"type": "loguniform", "low": 1e-5, "high": 1e-2})
            >>> builder.add_param("heads", SearchSpaceParamConfig(type=ParamType.INT, low=1, high=8))
        """
        self._ensure_not_frozen()
        self._validate_category(category)
        self._ensure_category(category)

        if isinstance(config, dict):
            config = self._dict_to_config(config)

        self._search_space[category][name] = config
        logger.debug(f"Added param '{category}.{name}' from config")

        return self

    def add_category(
        self, category: str, params: dict[str, SearchSpaceParamConfig | dict[str, Any]]
    ) -> "SearchSpaceBuilder":
        """
        Add multiple parameters for a category at once.

        Args:
            category: Category name
            params: Dict of parameter names to configs

        Returns:
            self for method chaining

        Example:
            >>> builder.add_category("optimizer", {
            ...     "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
            ...     "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-3}
            ... })
        """
        self._ensure_not_frozen()

        for param_name, param_config in params.items():
            self.add_param(param_name, param_config, category=category)

        return self

    def remove_param(self, name: str, category: str | None = None) -> "SearchSpaceBuilder":
        """
        Remove parameter from search space.

        Args:
            name: Parameter name
            category: Category to remove from (None = search all categories)

        Returns:
            self for method chaining
        """
        self._ensure_not_frozen()

        if category is not None:
            if category in self._search_space and name in self._search_space[category]:
                del self._search_space[category][name]
                logger.debug(f"Removed param '{category}.{name}'")
        else:
            for cat in self._search_space:
                if name in self._search_space[cat]:
                    del self._search_space[cat][name]
                    logger.debug(f"Removed param '{cat}.{name}'")
                    break

        return self

    def build(self) -> dict[str, dict[str, SearchSpaceParamConfig]]:
        """
        Build and return the search space.

        After calling build(), the builder is frozen and cannot be modified.

        Returns:
            Dict of category -> param_name -> SearchSpaceParamConfig

        Raises:
            SearchSpaceError: If search space is empty or invalid
        """
        if not self._search_space:
            raise SearchSpaceError("Cannot build empty search space. Add at least one parameter.")

        total_params = sum(len(params) for params in self._search_space.values())
        if total_params == 0:
            raise SearchSpaceError("Cannot build search space with no parameters.")

        self._frozen = True
        logger.info(
            f"Built search space with {total_params} parameters "
            f"across {len(self._search_space)} categories"
        )

        return deepcopy(self._search_space)

    def to_dict(self) -> dict[str, dict[str, dict[str, Any]]]:
        """
        Convert search space to plain dict format.

        Useful for serialization (YAML, JSON).

        Returns:
            Dict representation of search space
        """
        result = {}
        for category, params in self._search_space.items():
            result[category] = {}
            for name, config in params.items():
                result[category][name] = self._config_to_dict(config)
        return result

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @staticmethod
    def _dict_to_config(config_dict: dict[str, Any]) -> SearchSpaceParamConfig:
        """Convert dict to SearchSpaceParamConfig."""
        config_copy = config_dict.copy()

        if "type" in config_copy:
            type_val = config_copy["type"]
            if isinstance(type_val, str):
                config_copy["type"] = ParamType(type_val)

        return SearchSpaceParamConfig(**config_copy)

    @staticmethod
    def _config_to_dict(config: SearchSpaceParamConfig) -> dict[str, Any]:
        """Convert SearchSpaceParamConfig to dict."""
        result = {
            "type": config.type.value,
        }

        if config.low is not None:
            result["low"] = config.low
        if config.high is not None:
            result["high"] = config.high
        if config.step is not None:
            result["step"] = config.step
        if config.choices is not None:
            result["choices"] = config.choices
        if config.log:
            result["log"] = config.log

        return result

    # =========================================================================
    # CLASS METHODS FOR PREDEFINED SPACES
    # =========================================================================

    @classmethod
    def for_model(
        cls, model_name: str, include_optimizer: bool = True, include_scheduler: bool = False
    ) -> dict[str, dict[str, SearchSpaceParamConfig]]:
        """
        Get search space for a model architecture - dynamically generated.

        **Phase 6 Migration**: Now uses dynamic PyG introspection to generate
        search spaces for ANY PyG model, not just the 7 hardcoded ones.
        Falls back to legacy hardcoded spaces if introspector unavailable.

        Args:
            model_name: Name of model (GCN, GAT, GraphSAGE, ANY PyG model, etc.)
            include_optimizer: Include optimizer hyperparameters
            include_scheduler: Include scheduler hyperparameters

        Returns:
            Search space for the model (dynamically generated or hardcoded fallback)

        Raises:
            SearchSpaceError: If model_name not found and no fallback available

        Example:
            >>> # Works for ANY PyG model now
            >>> space = SearchSpaceBuilder.for_model("GAT", include_optimizer=True)
            >>> space = SearchSpaceBuilder.for_model("SchNet")
            >>> space = SearchSpaceBuilder.for_model("PMLP")  # NEW: works for any model
        """
        builder = cls()

        # =================================================================
        # PHASE 6: DYNAMIC SEARCH SPACE GENERATION
        # =================================================================
        if _INTROSPECTOR_AVAILABLE and get_introspector is not None:
            try:
                introspector = get_introspector()

                # Check if model exists in PyG
                if introspector.has_model(model_name):
                    # Get dynamically generated search space
                    dynamic_search_space = introspector.get_search_space(model_name)

                    # Convert to builder format
                    for category, params in dynamic_search_space.items():
                        for param_name, param_config in params.items():
                            builder.add_param(param_name, param_config, category=category)

                    logger.debug(
                        f"Generated dynamic search space for '{model_name}' "
                        f"with {sum(len(p) for p in dynamic_search_space.values())} parameters"
                    )

                    # Add optimizer/scheduler spaces
                    if include_optimizer:
                        builder = cls._add_optimizer_space(builder)

                    if include_scheduler:
                        builder = cls._add_scheduler_space(builder)

                    return builder.build()
                else:
                    logger.warning(
                        f"Model '{model_name}' not found in PyG introspector. "
                        f"Falling back to legacy hardcoded space."
                    )
            except Exception as e:
                logger.warning(
                    f"Dynamic introspection failed for '{model_name}': {e}. "
                    f"Falling back to legacy hardcoded space."
                )

        # =================================================================
        # LEGACY FALLBACK: Hardcoded model-specific spaces
        # =================================================================
        model_name_upper = model_name.upper()

        if model_name_upper in ("GCN", "GRAPHCONV"):
            builder = cls._build_gcn_space(builder)
        elif model_name_upper in ("GAT", "GATCONV"):
            builder = cls._build_gat_space(builder)
        elif model_name_upper in ("GRAPHSAGE", "SAGE", "SAGECONV"):
            builder = cls._build_graphsage_space(builder)
        elif model_name_upper in ("GIN", "GINCONV"):
            builder = cls._build_gin_space(builder)
        elif model_name_upper == "SCHNET":
            builder = cls._build_schnet_space(builder)
        elif model_name_upper == "DIMENET":
            builder = cls._build_dimenet_space(builder)
        elif model_name_upper in ("MPNN", "MPNNCONV"):
            builder = cls._build_mpnn_space(builder)
        else:
            builder = cls._build_generic_gnn_space(builder)
            logger.warning(
                f"No predefined space for '{model_name}', using generic GNN space. "
                f"Enable pyg_introspector for dynamic search spaces."
            )

        if include_optimizer:
            builder = cls._add_optimizer_space(builder)

        if include_scheduler:
            builder = cls._add_scheduler_space(builder)

        return builder.build()

    @classmethod
    def _build_gcn_space(cls, builder: "SearchSpaceBuilder") -> "SearchSpaceBuilder":
        """
        Build search space for GCN model.

        .. deprecated:: 2.0.0
            This hardcoded method is deprecated. Use dynamic introspection via
            `for_model()` which now generates search spaces from PyG model signatures.
            Kept as fallback when pyg_introspector is unavailable.
        """
        return (
            builder.add_int("hidden_channels", 32, 256, step=32, category="hyperparameters")
            .add_int("num_layers", 2, 6, category="hyperparameters")
            .add_float("dropout", 0.0, 0.6, category="hyperparameters")
            .add_categorical("aggregation", ["add", "mean", "max"], category="hyperparameters")
        )

    @classmethod
    def _build_gat_space(cls, builder: "SearchSpaceBuilder") -> "SearchSpaceBuilder":
        """
        Build search space for GAT model.

        .. deprecated:: 2.0.0
            Kept as fallback. Use dynamic introspection via `for_model()`.
        """
        return (
            builder.add_int("hidden_channels", 32, 256, step=32, category="hyperparameters")
            .add_int("num_layers", 2, 5, category="hyperparameters")
            .add_int("heads", 1, 8, category="hyperparameters")
            .add_float("dropout", 0.0, 0.6, category="hyperparameters")
            .add_float("attention_dropout", 0.0, 0.6, category="hyperparameters")
            .add_categorical("concat", [True, False], category="hyperparameters")
        )

    @classmethod
    def _build_graphsage_space(cls, builder: "SearchSpaceBuilder") -> "SearchSpaceBuilder":
        """
        Build search space for GraphSAGE model.

        .. deprecated:: 2.0.0
            Kept as fallback. Use dynamic introspection via `for_model()`.
        """
        return (
            builder.add_int("hidden_channels", 32, 256, step=32, category="hyperparameters")
            .add_int("num_layers", 2, 5, category="hyperparameters")
            .add_float("dropout", 0.0, 0.6, category="hyperparameters")
            .add_categorical("aggregation", ["mean", "max", "lstm"], category="hyperparameters")
            .add_categorical("normalize", [True, False], category="hyperparameters")
        )

    @classmethod
    def _build_gin_space(cls, builder: "SearchSpaceBuilder") -> "SearchSpaceBuilder":
        """
        Build search space for GIN model.

        .. deprecated:: 2.0.0
            Kept as fallback. Use dynamic introspection via `for_model()`.
        """
        return (
            builder.add_int("hidden_channels", 32, 256, step=32, category="hyperparameters")
            .add_int("num_layers", 2, 6, category="hyperparameters")
            .add_float("dropout", 0.0, 0.6, category="hyperparameters")
            .add_float("eps", 0.0, 1.0, category="hyperparameters")
            .add_categorical("train_eps", [True, False], category="hyperparameters")
        )

    @classmethod
    def _build_schnet_space(cls, builder: "SearchSpaceBuilder") -> "SearchSpaceBuilder":
        """
        Build search space for SchNet model (quantum chemistry).

        .. deprecated:: 2.0.0
            Kept as fallback. Use dynamic introspection via `for_model()`.
        """
        return (
            builder.add_int("hidden_channels", 64, 256, step=32, category="hyperparameters")
            .add_int("num_filters", 64, 256, step=32, category="hyperparameters")
            .add_int("num_interactions", 3, 6, category="hyperparameters")
            .add_int("num_gaussians", 25, 100, step=25, category="hyperparameters")
            .add_float("cutoff", 5.0, 10.0, category="hyperparameters")
        )

    @classmethod
    def _build_dimenet_space(cls, builder: "SearchSpaceBuilder") -> "SearchSpaceBuilder":
        """
        Build search space for DimeNet model.

        .. deprecated:: 2.0.0
            Kept as fallback. Use dynamic introspection via `for_model()`.
        """
        return (
            builder.add_int("hidden_channels", 64, 256, step=32, category="hyperparameters")
            .add_int("num_blocks", 3, 6, category="hyperparameters")
            .add_int("num_bilinear", 4, 8, category="hyperparameters")
            .add_int("num_spherical", 3, 7, category="hyperparameters")
            .add_int("num_radial", 3, 6, category="hyperparameters")
            .add_float("cutoff", 4.0, 6.0, category="hyperparameters")
        )

    @classmethod
    def _build_mpnn_space(cls, builder: "SearchSpaceBuilder") -> "SearchSpaceBuilder":
        """
        Build search space for MPNN model.

        .. deprecated:: 2.0.0
            Kept as fallback. Use dynamic introspection via `for_model()`.
        """
        return (
            builder.add_int("hidden_channels", 32, 256, step=32, category="hyperparameters")
            .add_int("num_layers", 2, 6, category="hyperparameters")
            .add_float("dropout", 0.0, 0.6, category="hyperparameters")
            .add_categorical("aggregation", ["add", "mean", "max"], category="hyperparameters")
        )

    @classmethod
    def _build_generic_gnn_space(cls, builder: "SearchSpaceBuilder") -> "SearchSpaceBuilder":
        """
        Build generic search space for GNN models.

        .. deprecated:: 2.0.0
            Kept as fallback. Use dynamic introspection via `for_model()`.
        """
        return (
            builder.add_int("hidden_channels", 32, 256, step=32, category="hyperparameters")
            .add_int("num_layers", 2, 6, category="hyperparameters")
            .add_float("dropout", 0.0, 0.6, category="hyperparameters")
        )

    @classmethod
    def _add_optimizer_space(cls, builder: "SearchSpaceBuilder") -> "SearchSpaceBuilder":
        """Add optimizer hyperparameters to search space."""
        return builder.add_loguniform("lr", 1e-5, 1e-2, category="optimizer").add_loguniform(
            "weight_decay", 1e-6, 1e-3, category="optimizer"
        )

    @classmethod
    def _add_scheduler_space(cls, builder: "SearchSpaceBuilder") -> "SearchSpaceBuilder":
        """Add scheduler hyperparameters to search space."""
        return builder.add_float("factor", 0.1, 0.9, category="scheduler").add_int(
            "patience", 5, 20, category="scheduler"
        )

    # =========================================================================
    # UTILITY CLASS METHODS
    # =========================================================================

    @classmethod
    def from_dict(
        cls, search_space_dict: dict[str, dict[str, dict[str, Any]]]
    ) -> dict[str, dict[str, SearchSpaceParamConfig]]:
        """
        Create search space from dictionary.

        Args:
            search_space_dict: Dict representation of search space

        Returns:
            Search space with SearchSpaceParamConfig objects

        Example:
            >>> space_dict = {
            ...     "hyperparameters": {
            ...         "hidden_channels": {"type": "int", "low": 32, "high": 256}
            ...     }
            ... }
            >>> space = SearchSpaceBuilder.from_dict(space_dict)
        """
        builder = cls()

        for category, params in search_space_dict.items():
            for param_name, param_config in params.items():
                builder.add_param(param_name, param_config, category=category)

        return builder.build()

    @classmethod
    def merge(
        cls,
        *search_spaces: dict[str, dict[str, SearchSpaceParamConfig]],
        conflict_resolution: str = "last",
    ) -> dict[str, dict[str, SearchSpaceParamConfig]]:
        """
        Merge multiple search spaces.

        Args:
            *search_spaces: Search spaces to merge
            conflict_resolution: How to handle conflicts:
                - "last": Later spaces override earlier (default)
                - "first": Keep first occurrence
                - "error": Raise error on conflict

        Returns:
            Merged search space

        Raises:
            SearchSpaceError: If conflict_resolution="error" and conflict found

        Example:
            >>> base_space = SearchSpaceBuilder.for_model("GCN")
            >>> custom = {"hyperparameters": {"heads": config}}
            >>> merged = SearchSpaceBuilder.merge(base_space, custom)
        """
        if not search_spaces:
            raise SearchSpaceError("At least one search space required for merge")

        builder = cls()
        seen_params: dict[str, str] = {}

        for space in search_spaces:
            for category, params in space.items():
                for param_name, param_config in params.items():
                    full_name = f"{category}.{param_name}"

                    if full_name in seen_params:
                        if conflict_resolution == "first":
                            continue
                        elif conflict_resolution == "error":
                            raise SearchSpaceError(
                                f"Conflict: parameter '{full_name}' defined in multiple spaces",
                                parameter_name=full_name,
                            )

                    seen_params[full_name] = category
                    builder.add_param(param_name, param_config, category=category)

        return builder.build()

    @classmethod
    def validate(
        cls, search_space: dict[str, dict[str, SearchSpaceParamConfig | dict[str, Any]]]
    ) -> tuple[bool, list[str]]:
        """
        Validate a search space configuration.

        Args:
            search_space: Search space to validate

        Returns:
            Tuple of (is_valid, list_of_errors)

        Example:
            >>> is_valid, errors = SearchSpaceBuilder.validate(my_space)
            >>> if not is_valid:
            ...     for error in errors:
            ...         print(f"Error: {error}")
        """
        errors = []

        if not search_space:
            errors.append("Search space is empty")
            return False, errors

        for category, params in search_space.items():
            if not isinstance(params, dict):
                errors.append(f"Category '{category}' must be a dict, got {type(params)}")
                continue

            for param_name, param_config in params.items():
                try:
                    if isinstance(param_config, dict):
                        cls._dict_to_config(param_config)
                except (ConfigurationError, ValueError, KeyError) as e:
                    errors.append(f"Invalid config for '{category}.{param_name}': {e}")

        return len(errors) == 0, errors

    @classmethod
    def get_param_count(
        cls, search_space: dict[str, dict[str, SearchSpaceParamConfig]]
    ) -> dict[str, int]:
        """
        Get parameter count per category.

        Args:
            search_space: Search space to analyze

        Returns:
            Dict of category -> param_count
        """
        return {category: len(params) for category, params in search_space.items()}

    @classmethod
    def estimate_search_space_size(
        cls, search_space: dict[str, dict[str, SearchSpaceParamConfig]], grid_points: int = 10
    ) -> int:
        """
        Estimate the size of the search space.

        For continuous parameters, assumes 'grid_points' discrete values.
        For categorical, uses actual number of choices.

        Args:
            search_space: Search space to analyze
            grid_points: Number of grid points for continuous params

        Returns:
            Estimated search space size (combinatorial)
        """
        total = 1

        for _category, params in search_space.items():
            for _param_name, config in params.items():
                if config.type == ParamType.CATEGORICAL:
                    total *= len(config.choices) if config.choices else 1
                elif config.type == ParamType.INT:
                    if config.step:
                        n_values = int((config.high - config.low) / config.step) + 1
                    else:
                        n_values = int(config.high - config.low) + 1
                    total *= n_values
                else:
                    total *= grid_points

        return total

    @classmethod
    def list_available_models(cls) -> list[str]:
        """
        List ALL available PyG models with search space support.

        **Phase 6 Migration**: Now returns ALL dynamically discovered PyG models,
        not just the 7 hardcoded ones. Falls back to legacy list if introspector
        unavailable.

        Returns:
            List of model names (dynamically discovered or legacy fallback)

        Example:
            >>> models = SearchSpaceBuilder.list_available_models()
            >>> print(f"Available: {len(models)} models")  # Now shows ALL PyG models
        """
        # Phase 6: Use dynamic introspection
        if _INTROSPECTOR_AVAILABLE and get_introspector is not None:
            try:
                introspector = get_introspector()
                models = introspector.get_all_model_names()
                logger.debug(f"Dynamic discovery: {len(models)} models available")
                return models
            except Exception as e:
                logger.warning(f"Dynamic model discovery failed: {e}. Using legacy list.")

        # Legacy fallback
        return [
            "GCN",
            "GAT",
            "GraphSAGE",
            "GIN",
            "SchNet",
            "DimeNet",
            "MPNN",
        ]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def build_search_space() -> SearchSpaceBuilder:
    """
    Create a new SearchSpaceBuilder instance.

    Returns:
        New SearchSpaceBuilder

    Example:
        >>> from milia_pipeline.models.hpo.search_spaces import build_search_space
        >>> space = (
        ...     build_search_space()
        ...     .add_int("hidden_channels", 32, 256)
        ...     .add_loguniform("lr", 1e-5, 1e-2, category="optimizer")
        ...     .build()
        ... )
    """
    return SearchSpaceBuilder()


def get_model_search_space(
    model_name: str, include_optimizer: bool = True
) -> dict[str, dict[str, SearchSpaceParamConfig]]:
    """
    Get predefined search space for a model.

    Convenience function wrapping SearchSpaceBuilder.for_model().

    Args:
        model_name: Model architecture name
        include_optimizer: Include optimizer hyperparameters

    Returns:
        Predefined search space

    Example:
        >>> from milia_pipeline.models.hpo.search_spaces import get_model_search_space
        >>> space = get_model_search_space("GAT")
    """
    return SearchSpaceBuilder.for_model(model_name, include_optimizer=include_optimizer)


def validate_search_space(search_space: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    """
    Validate a search space configuration.

    Convenience function wrapping SearchSpaceBuilder.validate().

    Args:
        search_space: Search space to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    return SearchSpaceBuilder.validate(search_space)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "SearchSpaceBuilder",
    "build_search_space",
    "get_model_search_space",
    "validate_search_space",
]


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Log module load with dynamic vs legacy status
if _INTROSPECTOR_AVAILABLE:
    logger.info(
        "search_space_builder module loaded (v2.0.0) - "
        "DYNAMIC search space generation enabled via pyg_introspector"
    )
else:
    logger.info(
        f"search_space_builder module loaded (v2.0.0) - "
        f"LEGACY mode: {len(SearchSpaceBuilder.list_available_models())} predefined model spaces "
        f"(enable pyg_introspector for dynamic support)"
    )
