"""
Neural Architecture Search Manager

Orchestrates architecture search for Graph Neural Networks using HPO infrastructure.

This module provides the NASManager class that coordinates neural architecture search
by leveraging the existing HPO infrastructure (HPOManager, HPOConfig) to search over
GNN architectures defined in GNNArchitectureSpace.

Pattern: Follows HPOManager (hpo_manager.py:77-735)

Key Features:
- Architecture search using HPO infrastructure
- Support for heterogeneous GNN architectures (mixed layer types)
- Dynamic model building from architecture configurations
- Integration with ModelFactory for model creation
- Configurable search spaces for GNN components

Dependencies:
    - hpo_manager.HPOManager: For running optimization
    - hpo_config.HPOConfig: For HPO configuration
    - search_space.GNNArchitectureSpace: For architecture search space definition
    - search_space.LayerType: For layer type enumeration
    - search_space.LayerConfig: For layer configuration
    - milia_pipeline.exceptions: For error handling

Location: milia_pipeline/models/hpo/nas/nas_manager.py
Blueprint Reference: MILIA_HPO_Implementation_Blueprint.md (lines 4288-4432)

Author: Milia Team
Version: 1.1.0

Pydantic V2 Migration (Phase 36):
    - Migrated NASConfig from @dataclass to mutable BaseModel
    - Uses @field_validator for individual field validation (n_trials, timeout, direction, cv_folds)
    - Replaces __post_init__ validation with Pydantic V2 validators
    - NON-BREAKING: Same constructor API, attribute access preserved
    - Added to_dict() method wrapping model_dump() for backward compatibility
    - Follows established patterns from search_space.py (Phase 14) and transfer_manager.py (Phase 11)
"""

import logging
from typing import Any

import torch.nn as nn
from pydantic import BaseModel, field_validator

from milia_pipeline.exceptions import HPOError

from ..hpo_config import HPOConfig, ParamType, SearchSpaceParamConfig
from ..hpo_manager import HPOManager
from .search_space import (
    GNNArchitectureSpace,
    LayerType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# NAS CONFIGURATION
# =============================================================================


class NASConfig(BaseModel):
    """
    Configuration for Neural Architecture Search.

    Pattern: Follows mutable BaseModel pattern from search_space.py (Pydantic V2)

    Extends HPO configuration with NAS-specific settings for
    architecture search over GNN components.

    Attributes:
        n_trials: Number of architecture trials to run
        timeout: Maximum time in seconds (None for no limit)
        metric: Metric to optimize (must match Trainer output keys)
        direction: Optimization direction ("minimize" or "maximize")
        cv_folds: Number of cross-validation folds (0 for no CV)
        study_name: Name for the NAS study
        storage: Storage URL for persistence (None for in-memory)

    Examples:
        >>> # Default NAS configuration
        >>> config = NASConfig(n_trials=100)

        >>> # NAS with cross-validation and persistence
        >>> config = NASConfig(
        ...     n_trials=200,
        ...     cv_folds=5,
        ...     storage="sqlite:///nas_study.db"
        ... )
    """

    n_trials: int = 100
    timeout: int | None = None
    metric: str = "val_loss"
    direction: str = "minimize"
    cv_folds: int = 0
    study_name: str = "milia_nas"
    storage: str | None = None

    @field_validator("n_trials")
    @classmethod
    def validate_n_trials(cls, v: int) -> int:
        """Validate n_trials is at least 1."""
        if v < 1:
            raise ValueError(f"n_trials must be at least 1, got {v}")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int | None) -> int | None:
        """Validate timeout is positive or None."""
        if v is not None and v < 1:
            raise ValueError(f"timeout must be positive or None, got {v}")
        return v

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        """Validate direction is 'minimize' or 'maximize'."""
        valid_directions = ("minimize", "maximize")
        if v not in valid_directions:
            raise ValueError(f"Invalid direction: '{v}'. Must be 'minimize' or 'maximize'")
        return v

    @field_validator("cv_folds")
    @classmethod
    def validate_cv_folds(cls, v: int) -> int:
        """Validate cv_folds is non-negative."""
        if v < 0:
            raise ValueError(f"cv_folds must be non-negative, got {v}")
        return v

    def to_dict(self) -> dict[str, Any]:
        """
        Backward compatible dict conversion.

        Returns:
            Dictionary representation of the NAS configuration.
        """
        return self.model_dump()


# =============================================================================
# NAS MANAGER CLASS
# =============================================================================


class NASManager:
    """
    Neural Architecture Search for GNNs.

    Uses HPO infrastructure to search over architectures defined in
    GNNArchitectureSpace. Builds models dynamically from architecture
    configurations found during search.

    Pattern: Follows HPOManager (hpo_manager.py:77-735)
    Blueprint Reference: MILIA_HPO_Implementation_Blueprint.md (lines 4304-4432)

    The NASManager coordinates:
    1. Converting GNNArchitectureSpace to HPO search space format
    2. Running architecture search via HPOManager
    3. Extracting architecture configuration from best parameters
    4. Building models from architecture configurations

    Attributes:
        arch_space: GNNArchitectureSpace defining searchable components
        hpo_manager: HPOManager for running optimization
        best_architecture: Best architecture found (set after search())

    Usage:
        >>> # Create architecture search space
        >>> arch_space = GNNArchitectureSpace(
        ...     min_layers=2,
        ...     max_layers=6,
        ...     layer_types=[LayerType.GCN, LayerType.GAT, LayerType.SAGE],
        ... )

        >>> # Create NAS manager
        >>> nas_manager = NASManager(arch_space)

        >>> # Run architecture search
        >>> best_arch = nas_manager.search(dataset=dataset)

        >>> # Build model from best architecture
        >>> model = nas_manager.build_model(
        ...     architecture=best_arch,
        ...     in_channels=10,
        ...     out_channels=1
        ... )

    Examples:
        >>> # Simple architecture search
        >>> from milia_pipeline.models.hpo.nas import NASManager, GNNArchitectureSpace
        >>> arch_space = GNNArchitectureSpace()
        >>> nas = NASManager(arch_space)
        >>> best_arch = nas.search(dataset=train_dataset)
        >>> model = nas.build_model(best_arch, in_channels=10, out_channels=1)

        >>> # Architecture search with custom HPO config
        >>> from milia_pipeline.models.hpo import HPOConfig
        >>> hpo_config = HPOConfig(enabled=True, n_trials=200, cv_folds=5)
        >>> nas = NASManager(arch_space, hpo_config=hpo_config)
        >>> best_arch = nas.search(dataset=dataset)
    """

    def __init__(
        self,
        arch_space: GNNArchitectureSpace,
        hpo_config: HPOConfig | None = None,
        nas_config: NASConfig | None = None,
    ):
        """
        Initialize NASManager.

        Args:
            arch_space: GNNArchitectureSpace defining the architecture search space.
                       Specifies searchable components like layer types, dimensions,
                       pooling strategies, etc.
            hpo_config: Optional HPOConfig for optimization settings. If provided,
                       the architecture search space will be merged with any existing
                       search space in the config. If None, a default config will be
                       created using nas_config settings.
            nas_config: Optional NASConfig for NAS-specific settings. Only used when
                       hpo_config is None. Defaults to NASConfig() if both are None.

        Raises:
            SearchSpaceError: If arch_space validation fails
            ConfigurationError: If configuration is invalid

        Examples:
            >>> # With default configuration
            >>> nas = NASManager(arch_space)

            >>> # With custom HPO configuration
            >>> hpo_config = HPOConfig(enabled=True, n_trials=200)
            >>> nas = NASManager(arch_space, hpo_config=hpo_config)

            >>> # With NAS-specific configuration
            >>> nas_config = NASConfig(n_trials=100, cv_folds=5)
            >>> nas = NASManager(arch_space, nas_config=nas_config)
        """
        self.arch_space = arch_space
        self.best_architecture: dict[str, Any] | None = None
        self._best_params: dict[str, Any] | None = None

        arch_search_space = self._convert_arch_space_to_hpo_format(arch_space)

        if hpo_config is None:
            nas_config = nas_config or NASConfig()
            hpo_config = self._create_hpo_config_from_nas_config(nas_config, arch_search_space)
        else:
            hpo_config = self._merge_search_spaces(hpo_config, arch_search_space)

        self.hpo_manager = HPOManager(hpo_config)

        logger.info(
            f"NASManager initialized with {arch_space.get_search_dimensions()} "
            f"search dimensions, ~{arch_space.estimate_search_space_size():,} "
            f"possible architectures"
        )

    def _convert_arch_space_to_hpo_format(
        self,
        arch_space: GNNArchitectureSpace,
    ) -> dict[str, dict[str, SearchSpaceParamConfig]]:
        """
        Convert GNNArchitectureSpace to HPO search space format.

        Transforms the architecture space definition into the format
        expected by HPOConfig.search_space for use with HPOManager.

        Args:
            arch_space: GNNArchitectureSpace to convert

        Returns:
            Dictionary in HPO search space format with SearchSpaceParamConfig objects

        Notes:
            The conversion handles:
            - Integer parameters (num_layers) -> ParamType.INT
            - Float parameters (dropout) -> ParamType.FLOAT
            - Categorical parameters (layer types, pooling) -> ParamType.CATEGORICAL
            - Per-layer parameters if allow_mixed_layers is True
        """
        search_space: dict[str, dict[str, SearchSpaceParamConfig]] = {"architecture": {}}

        arch_params = search_space["architecture"]

        arch_params["num_layers"] = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=arch_space.min_layers,
            high=arch_space.max_layers,
        )

        arch_params["hidden_channels"] = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=arch_space.hidden_channels,
        )

        arch_params["pooling"] = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=[p.value for p in arch_space.pooling_types],
        )

        arch_params["dropout"] = SearchSpaceParamConfig(
            type=ParamType.FLOAT,
            low=arch_space.dropout_range[0],
            high=arch_space.dropout_range[1],
        )

        arch_params["aggregation"] = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=[a.value for a in arch_space.aggregation_types],
        )

        arch_params["activation"] = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=[a.value for a in arch_space.activation_types],
        )

        arch_params["batch_norm"] = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=arch_space.batch_norm_options,
        )

        if arch_space.allow_skip_connections:
            arch_params["use_skip_connections"] = SearchSpaceParamConfig(
                type=ParamType.CATEGORICAL,
                choices=[True, False],
            )

        if arch_space.allow_dense_connections:
            arch_params["use_dense_connections"] = SearchSpaceParamConfig(
                type=ParamType.CATEGORICAL,
                choices=[True, False],
            )

        if arch_space.allow_mixed_layers:
            layer_type_choices = [lt.value for lt in arch_space.layer_types]
            attention_layer_types = [LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER]
            has_attention_layers = any(lt in arch_space.layer_types for lt in attention_layer_types)

            for i in range(arch_space.max_layers):
                arch_params[f"layer_{i}_type"] = SearchSpaceParamConfig(
                    type=ParamType.CATEGORICAL,
                    choices=layer_type_choices,
                )

                if has_attention_layers:
                    arch_params[f"layer_{i}_heads"] = SearchSpaceParamConfig(
                        type=ParamType.CATEGORICAL,
                        choices=arch_space.heads,
                    )
        else:
            arch_params["layer_type"] = SearchSpaceParamConfig(
                type=ParamType.CATEGORICAL,
                choices=[lt.value for lt in arch_space.layer_types],
            )

            attention_layer_types = [LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER]
            if any(lt in arch_space.layer_types for lt in attention_layer_types):
                arch_params["heads"] = SearchSpaceParamConfig(
                    type=ParamType.CATEGORICAL,
                    choices=arch_space.heads,
                )

        return search_space

    def _create_hpo_config_from_nas_config(
        self,
        nas_config: NASConfig,
        search_space: dict[str, dict[str, SearchSpaceParamConfig]],
    ) -> HPOConfig:
        """
        Create HPOConfig from NASConfig and search space.

        Args:
            nas_config: NAS-specific configuration
            search_space: Architecture search space in HPO format

        Returns:
            HPOConfig configured for architecture search
        """
        from ..hpo_config import (
            OptimizationDirection,
            PrunerConfig,
            SamplerConfig,
            StudyConfig,
        )

        direction = OptimizationDirection(nas_config.direction)

        study_config = StudyConfig(
            direction=direction,
            metric=nas_config.metric,
            study_name=nas_config.study_name,
            storage=nas_config.storage,
            load_if_exists=True,
        )

        return HPOConfig(
            enabled=True,
            backend="optuna",
            n_trials=nas_config.n_trials,
            timeout=nas_config.timeout,
            n_jobs=1,
            search_space=search_space,
            pruner=PrunerConfig(),
            sampler=SamplerConfig(),
            study=study_config,
            cv_folds=nas_config.cv_folds,
            cv_metric_aggregation="mean",
        )

    def _merge_search_spaces(
        self,
        hpo_config: HPOConfig,
        arch_search_space: dict[str, dict[str, SearchSpaceParamConfig]],
    ) -> HPOConfig:
        """
        Merge architecture search space into existing HPO config.

        Creates a new HPOConfig with the architecture search space merged
        with any existing search space from the provided config.

        Args:
            hpo_config: Existing HPO configuration
            arch_search_space: Architecture search space to merge

        Returns:
            New HPOConfig with merged search spaces

        Notes:
            Architecture parameters take precedence if there are key conflicts.
        """
        merged_space: dict[str, dict[str, SearchSpaceParamConfig]] = {}

        for category, params in hpo_config.search_space.items():
            merged_space[category] = dict(params)

        for category, params in arch_search_space.items():
            if category in merged_space:
                merged_space[category].update(params)
            else:
                merged_space[category] = dict(params)

        return HPOConfig(
            enabled=True,
            backend=hpo_config.backend,
            n_trials=hpo_config.n_trials,
            timeout=hpo_config.timeout,
            n_jobs=hpo_config.n_jobs,
            search_space=merged_space,
            pruner=hpo_config.pruner,
            sampler=hpo_config.sampler,
            study=hpo_config.study,
            cv_folds=hpo_config.cv_folds,
            cv_metric_aggregation=hpo_config.cv_metric_aggregation,
        )

    def search(
        self,
        dataset,
        base_hyperparameters: dict[str, Any] | None = None,
        trainer_kwargs: dict[str, Any] | None = None,
        callbacks: list | None = None,
        model_name: str = "DynamicGNN",
    ) -> dict[str, Any]:
        """
        Run architecture search.

        Searches over the architecture space to find the best GNN configuration
        for the given dataset. Uses HPOManager to run the optimization.

        Args:
            dataset: PyG dataset or DataLoader for training and evaluation.
                    Must be compatible with the Trainer class.
            base_hyperparameters: Fixed hyperparameters not being searched.
                                 These are passed directly to model creation.
            trainer_kwargs: Additional kwargs for Trainer initialization.
                           Controls training behavior (epochs, optimizer, etc.)
            callbacks: Additional callbacks for training.
                      HPO callback is added automatically.
            model_name: Name of the model to use for architecture search.
                       Default "DynamicGNN" uses dynamic architecture building.
                       Can be a specific model name from ModelRegistry.

        Returns:
            Dictionary containing the best architecture configuration with keys:
            - num_layers: Number of GNN layers
            - hidden_channels: Hidden dimension size
            - pooling: Graph pooling type
            - dropout: Dropout probability
            - aggregation: Message aggregation type
            - activation: Activation function
            - batch_norm: Whether to use batch normalization
            - use_skip_connections: Whether to use skip connections (if searched)
            - layers: List of per-layer configurations (if allow_mixed_layers)

        Raises:
            HPOError: If search fails or no valid architectures found

        Examples:
            >>> # Basic architecture search
            >>> best_arch = nas.search(dataset=train_dataset)
            >>> print(f"Best architecture: {best_arch}")

            >>> # Search with fixed hyperparameters
            >>> best_arch = nas.search(
            ...     dataset=train_dataset,
            ...     base_hyperparameters={"in_channels": 10, "out_channels": 1}
            ... )

            >>> # Search with custom trainer settings
            >>> best_arch = nas.search(
            ...     dataset=train_dataset,
            ...     trainer_kwargs={"max_epochs": 50, "patience": 10}
            ... )
        """
        logger.info(f"Starting architecture search with {self.hpo_manager.config.n_trials} trials")

        self._best_params = self.hpo_manager.optimize(
            model_name=model_name,
            dataset=dataset,
            base_hyperparameters=base_hyperparameters,
            trainer_kwargs=trainer_kwargs,
            callbacks=callbacks,
        )

        self.best_architecture = self._extract_architecture(self._best_params)

        logger.info("Architecture search completed")
        logger.info(f"Best architecture: {self.best_architecture}")

        return self.best_architecture

    def _extract_architecture(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extract architecture configuration from HPO parameters.

        Converts the flat parameter dictionary from HPO optimization
        into a structured architecture configuration.

        Args:
            params: Flat parameter dictionary from HPO optimization

        Returns:
            Structured architecture configuration dictionary
        """
        num_layers = params.get("num_layers", 3)
        hidden_channels = params.get("hidden_channels", 64)

        arch: dict[str, Any] = {
            "num_layers": num_layers,
            "hidden_channels": hidden_channels,
            "pooling": params.get("pooling", "mean"),
            "dropout": params.get("dropout", 0.0),
            "aggregation": params.get("aggregation", "mean"),
            "activation": params.get("activation", "relu"),
            "batch_norm": params.get("batch_norm", True),
            "layers": [],
        }

        if "use_skip_connections" in params:
            arch["use_skip_connections"] = params["use_skip_connections"]
        else:
            arch["use_skip_connections"] = self.arch_space.allow_skip_connections

        if "use_dense_connections" in params:
            arch["use_dense_connections"] = params["use_dense_connections"]
        else:
            arch["use_dense_connections"] = self.arch_space.allow_dense_connections

        if self.arch_space.allow_mixed_layers:
            for i in range(num_layers):
                layer_type = params.get(f"layer_{i}_type", "gcn")
                layer_heads = params.get(f"layer_{i}_heads", 1)

                layer_config = {
                    "type": layer_type,
                    "hidden_channels": hidden_channels,
                    "heads": layer_heads,
                    "dropout": arch["dropout"],
                    "activation": arch["activation"],
                    "batch_norm": arch["batch_norm"],
                    "residual": arch["use_skip_connections"],
                }
                arch["layers"].append(layer_config)
        else:
            layer_type = params.get("layer_type", "gcn")
            heads = params.get("heads", 1)

            for i in range(num_layers):
                layer_config = {
                    "type": layer_type,
                    "hidden_channels": hidden_channels,
                    "heads": heads,
                    "dropout": arch["dropout"],
                    "activation": arch["activation"],
                    "batch_norm": arch["batch_norm"],
                    "residual": arch["use_skip_connections"],
                }
                arch["layers"].append(layer_config)

        return arch

    def build_model(
        self,
        architecture: dict[str, Any],
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        """
        Build model from architecture configuration.

        Creates a PyTorch model based on the architecture configuration.
        For homogeneous architectures (all layers same type), uses standard
        models from ModelFactory. For heterogeneous architectures (mixed
        layer types), builds a custom model dynamically.

        Args:
            architecture: Architecture configuration from search() or manually created.
                         Must contain 'layers' list with layer configurations.
            in_channels: Number of input features per node
            out_channels: Number of output features (e.g., 1 for regression,
                         num_classes for classification)

        Returns:
            PyTorch nn.Module ready for training

        Raises:
            HPOError: If ModelFactory not available or model creation fails

        Examples:
            >>> # Build model from search results
            >>> best_arch = nas.search(dataset=train_dataset)
            >>> model = nas.build_model(
            ...     architecture=best_arch,
            ...     in_channels=10,
            ...     out_channels=1
            ... )

            >>> # Build model from manual architecture
            >>> custom_arch = {
            ...     'num_layers': 3,
            ...     'hidden_channels': 64,
            ...     'pooling': 'mean',
            ...     'dropout': 0.1,
            ...     'layers': [
            ...         {'type': 'gcn', 'hidden_channels': 64, 'heads': 1},
            ...         {'type': 'gat', 'hidden_channels': 64, 'heads': 4},
            ...         {'type': 'gcn', 'hidden_channels': 64, 'heads': 1},
            ...     ]
            ... }
            >>> model = nas.build_model(custom_arch, in_channels=10, out_channels=1)
        """
        try:
            from milia_pipeline.models.factory.model_factory import ModelFactory, get_factory
        except ImportError:
            raise HPOError(
                "ModelFactory not available",
                details="Cannot build model without model_factory module",
            )

        factory = get_factory() if get_factory else ModelFactory()

        hyperparameters = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "hidden_channels": architecture["hidden_channels"],
            "num_layers": architecture["num_layers"],
            "dropout": architecture.get("dropout", 0.0),
        }

        layer_types = [layer["type"] for layer in architecture["layers"]]
        unique_types = set(layer_types)

        if len(unique_types) == 1:
            model_name = layer_types[0].upper()

            if model_name in ("GAT", "GATV2", "TRANSFORMER"):
                hyperparameters["heads"] = architecture["layers"][0].get("heads", 1)

            logger.info(f"Building homogeneous {model_name} model")

            return factory.create_model(
                model_name=model_name,
                hyperparameters=hyperparameters,
            )

        logger.info(f"Building heterogeneous model with layer types: {layer_types}")
        return self._build_heterogeneous_model(architecture, in_channels, out_channels)

    def _build_heterogeneous_model(
        self,
        architecture: dict[str, Any],
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        """
        Build model with mixed layer types.

        Constructs a custom GNN model with different layer types in each
        position. This enables more flexible architectures that combine
        the strengths of different GNN operators.

        Args:
            architecture: Architecture configuration with 'layers' list
            in_channels: Number of input features
            out_channels: Number of output features

        Returns:
            HeterogeneousGNN module with mixed layer types

        Notes:
            The heterogeneous model:
            - Uses the specified layer type for each position
            - Handles dimension matching between different layer types
            - Supports attention heads for GAT/Transformer layers
            - Includes skip connections if specified
            - Applies global pooling for graph-level tasks
        """
        return HeterogeneousGNN(
            architecture=architecture,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def get_best_architecture(self) -> dict[str, Any]:
        """
        Get the best architecture found during search.

        Returns:
            Best architecture configuration dictionary

        Raises:
            HPOError: If search() has not been run yet
        """
        if self.best_architecture is None:
            raise HPOError(
                "No architecture search completed yet",
                details="Run search() before getting best architecture",
            )
        return self.best_architecture

    def get_best_params(self) -> dict[str, Any]:
        """
        Get raw HPO parameters from the best trial.

        Returns:
            Dictionary of hyperparameters from best trial

        Raises:
            HPOError: If search() has not been run yet
        """
        if self._best_params is None:
            raise HPOError(
                "No architecture search completed yet",
                details="Run search() before getting best parameters",
            )
        return self._best_params

    def get_search_summary(self) -> dict[str, Any]:
        """
        Get summary of the architecture search.

        Returns a dictionary containing information about the search
        space, best results, and optimization statistics.

        Returns:
            Dictionary with search summary including:
            - search_dimensions: Number of search dimensions
            - estimated_space_size: Approximate number of possible architectures
            - n_trials: Number of trials configured
            - best_value: Best objective value (if search completed)
            - best_architecture: Best architecture (if search completed)

        Examples:
            >>> summary = nas.get_search_summary()
            >>> print(f"Searched {summary['n_trials']} architectures")
            >>> print(f"Best validation loss: {summary['best_value']:.4f}")
        """
        summary = {
            "search_dimensions": self.arch_space.get_search_dimensions(),
            "estimated_space_size": self.arch_space.estimate_search_space_size(),
            "n_trials": self.hpo_manager.config.n_trials,
            "allow_mixed_layers": self.arch_space.allow_mixed_layers,
            "layer_types": [lt.value for lt in self.arch_space.layer_types],
        }

        if self.best_architecture is not None:
            summary["best_architecture"] = self.best_architecture

        if self.hpo_manager.study is not None:
            try:
                summary["best_value"] = self.hpo_manager.get_best_value()
                summary["completed_trials"] = len(
                    [
                        t
                        for t in self.hpo_manager.backend.get_all_trials(self.hpo_manager.study)
                        if t["state"] == "COMPLETE"
                    ]
                )
            except HPOError:
                pass

        return summary


# =============================================================================
# HETEROGENEOUS GNN MODEL
# =============================================================================


class HeterogeneousGNN(nn.Module):
    """
    GNN model with mixed layer types.

    A flexible GNN architecture that supports different layer types
    at each position, enabling architecture search over heterogeneous
    combinations of GNN operators.

    This model is built dynamically from an architecture configuration
    and supports:
    - Mixed layer types (GCN, GAT, SAGE, GIN, etc.)
    - Attention heads for attention-based layers
    - Skip connections
    - Configurable activation and normalization
    - Global graph pooling for graph-level tasks

    Attributes:
        layers: ModuleList of GNN layers
        batch_norms: ModuleList of BatchNorm layers (if enabled)
        pooling: Global pooling layer
        classifier: Final linear layer for output

    Args:
        architecture: Architecture configuration dictionary
        in_channels: Number of input features
        out_channels: Number of output features
    """

    def __init__(
        self,
        architecture: dict[str, Any],
        in_channels: int,
        out_channels: int,
    ):
        """
        Initialize HeterogeneousGNN.

        Args:
            architecture: Architecture configuration with 'layers' list
            in_channels: Number of input features
            out_channels: Number of output features
        """
        super().__init__()

        self.architecture = architecture
        self.use_skip = architecture.get("use_skip_connections", False)
        self.dropout_p = architecture.get("dropout", 0.0)
        self.activation_name = architecture.get("activation", "relu")
        self.use_batch_norm = architecture.get("batch_norm", True)
        self.pooling_type = architecture.get("pooling", "mean")

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        layer_configs = architecture["layers"]
        current_channels = in_channels

        for i, layer_config in enumerate(layer_configs):
            layer_type = layer_config["type"]
            hidden_channels = layer_config["hidden_channels"]
            heads = layer_config.get("heads", 1)

            layer = self._create_layer(
                layer_type=layer_type,
                in_channels=current_channels,
                out_channels=hidden_channels,
                heads=heads,
            )
            self.layers.append(layer)

            if layer_type in ("gat", "gatv2", "transformer"):
                layer_out_channels = hidden_channels * heads
            else:
                layer_out_channels = hidden_channels

            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(layer_out_channels))

            if self.use_skip and current_channels != layer_out_channels:
                self.skips.append(nn.Linear(current_channels, layer_out_channels))
            elif self.use_skip:
                self.skips.append(nn.Identity())

            current_channels = layer_out_channels

        self.dropout = nn.Dropout(self.dropout_p)
        self.activation = self._get_activation(self.activation_name)

        self.pooling = self._create_pooling(self.pooling_type)

        self.classifier = nn.Linear(current_channels, out_channels)

    def _create_layer(
        self,
        layer_type: str,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
    ) -> nn.Module:
        """
        Create a GNN layer based on type.

        Args:
            layer_type: Type of layer ('gcn', 'gat', 'sage', 'gin', etc.)
            in_channels: Number of input channels
            out_channels: Number of output channels
            heads: Number of attention heads (for attention-based layers)

        Returns:
            Configured GNN layer module
        """
        try:
            from torch_geometric.nn import (
                GATConv,
                GATv2Conv,
                GCNConv,
                GINConv,
                SAGEConv,
                TransformerConv,
            )
        except ImportError:
            raise HPOError(
                "PyTorch Geometric not available",
                details="Cannot create GNN layers without torch_geometric",
            )

        layer_type = layer_type.lower()

        if layer_type == "gcn":
            return GCNConv(in_channels, out_channels)

        elif layer_type == "gat":
            return GATConv(
                in_channels, out_channels, heads=heads, concat=True, dropout=self.dropout_p
            )

        elif layer_type == "gatv2":
            return GATv2Conv(
                in_channels, out_channels, heads=heads, concat=True, dropout=self.dropout_p
            )

        elif layer_type == "sage":
            return SAGEConv(in_channels, out_channels)

        elif layer_type == "gin":
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            return GINConv(mlp)

        elif layer_type == "transformer":
            return TransformerConv(
                in_channels, out_channels, heads=heads, concat=True, dropout=self.dropout_p
            )

        else:
            logger.warning(f"Unknown layer type '{layer_type}', using GCN")
            return GCNConv(in_channels, out_channels)

    def _get_activation(self, name: str) -> nn.Module:
        """
        Get activation function by name.

        Args:
            name: Activation function name

        Returns:
            Activation module
        """
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "prelu": nn.PReLU(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def _create_pooling(self, pooling_type: str):
        """
        Create global pooling layer.

        Args:
            pooling_type: Type of pooling ('mean', 'max', 'sum', 'attention')

        Returns:
            Pooling function or module
        """
        try:
            from torch_geometric.nn import (
                GlobalAttention,
                global_add_pool,
                global_max_pool,
                global_mean_pool,
            )
        except ImportError:
            raise HPOError(
                "PyTorch Geometric not available",
                details="Cannot create pooling without torch_geometric",
            )

        pooling_type = pooling_type.lower()

        if pooling_type == "mean":
            return global_mean_pool
        elif pooling_type == "max":
            return global_max_pool
        elif pooling_type == "sum":
            return global_add_pool
        elif pooling_type == "attention":
            gate_nn = nn.Linear(self.architecture["hidden_channels"], 1)
            return GlobalAttention(gate_nn)
        else:
            return global_mean_pool

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the heterogeneous GNN.

        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector for graph-level pooling [num_nodes]

        Returns:
            Output tensor [batch_size, out_channels] for graph-level tasks
            or [num_nodes, out_channels] for node-level tasks
        """
        for i, layer in enumerate(self.layers):
            identity = x

            x = layer(x, edge_index)

            if self.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)

            x = self.activation(x)
            x = self.dropout(x)

            if self.use_skip and i < len(self.skips):
                x = x + self.skips[i](identity)

        if batch is not None:
            x = self.pooling(x, batch)

        x = self.classifier(x)

        return x


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_nas_manager(
    arch_space: GNNArchitectureSpace | None = None, n_trials: int = 100, **kwargs
) -> NASManager:
    """
    Convenience function to create NASManager.

    Creates a NASManager with common default settings for quick setup.

    Args:
        arch_space: Architecture search space (default: GNNArchitectureSpace())
        n_trials: Number of trials to run
        **kwargs: Additional NASConfig parameters

    Returns:
        NASManager instance ready for architecture search

    Examples:
        >>> # Quick setup with defaults
        >>> nas = create_nas_manager()

        >>> # Custom architecture space
        >>> arch_space = GNNArchitectureSpace(
        ...     layer_types=[LayerType.GAT, LayerType.SAGE]
        ... )
        >>> nas = create_nas_manager(arch_space, n_trials=50)
    """
    if arch_space is None:
        arch_space = GNNArchitectureSpace()

    nas_config = NASConfig(n_trials=n_trials, **kwargs)

    return NASManager(arch_space, nas_config=nas_config)


def get_default_gnn_search_space() -> GNNArchitectureSpace:
    """
    Get the default GNN architecture search space.

    Returns a search space suitable for general-purpose GNN
    architecture search with reasonable defaults.

    Returns:
        Default GNNArchitectureSpace instance

    Examples:
        >>> space = get_default_gnn_search_space()
        >>> nas = NASManager(space)
    """
    return GNNArchitectureSpace()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "NASConfig",
    "NASManager",
    "HeterogeneousGNN",
    "create_nas_manager",
    "get_default_gnn_search_space",
]
