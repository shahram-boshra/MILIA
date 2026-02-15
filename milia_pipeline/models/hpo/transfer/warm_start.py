"""
Warm-Start Strategies for HPO Transfer Learning

This module provides the WarmStartStrategy class for warm-starting hyperparameter
optimization using knowledge transferred from previous studies.

Warm-starting enables:
- Faster convergence by starting with promising hyperparameter configurations
- Transfer of knowledge between related tasks/datasets
- Few-shot optimization on new tasks using prior experience
- Reduced computational cost for similar optimization problems

Warm-Start Methods:
1. weighted_transfer: Weight trials by source study similarity scores
2. filtered_transfer: Filter trials compatible with target search space
3. full_transfer: Transfer top-k trials from source studies
4. adaptive_transfer: Automatically select best strategy based on context

The module is designed to work with:
- Optuna trials and studies
- The HPOTransferManager for cross-study transfer
- MetaFeatureExtractor for similarity computation

Author: Milia Team
Version: 1.1.0

Pydantic V2 Migration (Phase 13):
    - Migrated WarmStartConfig from @dataclass(frozen=True) to BaseModel with frozen=True
    - Migrated TransferredTrial from @dataclass to mutable BaseModel
    - Uses @field_validator for individual field validation (n_trials, min_similarity, noise_scale)
    - NON-BREAKING: Same constructor API and attribute access
    - Follows established patterns from transfer_manager.py (Phase 11) and meta_features.py (Phase 12)

Pattern References:
- Frozen dataclass pattern: hpo_config.py (lines 59-131)
- Static method pattern: meta_features.py (lines 219-242)
- Error handling pattern: exceptions.py (lines 2892-3201)
- Transfer manager integration: MILIA_HPO_Implementation_Blueprint.md (lines 4010-4049)
"""

import logging
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# LAZY IMPORTS FOR OPTIONAL DEPENDENCIES
# =============================================================================


def _lazy_import_optuna():
    """Lazy import optuna to avoid import errors if not available."""
    try:
        import optuna

        return optuna
    except ImportError:
        return None


# =============================================================================
# WARM-START STRATEGY ENUM
# =============================================================================


class WarmStartMethod(Enum):
    """
    Available warm-start methods for HPO transfer.

    Each method has different trade-offs between transfer fidelity,
    computational cost, and adaptability to the target task.

    Attributes:
        WEIGHTED: Weight trials by similarity scores (recommended for related tasks)
        FILTERED: Filter trials compatible with target search space
        FULL: Transfer top-k trials without modification
        ADAPTIVE: Automatically select best strategy
    """

    WEIGHTED = "weighted"
    FILTERED = "filtered"
    FULL = "full"
    ADAPTIVE = "adaptive"


# =============================================================================
# WARM-START CONFIGURATION
# =============================================================================


class WarmStartConfig(BaseModel, frozen=True):
    """
    Configuration for warm-start strategy.

    Pattern: Follows frozen BaseModel pattern from transfer_manager.py (Pydantic V2)

    Controls how trials are transferred from source studies to
    target study during warm-starting.

    Attributes:
        method: Warm-start method to use
        n_trials: Maximum number of trials to transfer
        min_similarity: Minimum similarity threshold for weighted transfer
        weight_by_performance: Whether to also weight by trial performance
        filter_invalid: Whether to filter out invalid parameter combinations
        scale_to_bounds: Whether to scale transferred values to target bounds
        add_noise: Whether to add noise to transferred values for diversity
        noise_scale: Scale of noise to add (fraction of parameter range)

    Examples:
        >>> # Default configuration (weighted transfer)
        >>> config = WarmStartConfig()

        >>> # Filtered transfer with strict compatibility
        >>> config = WarmStartConfig(
        ...     method=WarmStartMethod.FILTERED,
        ...     filter_invalid=True
        ... )

        >>> # Transfer with diversity noise
        >>> config = WarmStartConfig(
        ...     method=WarmStartMethod.WEIGHTED,
        ...     add_noise=True,
        ...     noise_scale=0.1
        ... )
    """

    method: WarmStartMethod = WarmStartMethod.WEIGHTED
    n_trials: int = 10
    min_similarity: float = 0.5
    weight_by_performance: bool = True
    filter_invalid: bool = True
    scale_to_bounds: bool = True
    add_noise: bool = False
    noise_scale: float = 0.05

    @field_validator("n_trials")
    @classmethod
    def validate_n_trials(cls, v: int) -> int:
        """Validate n_trials is at least 1."""
        if v < 1:
            raise ValueError("n_trials must be at least 1")
        return v

    @field_validator("min_similarity")
    @classmethod
    def validate_min_similarity(cls, v: float) -> float:
        """Validate min_similarity is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("min_similarity must be between 0 and 1")
        return v

    @field_validator("noise_scale")
    @classmethod
    def validate_noise_scale(cls, v: float) -> float:
        """Validate noise_scale is between 0 and 1."""
        if v < 0.0 or v > 1.0:
            raise ValueError("noise_scale must be between 0 and 1")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()


# =============================================================================
# TRIAL INFO DATACLASS
# =============================================================================


class TransferredTrial(BaseModel):
    """
    Information about a trial prepared for transfer.

    Pattern: Follows mutable BaseModel pattern from transfer_manager.py (Pydantic V2)

    Contains the trial parameters and metadata needed for
    warm-starting a new study.

    Attributes:
        params: Hyperparameter dictionary
        value: Objective value from source study (if available)
        source_study: Name of source study
        similarity: Similarity score to target dataset
        weight: Combined weight for selection
        is_valid: Whether parameters are valid for target space
    """

    params: dict[str, Any]
    value: float | None = None
    source_study: str | None = None
    similarity: float = 1.0
    weight: float = 1.0
    is_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()


# =============================================================================
# WARM-START STRATEGY CLASS
# =============================================================================


class WarmStartStrategy:
    """
    Strategies for warm-starting hyperparameter optimization.

    Provides multiple methods for transferring trials from source studies
    to a target study, enabling faster convergence and knowledge transfer.

    This class implements the WarmStartStrategy interface defined in
    MILIA_HPO_Implementation_Blueprint.md (lines 4115-4133).

    Usage:
        >>> # Static methods for simple transfer
        >>> trials = WarmStartStrategy.weighted_transfer(source_trials, similarities)
        >>> trials = WarmStartStrategy.filtered_transfer(source_trials, target_space)

        >>> # Instance method for configured transfer
        >>> strategy = WarmStartStrategy(config)
        >>> trials = strategy.transfer(source_trials, target_space, similarities)

        >>> # Apply to Optuna study
        >>> n_enqueued = strategy.apply_to_study(target_study, transferred_trials)
    """

    def __init__(self, config: WarmStartConfig | None = None):
        """
        Initialize WarmStartStrategy with optional configuration.

        Args:
            config: Warm-start configuration (uses defaults if None)
        """
        self.config = config or WarmStartConfig()
        self._optuna = _lazy_import_optuna()

    # =========================================================================
    # PRIMARY STATIC METHODS (Match Blueprint API)
    # =========================================================================

    @staticmethod
    def weighted_transfer(
        source_trials: list[Any],
        similarities: list[float],
        n_trials: int | None = None,
        min_similarity: float = 0.0,
        weight_by_performance: bool = True,
    ) -> list[TransferredTrial]:
        """
        Weight trials by source study similarity.

        This method implements weighted transfer as specified in
        MILIA_HPO_Implementation_Blueprint.md (lines 4118-4124).

        Trials are weighted by:
        1. Similarity score between source and target datasets
        2. Optionally, the trial's performance in the source study

        The highest-weighted trials are selected for transfer.

        Args:
            source_trials: List of Optuna FrozenTrial objects or trial dicts
            similarities: List of similarity scores (one per trial or per source study)
            n_trials: Maximum number of trials to return (None for all)
            min_similarity: Minimum similarity threshold
            weight_by_performance: Whether to also weight by performance

        Returns:
            List of TransferredTrial objects sorted by weight (descending)

        Examples:
            >>> trials = WarmStartStrategy.weighted_transfer(
            ...     source_trials=study.trials,
            ...     similarities=[0.9] * len(study.trials),
            ...     n_trials=10
            ... )
        """
        if not source_trials:
            return []

        transferred: list[TransferredTrial] = []

        for i, trial in enumerate(source_trials):
            sim = similarities[i] if i < len(similarities) else similarities[-1]

            if sim < min_similarity:
                continue

            params = WarmStartStrategy._extract_params(trial)
            value = WarmStartStrategy._extract_value(trial)

            weight = sim
            if weight_by_performance and value is not None:
                perf_weight = 1.0 / (1.0 + abs(value))
                weight = sim * perf_weight

            transferred.append(
                TransferredTrial(
                    params=params,
                    value=value,
                    similarity=sim,
                    weight=weight,
                    is_valid=True,
                )
            )

        transferred.sort(key=lambda t: t.weight, reverse=True)

        if n_trials is not None:
            transferred = transferred[:n_trials]

        return transferred

    @staticmethod
    def filtered_transfer(
        source_trials: list[Any],
        target_search_space: dict[str, Any],
        n_trials: int | None = None,
        scale_to_bounds: bool = True,
    ) -> list[TransferredTrial]:
        """
        Filter trials compatible with target search space.

        This method implements filtered transfer as specified in
        MILIA_HPO_Implementation_Blueprint.md (lines 4126-4132).

        Trials are filtered based on:
        1. Parameter name compatibility (must exist in target space)
        2. Parameter value validity (must be within target bounds)
        3. Parameter type compatibility (categorical must match choices)

        Optionally scales numeric parameters to target bounds.

        Args:
            source_trials: List of Optuna FrozenTrial objects or trial dicts
            target_search_space: Target search space definition
            n_trials: Maximum number of trials to return (None for all)
            scale_to_bounds: Whether to scale values to target bounds

        Returns:
            List of TransferredTrial objects that are compatible

        Examples:
            >>> target_space = {
            ...     'lr': {'type': 'float', 'low': 1e-5, 'high': 1e-2},
            ...     'hidden_dim': {'type': 'int', 'low': 32, 'high': 256}
            ... }
            >>> trials = WarmStartStrategy.filtered_transfer(
            ...     source_trials=study.trials,
            ...     target_search_space=target_space
            ... )
        """
        if not source_trials:
            return []

        flat_space = WarmStartStrategy._flatten_search_space(target_search_space)
        transferred: list[TransferredTrial] = []

        for trial in source_trials:
            params = WarmStartStrategy._extract_params(trial)
            value = WarmStartStrategy._extract_value(trial)

            filtered_params, is_valid = WarmStartStrategy._filter_params(
                params, flat_space, scale_to_bounds
            )

            if is_valid and filtered_params:
                transferred.append(
                    TransferredTrial(
                        params=filtered_params,
                        value=value,
                        is_valid=True,
                        weight=1.0,
                    )
                )

        transferred.sort(key=lambda t: t.value if t.value is not None else float("inf"))

        if n_trials is not None:
            transferred = transferred[:n_trials]

        return transferred

    @staticmethod
    def full_transfer(
        source_trials: list[Any],
        n_trials: int = 10,
        sort_by_performance: bool = True,
    ) -> list[TransferredTrial]:
        """
        Transfer top-k trials from source studies without modification.

        Simple transfer strategy that takes the best trials from source
        studies based on their objective values.

        Args:
            source_trials: List of Optuna FrozenTrial objects or trial dicts
            n_trials: Number of trials to transfer
            sort_by_performance: Whether to sort by performance (best first)

        Returns:
            List of TransferredTrial objects

        Examples:
            >>> trials = WarmStartStrategy.full_transfer(
            ...     source_trials=study.trials,
            ...     n_trials=5
            ... )
        """
        if not source_trials:
            return []

        transferred: list[TransferredTrial] = []

        for trial in source_trials:
            params = WarmStartStrategy._extract_params(trial)
            value = WarmStartStrategy._extract_value(trial)

            transferred.append(
                TransferredTrial(
                    params=params,
                    value=value,
                    is_valid=True,
                    weight=1.0,
                )
            )

        if sort_by_performance:
            transferred.sort(key=lambda t: t.value if t.value is not None else float("inf"))

        return transferred[:n_trials]

    # =========================================================================
    # INSTANCE METHODS
    # =========================================================================

    def transfer(
        self,
        source_trials: list[Any],
        target_search_space: dict[str, Any] | None = None,
        similarities: list[float] | None = None,
    ) -> list[TransferredTrial]:
        """
        Transfer trials using the configured strategy.

        Automatically selects the appropriate transfer method based
        on the configuration and available inputs.

        Args:
            source_trials: List of source trials
            target_search_space: Target search space (required for filtered)
            similarities: Similarity scores (required for weighted)

        Returns:
            List of TransferredTrial objects ready for application

        Raises:
            ValueError: If required inputs for the method are missing
        """
        method = self.config.method

        if method == WarmStartMethod.ADAPTIVE:
            method = self._select_adaptive_method(source_trials, target_search_space, similarities)

        if method == WarmStartMethod.WEIGHTED:
            if similarities is None:
                similarities = [1.0] * len(source_trials)

            transferred = self.weighted_transfer(
                source_trials=source_trials,
                similarities=similarities,
                n_trials=self.config.n_trials,
                min_similarity=self.config.min_similarity,
                weight_by_performance=self.config.weight_by_performance,
            )

        elif method == WarmStartMethod.FILTERED:
            if target_search_space is None:
                raise ValueError("target_search_space required for filtered transfer")

            transferred = self.filtered_transfer(
                source_trials=source_trials,
                target_search_space=target_search_space,
                n_trials=self.config.n_trials,
                scale_to_bounds=self.config.scale_to_bounds,
            )

        elif method == WarmStartMethod.FULL:
            transferred = self.full_transfer(
                source_trials=source_trials,
                n_trials=self.config.n_trials,
                sort_by_performance=True,
            )

        else:
            raise ValueError(f"Unknown transfer method: {method}")

        if self.config.filter_invalid and target_search_space is not None:
            transferred = self._filter_invalid_trials(transferred, target_search_space)

        if self.config.add_noise and target_search_space is not None:
            transferred = self._add_noise_to_trials(transferred, target_search_space)

        return transferred

    def apply_to_study(
        self,
        study: Any,
        transferred_trials: list[TransferredTrial],
    ) -> int:
        """
        Apply transferred trials to an Optuna study.

        Enqueues the transferred trial parameters to the study
        so they will be evaluated first during optimization.

        Args:
            study: Optuna study object
            transferred_trials: List of trials to enqueue

        Returns:
            Number of trials successfully enqueued

        Examples:
            >>> strategy = WarmStartStrategy()
            >>> transferred = strategy.transfer(source_trials, target_space)
            >>> n_enqueued = strategy.apply_to_study(target_study, transferred)
            >>> print(f"Enqueued {n_enqueued} trials for warm-start")
        """
        if self._optuna is None:
            logger.warning("Optuna not available, cannot apply to study")
            return 0

        n_enqueued = 0

        for trial in transferred_trials:
            if not trial.is_valid:
                continue

            try:
                study.enqueue_trial(trial.params)
                n_enqueued += 1
                logger.debug(f"Enqueued trial with params: {trial.params}")
            except Exception as e:
                logger.warning(f"Failed to enqueue trial: {e}")

        logger.info(f"Warm-start: enqueued {n_enqueued}/{len(transferred_trials)} trials")
        return n_enqueued

    # =========================================================================
    # HELPER METHODS - PARAMETER EXTRACTION
    # =========================================================================

    @staticmethod
    def _extract_params(trial: Any) -> dict[str, Any]:
        """Extract parameters from various trial formats."""
        if isinstance(trial, dict):
            if "params" in trial:
                return dict(trial["params"])
            return dict(trial)
        elif hasattr(trial, "params") and not callable(getattr(trial, "params", None)):
            return dict(trial.params)
        else:
            return {}

    @staticmethod
    def _extract_value(trial: Any) -> float | None:
        """Extract objective value from various trial formats."""
        if isinstance(trial, dict):
            if "value" in trial:
                return trial["value"]
            if "values" in trial:
                values = trial["values"]
                if values:
                    return values[0] if isinstance(values, (list, tuple)) else values
        elif hasattr(trial, "value") and not callable(getattr(trial, "value", None)):
            return trial.value
        elif hasattr(trial, "values") and not callable(getattr(trial, "values", None)):
            values = trial.values
            if values:
                return values[0] if isinstance(values, (list, tuple)) else values
        return None

    # =========================================================================
    # HELPER METHODS - SEARCH SPACE PROCESSING
    # =========================================================================

    @staticmethod
    def _flatten_search_space(search_space: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        Flatten nested search space to {param_name: param_config} format.

        Handles both flat and nested (category -> params) search space formats.
        """
        flat: dict[str, dict[str, Any]] = {}

        for key, value in search_space.items():
            if isinstance(value, dict):
                if "type" in value:
                    flat[key] = value
                else:
                    for param_name, param_config in value.items():
                        if isinstance(param_config, dict):
                            flat[param_name] = param_config

        return flat

    @staticmethod
    def _filter_params(
        params: dict[str, Any],
        target_space: dict[str, dict[str, Any]],
        scale_to_bounds: bool = True,
    ) -> tuple[dict[str, Any], bool]:
        """
        Filter and adapt parameters to target search space.

        Returns:
            Tuple of (filtered_params, is_valid)
        """
        filtered: dict[str, Any] = {}
        is_valid = True

        for param_name, param_value in params.items():
            if param_name not in target_space:
                continue

            target_config = target_space[param_name]
            param_type = target_config.get("type", "float")

            if isinstance(param_type, Enum):
                param_type = param_type.value

            if param_type == "categorical":
                choices = target_config.get("choices", [])
                if param_value in choices:
                    filtered[param_name] = param_value
                else:
                    is_valid = False

            elif param_type in ("int", "int_uniform"):
                low = target_config.get("low")
                high = target_config.get("high")

                if low is not None and high is not None:
                    if scale_to_bounds:
                        param_value = max(low, min(high, int(param_value)))
                    elif not (low <= param_value <= high):
                        is_valid = False
                        continue

                filtered[param_name] = int(param_value)

            elif param_type in ("float", "loguniform", "uniform", "discrete_uniform"):
                low = target_config.get("low")
                high = target_config.get("high")

                if low is not None and high is not None:
                    if scale_to_bounds:
                        param_value = max(low, min(high, float(param_value)))
                    elif not (low <= param_value <= high):
                        is_valid = False
                        continue

                filtered[param_name] = float(param_value)

            else:
                filtered[param_name] = param_value

        return filtered, is_valid and len(filtered) > 0

    # =========================================================================
    # HELPER METHODS - TRIAL PROCESSING
    # =========================================================================

    def _filter_invalid_trials(
        self,
        trials: list[TransferredTrial],
        target_space: dict[str, Any],
    ) -> list[TransferredTrial]:
        """Filter out trials with invalid parameters."""
        flat_space = self._flatten_search_space(target_space)
        valid_trials: list[TransferredTrial] = []

        for trial in trials:
            _, is_valid = self._filter_params(trial.params, flat_space, scale_to_bounds=False)
            if is_valid:
                valid_trials.append(trial)

        return valid_trials

    def _add_noise_to_trials(
        self,
        trials: list[TransferredTrial],
        target_space: dict[str, Any],
    ) -> list[TransferredTrial]:
        """Add noise to numeric parameters for diversity."""
        flat_space = self._flatten_search_space(target_space)
        noisy_trials: list[TransferredTrial] = []

        for trial in trials:
            noisy_params = dict(trial.params)

            for param_name, param_value in trial.params.items():
                if param_name not in flat_space:
                    continue

                config = flat_space[param_name]
                param_type = config.get("type", "float")

                if isinstance(param_type, Enum):
                    param_type = param_type.value

                if param_type in ("float", "loguniform", "uniform"):
                    low = config.get("low", 0)
                    high = config.get("high", 1)
                    range_size = high - low
                    noise = np.random.normal(0, self.config.noise_scale * range_size)
                    noisy_value = max(low, min(high, param_value + noise))
                    noisy_params[param_name] = noisy_value

                elif param_type in ("int", "int_uniform"):
                    low = config.get("low", 0)
                    high = config.get("high", 100)
                    range_size = high - low
                    noise = int(np.random.normal(0, self.config.noise_scale * range_size))
                    noisy_value = max(low, min(high, int(param_value) + noise))
                    noisy_params[param_name] = noisy_value

            noisy_trials.append(
                TransferredTrial(
                    params=noisy_params,
                    value=trial.value,
                    source_study=trial.source_study,
                    similarity=trial.similarity,
                    weight=trial.weight,
                    is_valid=trial.is_valid,
                )
            )

        return noisy_trials

    def _select_adaptive_method(
        self,
        source_trials: list[Any],
        target_space: dict[str, Any] | None,
        similarities: list[float] | None,
    ) -> WarmStartMethod:
        """Select the best transfer method based on available inputs."""
        if similarities is not None and any(s < 1.0 for s in similarities):
            return WarmStartMethod.WEIGHTED
        elif target_space is not None:
            return WarmStartMethod.FILTERED
        else:
            return WarmStartMethod.FULL

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def get_transfer_summary(transferred_trials: list[TransferredTrial]) -> dict[str, Any]:
        """
        Get summary statistics for transferred trials.

        Args:
            transferred_trials: List of transferred trials

        Returns:
            Dictionary with summary statistics
        """
        if not transferred_trials:
            return {
                "n_trials": 0,
                "n_valid": 0,
                "mean_similarity": 0.0,
                "mean_weight": 0.0,
                "best_value": None,
            }

        valid_trials = [t for t in transferred_trials if t.is_valid]
        values = [t.value for t in transferred_trials if t.value is not None]

        return {
            "n_trials": len(transferred_trials),
            "n_valid": len(valid_trials),
            "mean_similarity": float(np.mean([t.similarity for t in transferred_trials])),
            "mean_weight": float(np.mean([t.weight for t in transferred_trials])),
            "best_value": min(values) if values else None,
            "param_names": list(transferred_trials[0].params.keys()) if transferred_trials else [],
        }

    @staticmethod
    def create_from_best_trials(
        study: Any,
        n_trials: int = 10,
        include_pruned: bool = False,
    ) -> list[TransferredTrial]:
        """
        Create transferred trials from the best trials of a study.

        Convenience method for extracting the best trials from
        a completed Optuna study for transfer.

        Args:
            study: Optuna study object
            n_trials: Number of best trials to extract
            include_pruned: Whether to include pruned trials

        Returns:
            List of TransferredTrial objects
        """
        optuna = _lazy_import_optuna()
        if optuna is None:
            logger.warning("Optuna not available")
            return []

        trials = study.trials

        if not include_pruned:
            trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

        trials = sorted(trials, key=lambda t: t.value if t.value is not None else float("inf"))

        return [
            TransferredTrial(
                params=dict(t.params),
                value=t.value,
                is_valid=True,
                weight=1.0,
            )
            for t in trials[:n_trials]
        ]


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "WarmStartMethod",
    "WarmStartConfig",
    "TransferredTrial",
    "WarmStartStrategy",
]
