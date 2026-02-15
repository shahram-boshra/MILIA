"""
HPO Transfer Manager for Cross-Study Hyperparameter Transfer Learning

This module provides the HPOTransferManager class for transferring hyperparameter
optimization knowledge between related studies, enabling faster convergence and
improved performance on new tasks.

Transfer Learning Use Cases:
1. Transfer from small dataset to large dataset
2. Transfer between related molecular properties (e.g., DFT to experimental)
3. Few-shot optimization on new tasks using prior experience
4. Cross-domain adaptation (e.g., different molecule families)

The module integrates with:
- MetaFeatureExtractor: For dataset similarity computation
- WarmStartStrategy: For trial transfer mechanisms
- Optuna studies: For trial management and enqueuing

Research Basis:
- Meta-learning for hyperparameter optimization
- Dataset similarity measures using meta-features
- Warm-starting Bayesian optimization
- Transfer learning for AutoML

Author: Milia Team
Version: 1.1.0

Pydantic V2 Migration (Phase 11):
    - Migrated TransferConfig from @dataclass(frozen=True) to BaseModel with frozen=True
    - Migrated RegisteredStudyInfo from @dataclass to mutable BaseModel
    - Uses @field_validator for individual field validation
    - Uses @model_validator(mode='before') for enum conversion and timestamp initialization
    - NON-BREAKING: Same constructor API and attribute access
    - Follows established patterns from hpo_config.py (Phase 6b) and device_manager.py (Phase 7)

Pattern References:
- Frozen BaseModel pattern: hpo_config.py (lines 64-156)
- Error handling pattern: exceptions.py (lines 2892-3201)
- Meta-feature extraction: meta_features.py (lines 165-242)
- Warm-start strategies: warm_start.py (lines 175-807)
- Blueprint specification: MILIA_HPO_Implementation_Blueprint.md (lines 3908-4133)
"""

import json
import logging
import os
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, field_validator, model_validator

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


def _lazy_import_meta_features():
    """Lazy import MetaFeatureExtractor from sibling module."""
    try:
        from .meta_features import MetaFeatureConfig, MetaFeatureExtractor

        return MetaFeatureExtractor, MetaFeatureConfig
    except ImportError:
        try:
            from meta_features import MetaFeatureConfig, MetaFeatureExtractor

            return MetaFeatureExtractor, MetaFeatureConfig
        except ImportError:
            return None, None


def _lazy_import_warm_start():
    """Lazy import WarmStartStrategy from sibling module."""
    try:
        from .warm_start import (
            TransferredTrial,
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        return WarmStartStrategy, WarmStartConfig, WarmStartMethod, TransferredTrial
    except ImportError:
        try:
            from warm_start import (
                TransferredTrial,
                WarmStartConfig,
                WarmStartMethod,
                WarmStartStrategy,
            )

            return WarmStartStrategy, WarmStartConfig, WarmStartMethod, TransferredTrial
        except ImportError:
            return None, None, None, None


def _lazy_import_hpo_error():
    """Lazy import HPOError from exceptions module."""
    try:
        from milia_pipeline.exceptions import HPOError

        return HPOError
    except ImportError:
        return None


# =============================================================================
# TRANSFER CONFIGURATION ENUMS
# =============================================================================


class MetaFeatureMethod(Enum):
    """
    Methods for computing meta-features for similarity.

    Attributes:
        STATISTICAL: Basic statistical features (size, density, target stats)
        LEARNED: Learned meta-features using neural networks (future)
        LANDMARK: Landmark-based meta-features using probe models (future)
    """

    STATISTICAL = "statistical"
    LEARNED = "learned"
    LANDMARK = "landmark"


class AdaptationMethod(Enum):
    """
    Methods for adapting transferred hyperparameters.

    Attributes:
        WEIGHTED: Weight trials by dataset similarity
        FILTERED: Filter trials compatible with target search space
        FULL: Transfer top-k trials without modification
        ADAPTIVE: Automatically select best method based on context
    """

    WEIGHTED = "weighted"
    FILTERED = "filtered"
    FULL = "full"
    ADAPTIVE = "adaptive"


# =============================================================================
# TRANSFER CONFIGURATION BASEMODEL
# =============================================================================


class TransferConfig(BaseModel, frozen=True):
    """
    Configuration for HPO transfer learning.

    Pattern: Follows frozen BaseModel pattern from hpo_config.py (Pydantic V2)

    Controls how hyperparameter knowledge is transferred between
    studies, including similarity computation and trial selection.

    Attributes:
        n_warm_start_trials: Maximum number of trials to transfer
        similarity_threshold: Minimum dataset similarity for transfer (0-1)
        meta_feature_method: Method for computing meta-features
        adaptation_method: Method for adapting transferred trials
        weight_by_performance: Whether to weight by trial performance
        scale_to_bounds: Whether to scale values to target bounds
        add_noise: Whether to add diversity noise to transferred trials
        noise_scale: Scale of noise (fraction of parameter range)
        persist_meta_db: Whether to persist meta-database to disk
        meta_db_path: Path for persisting meta-database

    Examples:
        >>> # Default configuration
        >>> config = TransferConfig()

        >>> # Strict similarity with weighted adaptation
        >>> config = TransferConfig(
        ...     similarity_threshold=0.8,
        ...     adaptation_method=AdaptationMethod.WEIGHTED,
        ...     weight_by_performance=True
        ... )

        >>> # Transfer with diversity noise
        >>> config = TransferConfig(
        ...     add_noise=True,
        ...     noise_scale=0.1
        ... )
    """

    n_warm_start_trials: int = 10
    similarity_threshold: float = 0.7
    meta_feature_method: MetaFeatureMethod = MetaFeatureMethod.STATISTICAL
    adaptation_method: AdaptationMethod = AdaptationMethod.WEIGHTED
    weight_by_performance: bool = True
    scale_to_bounds: bool = True
    add_noise: bool = False
    noise_scale: float = 0.05
    persist_meta_db: bool = False
    meta_db_path: str | None = None

    @field_validator("n_warm_start_trials")
    @classmethod
    def validate_n_warm_start_trials(cls, v: int) -> int:
        """Validate n_warm_start_trials is at least 1."""
        if v < 1:
            raise ValueError("n_warm_start_trials must be at least 1")
        return v

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v: float) -> float:
        """Validate similarity_threshold is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        return v

    @field_validator("noise_scale")
    @classmethod
    def validate_noise_scale(cls, v: float) -> float:
        """Validate noise_scale is between 0 and 1."""
        if v < 0.0 or v > 1.0:
            raise ValueError("noise_scale must be between 0 and 1")
        return v

    @model_validator(mode="before")
    @classmethod
    def convert_enums_and_validate_cross_fields(cls, data: Any) -> Any:
        """Convert string enums and validate cross-field dependencies."""
        if isinstance(data, dict):
            # Convert string to MetaFeatureMethod enum if needed
            if "meta_feature_method" in data and isinstance(data["meta_feature_method"], str):
                data["meta_feature_method"] = MetaFeatureMethod(data["meta_feature_method"])

            # Convert string to AdaptationMethod enum if needed
            if "adaptation_method" in data and isinstance(data["adaptation_method"], str):
                data["adaptation_method"] = AdaptationMethod(data["adaptation_method"])

            # Validate meta_db_path is required when persist_meta_db is True
            persist_meta_db = data.get("persist_meta_db", False)
            meta_db_path = data.get("meta_db_path")
            if persist_meta_db and meta_db_path is None:
                raise ValueError("meta_db_path must be provided when persist_meta_db is True")

        return data

    def to_dict(self) -> dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()


# =============================================================================
# STUDY INFO BASEMODEL
# =============================================================================


class RegisteredStudyInfo(BaseModel):
    """
    Information about a registered study for transfer.

    Pattern: Follows mutable BaseModel pattern from device_manager.py (Pydantic V2)

    Contains all metadata needed for similarity computation
    and trial transfer from a completed HPO study.

    Attributes:
        study_name: Unique identifier for the study
        meta_features: Computed meta-features of the dataset
        best_params: Best hyperparameters found
        best_value: Best objective value achieved
        n_trials: Total number of trials in study
        n_completed: Number of completed trials
        direction: Optimization direction ('minimize' or 'maximize')
        model_name: Name of the model being optimized
        dataset_info: Optional additional dataset information
        registered_at: Timestamp of registration
    """

    study_name: str
    meta_features: dict[str, float]
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    n_completed: int = 0
    direction: str = "minimize"
    model_name: str | None = None
    dataset_info: dict[str, Any] | None = None
    registered_at: str | None = None

    @model_validator(mode="before")
    @classmethod
    def set_registration_timestamp(cls, data: Any) -> Any:
        """Set registration timestamp if not provided."""
        if isinstance(data, dict):
            if data.get("registered_at") is None:
                data["registered_at"] = datetime.now().isoformat()
        return data

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Backward compatible method wrapping Pydantic V2's model_dump().
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegisteredStudyInfo":
        """Create from dictionary."""
        return cls.model_validate(data)


# =============================================================================
# HPO TRANSFER MANAGER CLASS
# =============================================================================


class HPOTransferManager:
    """
    Manages hyperparameter transfer between HPO studies.

    This class enables knowledge transfer between related optimization
    tasks using meta-learning principles. It maintains a database of
    completed studies and their meta-features, allowing new optimization
    tasks to leverage prior experience.

    Use cases:
    - Transfer from small dataset to large dataset
    - Transfer between related molecular properties
    - Transfer from DFT to experimental data
    - Few-shot optimization on new tasks

    Research basis:
    - Meta-learning for hyperparameter optimization
    - Dataset similarity measures
    - Warm-starting Bayesian optimization

    Integration:
    - Uses MetaFeatureExtractor for dataset characterization
    - Uses WarmStartStrategy for trial transfer mechanisms
    - Works with Optuna studies for trial management

    Usage:
        >>> # Initialize manager
        >>> config = TransferConfig(n_warm_start_trials=10)
        >>> manager = HPOTransferManager(config)

        >>> # Register a completed study
        >>> manager.register_study(
        ...     study_name="gcn_qm9",
        ...     study=completed_study,
        ...     dataset=qm9_dataset
        ... )

        >>> # Find similar studies for a new dataset
        >>> similar = manager.find_similar_studies(new_dataset, top_k=3)

        >>> # Warm-start a new study
        >>> n_transferred = manager.warm_start_study(
        ...     target_study=new_study,
        ...     source_studies=similar
        ... )

    Attributes:
        config: Transfer configuration
        _meta_db: Internal database of registered studies
        _optuna: Lazy-loaded Optuna module
        _meta_extractor: Lazy-loaded MetaFeatureExtractor
        _warm_start: Lazy-loaded WarmStartStrategy
    """

    def __init__(self, config: TransferConfig | None = None):
        """
        Initialize HPOTransferManager with optional configuration.

        Args:
            config: Transfer configuration (uses defaults if None)
        """
        self.config = config or TransferConfig()
        self._meta_db: dict[str, RegisteredStudyInfo] = {}
        self._study_cache: dict[str, Any] = {}

        self._optuna = _lazy_import_optuna()
        MetaFeatureExtractor, MetaFeatureConfig = _lazy_import_meta_features()
        self._meta_extractor_class = MetaFeatureExtractor
        self._meta_feature_config_class = MetaFeatureConfig
        WarmStartStrategy, WarmStartConfig, WarmStartMethod, TransferredTrial = (
            _lazy_import_warm_start()
        )
        self._warm_start_class = WarmStartStrategy
        self._warm_start_config_class = WarmStartConfig
        self._warm_start_method_class = WarmStartMethod
        self._transferred_trial_class = TransferredTrial
        self._hpo_error = _lazy_import_hpo_error()

        if self.config.persist_meta_db and self.config.meta_db_path:
            self._load_meta_db()

        logger.debug(
            f"HPOTransferManager initialized with config: "
            f"n_warm_start_trials={self.config.n_warm_start_trials}, "
            f"similarity_threshold={self.config.similarity_threshold}"
        )

    # =========================================================================
    # STUDY REGISTRATION
    # =========================================================================

    def register_study(
        self,
        study_name: str,
        study: Any,
        dataset: Any,
        meta_features: dict[str, float] | None = None,
        model_name: str | None = None,
        dataset_info: dict[str, Any] | None = None,
        cache_study: bool = True,
    ) -> RegisteredStudyInfo:
        """
        Register a completed study for future transfer.

        Stores the study metadata and computed meta-features in the
        internal database for later similarity computation and
        trial transfer.

        Args:
            study_name: Unique identifier for the study
            study: Completed Optuna study object
            dataset: Dataset used for the study (PyG dataset or similar)
            meta_features: Pre-computed meta-features (computed if None)
            model_name: Name of the model being optimized
            dataset_info: Additional dataset information to store
            cache_study: Whether to cache the study object for later access

        Returns:
            RegisteredStudyInfo with study metadata

        Raises:
            ValueError: If study_name already exists
            HPOError: If meta-feature extraction fails

        Examples:
            >>> manager.register_study(
            ...     study_name="gcn_qm9_v1",
            ...     study=completed_study,
            ...     dataset=qm9_dataset,
            ...     model_name="GCN"
            ... )
        """
        if study_name in self._meta_db:
            raise ValueError(f"Study '{study_name}' is already registered")

        if meta_features is None:
            meta_features = self._extract_meta_features(dataset)

        try:
            best_params = study.best_params
            best_value = study.best_value
            n_trials = len(study.trials)

            if self._optuna is not None:
                completed_trials = [
                    t for t in study.trials if t.state == self._optuna.trial.TrialState.COMPLETE
                ]
                n_completed = len(completed_trials)
            else:
                n_completed = n_trials

            direction = "minimize"
            if hasattr(study, "direction"):
                if hasattr(study.direction, "name"):
                    direction = study.direction.name.lower()
                else:
                    direction = str(study.direction).lower()

        except Exception as e:
            self._raise_hpo_error(
                f"Failed to extract study information: {e}", study_name=study_name, details=str(e)
            )

        study_info = RegisteredStudyInfo(
            study_name=study_name,
            meta_features=meta_features,
            best_params=best_params,
            best_value=best_value,
            n_trials=n_trials,
            n_completed=n_completed,
            direction=direction,
            model_name=model_name,
            dataset_info=dataset_info,
        )

        self._meta_db[study_name] = study_info

        if cache_study:
            self._study_cache[study_name] = study

        if self.config.persist_meta_db:
            self._save_meta_db()

        logger.info(
            f"Registered study '{study_name}' with {n_completed}/{n_trials} "
            f"completed trials, best_value={best_value:.6f}"
        )

        return study_info

    def unregister_study(self, study_name: str) -> bool:
        """
        Remove a study from the registry.

        Args:
            study_name: Name of the study to remove

        Returns:
            True if study was removed, False if not found
        """
        if study_name in self._meta_db:
            del self._meta_db[study_name]
            self._study_cache.pop(study_name, None)

            if self.config.persist_meta_db:
                self._save_meta_db()

            logger.info(f"Unregistered study '{study_name}'")
            return True

        return False

    def get_registered_studies(self) -> list[str]:
        """
        Get list of all registered study names.

        Returns:
            List of study names
        """
        return list(self._meta_db.keys())

    def get_study_info(self, study_name: str) -> RegisteredStudyInfo | None:
        """
        Get information about a registered study.

        Args:
            study_name: Name of the study

        Returns:
            RegisteredStudyInfo or None if not found
        """
        return self._meta_db.get(study_name)

    # =========================================================================
    # SIMILARITY COMPUTATION
    # =========================================================================

    def find_similar_studies(
        self,
        target_dataset: Any,
        top_k: int = 3,
        model_name: str | None = None,
        min_trials: int = 1,
    ) -> list[tuple[str, float]]:
        """
        Find most similar registered studies to target dataset.

        Computes similarity between the target dataset and all
        registered studies using meta-features, returning the
        most similar ones above the similarity threshold.

        Args:
            target_dataset: New dataset to optimize for
            top_k: Maximum number of similar studies to return
            model_name: Filter by model name (optional)
            min_trials: Minimum completed trials required

        Returns:
            List of (study_name, similarity) tuples, most similar first

        Examples:
            >>> similar = manager.find_similar_studies(
            ...     target_dataset=new_dataset,
            ...     top_k=5,
            ...     model_name="GCN"
            ... )
            >>> for name, sim in similar:
            ...     print(f"{name}: {sim:.3f}")
        """
        if not self._meta_db:
            logger.warning("No studies registered for similarity search")
            return []

        target_features = self._extract_meta_features(target_dataset)

        similarities: list[tuple[str, float]] = []

        for name, info in self._meta_db.items():
            if model_name is not None and info.model_name != model_name:
                continue

            if info.n_completed < min_trials:
                continue

            similarity = self._compute_similarity(target_features, info.meta_features)

            if similarity >= self.config.similarity_threshold:
                similarities.append((name, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        result = similarities[:top_k]

        logger.debug(
            f"Found {len(result)} similar studies above threshold "
            f"{self.config.similarity_threshold} (searched {len(self._meta_db)} studies)"
        )

        return result

    def compute_dataset_similarity(
        self,
        dataset_a: Any,
        dataset_b: Any,
    ) -> float:
        """
        Compute similarity between two datasets.

        Convenience method for direct dataset comparison without
        requiring study registration.

        Args:
            dataset_a: First dataset
            dataset_b: Second dataset

        Returns:
            Similarity score in range [0, 1]
        """
        features_a = self._extract_meta_features(dataset_a)
        features_b = self._extract_meta_features(dataset_b)

        return self._compute_similarity(features_a, features_b)

    def _compute_similarity(
        self,
        features_a: dict[str, float],
        features_b: dict[str, float],
    ) -> float:
        """
        Compute cosine similarity between meta-feature vectors.

        This method matches the similarity computation in
        MILIA_HPO_Implementation_Blueprint.md (lines 4051-4070).

        Args:
            features_a: Meta-features from first dataset
            features_b: Meta-features from second dataset

        Returns:
            Cosine similarity in range [0, 1]
        """
        if self._meta_extractor_class is not None:
            return self._meta_extractor_class.compute_similarity(features_a, features_b)

        common_keys = set(features_a.keys()) & set(features_b.keys())

        if not common_keys:
            return 0.0

        vec_a = np.array([features_a[k] for k in common_keys])
        vec_b = np.array([features_b[k] for k in common_keys])

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

        return max(0.0, min(1.0, similarity))

    # =========================================================================
    # WARM-START OPERATIONS
    # =========================================================================

    def warm_start_study(
        self,
        target_study: Any,
        source_studies: list[str] | None = None,
        target_dataset: Any | None = None,
        target_search_space: dict[str, Any] | None = None,
    ) -> int:
        """
        Warm-start target study with trials from source studies.

        Transfers the best hyperparameter configurations from source
        studies to the target study, enabling faster convergence.

        Args:
            target_study: New Optuna study to warm-start
            source_studies: List of source study names (auto-detected if None)
            target_dataset: Target dataset for auto-detection (required if source_studies is None)
            target_search_space: Target search space for filtering (optional)

        Returns:
            Number of trials successfully transferred

        Raises:
            ValueError: If no source studies provided and target_dataset is None
            HPOError: If warm-starting fails

        Examples:
            >>> # Explicit source studies
            >>> n = manager.warm_start_study(
            ...     target_study=new_study,
            ...     source_studies=['gcn_qm9', 'gcn_esol']
            ... )

            >>> # Auto-detect similar studies
            >>> n = manager.warm_start_study(
            ...     target_study=new_study,
            ...     target_dataset=new_dataset
            ... )
        """
        if source_studies is None:
            if target_dataset is None:
                raise ValueError("Either source_studies or target_dataset must be provided")
            similar = self.find_similar_studies(target_dataset)
            source_studies = [name for name, _ in similar]

        if not source_studies:
            logger.warning("No source studies available for warm-starting")
            return 0

        n_transferred = 0
        trials_per_source = max(1, self.config.n_warm_start_trials // len(source_studies))

        similarities = {}
        for source_name in source_studies:
            if target_dataset is not None:
                target_features = self._extract_meta_features(target_dataset)
                source_info = self._meta_db.get(source_name)
                if source_info:
                    similarities[source_name] = self._compute_similarity(
                        target_features, source_info.meta_features
                    )

        for source_name in source_studies:
            source_info = self._meta_db.get(source_name)
            if source_info is None:
                logger.warning(f"Source study '{source_name}' not found")
                continue

            source_study = self._study_cache.get(source_name)
            if source_study is None:
                logger.warning(f"Study '{source_name}' not in cache, skipping")
                continue

            transferred = self._transfer_trials(
                source_study=source_study,
                target_study=target_study,
                n_trials=trials_per_source,
                similarity=similarities.get(source_name, 1.0),
                target_search_space=target_search_space,
            )

            n_transferred += transferred

            if n_transferred >= self.config.n_warm_start_trials:
                break

        logger.info(
            f"Warm-started study with {n_transferred} trials "
            f"from {len(source_studies)} source studies"
        )

        return n_transferred

    def _transfer_trials(
        self,
        source_study: Any,
        target_study: Any,
        n_trials: int,
        similarity: float = 1.0,
        target_search_space: dict[str, Any] | None = None,
    ) -> int:
        """
        Transfer trials from source to target study.

        Args:
            source_study: Source Optuna study
            target_study: Target Optuna study
            n_trials: Maximum trials to transfer
            similarity: Similarity score for weighting
            target_search_space: Target search space for filtering

        Returns:
            Number of trials transferred
        """
        if self._optuna is None:
            logger.warning("Optuna not available for trial transfer")
            return 0

        completed_trials = [
            t for t in source_study.trials if t.state == self._optuna.trial.TrialState.COMPLETE
        ]

        sorted_trials = sorted(
            completed_trials, key=lambda t: t.value if t.value is not None else float("inf")
        )[:n_trials]

        n_transferred = 0

        for trial in sorted_trials:
            params = dict(trial.params)

            if target_search_space is not None:
                params = self._filter_params(params, target_search_space)

            if self.config.add_noise:
                params = self._add_noise_to_params(params, target_search_space)

            if not params:
                continue

            try:
                target_study.enqueue_trial(params)
                n_transferred += 1
            except Exception as e:
                logger.debug(f"Failed to enqueue trial: {e}")

        return n_transferred

    def _filter_params(
        self,
        params: dict[str, Any],
        search_space: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Filter parameters to match target search space.

        Args:
            params: Source parameters
            search_space: Target search space configuration

        Returns:
            Filtered parameters compatible with target space
        """
        if self._warm_start_class is not None:
            strategy = self._warm_start_class()
            filtered, is_valid = strategy._filter_params(
                params,
                strategy._flatten_search_space(search_space),
                scale_to_bounds=self.config.scale_to_bounds,
            )
            return filtered if is_valid else {}

        flat_space = self._flatten_search_space(search_space)
        filtered: dict[str, Any] = {}

        for param_name, param_value in params.items():
            if param_name not in flat_space:
                continue

            config = flat_space[param_name]
            param_type = config.get("type", "float")

            if hasattr(param_type, "value"):
                param_type = param_type.value

            if param_type == "categorical":
                choices = config.get("choices", [])
                if param_value in choices:
                    filtered[param_name] = param_value

            elif param_type in ("int", "int_uniform"):
                low = config.get("low")
                high = config.get("high")
                if low is not None and high is not None:
                    if self.config.scale_to_bounds:
                        param_value = max(low, min(high, int(param_value)))
                    elif not (low <= param_value <= high):
                        continue
                filtered[param_name] = int(param_value)

            elif param_type in ("float", "loguniform", "uniform"):
                low = config.get("low")
                high = config.get("high")
                if low is not None and high is not None:
                    if self.config.scale_to_bounds:
                        param_value = max(low, min(high, float(param_value)))
                    elif not (low <= param_value <= high):
                        continue
                filtered[param_name] = float(param_value)

            else:
                filtered[param_name] = param_value

        return filtered

    def _flatten_search_space(
        self,
        search_space: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Flatten nested search space to single-level dict.

        Args:
            search_space: Nested search space configuration

        Returns:
            Flattened parameter configurations
        """
        flat: dict[str, Any] = {}

        def _flatten(d: dict[str, Any], prefix: str = ""):
            for key, value in d.items():
                full_key = f"{prefix}{key}" if prefix else key

                if isinstance(value, dict):
                    if "type" in value:
                        flat[full_key] = value
                    else:
                        _flatten(value, f"{full_key}.")
                else:
                    flat[full_key] = value

        _flatten(search_space)
        return flat

    def _add_noise_to_params(
        self,
        params: dict[str, Any],
        search_space: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Add diversity noise to numeric parameters.

        Args:
            params: Parameters to add noise to
            search_space: Search space for bounds

        Returns:
            Parameters with noise added
        """
        if search_space is None:
            return params

        noisy_params = dict(params)
        flat_space = self._flatten_search_space(search_space)

        for param_name, param_value in params.items():
            if param_name not in flat_space:
                continue

            config = flat_space[param_name]
            param_type = config.get("type", "float")

            if hasattr(param_type, "value"):
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

        return noisy_params

    # =========================================================================
    # META-FEATURE EXTRACTION
    # =========================================================================

    def _extract_meta_features(self, dataset: Any) -> dict[str, float]:
        """
        Extract meta-features from a dataset.

        Uses the MetaFeatureExtractor if available, otherwise
        falls back to basic feature extraction.

        Args:
            dataset: PyG dataset or similar

        Returns:
            Dictionary of meta-features
        """
        if self._meta_extractor_class is not None:
            return self._meta_extractor_class.extract(dataset)

        return self._basic_meta_features(dataset)

    def _basic_meta_features(self, dataset: Any) -> dict[str, float]:
        """
        Extract basic meta-features without MetaFeatureExtractor.

        Fallback method for when meta_features.py is not available.

        Args:
            dataset: PyG dataset or similar

        Returns:
            Basic meta-feature dictionary
        """
        features: dict[str, float] = {}

        try:
            features["n_samples"] = float(len(dataset))
        except Exception:
            features["n_samples"] = 0.0

        try:
            first_data = dataset[0]

            if hasattr(first_data, "x") and first_data.x is not None:
                features["n_features"] = float(first_data.x.shape[1])

            if hasattr(first_data, "edge_index") and first_data.edge_index is not None:
                edge_counts = []
                node_counts = []

                for i in range(min(100, len(dataset))):
                    data = dataset[i]
                    if hasattr(data, "edge_index") and data.edge_index is not None:
                        edge_counts.append(data.edge_index.shape[1])
                    if hasattr(data, "x") and data.x is not None:
                        node_counts.append(data.x.shape[0])

                if edge_counts:
                    features["mean_edges"] = float(np.mean(edge_counts))
                if node_counts:
                    features["mean_nodes"] = float(np.mean(node_counts))

            if hasattr(first_data, "y") and first_data.y is not None:
                targets = []
                for i in range(min(100, len(dataset))):
                    data = dataset[i]
                    if hasattr(data, "y") and data.y is not None:
                        if hasattr(data.y, "numel") and data.y.numel() == 1:
                            targets.append(float(data.y.item()))
                        elif hasattr(data.y, "mean"):
                            targets.append(float(data.y.mean().item()))

                if targets:
                    features["target_mean"] = float(np.mean(targets))
                    features["target_std"] = float(np.std(targets))

        except Exception as e:
            logger.warning(f"Error extracting basic meta-features: {e}")

        return features

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _save_meta_db(self) -> None:
        """Save meta-database to disk."""
        if not self.config.meta_db_path:
            return

        try:
            data = {name: info.to_dict() for name, info in self._meta_db.items()}

            with open(self.config.meta_db_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved meta-database to {self.config.meta_db_path}")

        except Exception as e:
            logger.warning(f"Failed to save meta-database: {e}")

    def _load_meta_db(self) -> None:
        """Load meta-database from disk."""
        if not self.config.meta_db_path:
            return

        if not os.path.exists(self.config.meta_db_path):
            logger.debug(f"Meta-database not found at {self.config.meta_db_path}")
            return

        try:
            with open(self.config.meta_db_path) as f:
                data = json.load(f)

            self._meta_db = {
                name: RegisteredStudyInfo.from_dict(info) for name, info in data.items()
            }

            logger.info(f"Loaded {len(self._meta_db)} studies from {self.config.meta_db_path}")

        except Exception as e:
            logger.warning(f"Failed to load meta-database: {e}")

    def export_meta_db(self, path: str) -> None:
        """
        Export meta-database to a file.

        Args:
            path: Output file path
        """
        data = {name: info.to_dict() for name, info in self._meta_db.items()}

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(self._meta_db)} studies to {path}")

    def import_meta_db(self, path: str, merge: bool = True) -> int:
        """
        Import meta-database from a file.

        Args:
            path: Input file path
            merge: Whether to merge with existing (True) or replace (False)

        Returns:
            Number of studies imported
        """
        with open(path) as f:
            data = json.load(f)

        if not merge:
            self._meta_db.clear()
            self._study_cache.clear()

        n_imported = 0
        for name, info in data.items():
            if name not in self._meta_db:
                self._meta_db[name] = RegisteredStudyInfo.from_dict(info)
                n_imported += 1

        logger.info(f"Imported {n_imported} studies from {path}")

        return n_imported

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_transfer_summary(self) -> dict[str, Any]:
        """
        Get summary statistics about registered studies.

        Returns:
            Dictionary with summary statistics
        """
        if not self._meta_db:
            return {
                "n_studies": 0,
                "n_cached": 0,
                "total_trials": 0,
                "models": [],
            }

        models = set()
        total_trials = 0

        for info in self._meta_db.values():
            total_trials += info.n_trials
            if info.model_name:
                models.add(info.model_name)

        return {
            "n_studies": len(self._meta_db),
            "n_cached": len(self._study_cache),
            "total_trials": total_trials,
            "models": sorted(models),
            "similarity_threshold": self.config.similarity_threshold,
            "n_warm_start_trials": self.config.n_warm_start_trials,
        }

    def clear(self) -> None:
        """Clear all registered studies and cached data."""
        self._meta_db.clear()
        self._study_cache.clear()
        logger.info("Cleared all registered studies")

    def _raise_hpo_error(
        self,
        message: str,
        study_name: str | None = None,
        details: str | None = None,
    ) -> None:
        """
        Raise an HPOError with proper handling.

        Args:
            message: Error message
            study_name: Study name for context
            details: Additional details
        """
        if self._hpo_error is not None:
            raise self._hpo_error(message, study_name=study_name, details=details)
        else:
            raise RuntimeError(f"{message}. Study: {study_name}. Details: {details}")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HPOTransferManager("
            f"n_studies={len(self._meta_db)}, "
            f"n_cached={len(self._study_cache)}, "
            f"similarity_threshold={self.config.similarity_threshold})"
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "MetaFeatureMethod",
    "AdaptationMethod",
    "TransferConfig",
    "RegisteredStudyInfo",
    "HPOTransferManager",
]
