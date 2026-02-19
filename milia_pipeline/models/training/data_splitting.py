"""
Data Splitting Module

Comprehensive data splitting strategies for graph datasets:
- Random split
- Stratified split (for classification)
- Temporal split (for time-series)
- Scaffold split (for molecular data)

Author: milia Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from torch.utils.data import Dataset, Subset

# Import exceptions with fallback
try:
    from milia_pipeline.exceptions import DataError
except ImportError:

    class DataError(Exception):
        """Exception raised for data-related errors."""

        pass


logger = logging.getLogger(__name__)


# =============================================================================
# DATA SPLITTER CLASS
# =============================================================================


class DataSplitter:
    """
    Comprehensive data splitting utilities for graph datasets.

    Provides multiple splitting strategies:
    - Random split: Simple random shuffling
    - Stratified split: Maintains class distribution
    - Temporal split: Chronological ordering
    - Scaffold split: Molecular scaffold-based

    All methods return (train_subset, val_subset, test_subset).

    Usage:
        >>> from milia_pipeline.models import DataSplitter
        >>> train, val, test = DataSplitter.random_split(dataset)
    """

    @staticmethod
    def random_split(
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        shuffle: bool = True,
    ) -> tuple[Subset, Subset, Subset]:
        """
        Random split of dataset into train/val/test.

        Simple random splitting with optional shuffling. Best for general use
        when data has no specific ordering requirements.

        Args:
            dataset: PyTorch Dataset or PyG Dataset
            train_ratio: Proportion of data for training (default: 0.8)
            val_ratio: Proportion of data for validation (default: 0.1)
            test_ratio: Proportion of data for testing (default: 0.1)
            random_seed: Random seed for reproducibility (default: 42)
            shuffle: Whether to shuffle before splitting (default: True)

        Returns:
            Tuple of (train_subset, val_subset, test_subset)

        Raises:
            DataError: If ratios don't sum to 1.0 or dataset is empty

        Example:
            >>> dataset = MyGraphDataset(...)
            >>> train, val, test = DataSplitter.random_split(
            ...     dataset,
            ...     train_ratio=0.7,
            ...     val_ratio=0.15,
            ...     test_ratio=0.15,
            ...     random_seed=42
            ... )
            >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        """
        # Validation
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise DataError(
                f"Ratios must sum to 1.0. Got: train={train_ratio}, "
                f"val={val_ratio}, test={test_ratio} (sum={train_ratio + val_ratio + test_ratio})"
            )

        n = len(dataset)
        if n == 0:
            raise DataError("Dataset is empty")

        # Create indices
        indices = list(range(n))

        # Shuffle if requested
        if shuffle:
            random.seed(random_seed)
            random.shuffle(indices)

        # Calculate split points
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split indices
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        logger.info(
            f"Random split: Train={len(train_indices)}, "
            f"Val={len(val_indices)}, Test={len(test_indices)} "
            f"(seed={random_seed})"
        )

        return (
            Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            Subset(dataset, test_indices),
        )

    @staticmethod
    def stratified_split(
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        label_getter: callable | None = None,
    ) -> tuple[Subset, Subset, Subset]:
        """
        Stratified split maintaining class distribution across splits.

        Ensures each split has approximately the same proportion of samples
        from each class. Ideal for classification tasks with imbalanced classes.

        Requires: scikit-learn

        Args:
            dataset: PyTorch Dataset or PyG Dataset
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            random_seed: Random seed for reproducibility
            label_getter: Optional function to extract label from data object
                         Default: lambda data: data.y.item()

        Returns:
            Tuple of (train_subset, val_subset, test_subset)

        Raises:
            DataError: If ratios don't sum to 1.0 or sklearn not available

        Example:
            >>> dataset = MyGraphDataset(...)
            >>> train, val, test = DataSplitter.stratified_split(
            ...     dataset,
            ...     train_ratio=0.7,
            ...     val_ratio=0.15,
            ...     test_ratio=0.15
            ... )
        """
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise DataError(
                "stratified_split requires scikit-learn. Install with: pip install scikit-learn"
            ) from None

        # Validation
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise DataError(
                f"Ratios must sum to 1.0. Got: train={train_ratio}, "
                f"val={val_ratio}, test={test_ratio}"
            )

        n = len(dataset)
        if n == 0:
            raise DataError("Dataset is empty")

        # Extract labels
        if label_getter is None:
            # Default: assume data.y is label
            def label_getter(data):
                return (
                            data.y.item() if data.y.dim() == 0 else data.y.argmax().item()
                        )

        try:
            labels = [label_getter(dataset[i]) for i in range(len(dataset))]
        except Exception as e:
            raise DataError(f"Failed to extract labels: {e}") from e

        indices = list(range(n))

        # First split: train vs (val + test)
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_ratio, stratify=labels, random_state=random_seed
        )

        # Get labels for temp set
        temp_labels = [labels[i] for i in temp_idx]

        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_ratio_adjusted, stratify=temp_labels, random_state=random_seed
        )

        logger.info(
            f"Stratified split: Train={len(train_idx)}, "
            f"Val={len(val_idx)}, Test={len(test_idx)} "
            f"(seed={random_seed})"
        )

        return (Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx))

    @staticmethod
    def temporal_split(
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        time_field: str = "time",
        time_getter: callable | None = None,
    ) -> tuple[Subset, Subset, Subset]:
        """
        Temporal split based on chronological ordering.

        Splits data chronologically without shuffling, preserving temporal
        order. Train set contains earliest samples, test set contains latest.

        Args:
            dataset: PyTorch Dataset or PyG Dataset
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            time_field: Attribute name containing timestamp (default: "time")
            time_getter: Optional function to extract timestamp from data object
                        Default: lambda data: getattr(data, time_field)

        Returns:
            Tuple of (train_subset, val_subset, test_subset)

        Raises:
            DataError: If ratios don't sum to 1.0 or time field not found

        Example:
            >>> dataset = MyTemporalDataset(...)
            >>> train, val, test = DataSplitter.temporal_split(
            ...     dataset,
            ...     time_field="timestamp",
            ...     train_ratio=0.7
            ... )
        """
        # Validation
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise DataError(
                f"Ratios must sum to 1.0. Got: train={train_ratio}, "
                f"val={val_ratio}, test={test_ratio}"
            )

        n = len(dataset)
        if n == 0:
            raise DataError("Dataset is empty")

        # Extract timestamps
        if time_getter is None:
            def time_getter(data):
                return getattr(data, time_field)

        try:
            times = [time_getter(dataset[i]) for i in range(n)]
        except AttributeError:
            raise DataError(
                f"Dataset samples missing '{time_field}' attribute. "
                f"Provide custom time_getter function."
            ) from None
        except Exception as e:
            raise DataError(f"Failed to extract timestamps: {e}") from e

        # Sort by time
        sorted_indices = sorted(range(n), key=lambda i: times[i])

        # Calculate split points
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split indices (chronologically)
        train_indices = sorted_indices[:train_end]
        val_indices = sorted_indices[train_end:val_end]
        test_indices = sorted_indices[val_end:]

        logger.info(
            f"Temporal split: Train={len(train_indices)}, "
            f"Val={len(val_indices)}, Test={len(test_indices)} "
            f"(chronological order preserved)"
        )

        return (
            Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            Subset(dataset, test_indices),
        )

    @staticmethod
    def scaffold_split(
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        mol_getter: callable | None = None,
        include_chirality: bool = False,
    ) -> tuple[Subset, Subset, Subset]:
        """
        Scaffold split for molecular datasets.

        Groups molecules by their Murcko scaffold and assigns entire scaffold
        groups to splits. Ensures test set contains structurally different
        molecules than training set.

        Requires: rdkit

        Args:
            dataset: PyTorch Dataset or PyG Dataset with molecular data
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            random_seed: Random seed for reproducibility
            mol_getter: Optional function to extract RDKit mol from data object
                       Default: lambda data: data.mol
            include_chirality: Whether to consider chirality in scaffolds

        Returns:
            Tuple of (train_subset, val_subset, test_subset)

        Raises:
            DataError: If ratios don't sum to 1.0 or rdkit not available

        Example:
            >>> dataset = MyMolecularDataset(...)
            >>> train, val, test = DataSplitter.scaffold_split(
            ...     dataset,
            ...     train_ratio=0.8,
            ...     include_chirality=False
            ... )
        """
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
        except ImportError:
            raise DataError("scaffold_split requires rdkit. Install with: pip install rdkit") from None

        # Validation
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise DataError(
                f"Ratios must sum to 1.0. Got: train={train_ratio}, "
                f"val={val_ratio}, test={test_ratio}"
            )

        n = len(dataset)
        if n == 0:
            raise DataError("Dataset is empty")

        # Extract molecules and generate scaffolds
        if mol_getter is None:
            def mol_getter(data):
                return data.mol

        scaffolds = defaultdict(list)

        for idx in range(n):
            try:
                mol = mol_getter(dataset[idx])

                # Generate scaffold SMILES
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=include_chirality
                )

                scaffolds[scaffold].append(idx)

            except AttributeError:
                raise DataError(
                    "Dataset samples missing 'mol' attribute. Provide custom mol_getter function."
                ) from None
            except Exception as e:
                logger.warning(f"Failed to generate scaffold for sample {idx}: {e}")
                # Assign to unique scaffold
                scaffolds[f"__error_{idx}__"].append(idx)

        # Sort scaffolds by size (largest first)
        scaffold_sets = sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)

        logger.info(f"Found {len(scaffold_sets)} unique scaffolds")

        # Set random seed for reproducibility
        random.seed(random_seed)

        # Allocate scaffolds to splits
        train_idx, val_idx, test_idx = [], [], []
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        for scaffold, indices in scaffold_sets:
            if len(train_idx) < train_size:
                train_idx.extend(indices)
            elif len(val_idx) < val_size:
                val_idx.extend(indices)
            else:
                test_idx.extend(indices)

        logger.info(
            f"Scaffold split: Train={len(train_idx)}, "
            f"Val={len(val_idx)}, Test={len(test_idx)} "
            f"(seed={random_seed})"
        )

        return (Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx))

    @staticmethod
    def k_fold_split(
        dataset: Dataset, n_splits: int = 5, random_seed: int = 42, shuffle: bool = True
    ) -> list[tuple[Subset, Subset]]:
        """
        K-fold cross-validation split.

        Splits dataset into K folds for cross-validation. Returns list of
        (train, val) tuples, one for each fold.

        Args:
            dataset: PyTorch Dataset or PyG Dataset
            n_splits: Number of folds (default: 5)
            random_seed: Random seed for reproducibility
            shuffle: Whether to shuffle before splitting

        Returns:
            List of (train_subset, val_subset) tuples for each fold

        Raises:
            DataError: If n_splits < 2 or dataset too small

        Example:
            >>> dataset = MyGraphDataset(...)
            >>> folds = DataSplitter.k_fold_split(dataset, n_splits=5)
            >>> for fold_idx, (train, val) in enumerate(folds):
            ...     print(f"Fold {fold_idx}: Train={len(train)}, Val={len(val)}")
        """
        n = len(dataset)

        if n_splits < 2:
            raise DataError(f"n_splits must be >= 2, got {n_splits}")

        if n < n_splits:
            raise DataError(f"Dataset too small for {n_splits}-fold split. Dataset size: {n}")

        # Create indices
        indices = list(range(n))

        if shuffle:
            random.seed(random_seed)
            random.shuffle(indices)

        # Calculate fold size
        fold_size = n // n_splits

        folds = []
        for i in range(n_splits):
            # Validation fold
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < n_splits - 1 else n
            val_indices = indices[val_start:val_end]

            # Training folds
            train_indices = indices[:val_start] + indices[val_end:]

            folds.append((Subset(dataset, train_indices), Subset(dataset, val_indices)))

        logger.info(
            f"K-fold split: {n_splits} folds, ~{fold_size} samples per fold (seed={random_seed})"
        )

        return folds


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def random_split(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """
    Convenience function for random split.

    Example:
        >>> from milia_pipeline.models.training import random_split
        >>> train, val, test = random_split(dataset)
    """
    return DataSplitter.random_split(dataset, train_ratio, val_ratio, test_ratio, random_seed)


def stratified_split(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """
    Convenience function for stratified split.

    Example:
        >>> from milia_pipeline.models.training import stratified_split
        >>> train, val, test = stratified_split(dataset)
    """
    return DataSplitter.stratified_split(dataset, train_ratio, val_ratio, test_ratio, random_seed)


def temporal_split(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    time_field: str = "time",
    time_getter: callable | None = None,
) -> tuple[Subset, Subset, Subset]:
    """
    Convenience function for temporal split.

    Splits data chronologically without shuffling, preserving temporal order.
    Train set contains earliest samples, test set contains latest.

    Args:
        dataset: PyTorch Dataset or PyG Dataset with temporal data
        train_ratio: Proportion of data for training (default: 0.8)
        val_ratio: Proportion of data for validation (default: 0.1)
        test_ratio: Proportion of data for testing (default: 0.1)
        time_field: Attribute name containing timestamp (default: "time")
        time_getter: Optional function to extract timestamp from data object

    Returns:
        Tuple of (train_subset, val_subset, test_subset)

    Example:
        >>> from milia_pipeline.models.training import temporal_split
        >>> train, val, test = temporal_split(dataset, time_field="timestamp")
    """
    return DataSplitter.temporal_split(
        dataset, train_ratio, val_ratio, test_ratio, time_field, time_getter
    )


def scaffold_split(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    mol_getter: callable | None = None,
    include_chirality: bool = False,
) -> tuple[Subset, Subset, Subset]:
    """
    Convenience function for scaffold split.

    Groups molecules by their Murcko scaffold and assigns entire scaffold
    groups to splits. Ensures test set contains structurally different
    molecules than training set.

    Requires: rdkit

    Args:
        dataset: PyTorch Dataset or PyG Dataset with molecular data
        train_ratio: Proportion of data for training (default: 0.8)
        val_ratio: Proportion of data for validation (default: 0.1)
        test_ratio: Proportion of data for testing (default: 0.1)
        random_seed: Random seed for reproducibility (default: 42)
        mol_getter: Optional function to extract RDKit mol from data object
        include_chirality: Whether to consider chirality in scaffolds (default: False)

    Returns:
        Tuple of (train_subset, val_subset, test_subset)

    Example:
        >>> from milia_pipeline.models.training import scaffold_split
        >>> train, val, test = scaffold_split(dataset, include_chirality=True)
    """
    return DataSplitter.scaffold_split(
        dataset, train_ratio, val_ratio, test_ratio, random_seed, mol_getter, include_chirality
    )


def k_fold_split(
    dataset: Dataset, n_splits: int = 5, random_seed: int = 42, shuffle: bool = True
) -> list[tuple[Subset, Subset]]:
    """
    Convenience function for K-fold cross-validation split.

    Splits dataset into K folds for cross-validation. Returns list of
    (train, val) tuples, one for each fold.

    Args:
        dataset: PyTorch Dataset or PyG Dataset
        n_splits: Number of folds (default: 5)
        random_seed: Random seed for reproducibility (default: 42)
        shuffle: Whether to shuffle before splitting (default: True)

    Returns:
        List of (train_subset, val_subset) tuples for each fold

    Example:
        >>> from milia_pipeline.models.training import k_fold_split
        >>> folds = k_fold_split(dataset, n_splits=5)
        >>> for fold_idx, (train, val) in enumerate(folds):
        ...     print(f"Fold {fold_idx}: Train={len(train)}, Val={len(val)}")
    """
    return DataSplitter.k_fold_split(dataset, n_splits, random_seed, shuffle)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info("data_splitting module loaded")
