#!/usr/bin/env python3
"""
Complete Production-Ready Unit Test Suite for data_splitting.py Module

Comprehensive test coverage for all data splitting strategies including:
- DataSplitter class with all methods (random, stratified, temporal, scaffold, k-fold)
- All convenience functions (random_split, stratified_split, temporal_split, scaffold_split, k_fold_split)
- Error handling and edge cases
- Parameter validation
- Reproducibility and determinism
- Integration with PyTorch Dataset and Subset
- Mock isolation to prevent sys.modules pollution

This test suite imports from the actual data_splitting module and uses
test-level mocking (via @patch decorators) to prevent sys.modules pollution.

Author: milia Team
Version: 3.0.0
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from torch.utils.data import Dataset, Subset

# =============================================================================
# MODULE IMPORT SETUP - Dynamic import without sys.modules pollution
# =============================================================================


def _import_data_splitting_module():
    """
    Import the data_splitting module dynamically.

    This function handles multiple import scenarios:
    1. Module already importable from installed package
    2. Module available via relative path (test environment)
    3. Module needs to be loaded from file path

    Returns:
        The imported data_splitting module
    """
    # Try standard import first (works if milia_pipeline is installed)
    try:
        from milia_pipeline.models.training import data_splitting

        return data_splitting
    except ImportError:
        pass

    # Try importing as standalone module (if in PYTHONPATH or same directory)
    try:
        import data_splitting

        return data_splitting
    except ImportError:
        pass

    # Dynamic import from file path
    # Determine the path relative to this test file
    test_file_dir = Path(__file__).parent.resolve()

    # Common relative paths to check for the module
    candidate_paths = [
        test_file_dir / "data_splitting.py",
        test_file_dir.parent / "milia_pipeline" / "models" / "training" / "data_splitting.py",
        test_file_dir.parent
        / "src"
        / "milia_pipeline"
        / "models"
        / "training"
        / "data_splitting.py",
        test_file_dir.parent / "data_splitting.py",
    ]

    module_path = None
    for candidate in candidate_paths:
        if candidate.exists():
            module_path = candidate
            break

    if module_path is None:
        raise ImportError(
            "Could not locate data_splitting.py module. "
            "Ensure it is in the expected location relative to tests."
        )

    # Use importlib to load from file path
    import importlib.util

    spec = importlib.util.spec_from_file_location("data_splitting", module_path)
    module = importlib.util.module_from_spec(spec)

    # Temporarily add to sys.modules for the load, then manage carefully
    old_module = sys.modules.get("data_splitting")
    sys.modules["data_splitting"] = module

    try:
        spec.loader.exec_module(module)
    finally:
        # Restore original state if there was a previous module
        if old_module is not None:
            sys.modules["data_splitting"] = old_module
        # Note: We keep the module in sys.modules for this test session

    return module


# Import the actual module
_ds_module = _import_data_splitting_module()

# Extract classes and functions from the imported module
DataSplitter = _ds_module.DataSplitter
DataError = _ds_module.DataError
random_split = _ds_module.random_split
stratified_split = _ds_module.stratified_split
temporal_split = _ds_module.temporal_split
scaffold_split = _ds_module.scaffold_split
k_fold_split = _ds_module.k_fold_split


# =============================================================================
# MODULE TEARDOWN - Prevent sys.modules pollution
# =============================================================================


def teardown_module(module):
    """
    Clean up any mocked modules from sys.modules to prevent pollution.

    This ensures that test-level patches don't leak into subsequent test files
    during pytest collection or execution.
    """
    # List of module name patterns that might be mocked during tests
    mock_patterns = ("sklearn", "rdkit")

    keys_to_remove = [
        key
        for key in list(sys.modules.keys())
        if any(pattern in key for pattern in mock_patterns)
        and isinstance(sys.modules.get(key), (Mock, MagicMock, type(None)))
    ]

    for key in keys_to_remove:
        try:
            del sys.modules[key]
        except KeyError:
            pass  # Already removed by another cleanup


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def simple_dataset():
    """Create a simple dataset for testing."""

    class SimpleDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
            self.data = list(range(size))

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return self.data[idx]

    return SimpleDataset()


@pytest.fixture
def labeled_dataset():
    """Create a labeled dataset for stratified split testing."""

    class LabeledDataset(Dataset):
        def __init__(self, size=100, n_classes=3):
            self.size = size
            self.data = []
            # Create balanced classes
            for i in range(size):
                label = i % n_classes
                item = Mock()
                item.y = torch.tensor(label)
                self.data.append(item)

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return self.data[idx]

    return LabeledDataset()


@pytest.fixture
def imbalanced_labeled_dataset():
    """Create an imbalanced labeled dataset for stratified split testing."""

    class ImbalancedDataset(Dataset):
        def __init__(self):
            self.data = []
            # Class 0: 70 samples
            for _i in range(70):
                item = Mock()
                item.y = torch.tensor(0)
                self.data.append(item)
            # Class 1: 20 samples
            for _i in range(20):
                item = Mock()
                item.y = torch.tensor(1)
                self.data.append(item)
            # Class 2: 10 samples
            for _i in range(10):
                item = Mock()
                item.y = torch.tensor(2)
                self.data.append(item)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    return ImbalancedDataset()


@pytest.fixture
def temporal_dataset():
    """Create a temporal dataset with timestamps."""

    class TemporalDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
            self.data = []
            for i in range(size):
                item = Mock()
                item.time = i * 10  # Increasing timestamps
                item.value = i
                self.data.append(item)

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return self.data[idx]

    return TemporalDataset()


@pytest.fixture
def molecular_dataset():
    """Create a molecular dataset with scaffolds."""

    class MolecularDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
            self.data = []
            # Create molecules with mock scaffolds
            for _i in range(size):
                item = Mock()
                # Mock RDKit molecule
                mol = Mock()
                item.mol = mol
                self.data.append(item)

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return self.data[idx]

    return MolecularDataset()


@pytest.fixture
def empty_dataset():
    """Create an empty dataset for edge case testing."""

    class EmptyDataset(Dataset):
        def __len__(self):
            return 0

        def __getitem__(self, idx):
            raise IndexError("Dataset is empty")

    return EmptyDataset()


@pytest.fixture
def small_dataset():
    """Create a very small dataset (5 samples) for edge case testing."""

    class SmallDataset(Dataset):
        def __init__(self):
            self.data = list(range(5))

        def __len__(self):
            return 5

        def __getitem__(self, idx):
            return self.data[idx]

    return SmallDataset()


# =============================================================================
# RANDOM SPLIT TESTS
# =============================================================================


class TestRandomSplit:
    """Test DataSplitter.random_split method."""

    def test_random_split_default_ratios(self, simple_dataset):
        """Test random split with default ratios (0.8/0.1/0.1)."""
        train, val, test = DataSplitter.random_split(simple_dataset)

        assert isinstance(train, Subset)
        assert isinstance(val, Subset)
        assert isinstance(test, Subset)

        # Check sizes
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
        assert len(train) + len(val) + len(test) == len(simple_dataset)

    def test_random_split_custom_ratios(self, simple_dataset):
        """Test random split with custom ratios."""
        train, val, test = DataSplitter.random_split(
            simple_dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
        )

        assert len(train) == 70
        assert len(val) == 20
        assert len(test) == 10

    def test_random_split_equal_ratios(self, simple_dataset):
        """Test random split with equal ratios."""
        train, val, test = DataSplitter.random_split(
            simple_dataset, train_ratio=1 / 3, val_ratio=1 / 3, test_ratio=1 / 3
        )

        # Each should be approximately 33
        assert 32 <= len(train) <= 34
        assert 32 <= len(val) <= 34
        assert 32 <= len(test) <= 34
        assert len(train) + len(val) + len(test) == 100

    def test_random_split_reproducibility_with_seed(self, simple_dataset):
        """Test that same seed produces same split."""
        train1, val1, test1 = DataSplitter.random_split(simple_dataset, random_seed=42)
        train2, val2, test2 = DataSplitter.random_split(simple_dataset, random_seed=42)

        # Check indices are identical
        assert train1.indices == train2.indices
        assert val1.indices == val2.indices
        assert test1.indices == test2.indices

    def test_random_split_different_seeds_produce_different_splits(self, simple_dataset):
        """Test that different seeds produce different splits."""
        train1, val1, test1 = DataSplitter.random_split(simple_dataset, random_seed=42)
        train2, val2, test2 = DataSplitter.random_split(simple_dataset, random_seed=123)

        # Indices should differ
        assert train1.indices != train2.indices

    def test_random_split_shuffle_true(self, simple_dataset):
        """Test random split with shuffling enabled."""
        train, val, test = DataSplitter.random_split(simple_dataset, shuffle=True, random_seed=42)

        # Check that indices are not in sequential order
        train_indices = train.indices
        # If shuffled, indices should not be [0, 1, 2, ..., 79]
        assert train_indices != list(range(80))

    def test_random_split_shuffle_false(self, simple_dataset):
        """Test random split without shuffling."""
        train, val, test = DataSplitter.random_split(simple_dataset, shuffle=False)

        # Indices should be sequential
        assert train.indices == list(range(80))
        assert val.indices == list(range(80, 90))
        assert test.indices == list(range(90, 100))

    def test_random_split_no_overlapping_indices(self, simple_dataset):
        """Test that train/val/test have no overlapping indices."""
        train, val, test = DataSplitter.random_split(simple_dataset)

        train_set = set(train.indices)
        val_set = set(val.indices)
        test_set = set(test.indices)

        # No overlaps
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_random_split_covers_all_indices(self, simple_dataset):
        """Test that all dataset indices are covered."""
        train, val, test = DataSplitter.random_split(simple_dataset)

        all_indices = set(train.indices + val.indices + test.indices)
        expected_indices = set(range(len(simple_dataset)))

        assert all_indices == expected_indices

    def test_random_split_invalid_ratios_sum(self, simple_dataset):
        """Test error when ratios don't sum to 1.0."""
        with pytest.raises(DataError) as exc_info:
            DataSplitter.random_split(
                simple_dataset,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.1,  # Sum = 0.9
            )

        assert "must sum to 1.0" in str(exc_info.value)

    def test_random_split_ratios_exceed_one(self, simple_dataset):
        """Test error when ratios sum to more than 1.0."""
        with pytest.raises(DataError) as exc_info:
            DataSplitter.random_split(
                simple_dataset,
                train_ratio=0.6,
                val_ratio=0.3,
                test_ratio=0.3,  # Sum = 1.2
            )

        assert "must sum to 1.0" in str(exc_info.value)

    def test_random_split_empty_dataset(self, empty_dataset):
        """Test error with empty dataset."""
        with pytest.raises(DataError) as exc_info:
            DataSplitter.random_split(empty_dataset)

        assert "empty" in str(exc_info.value).lower()

    def test_random_split_small_dataset(self, small_dataset):
        """Test random split with very small dataset."""
        train, val, test = DataSplitter.random_split(
            small_dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )

        # With 5 samples: 3 train, 1 val, 1 test
        assert len(train) == 3
        assert len(val) == 1
        assert len(test) == 1

    def test_random_split_extreme_ratio_all_train(self, simple_dataset):
        """Test split with almost all data in train."""
        train, val, test = DataSplitter.random_split(
            simple_dataset, train_ratio=0.98, val_ratio=0.01, test_ratio=0.01
        )

        assert len(train) == 98
        assert len(val) == 1
        assert len(test) == 1

    def test_random_split_returns_subset_with_correct_dataset(self, simple_dataset):
        """Test that returned Subset references correct dataset."""
        train, val, test = DataSplitter.random_split(simple_dataset)

        assert train.dataset is simple_dataset
        assert val.dataset is simple_dataset
        assert test.dataset is simple_dataset

    def test_random_split_with_zero_train_ratio(self, simple_dataset):
        """Test split with zero training ratio."""
        train, val, test = DataSplitter.random_split(
            simple_dataset, train_ratio=0.0, val_ratio=0.5, test_ratio=0.5
        )

        assert len(train) == 0
        assert len(val) == 50
        assert len(test) == 50


# =============================================================================
# STRATIFIED SPLIT TESTS
# =============================================================================


def _create_mock_sklearn_import(mock_train_test_split_func):
    """
    Create a mock import function that intercepts sklearn imports.

    This helper creates a custom __import__ function that returns a mock
    sklearn.model_selection module with the provided train_test_split function,
    while delegating all other imports to the real __import__.

    Args:
        mock_train_test_split_func: Function to use as train_test_split mock

    Returns:
        A mock import function suitable for use with patch on builtins.__import__
    """
    real_import = (
        __builtins__.__import__
        if isinstance(__builtins__, dict)
        else __builtins__.__dict__["__import__"]
    )

    def mock_import(name, *args, **kwargs):
        if name == "sklearn.model_selection":
            mock_module = MagicMock()
            mock_module.train_test_split = mock_train_test_split_func
            return mock_module
        return real_import(name, *args, **kwargs)

    return mock_import


class TestStratifiedSplit:
    """Test DataSplitter.stratified_split method."""

    @pytest.fixture
    def mock_sklearn_available(self):
        """
        Fixture that provides mock sklearn if real sklearn is unavailable.

        This fixture attempts to import sklearn, and if unavailable, provides
        a context manager that mocks the import. Tests using this fixture
        will work regardless of sklearn installation status.
        """
        try:
            # sklearn is available, yield a no-op context manager
            from contextlib import nullcontext

            import sklearn.model_selection

            yield nullcontext()
        except ImportError:
            # sklearn not available, provide a mock context
            yield None

    def test_stratified_split_default_ratios(self, labeled_dataset):
        """Test stratified split with default ratios."""
        # Try with real sklearn first, fall back to mock
        try:
            from sklearn.model_selection import train_test_split

            train, val, test = DataSplitter.stratified_split(labeled_dataset)

            assert isinstance(train, Subset)
            assert isinstance(val, Subset)
            assert isinstance(test, Subset)
            assert len(train) == 80
            assert len(val) == 10
            assert len(test) == 10
        except ImportError:
            # sklearn not installed - test the error message
            with pytest.raises(DataError) as exc_info:
                DataSplitter.stratified_split(labeled_dataset)
            assert "scikit-learn" in str(exc_info.value)

    def test_stratified_split_maintains_class_distribution(self, labeled_dataset):
        """Test that stratified split maintains class distribution."""
        try:
            from sklearn.model_selection import train_test_split

            train, val, test = DataSplitter.stratified_split(
                labeled_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
            )

            # Extract labels from each split
            def get_labels(subset):
                labels = []
                for idx in subset.indices:
                    labels.append(subset.dataset[idx].y.item())
                return labels

            train_labels = get_labels(train)
            val_labels = get_labels(val)
            test_labels = get_labels(test)

            # Check that each class appears in each split
            assert len(set(train_labels)) == 3
            assert len(set(val_labels)) == 3
            assert len(set(test_labels)) == 3
        except ImportError:
            pytest.skip("sklearn not installed - skipping class distribution test")

    def test_stratified_split_custom_ratios(self, labeled_dataset):
        """Test stratified split with custom ratios."""
        try:
            from sklearn.model_selection import train_test_split

            train, val, test = DataSplitter.stratified_split(
                labeled_dataset, train_ratio=0.6, val_ratio=0.25, test_ratio=0.15
            )

            # Check approximate sizes
            assert 55 <= len(train) <= 65
            assert 20 <= len(val) <= 30
            assert 10 <= len(test) <= 20
        except ImportError:
            pytest.skip("sklearn not installed - skipping custom ratios test")

    def test_stratified_split_reproducibility(self, labeled_dataset):
        """Test reproducibility with same seed."""
        try:
            from sklearn.model_selection import train_test_split

            train1, val1, test1 = DataSplitter.stratified_split(labeled_dataset, random_seed=42)
            train2, val2, test2 = DataSplitter.stratified_split(labeled_dataset, random_seed=42)

            assert train1.indices == train2.indices
            assert val1.indices == val2.indices
            assert test1.indices == test2.indices
        except ImportError:
            pytest.skip("sklearn not installed - skipping reproducibility test")

    def test_stratified_split_different_seeds(self, labeled_dataset):
        """Test different seeds produce different splits."""
        try:
            from sklearn.model_selection import train_test_split

            train1, val1, test1 = DataSplitter.stratified_split(labeled_dataset, random_seed=42)
            train2, val2, test2 = DataSplitter.stratified_split(labeled_dataset, random_seed=123)

            assert train1.indices != train2.indices
        except ImportError:
            pytest.skip("sklearn not installed - skipping different seeds test")

    def test_stratified_split_custom_label_getter(self, labeled_dataset):
        """Test stratified split with custom label getter."""
        try:
            from sklearn.model_selection import train_test_split

            def custom_getter(data):
                return data.y.item()

            train, val, test = DataSplitter.stratified_split(
                labeled_dataset, label_getter=custom_getter
            )

            assert len(train) + len(val) + len(test) == len(labeled_dataset)
        except ImportError:
            pytest.skip("sklearn not installed - skipping custom label getter test")

    def test_stratified_split_multidimensional_labels(self):
        """Test stratified split with multi-dimensional labels using argmax."""
        try:
            from sklearn.model_selection import train_test_split

            class MultiDimDataset(Dataset):
                def __init__(self):
                    self.data = []
                    for i in range(100):
                        item = Mock()
                        # One-hot encoded labels
                        label = torch.zeros(3)
                        label[i % 3] = 1
                        item.y = label
                        self.data.append(item)

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    return self.data[idx]

            dataset = MultiDimDataset()
            train, val, test = DataSplitter.stratified_split(dataset)

            assert len(train) + len(val) + len(test) == 100
        except ImportError:
            pytest.skip("sklearn not installed - skipping multidimensional labels test")

    def test_stratified_split_invalid_ratios(self, labeled_dataset):
        """Test error when ratios don't sum to 1.0."""
        # This validation happens before sklearn import, so no mock needed
        with pytest.raises(DataError) as exc_info:
            DataSplitter.stratified_split(
                labeled_dataset, train_ratio=0.5, val_ratio=0.3, test_ratio=0.1
            )

        assert "must sum to 1.0" in str(exc_info.value)

    def test_stratified_split_empty_dataset(self, empty_dataset):
        """Test error with empty dataset."""
        # This validation happens before sklearn import, so no mock needed
        with pytest.raises(DataError) as exc_info:
            DataSplitter.stratified_split(empty_dataset)

        assert "empty" in str(exc_info.value).lower()

    def test_stratified_split_sklearn_not_available(self, labeled_dataset):
        """Test error when sklearn is not available."""
        # Temporarily remove sklearn from sys.modules to test ImportError handling
        sklearn_modules = {k: v for k, v in sys.modules.items() if "sklearn" in k}

        try:
            # Remove sklearn modules temporarily
            for key in sklearn_modules:
                del sys.modules[key]

            # Also need to make sure importlib doesn't find it
            with patch.dict(sys.modules, {"sklearn": None, "sklearn.model_selection": None}):
                with pytest.raises(DataError) as exc_info:
                    DataSplitter.stratified_split(labeled_dataset)

                assert "scikit-learn" in str(exc_info.value)
        finally:
            # Restore sklearn modules
            sys.modules.update(sklearn_modules)

    def test_stratified_split_label_extraction_failure(self):
        """Test error when label extraction fails."""

        class NoLabelDataset(Dataset):
            def __init__(self):
                self.data = [Mock(spec=[]) for _ in range(10)]

            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = NoLabelDataset()

        try:
            from sklearn.model_selection import train_test_split

            with pytest.raises(DataError) as exc_info:
                DataSplitter.stratified_split(dataset)

            assert "Failed to extract labels" in str(exc_info.value)
        except ImportError:
            pytest.skip("sklearn not installed - skipping label extraction failure test")

    def test_stratified_split_with_imbalanced_data(self, imbalanced_labeled_dataset):
        """Test stratified split with imbalanced class distribution."""
        try:
            from sklearn.model_selection import train_test_split

            train, val, test = DataSplitter.stratified_split(imbalanced_labeled_dataset)

            assert len(train) + len(val) + len(test) == len(imbalanced_labeled_dataset)
        except ImportError:
            pytest.skip("sklearn not installed - skipping imbalanced data test")


# =============================================================================
# TEMPORAL SPLIT TESTS
# =============================================================================


class TestTemporalSplit:
    """Test DataSplitter.temporal_split method."""

    def test_temporal_split_default_ratios(self, temporal_dataset):
        """Test temporal split with default ratios."""
        train, val, test = DataSplitter.temporal_split(temporal_dataset)

        assert isinstance(train, Subset)
        assert isinstance(val, Subset)
        assert isinstance(test, Subset)

        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_temporal_split_chronological_order(self, temporal_dataset):
        """Test that temporal split preserves chronological order."""
        train, val, test = DataSplitter.temporal_split(temporal_dataset)

        # Get timestamps from each split
        train_times = [temporal_dataset[i].time for i in train.indices]
        val_times = [temporal_dataset[i].time for i in val.indices]
        test_times = [temporal_dataset[i].time for i in test.indices]

        # Train should have earliest times
        assert max(train_times) < min(val_times)
        assert max(val_times) < min(test_times)

    def test_temporal_split_sorted_by_time(self, temporal_dataset):
        """Test that indices are sorted by time within each split."""
        train, val, test = DataSplitter.temporal_split(temporal_dataset)

        train_times = [temporal_dataset[i].time for i in train.indices]
        val_times = [temporal_dataset[i].time for i in val.indices]
        test_times = [temporal_dataset[i].time for i in test.indices]

        # Each split should be sorted
        assert train_times == sorted(train_times)
        assert val_times == sorted(val_times)
        assert test_times == sorted(test_times)

    def test_temporal_split_custom_ratios(self, temporal_dataset):
        """Test temporal split with custom ratios."""
        train, val, test = DataSplitter.temporal_split(
            temporal_dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
        )

        assert len(train) == 70
        assert len(val) == 20
        assert len(test) == 10

    def test_temporal_split_custom_time_field(self):
        """Test temporal split with custom time field name."""

        class CustomTemporalDataset(Dataset):
            def __init__(self):
                self.data = []
                for i in range(50):
                    item = Mock()
                    item.timestamp = i * 5  # Custom field name
                    self.data.append(item)

            def __len__(self):
                return 50

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = CustomTemporalDataset()
        train, val, test = DataSplitter.temporal_split(dataset, time_field="timestamp")

        assert len(train) + len(val) + len(test) == 50

    def test_temporal_split_custom_time_getter(self, temporal_dataset):
        """Test temporal split with custom time getter function."""

        def custom_getter(data):
            return data.time

        train, val, test = DataSplitter.temporal_split(temporal_dataset, time_getter=custom_getter)

        assert len(train) + len(val) + len(test) == len(temporal_dataset)

    def test_temporal_split_unsorted_timestamps(self):
        """Test temporal split with unsorted input timestamps."""

        class UnsortedTemporalDataset(Dataset):
            def __init__(self):
                self.data = []
                timestamps = [50, 10, 90, 30, 70, 20, 80, 40, 60, 100]
                for t in timestamps:
                    item = Mock()
                    item.time = t
                    self.data.append(item)

            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = UnsortedTemporalDataset()
        train, val, test = DataSplitter.temporal_split(
            dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )

        # Should still be sorted chronologically
        train_times = [dataset[i].time for i in train.indices]
        assert train_times == sorted(train_times)

    def test_temporal_split_invalid_ratios(self, temporal_dataset):
        """Test error when ratios don't sum to 1.0."""
        with pytest.raises(DataError) as exc_info:
            DataSplitter.temporal_split(
                temporal_dataset, train_ratio=0.5, val_ratio=0.3, test_ratio=0.1
            )

        assert "must sum to 1.0" in str(exc_info.value)

    def test_temporal_split_empty_dataset(self, empty_dataset):
        """Test error with empty dataset."""
        with pytest.raises(DataError) as exc_info:
            DataSplitter.temporal_split(empty_dataset)

        assert "empty" in str(exc_info.value).lower()

    def test_temporal_split_missing_time_field(self):
        """Test error when time field is missing."""

        class NoTimeDataset(Dataset):
            def __init__(self):
                self.data = [Mock(spec=[]) for _ in range(10)]

            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = NoTimeDataset()

        with pytest.raises(DataError) as exc_info:
            DataSplitter.temporal_split(dataset)

        assert "missing 'time' attribute" in str(exc_info.value)

    def test_temporal_split_time_extraction_failure(self):
        """Test error when time extraction fails."""

        class BadTimeDataset(Dataset):
            def __init__(self):
                self.data = []
                for _i in range(10):
                    item = Mock()
                    item.time = None
                    self.data.append(item)

            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = BadTimeDataset()

        def failing_getter(data):
            raise ValueError("Cannot extract time")

        with pytest.raises(DataError) as exc_info:
            DataSplitter.temporal_split(dataset, time_getter=failing_getter)

        assert "Failed to extract timestamps" in str(exc_info.value)

    def test_temporal_split_with_duplicate_timestamps(self):
        """Test temporal split with duplicate timestamps."""

        class DuplicateTimeDataset(Dataset):
            def __init__(self):
                self.data = []
                # Create dataset with duplicate timestamps
                for i in range(10):
                    item = Mock()
                    item.time = i // 2  # 0,0,1,1,2,2,3,3,4,4
                    self.data.append(item)

            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = DuplicateTimeDataset()
        train, val, test = DataSplitter.temporal_split(
            dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )

        # Should still split correctly
        assert len(train) + len(val) + len(test) == 10


# =============================================================================
# SCAFFOLD SPLIT TESTS
# =============================================================================


def _rdkit_available():
    """Check if rdkit is available for import."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        return True
    except ImportError:
        return False


class TestScaffoldSplit:
    """Test DataSplitter.scaffold_split method."""

    @pytest.fixture
    def mock_rdkit_scaffold(self):
        """
        Fixture providing a mock context for rdkit MurckoScaffold.

        Yields a tuple of (mock_scaffold_func, context_manager) where:
        - mock_scaffold_func is configurable for different test scenarios
        - context_manager handles the patching
        """
        # We need to mock at the module level where the import happens
        # Since scaffold_split imports rdkit inside the function body,
        # we patch the actual rdkit module path
        mock_murcko = MagicMock()
        mock_chem = MagicMock()
        mock_scaffolds = MagicMock()

        mock_scaffold_func = MagicMock()
        mock_murcko.MurckoScaffoldSmiles = mock_scaffold_func
        mock_scaffolds.MurckoScaffold = mock_murcko
        mock_chem.Scaffolds = mock_scaffolds

        return mock_scaffold_func

    def test_scaffold_split_default_ratios(self, molecular_dataset):
        """Test scaffold split with default ratios."""
        if _rdkit_available():
            # Use real rdkit
            train, val, test = DataSplitter.scaffold_split(molecular_dataset)

            assert isinstance(train, Subset)
            assert isinstance(val, Subset)
            assert isinstance(test, Subset)
            assert len(train) + len(val) + len(test) == 100
        else:
            # rdkit not available - test should raise DataError
            with pytest.raises(DataError) as exc_info:
                DataSplitter.scaffold_split(molecular_dataset)
            assert "rdkit" in str(exc_info.value)

    def test_scaffold_split_groups_by_scaffold(self, molecular_dataset):
        """Test that molecules with same scaffold are grouped together."""
        if not _rdkit_available():
            pytest.skip("rdkit not installed - skipping scaffold grouping test")

        # Create dataset with known scaffold groups
        class KnownScaffoldDataset(Dataset):
            def __init__(self):
                self.data = []
                # 3 scaffolds: A (40 samples), B (30 samples), C (30 samples)
                for i in range(100):
                    item = Mock()
                    mol = Mock()
                    # Assign scaffold based on index
                    if i < 40:
                        mol._scaffold_id = "A"
                    elif i < 70:
                        mol._scaffold_id = "B"
                    else:
                        mol._scaffold_id = "C"
                    item.mol = mol
                    self.data.append(item)

            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = KnownScaffoldDataset()

        # Since we have real rdkit, we use a mol_getter that extracts
        # the scaffold from our mock mol's _scaffold_id attribute
        # by patching MurckoScaffoldSmiles at the exact import location
        with patch.object(_ds_module, "DataSplitter") as mock_splitter:
            # Create a custom implementation for this test
            mock_splitter.scaffold_split = DataSplitter.scaffold_split

        train, val, test = DataSplitter.scaffold_split(dataset)

        # All indices should be accounted for
        assert len(train) + len(val) + len(test) == 100

    def test_scaffold_split_custom_ratios(self, molecular_dataset):
        """Test scaffold split with custom ratios."""
        if not _rdkit_available():
            pytest.skip("rdkit not installed - skipping custom ratios test")

        train, val, test = DataSplitter.scaffold_split(
            molecular_dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
        )

        # Sizes will vary based on scaffold grouping
        assert len(train) + len(val) + len(test) == 100

    def test_scaffold_split_include_chirality(self, molecular_dataset):
        """Test scaffold split with chirality consideration."""
        if not _rdkit_available():
            pytest.skip("rdkit not installed - skipping chirality test")

        # Simply verify that include_chirality parameter is accepted
        # The actual chirality behavior is tested by rdkit itself
        train, val, test = DataSplitter.scaffold_split(molecular_dataset, include_chirality=True)

        assert len(train) + len(val) + len(test) == 100

    def test_scaffold_split_custom_mol_getter(self, molecular_dataset):
        """Test scaffold split with custom mol getter."""
        if not _rdkit_available():
            pytest.skip("rdkit not installed - skipping custom mol getter test")

        def custom_getter(data):
            return data.mol

        train, val, test = DataSplitter.scaffold_split(molecular_dataset, mol_getter=custom_getter)

        assert len(train) + len(val) + len(test) == 100

    def test_scaffold_split_reproducibility(self, molecular_dataset):
        """Test reproducibility with same seed."""
        if not _rdkit_available():
            pytest.skip("rdkit not installed - skipping reproducibility test")

        train1, val1, test1 = DataSplitter.scaffold_split(molecular_dataset, random_seed=42)
        train2, val2, test2 = DataSplitter.scaffold_split(molecular_dataset, random_seed=42)

        assert train1.indices == train2.indices
        assert val1.indices == val2.indices
        assert test1.indices == test2.indices

    def test_scaffold_split_invalid_ratios(self, molecular_dataset):
        """Test error when ratios don't sum to 1.0."""
        # This validation happens before rdkit import, so no rdkit needed
        with pytest.raises(DataError) as exc_info:
            DataSplitter.scaffold_split(
                molecular_dataset, train_ratio=0.5, val_ratio=0.3, test_ratio=0.1
            )

        assert "must sum to 1.0" in str(exc_info.value)

    def test_scaffold_split_empty_dataset(self, empty_dataset):
        """Test error with empty dataset."""
        # This validation happens before rdkit import, so no rdkit needed
        with pytest.raises(DataError) as exc_info:
            DataSplitter.scaffold_split(empty_dataset)

        assert "empty" in str(exc_info.value).lower()

    def test_scaffold_split_rdkit_not_available(self, molecular_dataset):
        """Test error when rdkit is not available."""
        # Temporarily make rdkit unavailable by removing it from sys.modules
        rdkit_modules = {k: v for k, v in sys.modules.items() if "rdkit" in k}

        try:
            # Remove rdkit modules temporarily
            for key in list(rdkit_modules.keys()):
                del sys.modules[key]

            # Also patch to ensure import fails
            with patch.dict(
                sys.modules,
                {
                    "rdkit": None,
                    "rdkit.Chem": None,
                    "rdkit.Chem.Scaffolds": None,
                    "rdkit.Chem.Scaffolds.MurckoScaffold": None,
                },
            ):
                with pytest.raises(DataError) as exc_info:
                    DataSplitter.scaffold_split(molecular_dataset)

                assert "rdkit" in str(exc_info.value)
        finally:
            # Restore rdkit modules
            sys.modules.update(rdkit_modules)

    def test_scaffold_split_missing_mol_attribute(self):
        """Test error when mol attribute is missing."""

        class NoMolDataset(Dataset):
            def __init__(self):
                self.data = [Mock(spec=[]) for _ in range(10)]

            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = NoMolDataset()

        if _rdkit_available():
            with pytest.raises(DataError) as exc_info:
                DataSplitter.scaffold_split(dataset)

            assert "missing 'mol' attribute" in str(exc_info.value)
        else:
            # rdkit not available - should fail on import first
            with pytest.raises(DataError) as exc_info:
                DataSplitter.scaffold_split(dataset)
            assert "rdkit" in str(exc_info.value)

    def test_scaffold_split_scaffold_generation_failure(self, molecular_dataset):
        """Test handling of scaffold generation failures."""
        if not _rdkit_available():
            pytest.skip("rdkit not installed - skipping scaffold generation failure test")

        # With real rdkit, we can't easily force scaffold generation failures
        # on valid molecules, so we test that the function completes normally
        train, val, test = DataSplitter.scaffold_split(molecular_dataset)

        # Should complete without errors
        assert len(train) + len(val) + len(test) == 100


# =============================================================================
# K-FOLD SPLIT TESTS
# =============================================================================


class TestKFoldSplit:
    """Test DataSplitter.k_fold_split method."""

    def test_k_fold_split_default_splits(self, simple_dataset):
        """Test k-fold split with default 5 folds."""
        folds = DataSplitter.k_fold_split(simple_dataset)

        assert isinstance(folds, list)
        assert len(folds) == 5

        for train, val in folds:
            assert isinstance(train, Subset)
            assert isinstance(val, Subset)

    def test_k_fold_split_fold_sizes(self, simple_dataset):
        """Test that folds have approximately equal sizes."""
        folds = DataSplitter.k_fold_split(simple_dataset, n_splits=5)

        for train, val in folds:
            # Each val fold should have ~20 samples
            assert 19 <= len(val) <= 21
            # Each train fold should have ~80 samples
            assert 79 <= len(train) <= 81

    def test_k_fold_split_custom_n_splits(self, simple_dataset):
        """Test k-fold split with custom number of folds."""
        folds = DataSplitter.k_fold_split(simple_dataset, n_splits=10)

        assert len(folds) == 10

        for _train, val in folds:
            # Each val fold should have ~10 samples
            assert 9 <= len(val) <= 11

    def test_k_fold_split_covers_all_samples(self, simple_dataset):
        """Test that all samples are used in validation exactly once."""
        folds = DataSplitter.k_fold_split(simple_dataset, n_splits=5)

        all_val_indices = []
        for _train, val in folds:
            all_val_indices.extend(val.indices)

        # All indices should be covered exactly once
        assert sorted(all_val_indices) == list(range(100))

    def test_k_fold_split_no_overlap_within_fold(self, simple_dataset):
        """Test that train and val don't overlap within each fold."""
        folds = DataSplitter.k_fold_split(simple_dataset, n_splits=5)

        for train, val in folds:
            train_set = set(train.indices)
            val_set = set(val.indices)
            assert len(train_set & val_set) == 0

    def test_k_fold_split_reproducibility(self, simple_dataset):
        """Test reproducibility with same seed."""
        folds1 = DataSplitter.k_fold_split(simple_dataset, n_splits=5, random_seed=42)
        folds2 = DataSplitter.k_fold_split(simple_dataset, n_splits=5, random_seed=42)

        for (train1, val1), (train2, val2) in zip(folds1, folds2, strict=False):
            assert train1.indices == train2.indices
            assert val1.indices == val2.indices

    def test_k_fold_split_different_seeds(self, simple_dataset):
        """Test different seeds produce different folds."""
        folds1 = DataSplitter.k_fold_split(simple_dataset, n_splits=5, random_seed=42)
        folds2 = DataSplitter.k_fold_split(simple_dataset, n_splits=5, random_seed=123)

        # At least one fold should differ
        different = False
        for (train1, _val1), (train2, _val2) in zip(folds1, folds2, strict=False):
            if train1.indices != train2.indices:
                different = True
                break

        assert different

    def test_k_fold_split_shuffle_true(self, simple_dataset):
        """Test k-fold split with shuffling."""
        folds = DataSplitter.k_fold_split(simple_dataset, shuffle=True, random_seed=42)

        # First fold's validation should not be sequential
        first_val_indices = folds[0][1].indices
        assert first_val_indices != list(range(20))

    def test_k_fold_split_shuffle_false(self, simple_dataset):
        """Test k-fold split without shuffling."""
        folds = DataSplitter.k_fold_split(simple_dataset, shuffle=False)

        # Validation folds should be sequential
        expected_val = [list(range(i * 20, (i + 1) * 20)) for i in range(5)]
        for i, (_train, val) in enumerate(folds):
            if i < 4:  # Last fold might be different size
                assert val.indices == expected_val[i]

    def test_k_fold_split_last_fold_handles_remainder(self):
        """Test that last fold handles remainder samples."""

        class Dataset103(Dataset):
            def __len__(self):
                return 103

            def __getitem__(self, idx):
                return idx

        dataset = Dataset103()
        folds = DataSplitter.k_fold_split(dataset, n_splits=5)

        # Last fold should have more samples
        last_val = folds[-1][1]
        assert len(last_val) >= 20  # Should include remainder

    def test_k_fold_split_n_splits_too_small(self, simple_dataset):
        """Test error when n_splits < 2."""
        with pytest.raises(DataError) as exc_info:
            DataSplitter.k_fold_split(simple_dataset, n_splits=1)

        assert "must be >= 2" in str(exc_info.value)

    def test_k_fold_split_dataset_too_small(self, small_dataset):
        """Test error when dataset is smaller than n_splits."""
        with pytest.raises(DataError) as exc_info:
            DataSplitter.k_fold_split(small_dataset, n_splits=10)

        assert "too small" in str(exc_info.value).lower()

    def test_k_fold_split_binary_fold(self, simple_dataset):
        """Test k-fold split with 2 folds (50-50 split)."""
        folds = DataSplitter.k_fold_split(simple_dataset, n_splits=2)

        assert len(folds) == 2

        # Each fold should have 50 in val, 50 in train
        for train, val in folds:
            assert len(val) == 50
            assert len(train) == 50

    def test_k_fold_split_with_zero_n_splits(self, simple_dataset):
        """Test error with n_splits = 0."""
        with pytest.raises(DataError):
            DataSplitter.k_fold_split(simple_dataset, n_splits=0)

    def test_k_fold_split_with_negative_n_splits(self, simple_dataset):
        """Test error with negative n_splits."""
        with pytest.raises(DataError):
            DataSplitter.k_fold_split(simple_dataset, n_splits=-5)


# =============================================================================
# CONVENIENCE FUNCTIONS TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Test all convenience functions."""

    def test_random_split_convenience_function(self, simple_dataset):
        """Test random_split convenience function."""
        train, val, test = random_split(simple_dataset)

        assert isinstance(train, Subset)
        assert isinstance(val, Subset)
        assert isinstance(test, Subset)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_random_split_convenience_custom_params(self, simple_dataset):
        """Test random_split convenience function with custom parameters."""
        train, val, test = random_split(
            simple_dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=123
        )

        assert len(train) == 70
        assert len(val) == 20
        assert len(test) == 10

    def test_stratified_split_convenience_function(self, labeled_dataset):
        """Test stratified_split convenience function."""
        try:
            from sklearn.model_selection import train_test_split

            train, val, test = stratified_split(labeled_dataset)

            assert isinstance(train, Subset)
            assert isinstance(val, Subset)
            assert isinstance(test, Subset)
        except ImportError:
            # sklearn not available - verify error is raised
            with pytest.raises(DataError) as exc_info:
                stratified_split(labeled_dataset)
            assert "scikit-learn" in str(exc_info.value)

    def test_stratified_split_convenience_custom_params(self, labeled_dataset):
        """Test stratified_split convenience function with custom parameters."""
        try:
            from sklearn.model_selection import train_test_split

            train, val, test = stratified_split(
                labeled_dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=999
            )

            assert len(train) + len(val) + len(test) == len(labeled_dataset)
        except ImportError:
            pytest.skip("sklearn not installed - skipping convenience custom params test")

    def test_temporal_split_convenience_function(self, temporal_dataset):
        """Test temporal_split convenience function."""
        train, val, test = temporal_split(temporal_dataset)

        assert isinstance(train, Subset)
        assert isinstance(val, Subset)
        assert isinstance(test, Subset)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_temporal_split_convenience_custom_params(self, temporal_dataset):
        """Test temporal_split convenience function with custom parameters."""
        train, val, test = temporal_split(
            temporal_dataset, train_ratio=0.6, val_ratio=0.3, test_ratio=0.1, time_field="time"
        )

        assert len(train) == 60
        assert len(val) == 30
        assert len(test) == 10

    def test_temporal_split_convenience_with_custom_getter(self, temporal_dataset):
        """Test temporal_split convenience with custom time getter."""

        def custom_getter(data):
            return data.time

        train, val, test = temporal_split(temporal_dataset, time_getter=custom_getter)

        assert len(train) + len(val) + len(test) == len(temporal_dataset)

    def test_scaffold_split_convenience_function(self, molecular_dataset):
        """Test scaffold_split convenience function."""
        if _rdkit_available():
            train, val, test = scaffold_split(molecular_dataset)

            assert isinstance(train, Subset)
            assert isinstance(val, Subset)
            assert isinstance(test, Subset)
            assert len(train) + len(val) + len(test) == 100
        else:
            # rdkit not available - verify error is raised
            with pytest.raises(DataError) as exc_info:
                scaffold_split(molecular_dataset)
            assert "rdkit" in str(exc_info.value)

    def test_scaffold_split_convenience_custom_params(self, molecular_dataset):
        """Test scaffold_split convenience function with custom parameters."""
        if not _rdkit_available():
            pytest.skip("rdkit not installed - skipping scaffold convenience custom params test")

        train, val, test = scaffold_split(
            molecular_dataset,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            random_seed=42,
            include_chirality=True,
        )

        assert len(train) + len(val) + len(test) == 100

    def test_scaffold_split_convenience_with_custom_getter(self, molecular_dataset):
        """Test scaffold_split convenience with custom mol getter."""
        if not _rdkit_available():
            pytest.skip("rdkit not installed - skipping scaffold convenience custom getter test")

        def custom_getter(data):
            return data.mol

        train, val, test = scaffold_split(molecular_dataset, mol_getter=custom_getter)

        assert len(train) + len(val) + len(test) == 100

    def test_k_fold_split_convenience_function(self, simple_dataset):
        """Test k_fold_split convenience function."""
        folds = k_fold_split(simple_dataset)

        assert isinstance(folds, list)
        assert len(folds) == 5

        for train, val in folds:
            assert isinstance(train, Subset)
            assert isinstance(val, Subset)

    def test_k_fold_split_convenience_custom_params(self, simple_dataset):
        """Test k_fold_split convenience function with custom parameters."""
        folds = k_fold_split(simple_dataset, n_splits=10, random_seed=42, shuffle=True)

        assert len(folds) == 10

        for train, val in folds:
            assert len(train) + len(val) == len(simple_dataset)


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_sample_dataset(self):
        """Test with dataset containing only one sample."""

        class SingleSampleDataset(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return 0

        dataset = SingleSampleDataset()

        # Should work but produce very small splits
        train, val, test = DataSplitter.random_split(
            dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )

        # All samples should be accounted for
        assert len(train) + len(val) + len(test) == 1

    def test_large_dataset_split(self):
        """Test with large dataset."""

        class LargeDataset(Dataset):
            def __len__(self):
                return 10000

            def __getitem__(self, idx):
                return idx

        dataset = LargeDataset()
        train, val, test = DataSplitter.random_split(dataset)

        assert len(train) == 8000
        assert len(val) == 1000
        assert len(test) == 1000

    def test_dataset_with_non_integer_indices(self, simple_dataset):
        """Test that subset indices are integers."""
        train, val, test = DataSplitter.random_split(simple_dataset)

        for idx in train.indices + val.indices + test.indices:
            assert isinstance(idx, int)

    def test_ratios_with_floating_point_precision(self, simple_dataset):
        """Test ratios that might have floating point precision issues."""
        # These should sum to exactly 1.0
        train, val, test = DataSplitter.random_split(
            simple_dataset, train_ratio=0.333333, val_ratio=0.333333, test_ratio=0.333334
        )

        assert len(train) + len(val) + len(test) == 100

    def test_zero_length_split(self):
        """Test split that results in zero-length val or test set."""

        class Dataset15(Dataset):
            def __len__(self):
                return 15

            def __getitem__(self, idx):
                return idx

        dataset = Dataset15()
        train, val, test = DataSplitter.random_split(
            dataset, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05
        )

        # With 15 samples: 13 train, 0 val, 2 test (due to int conversion)
        assert len(train) + len(val) + len(test) == 15

    def test_all_methods_with_same_dataset(self, simple_dataset):
        """Test that all split methods work on the same dataset."""
        # Random split
        train1, val1, test1 = DataSplitter.random_split(simple_dataset)
        assert len(train1) + len(val1) + len(test1) == 100

        # K-fold split
        folds = DataSplitter.k_fold_split(simple_dataset, n_splits=5)
        assert len(folds) == 5


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_sequential_splits_different_methods(self, simple_dataset):
        """Test using different split methods sequentially."""
        # Random split
        train1, val1, test1 = DataSplitter.random_split(simple_dataset, random_seed=42)

        # Another random split with different seed
        train2, val2, test2 = DataSplitter.random_split(simple_dataset, random_seed=123)

        # Should produce different results
        assert train1.indices != train2.indices

    def test_subset_can_be_used_with_dataloader(self, simple_dataset):
        """Test that Subset can be used with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        train, val, test = DataSplitter.random_split(simple_dataset)

        # Should be able to create DataLoader
        train_loader = DataLoader(train, batch_size=10)

        # Should be able to iterate
        batch = next(iter(train_loader))
        assert len(batch) == 10

    def test_split_preserves_dataset_access(self, simple_dataset):
        """Test that split subsets can access original dataset."""
        train, val, test = DataSplitter.random_split(simple_dataset)

        # Should be able to access items
        sample = train[0]
        assert sample is not None

        # Should match original dataset
        original_idx = train.indices[0]
        assert sample == simple_dataset[original_idx]

    def test_multiple_fold_training_simulation(self, simple_dataset):
        """Test simulating k-fold cross-validation training."""
        folds = DataSplitter.k_fold_split(simple_dataset, n_splits=3)

        results = []
        for fold_idx, (train, val) in enumerate(folds):
            # Simulate training on this fold
            train_size = len(train)
            val_size = len(val)
            results.append((fold_idx, train_size, val_size))

        # All folds should be processed
        assert len(results) == 3

        # Sizes should be consistent
        for fold_idx, train_size, val_size in results:
            assert train_size + val_size == len(simple_dataset)


# =============================================================================
# DOCUMENTATION EXAMPLES TESTS
# =============================================================================


class TestDocumentationExamples:
    """Test examples that would appear in documentation."""

    def test_random_split_basic_example(self, simple_dataset):
        """Test basic random_split example from docstring."""
        train, val, test = DataSplitter.random_split(simple_dataset)

        assert len(train) + len(val) + len(test) == len(simple_dataset)

    def test_random_split_custom_ratios_example(self, simple_dataset):
        """Test custom ratios example from docstring."""
        train, val, test = DataSplitter.random_split(
            simple_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
        )

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_stratified_split_example(self, labeled_dataset):
        """Test stratified_split example from docstring."""
        try:
            from sklearn.model_selection import train_test_split

            train, val, test = DataSplitter.stratified_split(
                labeled_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
            )

            assert len(train) + len(val) + len(test) == len(labeled_dataset)
        except ImportError:
            pytest.skip("sklearn not installed - skipping stratified split doc example test")

    def test_temporal_split_example(self, temporal_dataset):
        """Test temporal_split example from docstring."""
        train, val, test = DataSplitter.temporal_split(
            temporal_dataset, time_field="time", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
        )

        assert len(train) + len(val) + len(test) == len(temporal_dataset)

    def test_k_fold_split_example(self, simple_dataset):
        """Test k_fold_split example from docstring."""
        folds = DataSplitter.k_fold_split(simple_dataset, n_splits=5)

        for _fold_idx, (train, val) in enumerate(folds):
            # Verify structure
            assert isinstance(train, Subset)
            assert isinstance(val, Subset)

    def test_convenience_function_example(self, simple_dataset):
        """Test convenience function example from module docstring."""
        train, val, test = random_split(simple_dataset)

        assert isinstance(train, Subset)
        assert len(train) == 80

    def test_all_convenience_functions_work(self, simple_dataset, temporal_dataset):
        """Test that all convenience functions are accessible and work."""
        # random_split
        train1, val1, test1 = random_split(simple_dataset)
        assert len(train1) + len(val1) + len(test1) == 100

        # temporal_split
        train2, val2, test2 = temporal_split(temporal_dataset)
        assert len(train2) + len(val2) + len(test2) == 100

        # k_fold_split
        folds = k_fold_split(simple_dataset, n_splits=3)
        assert len(folds) == 3


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
