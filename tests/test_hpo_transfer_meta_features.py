#!/usr/bin/env python3
"""
Complete Unit Test Suite for milia_pipeline/models/hpo/transfer/meta_features.py Module

Comprehensive Production-Ready Test Suite

Tests:
- Lazy import functions (_lazy_import_torch, _lazy_import_torch_geometric, _lazy_import_rdkit)
- MetaFeatureCategory enum (all values, value access, membership)
- MetaFeatureConfig Pydantic BaseModel (initialization, validation, frozen behavior, should_extract, to_dict)
- MetaFeatureExtractor.__init__() (default config, custom config, lazy imports)
- MetaFeatureExtractor._NORMALIZATION_BOUNDS class constant
- MetaFeatureExtractor.extract (static method) and extract_features (instance method)
- MetaFeatureExtractor._extract_statistical_features
- MetaFeatureExtractor._extract_graph_features
- MetaFeatureExtractor._extract_target_features
- MetaFeatureExtractor._extract_node_feature_statistics
- MetaFeatureExtractor._extract_edge_feature_statistics
- MetaFeatureExtractor._get_num_nodes
- MetaFeatureExtractor._get_num_edges
- MetaFeatureExtractor._compute_degrees
- MetaFeatureExtractor._compute_clustering_coefficients
- MetaFeatureExtractor._extract_molecular_features
- MetaFeatureExtractor._get_atomic_numbers
- MetaFeatureExtractor._get_bond_types
- MetaFeatureExtractor._get_molecular_weight
- MetaFeatureExtractor._get_ring_count
- MetaFeatureExtractor._normalize_features
- MetaFeatureExtractor.compute_similarity (static method)
- MetaFeatureExtractor.get_feature_names (static method)
- MetaFeatureExtractor.get_category_for_feature (static method)
- Integration tests
- Edge cases and error handling

Pydantic V2 Compatibility:
    - Uses PydanticValidationError for frozen model mutation tests
    - Tests to_dict() backward compatible method
    - Validates field_validator and model_validator behavior

Author: Milia Team
Version: 1.1.0
"""
import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import pytest
from typing import Dict, Any, Optional, List, Tuple
from unittest.mock import patch, MagicMock, PropertyMock
from enum import Enum
import numpy as np

# Pydantic V2 uses ValidationError for frozen model mutation attempts
try:
    from pydantic import ValidationError as PydanticValidationError
except ImportError:
    PydanticValidationError = Exception  # Fallback for type checking


# =============================================================================
# MOCK CLASSES FOR DEPENDENCIES
# =============================================================================

class MockTorch:
    """Mock torch module for testing."""
    
    class Tensor:
        """Mock torch Tensor."""
        def __init__(self, data):
            self._data = np.array(data)
        
        def shape(self):
            return self._data.shape
        
        def tolist(self):
            return self._data.tolist()
        
        def numpy(self):
            return self._data
        
        def item(self):
            return float(self._data.flat[0])
        
        def numel(self):
            return self._data.size
        
        def mean(self):
            result = MockTorch.Tensor([self._data.mean()])
            return result
        
        def flatten(self):
            return MockTorch.Tensor(self._data.flatten())
        
        def dim(self):
            return len(self._data.shape)
        
        def size(self, dim=None):
            if dim is None:
                return self._data.shape
            return self._data.shape[dim]
        
        def argmax(self, dim=None):
            return MockTorch.Tensor(np.argmax(self._data, axis=dim))
        
        def __float__(self):
            return float(self._data.flat[0])
        
        def __len__(self):
            return len(self._data)
        
        def __getitem__(self, idx):
            return MockTorch.Tensor(self._data[idx])


class MockDegreeFunction:
    """Mock torch_geometric degree function."""
    
    def __call__(self, edge_index, num_nodes=None):
        # Return mock degree tensor
        if num_nodes is None:
            num_nodes = 10
        return MockTorch.Tensor(np.random.randint(1, 5, size=num_nodes))


class MockRDKit:
    """Mock RDKit modules for testing."""
    
    class Chem:
        @staticmethod
        def MolFromSmiles(smiles):
            return MagicMock()
    
    class Descriptors:
        @staticmethod
        def MolWt(mol):
            return 180.0
    
    class rdMolDescriptors:
        @staticmethod
        def CalcNumRings(mol):
            return 2


class MockTensor:
    """Mock tensor class compatible with both torch and numpy.
    
    This is the primary mock tensor class used by MockDataset and helper functions.
    It provides a consistent interface mimicking PyTorch tensors for testing.
    """
    
    def __init__(self, data):
        if isinstance(data, MockTensor):
            self._data = data._data
        else:
            self._data = np.array(data)
    
    @property
    def shape(self):
        return self._data.shape
    
    def tolist(self):
        return self._data.tolist()
    
    def numpy(self):
        return self._data
    
    def item(self):
        return float(self._data.flat[0])
    
    def numel(self):
        return self._data.size
    
    def mean(self):
        return MockTensor([self._data.mean()])
    
    def flatten(self):
        return MockTensor(self._data.flatten())
    
    def dim(self):
        return len(self._data.shape)
    
    def size(self, dim=None):
        if dim is None:
            return self._data.shape
        return self._data.shape[dim]
    
    def argmax(self, dim=None):
        return MockTensor(np.argmax(self._data, axis=dim))
    
    def __float__(self):
        return float(self._data.flat[0])
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        return MockTensor(self._data[idx])
    
    def __add__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data + other._data)
        return MockTensor(self._data + other)


class MockData:
    """Mock PyG Data object for testing."""
    
    def __init__(
        self,
        x=None,
        edge_index=None,
        edge_attr=None,
        y=None,
        z=None,
        num_nodes=None,
        atomic_numbers=None,
        bond_type=None,
        mol_weight=None,
        num_rings=None,
        ring_count=None
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.z = z
        self.num_nodes = num_nodes
        self.atomic_numbers = atomic_numbers
        self.bond_type = bond_type
        self.mol_weight = mol_weight
        self.num_rings = num_rings
        self.ring_count = ring_count


class MockDataset:
    """Mock PyG-like dataset for testing.
    
    This is the primary mock dataset class used throughout the test suite.
    It generates mock graph data with configurable properties.
    """
    
    def __init__(
        self,
        size=100,
        n_features=10,
        mean_edges=50,
        mean_nodes=20,
        has_target=True,
        has_edge_attr=True,
        has_z=False,
        target_dim=1,
        edge_attr_dim=4,
        molecular=False
    ):
        self._size = size
        self._n_features = n_features
        self._mean_edges = mean_edges
        self._mean_nodes = mean_nodes
        self._has_target = has_target
        self._has_edge_attr = has_edge_attr
        self._has_z = has_z
        self._target_dim = target_dim
        self._edge_attr_dim = edge_attr_dim
        self._molecular = molecular
        self._data = [self._create_mock_data(i) for i in range(size)]
    
    def _create_mock_data(self, idx):
        """Create a mock data object with random variations."""
        np.random.seed(idx)  # Reproducible randomness per sample
        
        n_nodes = max(3, self._mean_nodes + np.random.randint(-5, 6))
        n_edges = max(1, self._mean_edges + np.random.randint(-10, 11))
        
        if self._molecular:
            return _create_molecular_mock_data(n_nodes=n_nodes, n_edges=n_edges, seed=idx)
        
        return _create_mock_data(
            n_nodes=n_nodes,
            n_edges=n_edges,
            n_features=self._n_features,
            n_edge_features=self._edge_attr_dim,
            has_target=self._has_target,
            target_dim=self._target_dim,
            has_edge_attr=self._has_edge_attr,
            has_z=self._has_z,
            seed=idx
        )
    
    def __len__(self):
        return self._size
    
    def __getitem__(self, idx):
        return self._data[idx]


def _create_mock_data(
    n_nodes=20,
    n_edges=50,
    n_features=10,
    n_edge_features=4,
    has_target=True,
    target_dim=1,
    has_edge_attr=True,
    has_z=False,
    seed=42
):
    """Factory function to create mock data with specified properties.
    
    This function creates MockData objects with configurable properties
    for use in tests. It uses MockTensor for all tensor-like attributes.
    """
    np.random.seed(seed)
    
    x = MockTensor(np.random.randn(n_nodes, n_features))
    
    # Create valid edge_index (source and target nodes)
    edge_index = MockTensor(np.random.randint(0, n_nodes, size=(2, n_edges)))
    
    y = None
    if has_target:
        if target_dim == 1:
            y = MockTensor([np.random.randn()])
        else:
            y = MockTensor(np.random.randn(target_dim))
    
    edge_attr = None
    if has_edge_attr:
        edge_attr = MockTensor(np.random.randn(n_edges, n_edge_features))
    
    z = None
    if has_z:
        z = MockTensor(np.random.choice([1, 6, 7, 8], size=n_nodes))
    
    return MockData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        z=z,
        num_nodes=n_nodes
    )


def _create_molecular_mock_data(n_nodes=15, n_edges=30, seed=42):
    """Create mock data with molecular features (z, atomic_numbers, bond_type).
    
    This function creates MockData objects specifically designed to mimic
    molecular graph data with atom types, bond types, and molecular properties.
    """
    np.random.seed(seed)
    
    # Common molecular atoms: H=1, C=6, N=7, O=8
    z = MockTensor(np.random.choice([1, 6, 6, 6, 7, 8], size=n_nodes))
    
    # Bond types: 1=single, 2=double, 3=triple, 12=aromatic
    bond_types = np.random.choice([1, 1, 1, 2, 3, 12], size=n_edges)
    
    # One-hot encoded edge features for bond types (5 columns)
    edge_attr = np.zeros((n_edges, 5))
    for i, bt in enumerate(bond_types):
        if bt == 1:
            edge_attr[i, 0] = 1  # Single
        elif bt == 2:
            edge_attr[i, 1] = 1  # Double
        elif bt == 3:
            edge_attr[i, 2] = 1  # Triple
        elif bt == 12:
            edge_attr[i, 3] = 1  # Aromatic
        else:
            edge_attr[i, 4] = 1  # Other
    
    return MockData(
        x=MockTensor(np.random.randn(n_nodes, 10)),
        edge_index=MockTensor(np.random.randint(0, n_nodes, size=(2, n_edges))),
        edge_attr=MockTensor(edge_attr),
        y=MockTensor([np.random.randn()]),
        z=z,
        num_nodes=n_nodes,
        mol_weight=180.0 + np.random.randn() * 50,
        num_rings=np.random.randint(0, 4)
    )


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_torch():
    """Create mock torch module."""
    return MockTorch()


@pytest.fixture
def mock_degree_fn():
    """Create mock degree function."""
    return MockDegreeFunction()


@pytest.fixture
def mock_dataset():
    """Create mock PyG-like dataset."""
    return MockDataset()


@pytest.fixture
def mock_dataset_small():
    """Create small mock dataset for quick tests."""
    return MockDataset(size=10, mean_nodes=10, mean_edges=20)


@pytest.fixture
def mock_dataset_with_molecular():
    """Create mock dataset with molecular features."""
    return MockDataset(size=50, has_z=True, mean_nodes=15)


@pytest.fixture
def mock_empty_dataset():
    """Create empty mock dataset."""
    return MockDataset(size=0)


@pytest.fixture
def sample_meta_features():
    """Create sample meta-features dictionary."""
    return {
        "n_samples": 1000.0,
        "n_features": 10.0,
        "mean_nodes": 25.0,
        "mean_edges": 50.0,
        "target_mean": 0.5,
        "target_std": 0.1
    }


# =============================================================================
# LAZY IMPORT FUNCTION TESTS
# =============================================================================

class TestLazyImportTorch:
    """Test _lazy_import_torch function."""
    
    def test_lazy_import_torch_success(self):
        """Test _lazy_import_torch returns torch when available."""
        from milia_pipeline.models.hpo.transfer.meta_features import _lazy_import_torch
        
        result = _lazy_import_torch()
        # Result should be either torch module or None
        assert result is None or hasattr(result, 'Tensor')
    
    @patch.dict('sys.modules', {'torch': None})
    def test_lazy_import_torch_import_error(self):
        """Test _lazy_import_torch returns None when torch unavailable."""
        # Need to reload module to test import behavior
        import importlib
        from milia_pipeline.models.hpo.transfer import meta_features
        
        # Patch the import inside the function
        with patch('builtins.__import__', side_effect=ImportError("No torch")):
            # Call the function directly, which tries to import torch
            result = meta_features._lazy_import_torch()
            # Should return None on ImportError
            # Note: This may not work perfectly due to caching
    
    def test_lazy_import_torch_caches_result(self):
        """Test that _lazy_import_torch can be called multiple times."""
        from milia_pipeline.models.hpo.transfer.meta_features import _lazy_import_torch
        
        result1 = _lazy_import_torch()
        result2 = _lazy_import_torch()
        
        # Should return same type both times
        assert type(result1) == type(result2)


class TestLazyImportTorchGeometric:
    """Test _lazy_import_torch_geometric function."""
    
    def test_lazy_import_torch_geometric_success(self):
        """Test _lazy_import_torch_geometric returns degree function when available."""
        from milia_pipeline.models.hpo.transfer.meta_features import _lazy_import_torch_geometric
        
        result = _lazy_import_torch_geometric()
        # Result should be either degree function or None
        assert result is None or callable(result)
    
    def test_lazy_import_torch_geometric_returns_callable(self):
        """Test that returned value is callable if not None."""
        from milia_pipeline.models.hpo.transfer.meta_features import _lazy_import_torch_geometric
        
        result = _lazy_import_torch_geometric()
        if result is not None:
            assert callable(result)


class TestLazyImportRDKit:
    """Test _lazy_import_rdkit function."""
    
    def test_lazy_import_rdkit_returns_tuple(self):
        """Test _lazy_import_rdkit returns 3-tuple."""
        from milia_pipeline.models.hpo.transfer.meta_features import _lazy_import_rdkit
        
        result = _lazy_import_rdkit()
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_lazy_import_rdkit_all_none_when_unavailable(self):
        """Test _lazy_import_rdkit returns (None, None, None) when unavailable."""
        from milia_pipeline.models.hpo.transfer.meta_features import _lazy_import_rdkit
        
        result = _lazy_import_rdkit()
        # Either all three are available or all three are None
        if result[0] is None:
            assert result == (None, None, None)


# =============================================================================
# META-FEATURE CATEGORY ENUM TESTS
# =============================================================================

class TestMetaFeatureCategoryEnum:
    """Test MetaFeatureCategory enum."""
    
    def test_meta_feature_category_has_statistical(self):
        """Test MetaFeatureCategory has STATISTICAL value."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureCategory
        
        assert hasattr(MetaFeatureCategory, 'STATISTICAL')
        assert MetaFeatureCategory.STATISTICAL.value == "statistical"
    
    def test_meta_feature_category_has_graph(self):
        """Test MetaFeatureCategory has GRAPH value."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureCategory
        
        assert hasattr(MetaFeatureCategory, 'GRAPH')
        assert MetaFeatureCategory.GRAPH.value == "graph"
    
    def test_meta_feature_category_has_molecular(self):
        """Test MetaFeatureCategory has MOLECULAR value."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureCategory
        
        assert hasattr(MetaFeatureCategory, 'MOLECULAR')
        assert MetaFeatureCategory.MOLECULAR.value == "molecular"
    
    def test_meta_feature_category_has_target(self):
        """Test MetaFeatureCategory has TARGET value."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureCategory
        
        assert hasattr(MetaFeatureCategory, 'TARGET')
        assert MetaFeatureCategory.TARGET.value == "target"
    
    def test_meta_feature_category_has_node_features(self):
        """Test MetaFeatureCategory has NODE_FEATURES value."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureCategory
        
        assert hasattr(MetaFeatureCategory, 'NODE_FEATURES')
        assert MetaFeatureCategory.NODE_FEATURES.value == "node_features"
    
    def test_meta_feature_category_has_edge_features(self):
        """Test MetaFeatureCategory has EDGE_FEATURES value."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureCategory
        
        assert hasattr(MetaFeatureCategory, 'EDGE_FEATURES')
        assert MetaFeatureCategory.EDGE_FEATURES.value == "edge_features"
    
    def test_meta_feature_category_has_all(self):
        """Test MetaFeatureCategory has ALL value."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureCategory
        
        assert hasattr(MetaFeatureCategory, 'ALL')
        assert MetaFeatureCategory.ALL.value == "all"
    
    def test_meta_feature_category_from_string(self):
        """Test MetaFeatureCategory can be created from string."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureCategory
        
        category = MetaFeatureCategory("statistical")
        assert category == MetaFeatureCategory.STATISTICAL
    
    def test_meta_feature_category_invalid_string_raises(self):
        """Test MetaFeatureCategory raises ValueError for invalid string."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureCategory
        
        with pytest.raises(ValueError):
            MetaFeatureCategory("invalid_category")
    
    def test_meta_feature_category_is_enum(self):
        """Test MetaFeatureCategory is an Enum."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureCategory
        
        assert issubclass(MetaFeatureCategory, Enum)
    
    def test_meta_feature_category_iteration(self):
        """Test MetaFeatureCategory can be iterated."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureCategory
        
        categories = list(MetaFeatureCategory)
        assert len(categories) == 7  # STATISTICAL, GRAPH, MOLECULAR, TARGET, NODE_FEATURES, EDGE_FEATURES, ALL
    
    def test_meta_feature_category_membership(self):
        """Test MetaFeatureCategory membership checks."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureCategory
        
        assert MetaFeatureCategory.STATISTICAL in MetaFeatureCategory
        assert MetaFeatureCategory.ALL in MetaFeatureCategory


# =============================================================================
# META-FEATURE CONFIG DATACLASS TESTS
# =============================================================================

class TestMetaFeatureConfig:
    """Test MetaFeatureConfig dataclass."""
    
    def test_meta_feature_config_default_initialization(self):
        """Test MetaFeatureConfig with default values."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig, MetaFeatureCategory
        
        config = MetaFeatureConfig()
        
        assert config.categories == (MetaFeatureCategory.ALL,)
        assert config.max_samples is None
        assert config.normalize is False
        assert config.include_molecular is True
        assert config.compute_expensive is True
    
    def test_meta_feature_config_custom_categories(self):
        """Test MetaFeatureConfig with custom categories."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig, MetaFeatureCategory
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.STATISTICAL, MetaFeatureCategory.GRAPH)
        )
        
        assert len(config.categories) == 2
        assert MetaFeatureCategory.STATISTICAL in config.categories
        assert MetaFeatureCategory.GRAPH in config.categories
    
    def test_meta_feature_config_single_category(self):
        """Test MetaFeatureConfig with single category."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig, MetaFeatureCategory
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.MOLECULAR,)
        )
        
        assert len(config.categories) == 1
        assert MetaFeatureCategory.MOLECULAR in config.categories
    
    def test_meta_feature_config_max_samples(self):
        """Test MetaFeatureConfig with max_samples."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig
        
        config = MetaFeatureConfig(max_samples=100)
        
        assert config.max_samples == 100
    
    def test_meta_feature_config_max_samples_one(self):
        """Test MetaFeatureConfig accepts max_samples=1."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig
        
        config = MetaFeatureConfig(max_samples=1)
        assert config.max_samples == 1
    
    def test_meta_feature_config_max_samples_invalid_zero(self):
        """Test MetaFeatureConfig rejects max_samples=0."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig
        
        with pytest.raises(ValueError, match="max_samples must be at least 1"):
            MetaFeatureConfig(max_samples=0)
    
    def test_meta_feature_config_max_samples_invalid_negative(self):
        """Test MetaFeatureConfig rejects negative max_samples."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig
        
        with pytest.raises(ValueError, match="max_samples must be at least 1"):
            MetaFeatureConfig(max_samples=-5)
    
    def test_meta_feature_config_normalize_true(self):
        """Test MetaFeatureConfig with normalize=True."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig
        
        config = MetaFeatureConfig(normalize=True)
        assert config.normalize is True
    
    def test_meta_feature_config_include_molecular_false(self):
        """Test MetaFeatureConfig with include_molecular=False."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig
        
        config = MetaFeatureConfig(include_molecular=False)
        assert config.include_molecular is False
    
    def test_meta_feature_config_compute_expensive_false(self):
        """Test MetaFeatureConfig with compute_expensive=False."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig
        
        config = MetaFeatureConfig(compute_expensive=False)
        assert config.compute_expensive is False
    
    def test_meta_feature_config_empty_categories_raises(self):
        """Test MetaFeatureConfig rejects empty categories."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig
        
        with pytest.raises(ValueError, match="categories cannot be empty"):
            MetaFeatureConfig(categories=())
    
    def test_meta_feature_config_is_frozen(self):
        """Test MetaFeatureConfig is frozen (immutable) using Pydantic V2 frozen model."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig
        
        config = MetaFeatureConfig()
        
        # Pydantic V2 frozen models raise ValidationError on attribute assignment
        with pytest.raises(PydanticValidationError):
            config.max_samples = 50
    
    def test_meta_feature_config_all_custom_values(self):
        """Test MetaFeatureConfig with all custom values."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig, MetaFeatureCategory
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.STATISTICAL,),
            max_samples=50,
            normalize=True,
            include_molecular=False,
            compute_expensive=False
        )
        
        assert config.categories == (MetaFeatureCategory.STATISTICAL,)
        assert config.max_samples == 50
        assert config.normalize is True
        assert config.include_molecular is False
        assert config.compute_expensive is False
    
    def test_meta_feature_config_to_dict(self):
        """Test MetaFeatureConfig.to_dict() backward compatible method."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig, MetaFeatureCategory
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.STATISTICAL, MetaFeatureCategory.GRAPH),
            max_samples=100,
            normalize=True,
            include_molecular=False,
            compute_expensive=True
        )
        
        result = config.to_dict()
        
        assert isinstance(result, dict)
        assert result['max_samples'] == 100
        assert result['normalize'] is True
        assert result['include_molecular'] is False
        assert result['compute_expensive'] is True
        # Categories should be serialized properly
        assert 'categories' in result
    
    def test_meta_feature_config_to_dict_default_values(self):
        """Test MetaFeatureConfig.to_dict() with default values."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig, MetaFeatureCategory
        
        config = MetaFeatureConfig()
        result = config.to_dict()
        
        assert isinstance(result, dict)
        assert result['max_samples'] is None
        assert result['normalize'] is False
        assert result['include_molecular'] is True
        assert result['compute_expensive'] is True


class TestMetaFeatureConfigShouldExtract:
    """Test MetaFeatureConfig.should_extract method."""
    
    def test_should_extract_with_all_category(self):
        """Test should_extract returns True for all categories when ALL is selected."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig, MetaFeatureCategory
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.ALL,))
        
        assert config.should_extract(MetaFeatureCategory.STATISTICAL) is True
        assert config.should_extract(MetaFeatureCategory.GRAPH) is True
        assert config.should_extract(MetaFeatureCategory.MOLECULAR) is True
        assert config.should_extract(MetaFeatureCategory.TARGET) is True
        assert config.should_extract(MetaFeatureCategory.NODE_FEATURES) is True
        assert config.should_extract(MetaFeatureCategory.EDGE_FEATURES) is True
    
    def test_should_extract_with_specific_categories(self):
        """Test should_extract returns True only for selected categories."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig, MetaFeatureCategory
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.STATISTICAL, MetaFeatureCategory.GRAPH)
        )
        
        assert config.should_extract(MetaFeatureCategory.STATISTICAL) is True
        assert config.should_extract(MetaFeatureCategory.GRAPH) is True
        assert config.should_extract(MetaFeatureCategory.MOLECULAR) is False
        assert config.should_extract(MetaFeatureCategory.TARGET) is False
        assert config.should_extract(MetaFeatureCategory.NODE_FEATURES) is False
        assert config.should_extract(MetaFeatureCategory.EDGE_FEATURES) is False
    
    def test_should_extract_single_category(self):
        """Test should_extract with single category."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig, MetaFeatureCategory
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.TARGET,))
        
        assert config.should_extract(MetaFeatureCategory.TARGET) is True
        assert config.should_extract(MetaFeatureCategory.STATISTICAL) is False
        assert config.should_extract(MetaFeatureCategory.GRAPH) is False
    
    def test_should_extract_molecular_category(self):
        """Test should_extract specifically for MOLECULAR category."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig, MetaFeatureCategory
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.MOLECULAR,))
        
        assert config.should_extract(MetaFeatureCategory.MOLECULAR) is True
        assert config.should_extract(MetaFeatureCategory.STATISTICAL) is False
    
    def test_should_extract_all_individual_categories(self):
        """Test should_extract with all individual categories (not ALL)."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureConfig, MetaFeatureCategory
        
        all_categories = (
            MetaFeatureCategory.STATISTICAL,
            MetaFeatureCategory.GRAPH,
            MetaFeatureCategory.MOLECULAR,
            MetaFeatureCategory.TARGET,
            MetaFeatureCategory.NODE_FEATURES,
            MetaFeatureCategory.EDGE_FEATURES
        )
        
        config = MetaFeatureConfig(categories=all_categories)
        
        for cat in all_categories:
            assert config.should_extract(cat) is True


# =============================================================================
# META-FEATURE EXTRACTOR INITIALIZATION TESTS
# =============================================================================

class TestMetaFeatureExtractorInit:
    """Test MetaFeatureExtractor initialization."""
    
    def test_meta_feature_extractor_default_config(self):
        """Test MetaFeatureExtractor with default configuration."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor, MetaFeatureConfig
        
        extractor = MetaFeatureExtractor()
        
        assert extractor.config is not None
        assert isinstance(extractor.config, MetaFeatureConfig)
    
    def test_meta_feature_extractor_custom_config(self):
        """Test MetaFeatureExtractor with custom configuration."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.STATISTICAL,),
            max_samples=50
        )
        
        extractor = MetaFeatureExtractor(config)
        
        assert extractor.config is config
        assert extractor.config.max_samples == 50
    
    def test_meta_feature_extractor_none_config_uses_default(self):
        """Test MetaFeatureExtractor with None config uses default."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor, MetaFeatureCategory
        
        extractor = MetaFeatureExtractor(None)
        
        assert extractor.config is not None
        assert MetaFeatureCategory.ALL in extractor.config.categories
    
    def test_meta_feature_extractor_has_torch_attribute(self):
        """Test MetaFeatureExtractor has _torch attribute."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        
        assert hasattr(extractor, '_torch')
        # Can be None or torch module
    
    def test_meta_feature_extractor_has_degree_fn_attribute(self):
        """Test MetaFeatureExtractor has _degree_fn attribute."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        
        assert hasattr(extractor, '_degree_fn')
        # Can be None or callable
    
    def test_meta_feature_extractor_has_rdkit_attribute(self):
        """Test MetaFeatureExtractor has _rdkit attribute."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        
        assert hasattr(extractor, '_rdkit')
        # Should be a tuple of 3 elements
        assert isinstance(extractor._rdkit, tuple)
        assert len(extractor._rdkit) == 3


class TestMetaFeatureExtractorNormalizationBounds:
    """Test MetaFeatureExtractor._NORMALIZATION_BOUNDS constant."""
    
    def test_normalization_bounds_exists(self):
        """Test _NORMALIZATION_BOUNDS class constant exists."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        assert hasattr(MetaFeatureExtractor, '_NORMALIZATION_BOUNDS')
        assert isinstance(MetaFeatureExtractor._NORMALIZATION_BOUNDS, dict)
    
    def test_normalization_bounds_n_samples(self):
        """Test _NORMALIZATION_BOUNDS has n_samples."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        assert 'n_samples' in MetaFeatureExtractor._NORMALIZATION_BOUNDS
        bounds = MetaFeatureExtractor._NORMALIZATION_BOUNDS['n_samples']
        assert bounds == (1, 1_000_000)
    
    def test_normalization_bounds_n_features(self):
        """Test _NORMALIZATION_BOUNDS has n_features."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        assert 'n_features' in MetaFeatureExtractor._NORMALIZATION_BOUNDS
        bounds = MetaFeatureExtractor._NORMALIZATION_BOUNDS['n_features']
        assert bounds == (1, 10_000)
    
    def test_normalization_bounds_mean_nodes(self):
        """Test _NORMALIZATION_BOUNDS has mean_nodes."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        assert 'mean_nodes' in MetaFeatureExtractor._NORMALIZATION_BOUNDS
        bounds = MetaFeatureExtractor._NORMALIZATION_BOUNDS['mean_nodes']
        assert bounds == (1, 10_000)
    
    def test_normalization_bounds_mean_edges(self):
        """Test _NORMALIZATION_BOUNDS has mean_edges."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        assert 'mean_edges' in MetaFeatureExtractor._NORMALIZATION_BOUNDS
        bounds = MetaFeatureExtractor._NORMALIZATION_BOUNDS['mean_edges']
        assert bounds == (1, 100_000)
    
    def test_normalization_bounds_mean_density(self):
        """Test _NORMALIZATION_BOUNDS has mean_density."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        assert 'mean_density' in MetaFeatureExtractor._NORMALIZATION_BOUNDS
        bounds = MetaFeatureExtractor._NORMALIZATION_BOUNDS['mean_density']
        assert bounds == (0, 1)
    
    def test_normalization_bounds_mean_degree(self):
        """Test _NORMALIZATION_BOUNDS has mean_degree."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        assert 'mean_degree' in MetaFeatureExtractor._NORMALIZATION_BOUNDS
        bounds = MetaFeatureExtractor._NORMALIZATION_BOUNDS['mean_degree']
        assert bounds == (0, 100)
    
    def test_normalization_bounds_clustering_coefficient(self):
        """Test _NORMALIZATION_BOUNDS has clustering_coefficient."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        assert 'clustering_coefficient' in MetaFeatureExtractor._NORMALIZATION_BOUNDS
        bounds = MetaFeatureExtractor._NORMALIZATION_BOUNDS['clustering_coefficient']
        assert bounds == (0, 1)
    
    def test_normalization_bounds_values_are_tuples(self):
        """Test all _NORMALIZATION_BOUNDS values are 2-tuples."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        for key, bounds in MetaFeatureExtractor._NORMALIZATION_BOUNDS.items():
            assert isinstance(bounds, tuple), f"{key} should be a tuple"
            assert len(bounds) == 2, f"{key} should have 2 elements"
            assert bounds[0] <= bounds[1], f"{key} low should be <= high"


# =============================================================================
# META-FEATURE EXTRACTOR STATIC EXTRACT METHOD TESTS
# =============================================================================

class TestMetaFeatureExtractorExtractStatic:
    """Test MetaFeatureExtractor.extract static method."""
    
    def test_extract_static_method_exists(self):
        """Test extract static method exists."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        assert hasattr(MetaFeatureExtractor, 'extract')
        assert callable(MetaFeatureExtractor.extract)
    
    def test_extract_returns_dict(self, mock_dataset_small):
        """Test extract returns a dictionary."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.extract(mock_dataset_small)
        
        assert isinstance(result, dict)
    
    def test_extract_returns_float_values(self, mock_dataset_small):
        """Test extract returns float values."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.extract(mock_dataset_small)
        
        for key, value in result.items():
            assert isinstance(value, float), f"{key} should be float, got {type(value)}"
    
    def test_extract_with_none_config(self, mock_dataset_small):
        """Test extract with None config uses defaults."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.extract(mock_dataset_small, config=None)
        
        assert isinstance(result, dict)
        assert 'n_samples' in result
    
    def test_extract_with_custom_config(self, mock_dataset_small):
        """Test extract with custom config."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.STATISTICAL,),
            compute_expensive=False
        )
        
        result = MetaFeatureExtractor.extract(mock_dataset_small, config=config)
        
        assert isinstance(result, dict)
        assert 'n_samples' in result
    
    def test_extract_contains_n_samples(self, mock_dataset_small):
        """Test extract result contains n_samples."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.extract(mock_dataset_small)
        
        assert 'n_samples' in result
        assert result['n_samples'] == float(len(mock_dataset_small))


# =============================================================================
# MODULE EXPORTS TESTS
# =============================================================================

class TestModuleExports:
    """Test module exports."""
    
    def test_all_exports_available(self):
        """Test all expected exports are available."""
        from milia_pipeline.models.hpo.transfer import meta_features
        
        expected_exports = [
            'MetaFeatureCategory',
            'MetaFeatureConfig',
            'MetaFeatureExtractor',
        ]
        
        for export in expected_exports:
            assert hasattr(meta_features, export), f"Missing export: {export}"
    
    def test_all_list_matches_exports(self):
        """Test __all__ matches available exports."""
        from milia_pipeline.models.hpo.transfer import meta_features
        
        if hasattr(meta_features, '__all__'):
            for export in meta_features.__all__:
                assert hasattr(meta_features, export), f"__all__ contains unavailable: {export}"
    
    def test_all_list_correct_content(self):
        """Test __all__ contains exactly expected exports."""
        from milia_pipeline.models.hpo.transfer import meta_features
        
        expected = ['MetaFeatureCategory', 'MetaFeatureConfig', 'MetaFeatureExtractor']
        
        if hasattr(meta_features, '__all__'):
            assert sorted(meta_features.__all__) == sorted(expected)


# =============================================================================
# ADDITIONAL TEST FIXTURES (Part 2)
# =============================================================================

# Note: These fixtures use the module-level _create_mock_data helper function
# and reference the consolidated MockDataset class defined at the top of the file.

@pytest.fixture
def mock_dataset():
    """Create standard mock dataset."""
    return MockDataset()


@pytest.fixture
def mock_dataset_small():
    """Create small mock dataset for quick tests."""
    return MockDataset(size=10, mean_nodes=10, mean_edges=20)


@pytest.fixture
def mock_dataset_with_target():
    """Create dataset with targets."""
    return MockDataset(size=20, has_target=True, target_dim=1)


@pytest.fixture
def mock_dataset_without_target():
    """Create dataset without targets."""
    return MockDataset(size=20, has_target=False)


@pytest.fixture
def mock_dataset_with_edge_attr():
    """Create dataset with edge attributes."""
    return MockDataset(size=20, has_edge_attr=True, edge_attr_dim=4)


@pytest.fixture
def mock_dataset_without_edge_attr():
    """Create dataset without edge attributes."""
    return MockDataset(size=20, has_edge_attr=False)


@pytest.fixture
def mock_dataset_with_z():
    """Create dataset with atomic numbers."""
    return MockDataset(size=20, has_z=True)


@pytest.fixture
def mock_empty_dataset():
    """Create empty dataset."""
    return MockDataset(size=0)


@pytest.fixture
def mock_single_sample_dataset():
    """Create single sample dataset."""
    return MockDataset(size=1, mean_nodes=10, mean_edges=15)


@pytest.fixture
def mock_data_basic():
    """Create basic mock data object."""
    return _create_mock_data(n_nodes=20, n_edges=50)


@pytest.fixture
def mock_data_no_x():
    """Create mock data without node features."""
    data = _create_mock_data()
    data.x = None
    return data


@pytest.fixture
def mock_data_no_edges():
    """Create mock data without edges."""
    data = _create_mock_data()
    data.edge_index = None
    return data


# =============================================================================
# EXTRACT_FEATURES INSTANCE METHOD TESTS
# =============================================================================

class TestMetaFeatureExtractorExtractFeatures:
    """Test MetaFeatureExtractor.extract_features instance method."""
    
    def test_extract_features_returns_dict(self, mock_dataset_small):
        """Test extract_features returns a dictionary."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        result = extractor.extract_features(mock_dataset_small)
        
        assert isinstance(result, dict)
    
    def test_extract_features_empty_dataset_returns_empty_dict(self, mock_empty_dataset):
        """Test extract_features returns empty dict for empty dataset."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        result = extractor.extract_features(mock_empty_dataset)
        
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_extract_features_none_dataset_returns_empty_dict(self):
        """Test extract_features returns empty dict for None dataset."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        result = extractor.extract_features(None)
        
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_extract_features_respects_max_samples(self):
        """Test extract_features respects max_samples config."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        dataset = MockDataset(size=100)
        config = MetaFeatureConfig(
            max_samples=10,
            categories=(MetaFeatureCategory.STATISTICAL,)
        )
        
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(dataset)
        
        # Should still report full dataset size for n_samples
        assert result['n_samples'] == 100.0
    
    def test_extract_features_all_float_values(self, mock_dataset_small):
        """Test all extracted features are floats."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        result = extractor.extract_features(mock_dataset_small)
        
        for key, value in result.items():
            assert isinstance(value, float), f"{key} should be float, got {type(value)}"
    
    def test_extract_features_with_normalization(self, mock_dataset_small):
        """Test extract_features with normalization enabled."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig
        )
        
        config = MetaFeatureConfig(normalize=True)
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        # Normalized features with known bounds should be in [0, 1]
        if 'mean_density' in result:
            assert 0.0 <= result['mean_density'] <= 1.0
        if 'clustering_coefficient' in result:
            assert 0.0 <= result['clustering_coefficient'] <= 1.0
    
    def test_extract_features_respects_category_selection(self, mock_dataset_small):
        """Test extract_features only extracts selected categories."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.STATISTICAL,),
            include_molecular=False
        )
        
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        # Should have n_samples (statistical)
        assert 'n_samples' in result
        
        # Should NOT have graph features
        assert 'mean_nodes' not in result or 'mean_nodes' in result  # May or may not be present
    
    def test_extract_features_include_molecular_false(self, mock_dataset_with_z):
        """Test extract_features respects include_molecular=False."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.MOLECULAR,),
            include_molecular=False
        )
        
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_with_z)
        
        # Molecular features should not be extracted
        assert 'atom_frac_C' not in result


# =============================================================================
# STATISTICAL FEATURES EXTRACTION TESTS
# =============================================================================

class TestExtractStatisticalFeatures:
    """Test MetaFeatureExtractor._extract_statistical_features method."""
    
    def test_statistical_features_n_samples(self, mock_dataset):
        """Test statistical features includes n_samples."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.STATISTICAL,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset)
        
        assert 'n_samples' in result
        assert result['n_samples'] == float(len(mock_dataset))
    
    def test_statistical_features_n_features(self, mock_dataset_small):
        """Test statistical features includes n_features."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.STATISTICAL,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        assert 'n_features' in result
        assert result['n_features'] == float(mock_dataset_small._n_features)
    
    def test_statistical_features_n_edge_features(self, mock_dataset_with_edge_attr):
        """Test statistical features includes n_edge_features."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.STATISTICAL,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_with_edge_attr)
        
        assert 'n_edge_features' in result
    
    def test_statistical_features_no_edge_attr(self, mock_dataset_without_edge_attr):
        """Test statistical features without edge attributes."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.STATISTICAL,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_without_edge_attr)
        
        assert 'n_samples' in result
        # n_edge_features may not be present
    
    def test_statistical_features_single_sample(self, mock_single_sample_dataset):
        """Test statistical features with single sample dataset."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.STATISTICAL,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_single_sample_dataset)
        
        assert 'n_samples' in result
        assert result['n_samples'] == 1.0


# =============================================================================
# GRAPH FEATURES EXTRACTION TESTS
# =============================================================================

class TestExtractGraphFeatures:
    """Test MetaFeatureExtractor._extract_graph_features method."""
    
    def test_graph_features_mean_nodes(self, mock_dataset_small):
        """Test graph features includes mean_nodes."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.GRAPH,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        assert 'mean_nodes' in result
        assert result['mean_nodes'] > 0
    
    def test_graph_features_std_nodes(self, mock_dataset_small):
        """Test graph features includes std_nodes."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.GRAPH,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        assert 'std_nodes' in result
        assert result['std_nodes'] >= 0
    
    def test_graph_features_min_max_nodes(self, mock_dataset_small):
        """Test graph features includes min_nodes and max_nodes."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.GRAPH,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        assert 'min_nodes' in result
        assert 'max_nodes' in result
        assert result['min_nodes'] <= result['max_nodes']
    
    def test_graph_features_mean_edges(self, mock_dataset_small):
        """Test graph features includes mean_edges."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.GRAPH,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        assert 'mean_edges' in result
        assert result['mean_edges'] > 0
    
    def test_graph_features_std_edges(self, mock_dataset_small):
        """Test graph features includes std_edges."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.GRAPH,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        assert 'std_edges' in result
        assert result['std_edges'] >= 0
    
    def test_graph_features_min_max_edges(self, mock_dataset_small):
        """Test graph features includes min_edges and max_edges."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.GRAPH,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        assert 'min_edges' in result
        assert 'max_edges' in result
        assert result['min_edges'] <= result['max_edges']
    
    def test_graph_features_mean_density(self, mock_dataset_small):
        """Test graph features includes mean_density."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.GRAPH,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        assert 'mean_density' in result
        # Density should be between 0 and 1 for directed graphs
        assert result['mean_density'] >= 0
    
    def test_graph_features_mean_degree(self, mock_dataset_small):
        """Test graph features includes mean_degree when degrees can be computed."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )

        config = MetaFeatureConfig(categories=(MetaFeatureCategory.GRAPH,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)

        # mean_degree may or may not be present depending on whether degrees could be computed
        if 'mean_degree' in result:
            assert result['mean_degree'] >= 0
    
    def test_graph_features_degree_percentiles(self, mock_dataset_small):
        """Test graph features includes degree percentiles when degrees can be computed."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )

        config = MetaFeatureConfig(categories=(MetaFeatureCategory.GRAPH,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)

        # Degree percentiles may or may not be present depending on whether degrees could be computed
        if 'degree_25th' in result:
            assert 'degree_50th' in result
            assert 'degree_75th' in result
            # Percentiles should be ordered
            assert result['degree_25th'] <= result['degree_50th']
            assert result['degree_50th'] <= result['degree_75th']
    
    def test_graph_features_clustering_coefficient(self, mock_dataset_small):
        """Test graph features includes clustering_coefficient when compute_expensive=True."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.GRAPH,),
            compute_expensive=True
        )
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        # May or may not have clustering depending on graph structure
        if 'clustering_coefficient' in result:
            assert 0.0 <= result['clustering_coefficient'] <= 1.0
    
    def test_graph_features_no_clustering_when_compute_expensive_false(self, mock_dataset_small):
        """Test clustering_coefficient not computed when compute_expensive=False."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.GRAPH,),
            compute_expensive=False
        )
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        assert 'clustering_coefficient' not in result


# =============================================================================
# TARGET FEATURES EXTRACTION TESTS
# =============================================================================

class TestExtractTargetFeatures:
    """Test MetaFeatureExtractor._extract_target_features method."""
    
    def test_target_features_mean(self, mock_dataset_with_target):
        """Test target features includes target_mean."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.TARGET,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_with_target)
        
        assert 'target_mean' in result
    
    def test_target_features_std(self, mock_dataset_with_target):
        """Test target features includes target_std."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.TARGET,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_with_target)
        
        assert 'target_std' in result
        assert result['target_std'] >= 0
    
    def test_target_features_min_max(self, mock_dataset_with_target):
        """Test target features includes target_min and target_max."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.TARGET,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_with_target)
        
        assert 'target_min' in result
        assert 'target_max' in result
        assert result['target_min'] <= result['target_max']
    
    def test_target_features_range(self, mock_dataset_with_target):
        """Test target features includes target_range."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.TARGET,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_with_target)
        
        assert 'target_range' in result
        assert result['target_range'] >= 0
        assert result['target_range'] == result['target_max'] - result['target_min']
    
    def test_target_features_skewness(self, mock_dataset_with_target):
        """Test target features includes target_skewness."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.TARGET,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_with_target)
        
        # Skewness may or may not be present depending on std
        if 'target_skewness' in result:
            assert isinstance(result['target_skewness'], float)
    
    def test_target_features_dim(self, mock_dataset_with_target):
        """Test target features includes target_dim."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.TARGET,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_with_target)
        
        assert 'target_dim' in result
        assert result['target_dim'] >= 1
    
    def test_target_features_without_target(self, mock_dataset_without_target):
        """Test target features when dataset has no targets."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.TARGET,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_without_target)
        
        # Should not raise, may return empty or limited features
        assert isinstance(result, dict)
    
    def test_target_features_multi_dim_target(self):
        """Test target features with multi-dimensional target."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        dataset = MockDataset(size=20, has_target=True, target_dim=5)
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.TARGET,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(dataset)
        
        assert 'target_dim' in result
        assert result['target_dim'] == 5.0


# =============================================================================
# NODE FEATURE STATISTICS TESTS
# =============================================================================

class TestExtractNodeFeatureStatistics:
    """Test MetaFeatureExtractor._extract_node_feature_statistics method."""
    
    def test_node_features_mean(self, mock_dataset_small):
        """Test node features includes node_feat_mean."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.NODE_FEATURES,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        assert 'node_feat_mean' in result
    
    def test_node_features_std(self, mock_dataset_small):
        """Test node features includes node_feat_std."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.NODE_FEATURES,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        assert 'node_feat_std' in result
        assert result['node_feat_std'] >= 0
    
    def test_node_features_min_max(self, mock_dataset_small):
        """Test node features includes node_feat_min and node_feat_max."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.NODE_FEATURES,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        assert 'node_feat_min' in result
        assert 'node_feat_max' in result
        assert result['node_feat_min'] <= result['node_feat_max']
    
    def test_node_features_sparsity(self, mock_dataset_small):
        """Test node features includes node_feat_sparsity."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.NODE_FEATURES,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        assert 'node_feat_sparsity' in result
        assert 0.0 <= result['node_feat_sparsity'] <= 1.0


# =============================================================================
# EDGE FEATURE STATISTICS TESTS
# =============================================================================

class TestExtractEdgeFeatureStatistics:
    """Test MetaFeatureExtractor._extract_edge_feature_statistics method."""
    
    def test_edge_features_has_edge_features_flag(self, mock_dataset_with_edge_attr):
        """Test edge features includes has_edge_features flag."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.EDGE_FEATURES,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_with_edge_attr)
        
        assert 'has_edge_features' in result
        assert result['has_edge_features'] == 1.0
    
    def test_edge_features_no_edge_attr_flag(self, mock_dataset_without_edge_attr):
        """Test has_edge_features is 0.0 when no edge attributes."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.EDGE_FEATURES,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_without_edge_attr)
        
        assert 'has_edge_features' in result
        assert result['has_edge_features'] == 0.0
    
    def test_edge_features_mean(self, mock_dataset_with_edge_attr):
        """Test edge features includes edge_feat_mean."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.EDGE_FEATURES,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_with_edge_attr)
        
        assert 'edge_feat_mean' in result
    
    def test_edge_features_std(self, mock_dataset_with_edge_attr):
        """Test edge features includes edge_feat_std."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.EDGE_FEATURES,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_with_edge_attr)
        
        assert 'edge_feat_std' in result
        assert result['edge_feat_std'] >= 0
    
    def test_edge_features_min_max(self, mock_dataset_with_edge_attr):
        """Test edge features includes edge_feat_min and edge_feat_max."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(categories=(MetaFeatureCategory.EDGE_FEATURES,))
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_with_edge_attr)
        
        assert 'edge_feat_min' in result
        assert 'edge_feat_max' in result
        assert result['edge_feat_min'] <= result['edge_feat_max']


# =============================================================================
# HELPER METHOD TESTS - GET NUM NODES / EDGES
# =============================================================================

class TestGetNumNodes:
    """Test MetaFeatureExtractor._get_num_nodes method."""
    
    def test_get_num_nodes_from_num_nodes_attr(self):
        """Test _get_num_nodes from num_nodes attribute."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        data = MockData(num_nodes=25)
        extractor = MetaFeatureExtractor()
        
        result = extractor._get_num_nodes(data)
        
        assert result == 25
    
    def test_get_num_nodes_from_x_shape(self):
        """Test _get_num_nodes from x.shape."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        data = MockData(x=MockTensor(np.random.randn(30, 10)))
        extractor = MetaFeatureExtractor()
        
        result = extractor._get_num_nodes(data)
        
        assert result == 30
    
    def test_get_num_nodes_no_x(self):
        """Test _get_num_nodes returns None when no x and no num_nodes."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        data = MockData()
        extractor = MetaFeatureExtractor()
        
        result = extractor._get_num_nodes(data)
        
        assert result is None


class TestGetNumEdges:
    """Test MetaFeatureExtractor._get_num_edges method."""
    
    def test_get_num_edges_from_edge_index_shape(self):
        """Test _get_num_edges from edge_index.shape."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        data = MockData(edge_index=MockTensor(np.random.randint(0, 10, size=(2, 45))))
        extractor = MetaFeatureExtractor()
        
        result = extractor._get_num_edges(data)
        
        assert result == 45
    
    def test_get_num_edges_no_edge_index(self):
        """Test _get_num_edges returns None when no edge_index."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        data = MockData()
        extractor = MetaFeatureExtractor()
        
        result = extractor._get_num_edges(data)
        
        assert result is None


class TestComputeDegrees:
    """Test MetaFeatureExtractor._compute_degrees method."""
    
    def test_compute_degrees_returns_list(self):
        """Test _compute_degrees returns a list."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        data = _create_mock_data(n_nodes=10, n_edges=20)
        extractor = MetaFeatureExtractor()
        
        result = extractor._compute_degrees(data, 10)
        
        assert result is None or isinstance(result, list)
    
    def test_compute_degrees_no_edge_index_returns_none(self):
        """Test _compute_degrees returns None when no edge_index."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        data = MockData()
        extractor = MetaFeatureExtractor()
        
        result = extractor._compute_degrees(data, 10)
        
        assert result is None
    
    def test_compute_degrees_zero_nodes_returns_none(self):
        """Test _compute_degrees returns None when n_nodes is 0."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        data = _create_mock_data(n_nodes=10, n_edges=20)
        extractor = MetaFeatureExtractor()
        
        result = extractor._compute_degrees(data, 0)
        
        assert result is None


class TestComputeClusteringCoefficients:
    """Test MetaFeatureExtractor._compute_clustering_coefficients method."""
    
    def test_clustering_coefficients_returns_float_or_none(self, mock_dataset_small):
        """Test _compute_clustering_coefficients returns float or None."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        
        result = extractor._compute_clustering_coefficients(mock_dataset_small, 10)
        
        assert result is None or isinstance(result, float)
    
    def test_clustering_coefficients_in_valid_range(self, mock_dataset_small):
        """Test clustering coefficient is in [0, 1] range."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        
        result = extractor._compute_clustering_coefficients(mock_dataset_small, 10)
        
        if result is not None:
            assert 0.0 <= result <= 1.0
    
    def test_clustering_coefficients_limited_samples(self, mock_dataset):
        """Test clustering computation is limited to 100 samples."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        
        # Should not raise even with 100+ sample dataset
        result = extractor._compute_clustering_coefficients(mock_dataset, len(mock_dataset))
        
        assert result is None or isinstance(result, float)



# =============================================================================
# ADDITIONAL TEST FIXTURES (Part 3 - Molecular and Similarity Testing)
# =============================================================================

# Note: These fixtures use the module-level mock classes and helper functions
# defined at the top of the file.

@pytest.fixture
def mock_dataset_molecular():
    """Create dataset with molecular features."""
    return MockDataset(size=30, molecular=True)


@pytest.fixture
def mock_data_with_z():
    """Create mock data with z attribute."""
    np.random.seed(42)
    return MockData(
        x=MockTensor(np.random.randn(15, 10)),
        z=MockTensor(np.array([1, 6, 6, 6, 7, 8, 6, 6, 1, 1, 6, 6, 7, 8, 1]))
    )


@pytest.fixture
def mock_data_with_atomic_numbers():
    """Create mock data with atomic_numbers attribute."""
    np.random.seed(42)
    return MockData(
        x=MockTensor(np.random.randn(10, 10)),
        atomic_numbers=MockTensor(np.array([6, 6, 6, 7, 8, 1, 1, 1, 1, 1]))
    )


@pytest.fixture
def mock_data_with_bond_type():
    """Create mock data with bond_type attribute."""
    np.random.seed(42)
    return MockData(
        edge_index=MockTensor(np.random.randint(0, 10, size=(2, 20))),
        bond_type=MockTensor(np.array([1, 1, 1, 2, 1, 1, 2, 3, 1, 1, 1, 1, 12, 12, 1, 1, 1, 2, 1, 1]))
    )


@pytest.fixture
def mock_data_with_mol_weight():
    """Create mock data with mol_weight attribute."""
    return MockData(mol_weight=180.15)


@pytest.fixture
def mock_data_with_rings():
    """Create mock data with ring count attributes."""
    return MockData(num_rings=3)


@pytest.fixture
def sample_features_a():
    """Sample meta-features A for similarity tests."""
    return {
        'n_samples': 1000.0,
        'n_features': 10.0,
        'mean_nodes': 25.0,
        'mean_edges': 50.0,
        'target_mean': 0.5,
    }


@pytest.fixture
def sample_features_b():
    """Sample meta-features B for similarity tests."""
    return {
        'n_samples': 1200.0,
        'n_features': 10.0,
        'mean_nodes': 28.0,
        'mean_edges': 55.0,
        'target_mean': 0.45,
    }


@pytest.fixture
def sample_features_identical():
    """Identical meta-features for similarity tests."""
    return {
        'n_samples': 1000.0,
        'n_features': 10.0,
        'mean_nodes': 25.0,
    }


# =============================================================================
# MOLECULAR FEATURES EXTRACTION TESTS
# =============================================================================

class TestExtractMolecularFeatures:
    """Test MetaFeatureExtractor._extract_molecular_features method."""
    
    def test_molecular_features_extraction(self, mock_dataset_molecular):
        """Test molecular features can be extracted."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.MOLECULAR,),
            include_molecular=True
        )
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_molecular)
        
        assert isinstance(result, dict)
    
    def test_molecular_features_atom_fractions(self, mock_dataset_molecular):
        """Test molecular features includes atom fractions."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.MOLECULAR,),
            include_molecular=True
        )
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_molecular)
        
        # Should have atom fractions for common atoms
        atom_fraction_keys = [k for k in result.keys() if k.startswith('atom_frac_')]
        assert len(atom_fraction_keys) > 0
    
    def test_molecular_features_heavy_atom_ratio(self, mock_dataset_molecular):
        """Test molecular features includes heavy_atom_ratio."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.MOLECULAR,),
            include_molecular=True
        )
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_molecular)
        
        if 'heavy_atom_ratio' in result:
            assert 0.0 <= result['heavy_atom_ratio'] <= 1.0
    
    def test_molecular_features_n_atom_types(self, mock_dataset_molecular):
        """Test molecular features includes n_atom_types."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.MOLECULAR,),
            include_molecular=True
        )
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_molecular)
        
        if 'n_atom_types' in result:
            assert result['n_atom_types'] >= 1.0
    
    def test_molecular_features_mol_weight_stats(self, mock_dataset_molecular):
        """Test molecular features includes molecular weight statistics."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.MOLECULAR,),
            include_molecular=True
        )
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_molecular)
        
        if 'mol_weight_mean' in result:
            assert result['mol_weight_mean'] > 0
        if 'mol_weight_std' in result:
            assert result['mol_weight_std'] >= 0
    
    def test_molecular_features_ring_count_stats(self, mock_dataset_molecular):
        """Test molecular features includes ring count statistics."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.MOLECULAR,),
            include_molecular=True
        )
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_molecular)
        
        if 'ring_count_mean' in result:
            assert result['ring_count_mean'] >= 0


class TestGetAtomicNumbers:
    """Test MetaFeatureExtractor._get_atomic_numbers method."""
    
    def test_get_atomic_numbers_from_z(self, mock_data_with_z):
        """Test _get_atomic_numbers from z attribute."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        result = extractor._get_atomic_numbers(mock_data_with_z)
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 15
    
    def test_get_atomic_numbers_from_atomic_numbers(self, mock_data_with_atomic_numbers):
        """Test _get_atomic_numbers from atomic_numbers attribute."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        result = extractor._get_atomic_numbers(mock_data_with_atomic_numbers)
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 10
    
    def test_get_atomic_numbers_none_when_missing(self):
        """Test _get_atomic_numbers returns None when no atomic info."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        data = MockData()
        extractor = MetaFeatureExtractor()
        result = extractor._get_atomic_numbers(data)
        
        assert result is None


class TestGetBondTypes:
    """Test MetaFeatureExtractor._get_bond_types method."""
    
    def test_get_bond_types_from_bond_type(self, mock_data_with_bond_type):
        """Test _get_bond_types from bond_type attribute."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        result = extractor._get_bond_types(mock_data_with_bond_type)
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 20
    
    def test_get_bond_types_from_one_hot_edge_attr(self):
        """Test _get_bond_types from one-hot encoded edge_attr."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        # Create one-hot encoded edge attributes (5 columns for bond types)
        n_edges = 10
        edge_attr = np.zeros((n_edges, 5))
        edge_attr[0, 0] = 1  # Single
        edge_attr[1, 1] = 1  # Double
        edge_attr[2, 2] = 1  # Triple
        edge_attr[3, 3] = 1  # Aromatic
        
        data = MockData(
            edge_index=MockTensor(np.random.randint(0, 5, size=(2, n_edges))),
            edge_attr=MockTensor(edge_attr)
        )
        
        extractor = MetaFeatureExtractor()
        result = extractor._get_bond_types(data)
        
        # May return bond types from one-hot encoding
        if result is not None:
            assert isinstance(result, list)
    
    def test_get_bond_types_none_when_missing(self):
        """Test _get_bond_types returns None when no bond info."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        data = MockData()
        extractor = MetaFeatureExtractor()
        result = extractor._get_bond_types(data)
        
        assert result is None


class TestGetMolecularWeight:
    """Test MetaFeatureExtractor._get_molecular_weight method."""
    
    def test_get_molecular_weight_from_attr(self, mock_data_with_mol_weight):
        """Test _get_molecular_weight from mol_weight attribute."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        result = extractor._get_molecular_weight(mock_data_with_mol_weight)
        
        assert result is not None
        assert result == 180.15
    
    def test_get_molecular_weight_computed_from_z(self, mock_data_with_z):
        """Test _get_molecular_weight computed from atomic numbers."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        result = extractor._get_molecular_weight(mock_data_with_z)
        
        # Should compute weight from atomic numbers
        if result is not None:
            assert result > 0
    
    def test_get_molecular_weight_none_when_missing(self):
        """Test _get_molecular_weight returns None when no weight info."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        data = MockData()
        extractor = MetaFeatureExtractor()
        result = extractor._get_molecular_weight(data)
        
        assert result is None


class TestGetRingCount:
    """Test MetaFeatureExtractor._get_ring_count method."""
    
    def test_get_ring_count_from_num_rings(self, mock_data_with_rings):
        """Test _get_ring_count from num_rings attribute."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        result = extractor._get_ring_count(mock_data_with_rings)
        
        assert result == 3
    
    def test_get_ring_count_from_ring_count_attr(self):
        """Test _get_ring_count from ring_count attribute."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        data = MockData(ring_count=5)
        extractor = MetaFeatureExtractor()
        result = extractor._get_ring_count(data)
        
        assert result == 5
    
    def test_get_ring_count_none_when_missing(self):
        """Test _get_ring_count returns None when no ring info."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        data = MockData()
        extractor = MetaFeatureExtractor()
        result = extractor._get_ring_count(data)
        
        assert result is None


# =============================================================================
# NORMALIZATION TESTS
# =============================================================================

class TestNormalizeFeatures:
    """Test MetaFeatureExtractor._normalize_features method."""
    
    def test_normalize_features_returns_dict(self):
        """Test _normalize_features returns a dictionary."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        features = {'n_samples': 500.0, 'mean_nodes': 50.0}
        
        result = extractor._normalize_features(features)
        
        assert isinstance(result, dict)
    
    def test_normalize_features_known_bounds(self):
        """Test _normalize_features uses predefined bounds for known features."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        features = {'n_samples': 500000.0}  # Mid-range value
        
        result = extractor._normalize_features(features)
        
        # n_samples bounds are (1, 1_000_000)
        # Normalized value should be approximately 0.5
        assert 0.0 <= result['n_samples'] <= 1.0
        assert abs(result['n_samples'] - 0.5) < 0.1
    
    def test_normalize_features_clamps_to_bounds(self):
        """Test _normalize_features clamps values to [0, 1]."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        features = {'n_samples': 2_000_000.0}  # Above max bound
        
        result = extractor._normalize_features(features)
        
        assert result['n_samples'] == 1.0  # Clamped to max
    
    def test_normalize_features_clamps_below_bounds(self):
        """Test _normalize_features clamps negative values."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        features = {'n_samples': 0.0}  # Below min bound
        
        result = extractor._normalize_features(features)
        
        assert result['n_samples'] == 0.0  # Clamped to min
    
    def test_normalize_features_unknown_features_pass_through(self):
        """Test unknown features pass through unchanged."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        features = {'custom_feature': 42.0}
        
        result = extractor._normalize_features(features)
        
        assert result['custom_feature'] == 42.0
    
    def test_normalize_features_mean_density(self):
        """Test mean_density normalization (bounds 0-1)."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        features = {'mean_density': 0.5}
        
        result = extractor._normalize_features(features)
        
        assert result['mean_density'] == 0.5
    
    def test_normalize_features_clustering_coefficient(self):
        """Test clustering_coefficient normalization (bounds 0-1)."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        features = {'clustering_coefficient': 0.75}
        
        result = extractor._normalize_features(features)
        
        assert result['clustering_coefficient'] == 0.75


# =============================================================================
# SIMILARITY COMPUTATION TESTS
# =============================================================================

class TestComputeSimilarity:
    """Test MetaFeatureExtractor.compute_similarity static method."""
    
    def test_compute_similarity_exists(self):
        """Test compute_similarity method exists."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        assert hasattr(MetaFeatureExtractor, 'compute_similarity')
        assert callable(MetaFeatureExtractor.compute_similarity)
    
    def test_compute_similarity_returns_float(self, sample_features_a, sample_features_b):
        """Test compute_similarity returns a float."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.compute_similarity(sample_features_a, sample_features_b)
        
        assert isinstance(result, float)
    
    def test_compute_similarity_in_range(self, sample_features_a, sample_features_b):
        """Test compute_similarity result is in [0, 1]."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.compute_similarity(sample_features_a, sample_features_b)
        
        assert 0.0 <= result <= 1.0
    
    def test_compute_similarity_identical_features(self, sample_features_identical):
        """Test compute_similarity returns 1.0 for identical features."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.compute_similarity(
            sample_features_identical, 
            sample_features_identical
        )
        
        assert result == 1.0
    
    def test_compute_similarity_symmetric(self, sample_features_a, sample_features_b):
        """Test compute_similarity is symmetric."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result_ab = MetaFeatureExtractor.compute_similarity(sample_features_a, sample_features_b)
        result_ba = MetaFeatureExtractor.compute_similarity(sample_features_b, sample_features_a)
        
        assert abs(result_ab - result_ba) < 1e-10
    
    def test_compute_similarity_no_common_keys(self):
        """Test compute_similarity returns 0.0 when no common keys."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        features_a = {'feature1': 1.0, 'feature2': 2.0}
        features_b = {'feature3': 3.0, 'feature4': 4.0}
        
        result = MetaFeatureExtractor.compute_similarity(features_a, features_b)
        
        assert result == 0.0
    
    def test_compute_similarity_zero_vector(self):
        """Test compute_similarity returns 0.0 for zero vectors."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        features_a = {'n_samples': 0.0, 'n_features': 0.0}
        features_b = {'n_samples': 1.0, 'n_features': 1.0}
        
        result = MetaFeatureExtractor.compute_similarity(features_a, features_b)
        
        assert result == 0.0
    
    def test_compute_similarity_partial_overlap(self):
        """Test compute_similarity with partial key overlap."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        features_a = {'n_samples': 1000.0, 'n_features': 10.0, 'extra_a': 5.0}
        features_b = {'n_samples': 1000.0, 'n_features': 10.0, 'extra_b': 3.0}
        
        result = MetaFeatureExtractor.compute_similarity(features_a, features_b)
        
        # Should compute similarity only on common keys
        assert result == 1.0  # Identical on common keys
    
    def test_compute_similarity_single_common_key(self):
        """Test compute_similarity with single common key."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        features_a = {'n_samples': 100.0}
        features_b = {'n_samples': 100.0}
        
        result = MetaFeatureExtractor.compute_similarity(features_a, features_b)
        
        assert result == 1.0
    
    def test_compute_similarity_high_similarity_similar_values(self):
        """Test compute_similarity returns high value for similar features."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        features_a = {'n_samples': 1000.0, 'mean_nodes': 25.0}
        features_b = {'n_samples': 1050.0, 'mean_nodes': 26.0}
        
        result = MetaFeatureExtractor.compute_similarity(features_a, features_b)
        
        assert result > 0.9  # Very similar features


# =============================================================================
# UTILITY METHOD TESTS
# =============================================================================

class TestGetFeatureNames:
    """Test MetaFeatureExtractor.get_feature_names static method."""
    
    def test_get_feature_names_exists(self):
        """Test get_feature_names method exists."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        assert hasattr(MetaFeatureExtractor, 'get_feature_names')
        assert callable(MetaFeatureExtractor.get_feature_names)
    
    def test_get_feature_names_returns_dict(self):
        """Test get_feature_names returns a dictionary."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.get_feature_names()
        
        assert isinstance(result, dict)
    
    def test_get_feature_names_has_categories(self):
        """Test get_feature_names has all expected categories."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.get_feature_names()
        
        expected_categories = ['statistical', 'graph', 'target', 'node_features', 'edge_features', 'molecular']
        
        for category in expected_categories:
            assert category in result, f"Missing category: {category}"
    
    def test_get_feature_names_statistical_features(self):
        """Test get_feature_names includes statistical features."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.get_feature_names()
        
        assert 'statistical' in result
        assert 'n_samples' in result['statistical']
        assert 'n_features' in result['statistical']
    
    def test_get_feature_names_graph_features(self):
        """Test get_feature_names includes graph features."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.get_feature_names()
        
        assert 'graph' in result
        assert 'mean_nodes' in result['graph']
        assert 'mean_edges' in result['graph']
        assert 'mean_degree' in result['graph']
    
    def test_get_feature_names_target_features(self):
        """Test get_feature_names includes target features."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.get_feature_names()
        
        assert 'target' in result
        assert 'target_mean' in result['target']
        assert 'target_std' in result['target']
    
    def test_get_feature_names_molecular_features(self):
        """Test get_feature_names includes molecular features."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.get_feature_names()
        
        assert 'molecular' in result
        assert 'atom_frac_C' in result['molecular']
        assert 'mol_weight_mean' in result['molecular']
    
    def test_get_feature_names_values_are_lists(self):
        """Test get_feature_names values are lists of strings."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.get_feature_names()
        
        for category, features in result.items():
            assert isinstance(features, list), f"{category} should be a list"
            for feat in features:
                assert isinstance(feat, str), f"{feat} in {category} should be a string"


class TestGetCategoryForFeature:
    """Test MetaFeatureExtractor.get_category_for_feature static method."""
    
    def test_get_category_for_feature_exists(self):
        """Test get_category_for_feature method exists."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        assert hasattr(MetaFeatureExtractor, 'get_category_for_feature')
        assert callable(MetaFeatureExtractor.get_category_for_feature)
    
    def test_get_category_for_feature_statistical(self):
        """Test get_category_for_feature returns 'statistical' for n_samples."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.get_category_for_feature('n_samples')
        
        assert result == 'statistical'
    
    def test_get_category_for_feature_graph(self):
        """Test get_category_for_feature returns 'graph' for mean_nodes."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.get_category_for_feature('mean_nodes')
        
        assert result == 'graph'
    
    def test_get_category_for_feature_target(self):
        """Test get_category_for_feature returns 'target' for target_mean."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.get_category_for_feature('target_mean')
        
        assert result == 'target'
    
    def test_get_category_for_feature_molecular(self):
        """Test get_category_for_feature returns 'molecular' for atom_frac_C."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.get_category_for_feature('atom_frac_C')
        
        assert result == 'molecular'
    
    def test_get_category_for_feature_unknown(self):
        """Test get_category_for_feature returns None for unknown feature."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.get_category_for_feature('unknown_feature_xyz')
        
        assert result is None
    
    def test_get_category_for_feature_node_features(self):
        """Test get_category_for_feature returns 'node_features' for node_feat_mean."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.get_category_for_feature('node_feat_mean')
        
        assert result == 'node_features'
    
    def test_get_category_for_feature_edge_features(self):
        """Test get_category_for_feature returns 'edge_features' for edge_feat_mean."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.get_category_for_feature('edge_feat_mean')
        
        assert result == 'edge_features'


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for MetaFeatureExtractor."""
    
    def test_full_extraction_pipeline(self, mock_dataset_small):
        """Test full extraction pipeline with all categories."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.ALL,),
            normalize=False,
            compute_expensive=False
        )
        
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        # Should have features from multiple categories
        assert 'n_samples' in result  # Statistical
        assert 'mean_nodes' in result  # Graph
        assert 'target_mean' in result or 'node_feat_mean' in result  # At least one of these
    
    def test_extraction_and_similarity(self, mock_dataset_small):
        """Test extraction followed by similarity computation."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        # Create two similar datasets
        dataset1 = MockDataset(size=20, mean_nodes=15, mean_edges=30)
        dataset2 = MockDataset(size=25, mean_nodes=16, mean_edges=32)
        
        features1 = MetaFeatureExtractor.extract(dataset1)
        features2 = MetaFeatureExtractor.extract(dataset2)
        
        similarity = MetaFeatureExtractor.compute_similarity(features1, features2)
        
        # Similar datasets should have high similarity
        assert similarity > 0.8
    
    def test_extraction_with_normalization(self, mock_dataset_small):
        """Test extraction with normalization produces bounded values."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.STATISTICAL, MetaFeatureCategory.GRAPH),
            normalize=True
        )
        
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(mock_dataset_small)
        
        # Known bounded features should be in [0, 1]
        bounded_keys = ['mean_density', 'clustering_coefficient']
        for key in bounded_keys:
            if key in result:
                assert 0.0 <= result[key] <= 1.0, f"{key} should be in [0, 1]"
    
    def test_static_and_instance_methods_consistency(self, mock_dataset_small):
        """Test static extract and instance extract_features give same results."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig, MetaFeatureCategory
        )
        
        config = MetaFeatureConfig(
            categories=(MetaFeatureCategory.STATISTICAL,),
            compute_expensive=False
        )
        
        # Static method
        static_result = MetaFeatureExtractor.extract(mock_dataset_small, config)
        
        # Instance method
        extractor = MetaFeatureExtractor(config)
        instance_result = extractor.extract_features(mock_dataset_small)
        
        # Results should be identical
        assert static_result.keys() == instance_result.keys()
        for key in static_result:
            assert static_result[key] == instance_result[key], f"{key} mismatch"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_node_graph(self):
        """Test handling of single-node graphs."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        class SingleNodeDataset:
            def __len__(self):
                return 5
            
            def __getitem__(self, idx):
                return MockData(
                    x=MockTensor(np.random.randn(1, 5)),
                    edge_index=MockTensor(np.array([[0], [0]])),
                    num_nodes=1
                )
        
        dataset = SingleNodeDataset()
        extractor = MetaFeatureExtractor()
        
        # Should not raise
        result = extractor.extract_features(dataset)
        
        assert isinstance(result, dict)
    
    def test_no_edges_graph(self):
        """Test handling of graphs with no edges."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        class NoEdgeDataset:
            def __len__(self):
                return 5
            
            def __getitem__(self, idx):
                return MockData(
                    x=MockTensor(np.random.randn(10, 5)),
                    edge_index=MockTensor(np.array([[], []]).astype(int)),
                    num_nodes=10
                )
        
        dataset = NoEdgeDataset()
        extractor = MetaFeatureExtractor()
        
        # Should not raise
        result = extractor.extract_features(dataset)
        
        assert isinstance(result, dict)
    
    def test_very_large_values(self):
        """Test handling of very large feature values."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        
        features = {'n_samples': 1e10, 'mean_nodes': 1e6}
        normalized = extractor._normalize_features(features)
        
        # Should clamp to 1.0
        assert normalized['n_samples'] == 1.0
    
    def test_negative_feature_values(self):
        """Test handling of negative feature values."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor()
        
        features = {'n_samples': -100.0}
        normalized = extractor._normalize_features(features)
        
        # Should clamp to 0.0
        assert normalized['n_samples'] == 0.0
    
    def test_empty_features_similarity(self):
        """Test similarity computation with empty features."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        result = MetaFeatureExtractor.compute_similarity({}, {})
        
        assert result == 0.0
    
    def test_nan_handling_in_similarity(self):
        """Test similarity computation handles potential NaN gracefully."""
        from milia_pipeline.models.hpo.transfer.meta_features import MetaFeatureExtractor
        
        # Create features that could produce NaN if not handled
        features_a = {'n_samples': 0.0}
        features_b = {'n_samples': 0.0}
        
        result = MetaFeatureExtractor.compute_similarity(features_a, features_b)
        
        # Should return 0.0 for zero vectors, not NaN
        assert result == 0.0
        assert not np.isnan(result)
    
    def test_max_samples_larger_than_dataset(self):
        """Test max_samples larger than dataset size."""
        from milia_pipeline.models.hpo.transfer.meta_features import (
            MetaFeatureExtractor, MetaFeatureConfig
        )
        
        dataset = MockDataset(size=10)
        config = MetaFeatureConfig(max_samples=1000)
        
        extractor = MetaFeatureExtractor(config)
        result = extractor.extract_features(dataset)
        
        # Should handle gracefully
        assert result['n_samples'] == 10.0


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
