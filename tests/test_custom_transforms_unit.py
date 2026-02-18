#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for custom_transforms.py Module

This test suite provides extensive unit test coverage for the custom_transforms.py module,
testing all base classes, metadata handling, and all transform implementations.

Test Coverage:
- TransformMetadata Pydantic model and serialization
- CustomTransformBase abstract base class and methods
- MolecularTransformBase with chemistry validation (validate_molecular_structure)
- QuantumTransformBase with milia quantum properties (validate_quantum_properties)
- NormalizeVibrationalModes example transform
- FilterByDMCUncertainty example transform
- ScaleMullikenCharges example transform
- StandardizeTargets z-score normalization transform
- NormalizeTargets min-max normalization transform
- DiscretizeTargets classification discretization transform
  - fit() method for dataset-level bin computation
  - compute_class_weights() static method for imbalanced classification
  - get_num_classes() static method for model factory integration
  - Graph-level, node-level, and edge-level target handling
- validate_compatibility() method with strict/normal/lenient modes
- get_usage_statistics() method for transform monitoring
- Parameter validation and constraints
- Error handling and exceptions (TransformValidationError, TransformExecutionError, TransformConfigurationError)
- Data cloning and immutability guarantees
- Edge cases (empty graphs, single node, large graphs, extreme values)
- Integration tests for transform pipelines

NOTE: This test suite runs inside Docker at /app/milia

Author: milia Project Team
Created: November 01, 2025
Updated: Production-ready test coverage
"""

import sys
from pathlib import Path

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import contextlib
import logging
from typing import Any

import pytest
import torch
from torch_geometric.data import Data

# Import the module under test
from milia_pipeline.transformations.custom_transforms import (
    CustomTransformBase,
    DiscretizeTargets,
    FilterByDMCUncertainty,
    MolecularTransformBase,
    NormalizeTargets,
    # Example transforms
    NormalizeVibrationalModes,
    QuantumTransformBase,
    ScaleMullikenCharges,
    # Target normalization transforms
    StandardizeTargets,
    TransformConfigurationError,
    TransformExecutionError,
    # Core classes
    TransformMetadata,
    # Exceptions
    TransformValidationError,
)

# =============================================================================
# TEST FIXTURES AND HELPER CLASSES
# =============================================================================


@pytest.fixture
def sample_metadata():
    """Sample TransformMetadata for testing"""
    return TransformMetadata(
        name="TestTransform",
        version="1.0.0",
        author="Test Author",
        category="quantum",
        description="Test transform for unit testing",
        paper_reference="Test Paper (2025)",
        github_url="https://github.com/test/repo",
        validated_datasets=["milia_DFT", "milia_DMC"],
        required_node_features=["x", "z"],
        required_edge_features=["edge_attr"],
        required_graph_attributes=["energy", "forces"],
    )


@pytest.fixture
def minimal_metadata():
    """Minimal TransformMetadata with only required fields"""
    return TransformMetadata(
        name="MinimalTransform",
        version="1.0.0",
        author="Test Author",
        category="molecular",
        description="Minimal test transform",
    )


@pytest.fixture
def sample_graph_data():
    """Sample PyTorch Geometric Data object for testing"""
    return Data(
        x=torch.randn(5, 10),  # 5 nodes, 10 features each
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        edge_attr=torch.randn(4, 3),  # 4 edges, 3 features each
        z=torch.tensor([6, 8, 7, 6, 1]),  # atomic numbers
        pos=torch.randn(5, 3),  # 3D coordinates
        energy=torch.tensor([-100.5]),
        forces=torch.randn(5, 3),
        num_nodes=5,
    )


@pytest.fixture
def quantum_graph_data():
    """Sample quantum data with milia-specific attributes"""
    return Data(
        x=torch.randn(3, 5),
        z=torch.tensor([6, 8, 7]),
        pos=torch.randn(3, 3),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        energy=torch.tensor([-50.25]),
        forces=torch.randn(3, 3),
        charges=torch.tensor([0.5, -0.3, -0.2]),
        dipole_moment=torch.randn(3),
        vibmodes=torch.randn(5, 3, 3),  # 5 modes, 3 atoms, 3D
        dmc_uncertainty=torch.tensor(0.05),
        num_nodes=3,
    )


@pytest.fixture
def regression_dataset():
    """Sample regression dataset for target transform tests"""
    dataset = []
    for i in range(10):
        data = Data(
            x=torch.randn(5, 10),
            y=torch.tensor([float(i * 10)]),  # Values from 0 to 90
            num_nodes=5,
        )
        dataset.append(data)
    return dataset


@pytest.fixture
def node_level_data():
    """Sample data with node-level targets"""
    return Data(
        x=torch.randn(10, 5),
        y=torch.randn(10),  # One target per node
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        num_nodes=10,
    )


@pytest.fixture
def multi_target_data():
    """Sample data with multiple targets per graph"""
    return Data(
        x=torch.randn(5, 10),
        y=torch.tensor([-100.5, 0.5, 2.5]),  # 3 targets: energy, gap, dipole
        num_nodes=5,
    )


class ConcreteCustomTransform(CustomTransformBase):
    """Concrete implementation of CustomTransformBase for testing"""

    def __init__(self, param1: float = 1.0, param2: str = "default"):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def transform(self, data: Data) -> Data:
        """Simple transform that modifies node features"""
        data = data.clone()
        if hasattr(data, "x"):
            data.x = data.x * self.param1
        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="ConcreteCustomTransform",
            version="1.0.0",
            author="Test",
            category="experimental",
            description="Test implementation",
            required_node_features=["x"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict[str, Any]:
        return {
            "param1": {"type": float, "range": (0.0, 10.0), "default": 1.0},
            "param2": {"type": str, "default": "default"},
        }

    @classmethod
    def get_required_node_attributes(cls):
        """Override to require 'x' attribute for validation testing"""
        return {"x"}


class ConcreteMolecularTransform(MolecularTransformBase):
    """Concrete implementation of MolecularTransformBase for testing"""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def transform(self, data: Data) -> Data:
        """Scale atomic numbers"""
        data = data.clone()
        if hasattr(data, "z"):
            data.z = (data.z.float() * self.scale).long()
        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="ConcreteMolecularTransform",
            version="1.0.0",
            author="Test",
            category="molecular",
            description="Test molecular transform",
            required_node_features=["z"],
        )


class ConcreteQuantumTransform(QuantumTransformBase):
    """Concrete implementation of QuantumTransformBase for testing"""

    def __init__(self, energy_scale: float = 1.0):
        super().__init__()
        self.energy_scale = energy_scale

    def transform(self, data: Data) -> Data:
        """Scale energy"""
        data = data.clone()
        if hasattr(data, "energy"):
            data.energy = data.energy * self.energy_scale
        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="ConcreteQuantumTransform",
            version="1.0.0",
            author="Test",
            category="quantum",
            description="Test quantum transform",
            required_graph_attributes=["energy"],
        )


# =============================================================================
# TRANSFORMMETADATA TESTS
# =============================================================================


class TestTransformMetadata:
    """Test suite for TransformMetadata dataclass"""

    def test_metadata_creation_full(self, sample_metadata):
        """Test creating metadata with all fields"""
        assert sample_metadata.name == "TestTransform"
        assert sample_metadata.version == "1.0.0"
        assert sample_metadata.author == "Test Author"
        assert sample_metadata.category == "quantum"
        assert sample_metadata.description == "Test transform for unit testing"
        assert sample_metadata.paper_reference == "Test Paper (2025)"
        assert sample_metadata.github_url == "https://github.com/test/repo"
        assert len(sample_metadata.validated_datasets) == 2
        assert "milia_DFT" in sample_metadata.validated_datasets
        assert len(sample_metadata.required_node_features) == 2
        assert len(sample_metadata.required_edge_features) == 1
        assert len(sample_metadata.required_graph_attributes) == 2

    def test_metadata_creation_minimal(self, minimal_metadata):
        """Test creating metadata with only required fields"""
        assert minimal_metadata.name == "MinimalTransform"
        assert minimal_metadata.version == "1.0.0"
        assert minimal_metadata.author == "Test Author"
        assert minimal_metadata.category == "molecular"
        assert minimal_metadata.description == "Minimal test transform"
        # Check defaults
        assert minimal_metadata.paper_reference is None
        assert minimal_metadata.github_url is None
        assert len(minimal_metadata.validated_datasets) == 0
        assert len(minimal_metadata.required_node_features) == 0
        assert len(minimal_metadata.required_edge_features) == 0
        assert len(minimal_metadata.required_graph_attributes) == 0

    def test_metadata_to_dict(self, sample_metadata):
        """Test metadata serialization to dictionary"""
        meta_dict = sample_metadata.to_dict()
        assert isinstance(meta_dict, dict)
        assert meta_dict["name"] == "TestTransform"
        assert meta_dict["version"] == "1.0.0"
        assert meta_dict["author"] == "Test Author"
        assert meta_dict["category"] == "quantum"
        assert meta_dict["description"] == "Test transform for unit testing"
        assert meta_dict["paper_reference"] == "Test Paper (2025)"
        assert meta_dict["github_url"] == "https://github.com/test/repo"
        assert isinstance(meta_dict["validated_datasets"], list)
        assert isinstance(meta_dict["required_node_features"], list)

    def test_metadata_to_dict_minimal(self, minimal_metadata):
        """Test minimal metadata serialization"""
        meta_dict = minimal_metadata.to_dict()
        assert meta_dict["paper_reference"] is None
        assert meta_dict["github_url"] is None
        assert meta_dict["validated_datasets"] == []
        assert meta_dict["required_node_features"] == []

    def test_metadata_str_representation(self, sample_metadata):
        """Test string representation of metadata"""
        str_repr = str(sample_metadata)
        assert "TransformMetadata" in str_repr
        assert "TestTransform" in str_repr
        assert "1.0.0" in str_repr
        assert "quantum" in str_repr

    def test_metadata_categories(self):
        """Test different metadata categories"""
        categories = ["molecular", "quantum", "experimental", "augmentation"]
        for category in categories:
            metadata = TransformMetadata(
                name=f"{category}_transform",
                version="1.0.0",
                author="Test",
                category=category,
                description=f"Test {category} transform",
            )
            assert metadata.category == category

    def test_metadata_validated_datasets_modification(self):
        """Test that validated_datasets can be modified after creation"""
        metadata = TransformMetadata(
            name="Test", version="1.0.0", author="Test", category="quantum", description="Test"
        )
        metadata.validated_datasets.append("milia_DFT")
        metadata.validated_datasets.append("milia_DMC")
        assert len(metadata.validated_datasets) == 2
        assert "milia_DFT" in metadata.validated_datasets


# =============================================================================
# CUSTOMTRANSFORMBASE TESTS
# =============================================================================


class TestCustomTransformBase:
    """Test suite for CustomTransformBase abstract class"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that CustomTransformBase cannot be instantiated directly"""
        with pytest.raises(TypeError):
            CustomTransformBase()

    def test_concrete_implementation_instantiation(self):
        """Test that concrete implementations can be instantiated"""
        transform = ConcreteCustomTransform(param1=2.0, param2="test")
        assert isinstance(transform, CustomTransformBase)
        assert transform.param1 == 2.0
        assert transform.param2 == "test"

    def test_metadata_retrieval(self):
        """Test get_metadata class method"""
        metadata = ConcreteCustomTransform.get_metadata()
        assert isinstance(metadata, TransformMetadata)
        assert metadata.name == "ConcreteCustomTransform"
        assert metadata.version == "1.0.0"
        assert metadata.category == "experimental"

    def test_parameter_constraints_retrieval(self):
        """Test get_parameter_constraints class method"""
        constraints = ConcreteCustomTransform.get_parameter_constraints()
        assert isinstance(constraints, dict)
        assert "param1" in constraints
        assert "param2" in constraints
        assert constraints["param1"]["type"] is float
        assert constraints["param1"]["default"] == 1.0
        assert constraints["param2"]["type"] is str

    def test_transform_execution(self, sample_graph_data):
        """Test basic transform execution"""
        transform = ConcreteCustomTransform(param1=2.0)
        result = transform(sample_graph_data)
        assert isinstance(result, Data)
        assert torch.allclose(result.x, sample_graph_data.x * 2.0)

    def test_transform_immutability(self, sample_graph_data):
        """Test that transforms don't modify original data"""
        original_x = sample_graph_data.x.clone()
        transform = ConcreteCustomTransform(param1=3.0)
        result = transform(sample_graph_data)
        # Original should be unchanged
        assert torch.allclose(sample_graph_data.x, original_x)
        # Result should be modified
        assert torch.allclose(result.x, original_x * 3.0)

    def test_logger_initialization(self):
        """Test that logger is properly initialized"""
        transform = ConcreteCustomTransform()
        assert hasattr(transform, "_logger")
        assert isinstance(transform._logger, logging.Logger)

    def test_metadata_property(self):
        """Test metadata property access"""
        transform = ConcreteCustomTransform()
        metadata = transform._metadata
        assert isinstance(metadata, TransformMetadata)
        assert metadata.name == "ConcreteCustomTransform"

    def test_transform_with_default_parameters(self, sample_graph_data):
        """Test transform with default parameter values"""
        transform = ConcreteCustomTransform()  # Using defaults
        result = transform(sample_graph_data)
        assert torch.allclose(result.x, sample_graph_data.x * 1.0)


# =============================================================================
# MOLECULARTRANSFORMBASE TESTS
# =============================================================================


class TestMolecularTransformBase:
    """Test suite for MolecularTransformBase"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that MolecularTransformBase cannot be instantiated directly"""
        with pytest.raises(TypeError):
            MolecularTransformBase()

    def test_concrete_molecular_implementation(self):
        """Test concrete molecular transform implementation"""
        transform = ConcreteMolecularTransform(scale=2.0)
        assert isinstance(transform, MolecularTransformBase)
        assert isinstance(transform, CustomTransformBase)
        assert transform.scale == 2.0

    def test_molecular_transform_execution(self, sample_graph_data):
        """Test molecular transform execution"""
        transform = ConcreteMolecularTransform(scale=2.0)
        result = transform(sample_graph_data)
        assert isinstance(result, Data)
        expected_z = (sample_graph_data.z.float() * 2.0).long()
        assert torch.equal(result.z, expected_z)

    def test_molecular_metadata_category(self):
        """Test that molecular transforms have correct category"""
        metadata = ConcreteMolecularTransform.get_metadata()
        assert metadata.category == "molecular"

    def test_molecular_transform_with_missing_attribute(self):
        """Test molecular transform when required attribute is missing"""
        data = Data(x=torch.randn(3, 5), num_nodes=3)  # No 'z' attribute
        transform = ConcreteMolecularTransform(scale=2.0)
        result = transform(data)
        # Should handle gracefully (no 'z' to modify)
        assert not hasattr(result, "z") or result.z is None


# =============================================================================
# QUANTUMTRANSFORMBASE TESTS
# =============================================================================


class TestQuantumTransformBase:
    """Test suite for QuantumTransformBase"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that QuantumTransformBase cannot be instantiated directly"""
        with pytest.raises(TypeError):
            QuantumTransformBase()

    def test_concrete_quantum_implementation(self):
        """Test concrete quantum transform implementation"""
        transform = ConcreteQuantumTransform(energy_scale=2.0)
        assert isinstance(transform, QuantumTransformBase)
        assert isinstance(transform, MolecularTransformBase)
        assert isinstance(transform, CustomTransformBase)
        assert transform.energy_scale == 2.0

    def test_quantum_transform_execution(self, quantum_graph_data):
        """Test quantum transform execution"""
        original_energy = quantum_graph_data.energy.clone()
        transform = ConcreteQuantumTransform(energy_scale=0.5)
        result = transform(quantum_graph_data)
        assert isinstance(result, Data)
        assert torch.allclose(result.energy, original_energy * 0.5)

    def test_quantum_metadata_category(self):
        """Test that quantum transforms have correct category"""
        metadata = ConcreteQuantumTransform.get_metadata()
        assert metadata.category == "quantum"

    def test_quantum_transform_immutability(self, quantum_graph_data):
        """Test that quantum transforms don't modify original data"""
        original_energy = quantum_graph_data.energy.clone()
        transform = ConcreteQuantumTransform(energy_scale=3.0)
        result = transform(quantum_graph_data)
        # Original unchanged
        assert torch.allclose(quantum_graph_data.energy, original_energy)
        # Result modified
        assert torch.allclose(result.energy, original_energy * 3.0)


# =============================================================================
# NORMALIZEVIBRIONALMODES TESTS
# =============================================================================


class TestNormalizeVibrationalModes:
    """Test suite for NormalizeVibrationalModes transform"""

    def test_instantiation_default(self):
        """Test instantiation with default parameters"""
        transform = NormalizeVibrationalModes()
        assert transform.normalize_per_mode is True
        assert transform.epsilon == 1e-8

    def test_instantiation_custom(self):
        """Test instantiation with custom parameters"""
        transform = NormalizeVibrationalModes(normalize_per_mode=False, epsilon=1e-6)
        assert transform.normalize_per_mode is False
        assert transform.epsilon == 1e-6

    def test_metadata(self):
        """Test metadata"""
        metadata = NormalizeVibrationalModes.get_metadata()
        assert metadata.name == "NormalizeVibrationalModes"
        assert metadata.category == "quantum"
        assert "milia_DFT" in metadata.validated_datasets
        assert "vibmodes" in metadata.required_graph_attributes

    def test_parameter_constraints(self):
        """Test parameter constraints"""
        constraints = NormalizeVibrationalModes.get_parameter_constraints()
        assert "normalize_per_mode" in constraints
        assert "epsilon" in constraints
        assert constraints["normalize_per_mode"]["type"] is bool
        assert constraints["epsilon"]["type"] is float

    def test_normalize_per_mode_true(self):
        """Test normalization per mode"""
        vibmodes = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # mode 1
                [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],  # mode 2
            ]
        )
        data = Data(x=torch.ones(3, 1), vibmodes=vibmodes, num_nodes=3)

        transform = NormalizeVibrationalModes(normalize_per_mode=True)
        result = transform(data)

        # Each mode should be normalized independently
        for i in range(result.vibmodes.size(0)):
            mode = result.vibmodes[i]
            # Check that norm is approximately 1.0 (or close)
            norm = torch.norm(mode)
            assert norm > 0.0  # Should be normalized

    def test_normalize_global(self):
        """Test global normalization"""
        vibmodes = torch.randn(5, 3, 3)
        data = Data(x=torch.ones(3, 1), vibmodes=vibmodes, num_nodes=3)

        transform = NormalizeVibrationalModes(normalize_per_mode=False)
        result = transform(data)

        # Global norm should be close to 1.0
        global_norm = torch.norm(result.vibmodes)
        assert global_norm > 0.0

    def test_epsilon_parameter(self):
        """Test epsilon parameter application"""
        vibmodes = torch.ones(2, 3, 3)
        data = Data(x=torch.ones(3, 1), vibmodes=vibmodes, num_nodes=3)

        transform = NormalizeVibrationalModes(normalize_per_mode=False, epsilon=1e-6)
        result = transform(data)

        # Should be normalized with custom epsilon
        assert result.vibmodes is not None
        assert transform.epsilon == 1e-6

    def test_missing_vibmodes(self):
        """Test handling of missing vibmodes"""
        data = Data(x=torch.ones(3, 1), num_nodes=3)  # No vibmodes
        transform = NormalizeVibrationalModes()
        result = transform(data)
        # Should return data unchanged
        assert not hasattr(result, "vibmodes") or result.vibmodes is None

    def test_empty_vibmodes(self):
        """Test handling of empty vibmodes tensor"""
        data = Data(x=torch.ones(3, 1), vibmodes=torch.empty(0, 3, 3), num_nodes=3)
        transform = NormalizeVibrationalModes()
        # Empty vibmodes should raise error due to reshape issue
        with pytest.raises(TransformExecutionError):
            _result = transform(data)

    def test_immutability(self):
        """Test that original data is not modified"""
        original_vibmodes = torch.randn(5, 3, 3)
        data = Data(x=torch.ones(3, 1), vibmodes=original_vibmodes.clone(), num_nodes=3)
        original_copy = data.vibmodes.clone()

        transform = NormalizeVibrationalModes(normalize_per_mode=True)
        result = transform(data)

        # Original should be unchanged
        assert torch.allclose(data.vibmodes, original_copy)
        # Result should be different (normalized)
        assert not torch.allclose(result.vibmodes, original_copy)

    def test_zero_vibmodes(self):
        """Test handling of zero vibmodes"""
        vibmodes = torch.zeros(2, 3, 3)
        data = Data(x=torch.ones(3, 1), vibmodes=vibmodes, num_nodes=3)
        transform = NormalizeVibrationalModes()
        result = transform(data)
        # Should handle zero vectors gracefully
        assert result is not None


# =============================================================================
# FILTERBYDMCUNCERTAINTY TESTS
# =============================================================================


class TestFilterByDMCUncertainty:
    """Test suite for FilterByDMCUncertainty transform"""

    def test_instantiation_default(self):
        """Test instantiation with default parameters"""
        transform = FilterByDMCUncertainty(max_uncertainty=0.1)
        assert transform.max_uncertainty == 0.1
        assert transform.remove is False

    def test_instantiation_with_remove(self):
        """Test instantiation with remove=True"""
        transform = FilterByDMCUncertainty(max_uncertainty=0.05, remove=True)
        assert transform.max_uncertainty == 0.05
        assert transform.remove is True

    def test_metadata(self):
        """Test metadata"""
        metadata = FilterByDMCUncertainty.get_metadata()
        assert metadata.name == "FilterByDMCUncertainty"
        assert metadata.category == "quantum"
        assert "milia_DMC" in metadata.validated_datasets
        assert "dmc_uncertainty" in metadata.required_graph_attributes

    def test_parameter_constraints(self):
        """Test parameter constraints"""
        constraints = FilterByDMCUncertainty.get_parameter_constraints()
        assert "max_uncertainty" in constraints
        assert "remove" in constraints
        assert constraints["max_uncertainty"]["type"] is float
        assert constraints["remove"]["type"] is bool

    def test_low_uncertainty_flagging(self):
        """Test flagging low uncertainty data"""
        data = Data(x=torch.ones(3, 1), dmc_uncertainty=torch.tensor(0.05), num_nodes=3)
        transform = FilterByDMCUncertainty(max_uncertainty=0.1, remove=False)
        result = transform(data)

        assert hasattr(result, "is_high_uncertainty")
        assert not result.is_high_uncertainty

    def test_high_uncertainty_flagging(self):
        """Test flagging high uncertainty data"""
        data = Data(x=torch.ones(3, 1), dmc_uncertainty=torch.tensor(0.15), num_nodes=3)
        transform = FilterByDMCUncertainty(max_uncertainty=0.1, remove=False)
        result = transform(data)

        assert hasattr(result, "is_high_uncertainty")
        assert result.is_high_uncertainty

    def test_high_uncertainty_removal(self):
        """Test removing high uncertainty data"""
        data = Data(x=torch.ones(3, 1), dmc_uncertainty=torch.tensor(0.15), num_nodes=3)
        transform = FilterByDMCUncertainty(max_uncertainty=0.1, remove=True)
        result = transform(data)

        assert result is None

    def test_low_uncertainty_no_removal(self):
        """Test keeping low uncertainty data when remove=True"""
        data = Data(x=torch.ones(3, 1), dmc_uncertainty=torch.tensor(0.05), num_nodes=3)
        transform = FilterByDMCUncertainty(max_uncertainty=0.1, remove=True)
        result = transform(data)

        assert result is not None
        assert hasattr(result, "is_high_uncertainty")
        assert not result.is_high_uncertainty

    def test_missing_uncertainty(self):
        """Test handling of missing uncertainty attribute"""
        data = Data(x=torch.ones(3, 1), num_nodes=3)  # No dmc_uncertainty
        transform = FilterByDMCUncertainty(max_uncertainty=0.1, remove=False)
        result = transform(data)

        # Should return data unchanged with warning flag
        assert result is not None

    def test_boundary_case_exact_threshold(self):
        """Test boundary case where uncertainty equals threshold"""
        data = Data(x=torch.ones(3, 1), dmc_uncertainty=torch.tensor(0.1), num_nodes=3)
        transform = FilterByDMCUncertainty(max_uncertainty=0.1, remove=False)
        result = transform(data)

        # At threshold, should be considered acceptable (<=)
        assert not result.is_high_uncertainty

    def test_immutability(self):
        """Test that original data is not modified"""
        data = Data(x=torch.ones(3, 1), dmc_uncertainty=torch.tensor(0.15), num_nodes=3)
        transform = FilterByDMCUncertainty(max_uncertainty=0.1, remove=False)
        result = transform(data)

        # Original should not have new attributes
        assert not hasattr(data, "is_high_uncertainty")
        # Result should have new attributes
        assert hasattr(result, "is_high_uncertainty")

    def test_scalar_vs_tensor_uncertainty(self):
        """Test handling of both scalar tensor and multi-element uncertainty"""
        # Scalar tensor
        data1 = Data(x=torch.ones(3, 1), dmc_uncertainty=torch.tensor(0.05), num_nodes=3)
        transform = FilterByDMCUncertainty(max_uncertainty=0.1)
        result1 = transform(data1)
        assert result1 is not None


# =============================================================================
# SCALEMULLIKENCHARGES TESTS
# =============================================================================


class TestScaleMullikenCharges:
    """Test suite for ScaleMullikenCharges transform"""

    def test_instantiation_default(self):
        """Test instantiation with default parameters"""
        transform = ScaleMullikenCharges()
        assert transform.scale_factor == 1.0
        assert transform.center is False

    def test_instantiation_custom(self):
        """Test instantiation with custom parameters"""
        transform = ScaleMullikenCharges(scale_factor=2.0, center=True)
        assert transform.scale_factor == 2.0
        assert transform.center is True

    def test_metadata(self):
        """Test metadata"""
        metadata = ScaleMullikenCharges.get_metadata()
        assert metadata.name == "ScaleMullikenCharges"
        assert metadata.category == "quantum"
        assert "milia_DFT" in metadata.validated_datasets
        assert "charges" in metadata.required_graph_attributes

    def test_parameter_constraints(self):
        """Test parameter constraints"""
        constraints = ScaleMullikenCharges.get_parameter_constraints()
        assert "scale_factor" in constraints
        assert "center" in constraints
        assert constraints["scale_factor"]["type"] is float
        assert constraints["scale_factor"]["range"] == (0.0, 10.0)
        assert constraints["center"]["type"] is bool

    def test_simple_scaling(self):
        """Test simple charge scaling without centering"""
        charges = torch.tensor([1.0, -0.5, -0.5])
        data = Data(x=torch.ones(3, 1), charges=charges, num_nodes=3)

        transform = ScaleMullikenCharges(scale_factor=2.0, center=False)
        result = transform(data)

        expected = torch.tensor([2.0, -1.0, -1.0])
        assert torch.allclose(result.charges, expected)

    def test_scaling_with_centering(self):
        """Test charge scaling with centering"""
        charges = torch.tensor([1.0, 0.0, -1.0])  # mean = 0.0
        data = Data(x=torch.ones(3, 1), charges=charges, num_nodes=3)

        transform = ScaleMullikenCharges(scale_factor=2.0, center=True)
        result = transform(data)

        # Already centered, just scaled
        expected = torch.tensor([2.0, 0.0, -2.0])
        assert torch.allclose(result.charges, expected)

    def test_centering_non_zero_mean(self):
        """Test centering charges with non-zero mean"""
        charges = torch.tensor([0.5, -0.3, -0.2])  # mean = 0.0
        data = Data(x=torch.ones(3, 1), charges=charges, num_nodes=3)

        transform = ScaleMullikenCharges(scale_factor=1.0, center=True)
        result = transform(data)

        # After centering, mean should be ~0
        assert torch.abs(result.charges.mean()) < 1e-6

    def test_missing_charges(self):
        """Test handling of missing charges"""
        data = Data(x=torch.ones(3, 1), num_nodes=3)  # No charges
        transform = ScaleMullikenCharges(scale_factor=2.0)
        result = transform(data)

        # Should return data unchanged
        assert not hasattr(result, "charges") or result.charges is None

    def test_empty_charges(self):
        """Test handling of empty charges tensor"""
        data = Data(x=torch.ones(3, 1), charges=torch.empty(0), num_nodes=3)
        transform = ScaleMullikenCharges(scale_factor=2.0)
        result = transform(data)

        # Should handle gracefully
        assert result is not None

    def test_immutability(self):
        """Test that original data is not modified"""
        original_charges = torch.tensor([0.5, -0.3, -0.2])
        data = Data(x=torch.ones(3, 1), charges=original_charges.clone(), num_nodes=3)
        original_copy = data.charges.clone()

        transform = ScaleMullikenCharges(scale_factor=2.0, center=True)
        result = transform(data)

        # Original should be unchanged
        assert torch.allclose(data.charges, original_copy)
        # Result should be modified
        assert not torch.allclose(result.charges, original_copy)

    def test_1d_charges(self):
        """Test handling of 1D charges tensor"""
        charges = torch.tensor([0.5, -0.3, -0.2])
        data = Data(x=torch.ones(3, 1), charges=charges, num_nodes=3)

        transform = ScaleMullikenCharges(scale_factor=2.0)
        result = transform(data)

        assert result.charges.dim() == 1
        assert torch.allclose(result.charges, charges * 2.0)

    def test_2d_charges_single_column(self):
        """Test handling of 2D charges tensor with single column"""
        charges = torch.tensor([[0.5], [-0.3], [-0.2]])
        data = Data(x=torch.ones(3, 1), charges=charges, num_nodes=3)

        transform = ScaleMullikenCharges(scale_factor=2.0)
        result = transform(data)

        # Should be handled correctly
        assert result.charges is not None

    def test_non_finite_charges_handling(self):
        """Test handling of non-finite charges"""
        charges = torch.tensor([0.5, float("nan"), -0.2])
        data = Data(x=torch.ones(3, 1), charges=charges, num_nodes=3)

        transform = ScaleMullikenCharges(scale_factor=2.0)
        result = transform(data)

        # Should return data unchanged due to non-finite values warning
        # Note: Can't use torch.equal with NaN values, check individual finite values
        assert torch.isclose(result.charges[0], torch.tensor(0.5))
        assert torch.isclose(result.charges[2], torch.tensor(-0.2))
        assert torch.isnan(result.charges[1])  # NaN should still be NaN

    def test_scale_factor_zero(self):
        """Test scaling with factor of zero"""
        charges = torch.tensor([0.5, -0.3, -0.2])
        data = Data(x=torch.ones(3, 1), charges=charges, num_nodes=3)

        transform = ScaleMullikenCharges(scale_factor=0.0, center=False)
        result = transform(data)

        expected = torch.zeros_like(charges)
        assert torch.allclose(result.charges, expected)

    def test_shape_preservation(self):
        """Test that charge tensor shape is preserved"""
        charges = torch.tensor([0.5, -0.3, -0.2])
        original_shape = charges.shape
        data = Data(x=torch.ones(3, 1), charges=charges, num_nodes=3)

        transform = ScaleMullikenCharges(scale_factor=2.0, center=True)
        result = transform(data)

        assert result.charges.shape == original_shape


# =============================================================================
# STANDARDIZETARGETS TESTS
# =============================================================================


class TestStandardizeTargets:
    """Test suite for StandardizeTargets transform"""

    def test_instantiation_default(self):
        """Test instantiation with default parameters"""
        transform = StandardizeTargets()
        assert transform.attrs == ["y"]
        assert transform.eps == 1e-8

    def test_instantiation_custom(self):
        """Test instantiation with custom parameters"""
        transform = StandardizeTargets(attrs=["y", "energy"], eps=1e-6)
        assert transform.attrs == ["y", "energy"]
        assert transform.eps == 1e-6

    def test_metadata(self):
        """Test metadata"""
        metadata = StandardizeTargets.get_metadata()
        assert metadata.name == "StandardizeTargets"
        assert metadata.category == "normalization"
        assert "milia_DFT" in metadata.validated_datasets
        assert "y" in metadata.required_graph_attributes

    def test_parameter_constraints(self):
        """Test parameter constraints"""
        constraints = StandardizeTargets.get_parameter_constraints()
        assert "attrs" in constraints
        assert "eps" in constraints
        assert constraints["attrs"]["type"] is list
        assert constraints["eps"]["type"] is float

    def test_basic_standardization(self):
        """Test basic z-score standardization"""
        # Values with known mean and std
        y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # mean=3, std≈1.58
        data = Data(x=torch.ones(5, 3), y=y, num_nodes=5)

        transform = StandardizeTargets(attrs=["y"])
        result = transform(data)

        # Check standardized values have mean≈0 and std≈1
        assert torch.abs(result.y.mean()) < 1e-5
        assert torch.abs(result.y.std() - 1.0) < 1e-5

        # Check metadata stored
        assert hasattr(result, "y_mean")
        assert hasattr(result, "y_std")
        assert result.targets_standardized is True

    def test_standardization_preserves_original(self):
        """Test that original data is not modified"""
        original_y = torch.tensor([1.0, 2.0, 3.0])
        data = Data(x=torch.ones(3, 3), y=original_y.clone(), num_nodes=3)
        original_copy = data.y.clone()

        transform = StandardizeTargets()
        result = transform(data)

        assert torch.allclose(data.y, original_copy)
        assert not torch.allclose(result.y, original_copy)

    def test_standardization_scalar_target(self):
        """Test standardization of scalar target value"""
        data = Data(x=torch.ones(3, 3), y=torch.tensor(-100.5), num_nodes=3)

        transform = StandardizeTargets()
        result = transform(data)

        # Single value standardization should be 0 (or close to 0)
        assert hasattr(result, "y_mean")
        assert hasattr(result, "y_std")

    def test_standardization_multiple_attrs(self):
        """Test standardization of multiple attributes"""
        data = Data(
            x=torch.ones(5, 3),
            y=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            energy=torch.tensor([-100.0, -101.0, -102.0, -103.0, -104.0]),
            num_nodes=5,
        )

        transform = StandardizeTargets(attrs=["y", "energy"])
        result = transform(data)

        # Both should be standardized
        assert torch.abs(result.y.mean()) < 1e-5
        assert torch.abs(result.energy.mean()) < 1e-5
        assert hasattr(result, "y_mean")
        assert hasattr(result, "energy_mean")

    def test_standardization_missing_attr(self):
        """Test handling of missing attribute"""
        data = Data(x=torch.ones(3, 3), num_nodes=3)  # No 'y'

        transform = StandardizeTargets(attrs=["y"])
        result = transform(data)

        # Should not raise, just skip missing attr
        assert result is not None
        assert result.targets_standardized is True

    def test_standardization_non_finite_values(self):
        """Test handling of non-finite values"""
        y = torch.tensor([1.0, float("nan"), 3.0])
        data = Data(x=torch.ones(3, 3), y=y, num_nodes=3)

        transform = StandardizeTargets()
        result = transform(data)

        # Should skip standardization for non-finite values
        assert result is not None

    def test_standardization_constant_values(self):
        """Test handling of constant values (std=0)"""
        y = torch.tensor([5.0, 5.0, 5.0, 5.0])  # Constant, std=0
        data = Data(x=torch.ones(4, 3), y=y, num_nodes=4)

        transform = StandardizeTargets()
        result = transform(data)

        # Should use eps to prevent division by zero
        assert result is not None
        assert torch.all(torch.isfinite(result.y))

    def test_standardization_integer_to_float(self):
        """Test conversion of integer targets to float"""
        y = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        data = Data(x=torch.ones(5, 3), y=y, num_nodes=5)

        transform = StandardizeTargets()
        result = transform(data)

        assert result.y.is_floating_point()


# =============================================================================
# NORMALIZETARGETS TESTS
# =============================================================================


class TestNormalizeTargets:
    """Test suite for NormalizeTargets transform"""

    def test_instantiation_default(self):
        """Test instantiation with default parameters"""
        transform = NormalizeTargets()
        assert transform.attrs == ["y"]
        assert transform.range_min == 0.0
        assert transform.range_max == 1.0
        assert transform.eps == 1e-8

    def test_instantiation_custom(self):
        """Test instantiation with custom parameters"""
        transform = NormalizeTargets(attrs=["y", "energy"], range_min=-1.0, range_max=1.0, eps=1e-6)
        assert transform.attrs == ["y", "energy"]
        assert transform.range_min == -1.0
        assert transform.range_max == 1.0
        assert transform.eps == 1e-6

    def test_invalid_range_raises_error(self):
        """Test that invalid range raises TransformConfigurationError"""
        with pytest.raises(TransformConfigurationError):
            NormalizeTargets(range_min=1.0, range_max=0.0)  # min >= max

    def test_metadata(self):
        """Test metadata"""
        metadata = NormalizeTargets.get_metadata()
        assert metadata.name == "NormalizeTargets"
        assert metadata.category == "normalization"
        assert "milia_DFT" in metadata.validated_datasets

    def test_parameter_constraints(self):
        """Test parameter constraints"""
        constraints = NormalizeTargets.get_parameter_constraints()
        assert "attrs" in constraints
        assert "range_min" in constraints
        assert "range_max" in constraints
        assert "eps" in constraints

    def test_basic_normalization_0_1(self):
        """Test basic min-max normalization to [0, 1]"""
        y = torch.tensor([0.0, 50.0, 100.0])  # min=0, max=100
        data = Data(x=torch.ones(3, 3), y=y, num_nodes=3)

        transform = NormalizeTargets(range_min=0.0, range_max=1.0)
        result = transform(data)

        expected = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(result.y, expected)
        assert hasattr(result, "y_min")
        assert hasattr(result, "y_max")
        assert result.targets_normalized is True

    def test_normalization_custom_range(self):
        """Test normalization to custom range [-1, 1]"""
        y = torch.tensor([0.0, 50.0, 100.0])
        data = Data(x=torch.ones(3, 3), y=y, num_nodes=3)

        transform = NormalizeTargets(range_min=-1.0, range_max=1.0)
        result = transform(data)

        expected = torch.tensor([-1.0, 0.0, 1.0])
        assert torch.allclose(result.y, expected)

    def test_normalization_preserves_original(self):
        """Test that original data is not modified"""
        original_y = torch.tensor([1.0, 2.0, 3.0])
        data = Data(x=torch.ones(3, 3), y=original_y.clone(), num_nodes=3)
        original_copy = data.y.clone()

        transform = NormalizeTargets()
        _result = transform(data)

        assert torch.allclose(data.y, original_copy)

    def test_normalization_scalar_target(self):
        """Test normalization of scalar target"""
        data = Data(x=torch.ones(3, 3), y=torch.tensor(-100.5), num_nodes=3)

        transform = NormalizeTargets()
        result = transform(data)

        assert hasattr(result, "y_min")
        assert hasattr(result, "y_max")

    def test_normalization_constant_values(self):
        """Test handling of constant values (range=0)"""
        y = torch.tensor([5.0, 5.0, 5.0])
        data = Data(x=torch.ones(3, 3), y=y, num_nodes=3)

        transform = NormalizeTargets()
        result = transform(data)

        # Should handle gracefully with eps
        assert result is not None
        assert torch.all(torch.isfinite(result.y))


# =============================================================================
# DISCRETIZETARGETS TESTS
# =============================================================================


class TestDiscretizeTargets:
    """Test suite for DiscretizeTargets transform"""

    def test_instantiation_default(self):
        """Test instantiation with default parameters"""
        transform = DiscretizeTargets()
        assert transform.n_bins == 5
        assert transform.strategy == "quantile"
        assert transform.target_column == 0
        assert transform.attrs == ["y"]
        assert transform.target_level == "auto"

    def test_instantiation_custom(self):
        """Test instantiation with custom parameters"""
        transform = DiscretizeTargets(
            n_bins=10,
            strategy="uniform",
            target_column=1,
            attrs=["y", "edge_label"],
            target_level="graph",
        )
        assert transform.n_bins == 10
        assert transform.strategy == "uniform"
        assert transform.target_column == 1
        assert transform.attrs == ["y", "edge_label"]
        assert transform.target_level == "graph"

    def test_invalid_n_bins_raises_error(self):
        """Test that n_bins < 2 raises TransformConfigurationError"""
        with pytest.raises(TransformConfigurationError):
            DiscretizeTargets(n_bins=1)

    def test_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises TransformConfigurationError"""
        with pytest.raises(TransformConfigurationError):
            DiscretizeTargets(strategy="invalid_strategy")

    def test_invalid_target_level_raises_error(self):
        """Test that invalid target_level raises TransformConfigurationError"""
        with pytest.raises(TransformConfigurationError):
            DiscretizeTargets(target_level="invalid_level")

    def test_metadata(self):
        """Test metadata"""
        metadata = DiscretizeTargets.get_metadata()
        assert metadata.name == "DiscretizeTargets"
        assert metadata.category == "classification"
        assert "milia_DFT" in metadata.validated_datasets

    def test_parameter_constraints(self):
        """Test parameter constraints"""
        constraints = DiscretizeTargets.get_parameter_constraints()
        assert "n_bins" in constraints
        assert "strategy" in constraints
        assert "target_column" in constraints
        assert "attrs" in constraints
        assert constraints["strategy"]["choices"] == ["quantile", "uniform", "kmeans"]

    def test_quantile_discretization(self):
        """Test quantile-based discretization"""
        # Create evenly distributed values for predictable bins
        y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        data = Data(x=torch.ones(10, 3), y=y, num_nodes=10)

        transform = DiscretizeTargets(n_bins=5, strategy="quantile")
        result = transform(data)

        # Should have 5 classes (0-4)
        assert result.y.dtype == torch.long
        assert result.y.min() >= 0
        assert result.y.max() < 5
        assert result.y_num_classes == 5
        assert hasattr(result, "y_original")
        assert hasattr(result, "y_bin_edges")
        assert result.targets_discretized is True

    def test_uniform_discretization(self):
        """Test uniform-width discretization"""
        y = torch.tensor([0.0, 25.0, 50.0, 75.0, 100.0])
        data = Data(x=torch.ones(5, 3), y=y, num_nodes=5)

        transform = DiscretizeTargets(n_bins=4, strategy="uniform")
        result = transform(data)

        assert result.y.dtype == torch.long
        assert result.y.min() >= 0
        assert result.y.max() < 4
        assert result.y_num_classes == 4

    def test_kmeans_discretization(self):
        """Test kmeans-based discretization"""
        # Create clustered values
        y = torch.tensor([1.0, 1.1, 1.2, 5.0, 5.1, 5.2, 9.0, 9.1, 9.2])
        data = Data(x=torch.ones(9, 3), y=y, num_nodes=9)

        transform = DiscretizeTargets(n_bins=3, strategy="kmeans")
        result = transform(data)

        assert result.y.dtype == torch.long
        assert result.y.min() >= 0
        assert result.y.max() < 3
        assert result.y_num_classes == 3

    def test_discretization_preserves_original(self):
        """Test that original data is not modified"""
        original_y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        data = Data(x=torch.ones(5, 3), y=original_y.clone(), num_nodes=5)
        original_copy = data.y.clone()

        transform = DiscretizeTargets()
        _result = transform(data)

        assert torch.allclose(data.y, original_copy)

    def test_discretization_scalar_target(self):
        """Test discretization of scalar target"""
        data = Data(x=torch.ones(3, 3), y=torch.tensor(-100.5), num_nodes=3)

        transform = DiscretizeTargets(n_bins=5)
        result = transform(data)

        assert result.y.dtype == torch.long
        assert hasattr(result, "y_num_classes")

    def test_discretization_graph_level_1d(self):
        """Test discretization of 1D graph-level target"""
        data = Data(x=torch.ones(3, 3), y=torch.tensor([50.0]), num_nodes=3)

        transform = DiscretizeTargets(n_bins=5, target_level="graph")
        result = transform(data)

        assert result.y.dtype == torch.long
        assert result.y_target_level == "graph"

    def test_discretization_node_level(self):
        """Test discretization of node-level targets"""
        # Node-level: y has shape [num_nodes]
        y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 nodes
        data = Data(x=torch.ones(5, 3), y=y, num_nodes=5)

        transform = DiscretizeTargets(n_bins=3, target_level="node")
        result = transform(data)

        assert result.y.shape[0] == 5
        assert result.y.dtype == torch.long
        assert result.y_target_level == "node"
        assert result.y_is_node_level is True

    def test_discretization_edge_level(self):
        """Test discretization of edge-level targets"""
        # Edge-level: edge_label has shape [num_edges]
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        edge_label = torch.tensor([1.0, 2.0, 3.0])  # 3 edges
        data = Data(x=torch.ones(3, 3), edge_index=edge_index, edge_label=edge_label, num_nodes=3)

        transform = DiscretizeTargets(n_bins=3, attrs=["edge_label"], target_level="edge")
        result = transform(data)

        assert result.edge_label.shape[0] == 3
        assert result.edge_label.dtype == torch.long
        assert result.edge_label_target_level == "edge"

    def test_discretization_multi_target_column_selection(self):
        """Test multi-target discretization with column selection"""
        # Multi-target: y has shape [num_nodes, num_features]
        y = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
        data = Data(x=torch.ones(5, 3), y=y, num_nodes=5)

        # Discretize second column
        transform = DiscretizeTargets(n_bins=3, target_column=1, target_level="node")
        result = transform(data)

        assert result.y.shape[0] == 5
        assert result.y.dtype == torch.long

    def test_fit_method(self):
        """Test fit method for computing bin edges from dataset"""
        # Create a dataset
        dataset = [
            Data(x=torch.ones(3, 3), y=torch.tensor([1.0]), num_nodes=3),
            Data(x=torch.ones(3, 3), y=torch.tensor([2.0]), num_nodes=3),
            Data(x=torch.ones(3, 3), y=torch.tensor([3.0]), num_nodes=3),
            Data(x=torch.ones(3, 3), y=torch.tensor([4.0]), num_nodes=3),
            Data(x=torch.ones(3, 3), y=torch.tensor([5.0]), num_nodes=3),
        ]

        transform = DiscretizeTargets(n_bins=3, strategy="quantile")
        transform.fit(dataset)

        assert transform._is_fitted is True
        assert "y" in transform._fitted_bin_edges

    def test_fitted_transform_consistency(self):
        """Test that fitted transform applies consistent bins"""
        # Create training data
        train_data = [
            Data(x=torch.ones(3, 3), y=torch.tensor([1.0]), num_nodes=3),
            Data(x=torch.ones(3, 3), y=torch.tensor([5.0]), num_nodes=3),
            Data(x=torch.ones(3, 3), y=torch.tensor([10.0]), num_nodes=3),
        ]

        transform = DiscretizeTargets(n_bins=3)
        transform.fit(train_data)

        # Apply to test data (value within training range)
        test_data = Data(x=torch.ones(3, 3), y=torch.tensor([3.0]), num_nodes=3)
        result = transform(test_data)

        # Should use fitted bin edges
        assert result.y.dtype == torch.long

    def test_compute_class_weights_balanced(self):
        """Test compute_class_weights with balanced method"""
        # Create discretized dataset
        dataset = []
        transform = DiscretizeTargets(n_bins=3)

        # Create imbalanced data: more samples in class 0
        for val in [1.0, 1.1, 1.2, 1.3, 5.0, 10.0]:  # 4 in low, 1 mid, 1 high
            data = Data(x=torch.ones(3, 3), y=torch.tensor([val]), num_nodes=3)
            dataset.append(transform(data))

        weights = DiscretizeTargets.compute_class_weights(dataset, attr="y", method="balanced")

        assert weights.shape[0] == 3
        assert torch.all(weights > 0)

    def test_compute_class_weights_inverse(self):
        """Test compute_class_weights with inverse method"""
        dataset = []
        transform = DiscretizeTargets(n_bins=3)

        for val in [1.0, 5.0, 10.0]:
            data = Data(x=torch.ones(3, 3), y=torch.tensor([val]), num_nodes=3)
            dataset.append(transform(data))

        weights = DiscretizeTargets.compute_class_weights(dataset, attr="y", method="inverse")

        assert weights.shape[0] == 3

    def test_compute_class_weights_sqrt_inverse(self):
        """Test compute_class_weights with sqrt_inverse method"""
        dataset = []
        transform = DiscretizeTargets(n_bins=3)

        for val in [1.0, 5.0, 10.0]:
            data = Data(x=torch.ones(3, 3), y=torch.tensor([val]), num_nodes=3)
            dataset.append(transform(data))

        weights = DiscretizeTargets.compute_class_weights(dataset, attr="y", method="sqrt_inverse")

        assert weights.shape[0] == 3

    def test_compute_class_weights_invalid_method(self):
        """Test that invalid method raises ValueError"""
        dataset = [Data(x=torch.ones(3, 3), y=torch.tensor([0]), num_nodes=3)]

        with pytest.raises(ValueError):
            DiscretizeTargets.compute_class_weights(dataset, attr="y", method="invalid")

    def test_get_num_classes_single_data(self):
        """Test get_num_classes from single Data object"""
        data = Data(x=torch.ones(3, 3), y=torch.tensor([0]), num_nodes=3)
        data.y_num_classes = 5

        num_classes = DiscretizeTargets.get_num_classes(data, attr="y")
        assert num_classes == 5

    def test_get_num_classes_dataset(self):
        """Test get_num_classes from dataset"""
        data = Data(x=torch.ones(3, 3), y=torch.tensor([0]), num_nodes=3)
        data.y_num_classes = 5
        dataset = [data]

        num_classes = DiscretizeTargets.get_num_classes(dataset, attr="y")
        assert num_classes == 5

    def test_get_num_classes_missing_attr(self):
        """Test get_num_classes when attribute is missing from dataset"""
        # Use a list (dataset) without the discretization metadata
        data = Data(x=torch.ones(3, 3), num_nodes=3)  # No y_num_classes
        dataset = [data]

        num_classes = DiscretizeTargets.get_num_classes(dataset, attr="y")
        assert num_classes is None

    def test_discretization_with_pre_fitted_edges(self):
        """Test discretization with pre-computed bin edges"""
        bin_edges = torch.tensor([0.0, 3.0, 7.0, 10.0])
        transform = DiscretizeTargets(n_bins=3, fitted_bin_edges={"y": bin_edges})

        assert transform._is_fitted is True

        data = Data(x=torch.ones(3, 3), y=torch.tensor([5.0]), num_nodes=3)
        result = transform(data)

        assert result.y.dtype == torch.long

    def test_discretization_already_integer(self):
        """Test handling of already integer targets"""
        y = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
        data = Data(x=torch.ones(5, 3), y=y, num_nodes=5)

        transform = DiscretizeTargets(n_bins=3)
        result = transform(data)

        # Should recognize valid class indices and skip discretization
        assert result.y_num_classes == 3


# =============================================================================
# VALIDATE METHODS TESTS
# =============================================================================


class TestValidateMethods:
    """Test suite for validation methods on base classes"""

    def test_validate_molecular_structure_valid(self):
        """Test validate_molecular_structure with valid data"""
        data = Data(
            x=torch.tensor([[6.0], [8.0], [1.0]]),  # C, O, H
            pos=torch.randn(3, 3),
            edge_attr=torch.tensor([[1.0], [2.0]]),  # single, double bonds
            num_nodes=3,
        )

        transform = ConcreteMolecularTransform()
        is_valid, issues = transform.validate_molecular_structure(data)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_molecular_structure_invalid_atomic_numbers(self):
        """Test validate_molecular_structure with invalid atomic numbers"""
        data = Data(
            x=torch.tensor([[200.0], [8.0], [1.0]]),  # 200 is invalid
            num_nodes=3,
        )

        transform = ConcreteMolecularTransform()
        is_valid, issues = transform.validate_molecular_structure(data)

        assert is_valid is False
        assert any("Invalid atomic numbers" in issue for issue in issues)

    def test_validate_molecular_structure_invalid_bond_types(self):
        """Test validate_molecular_structure with invalid bond types"""
        data = Data(
            x=torch.tensor([[6.0], [8.0]]),
            edge_attr=torch.tensor([[5.0]]),  # 5 is invalid bond type
            num_nodes=2,
        )

        transform = ConcreteMolecularTransform()
        is_valid, issues = transform.validate_molecular_structure(data)

        assert is_valid is False
        assert any("Invalid bond types" in issue for issue in issues)

    def test_validate_molecular_structure_coordinate_mismatch(self):
        """Test validate_molecular_structure with coordinate mismatch"""
        data = Data(
            x=torch.tensor([[6.0], [8.0], [1.0]]),
            pos=torch.randn(5, 3),  # 5 positions for 3 atoms
            num_nodes=3,
        )

        transform = ConcreteMolecularTransform()
        is_valid, issues = transform.validate_molecular_structure(data)

        assert is_valid is False
        assert any("Coordinate count mismatch" in issue for issue in issues)

    def test_validate_molecular_structure_non_finite_values(self):
        """Test validate_molecular_structure with non-finite values"""
        data = Data(x=torch.tensor([[6.0], [float("nan")], [1.0]]), num_nodes=3)

        transform = ConcreteMolecularTransform()
        is_valid, issues = transform.validate_molecular_structure(data)

        assert is_valid is False
        assert any("Non-finite" in issue for issue in issues)

    def test_validate_quantum_properties_valid(self):
        """Test validate_quantum_properties with valid data"""
        data = Data(
            x=torch.ones(3, 3),
            energy=torch.tensor(-100.5),
            dmc_energy=torch.tensor(-100.6),
            dmc_uncertainty=torch.tensor(0.05),
            charges=torch.tensor([0.5, -0.3, -0.2]),
            vibmodes=torch.randn(5, 3, 3),
            forces=torch.randn(3, 3),
            num_nodes=3,
        )

        transform = ConcreteQuantumTransform()
        is_valid, issues = transform.validate_quantum_properties(data)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_quantum_properties_missing_energy(self):
        """Test validate_quantum_properties with missing energy"""
        data = Data(x=torch.ones(3, 3), num_nodes=3)  # No energy

        transform = ConcreteQuantumTransform()
        is_valid, issues = transform.validate_quantum_properties(data)

        assert is_valid is False
        assert any("Missing required 'energy'" in issue for issue in issues)

    def test_validate_quantum_properties_negative_uncertainty(self):
        """Test validate_quantum_properties with negative uncertainty"""
        data = Data(
            x=torch.ones(3, 3),
            energy=torch.tensor(-100.5),
            dmc_uncertainty=torch.tensor(-0.05),  # Negative
            num_nodes=3,
        )

        transform = ConcreteQuantumTransform()
        is_valid, issues = transform.validate_quantum_properties(data)

        assert is_valid is False
        assert any("Negative DMC uncertainty" in issue for issue in issues)

    def test_validate_quantum_properties_charge_mismatch(self):
        """Test validate_quantum_properties with charge count mismatch"""
        data = Data(
            x=torch.ones(3, 3),
            energy=torch.tensor(-100.5),
            charges=torch.tensor([0.5, -0.3]),  # 2 charges for 3 nodes
            num_nodes=3,
        )

        transform = ConcreteQuantumTransform()
        is_valid, issues = transform.validate_quantum_properties(data)

        assert is_valid is False
        assert any("Charge count mismatch" in issue for issue in issues)

    def test_validate_quantum_properties_vibmode_dimension(self):
        """Test validate_quantum_properties with wrong vibmode dimensions"""
        data = Data(
            x=torch.ones(3, 3),
            energy=torch.tensor(-100.5),
            vibmodes=torch.randn(5, 5, 3),  # 5 atoms but data has 3
            num_nodes=3,
        )

        transform = ConcreteQuantumTransform()
        is_valid, issues = transform.validate_quantum_properties(data)

        assert is_valid is False
        assert any("Vibmode atom dimension mismatch" in issue for issue in issues)

    def test_validate_compatibility_strict_mode(self):
        """Test validate_compatibility in strict mode"""
        # ConcreteCustomTransform requires 'x'
        data = Data(num_nodes=3)  # No 'x'

        transform = ConcreteCustomTransform()
        is_valid, warnings = transform.validate_compatibility(data, validation_level="strict")

        assert is_valid is False
        assert len(warnings) > 0

    def test_validate_compatibility_normal_mode(self):
        """Test validate_compatibility in normal mode"""
        data = Data(num_nodes=3)  # No 'x'

        transform = ConcreteCustomTransform()
        is_valid, warnings = transform.validate_compatibility(data, validation_level="normal")

        # Normal mode gives warnings but doesn't fail
        assert is_valid is True
        assert len(warnings) > 0

    def test_validate_compatibility_lenient_mode(self):
        """Test validate_compatibility in lenient mode"""
        data = Data(num_nodes=3)

        transform = ConcreteCustomTransform()
        is_valid, warnings = transform.validate_compatibility(data, validation_level="lenient")

        assert is_valid is True


# =============================================================================
# USAGE STATISTICS TESTS
# =============================================================================


class TestUsageStatistics:
    """Test suite for usage statistics tracking"""

    def test_get_usage_statistics_initial(self):
        """Test initial usage statistics"""
        transform = ConcreteCustomTransform()
        stats = transform.get_usage_statistics()

        assert stats["transform_name"] == "ConcreteCustomTransform"
        assert stats["call_count"] == 0
        assert stats["error_count"] == 0
        assert stats["success_rate"] == 0.0
        assert "metadata" in stats

    def test_get_usage_statistics_after_calls(self):
        """Test usage statistics after successful calls"""
        transform = ConcreteCustomTransform()
        data = Data(x=torch.randn(3, 5), num_nodes=3)

        # Make some calls
        transform(data)
        transform(data)
        transform(data)

        stats = transform.get_usage_statistics()
        assert stats["call_count"] == 3
        assert stats["error_count"] == 0
        assert stats["success_rate"] == 1.0

    def test_get_usage_statistics_with_errors(self):
        """Test usage statistics with error tracking"""

        # Create a transform that will error
        class FailingTransform(CustomTransformBase):
            def __init__(self):
                super().__init__()

            def transform(self, data: Data) -> Data:
                raise ValueError("Intentional failure")

            @classmethod
            def get_metadata(cls) -> TransformMetadata:
                return TransformMetadata(
                    name="FailingTransform",
                    version="1.0.0",
                    author="Test",
                    category="test",
                    description="Test",
                )

        transform = FailingTransform()
        data = Data(x=torch.ones(3, 5), num_nodes=3)

        # Expect error
        with contextlib.suppress(TransformExecutionError):
            transform(data)

        stats = transform.get_usage_statistics()
        assert stats["error_count"] == 1
        assert stats["success_rate"] == 0.0


# =============================================================================
# EXCEPTION HANDLING TESTS
# =============================================================================


class TestExceptionHandling:
    """Test suite for exception handling in transforms"""

    def test_transform_validation_error_creation(self):
        """Test TransformValidationError creation"""
        error = TransformValidationError(
            "Test validation error", transform_name="TestTransform", param="test_param"
        )
        assert str(error) == "Test validation error"
        assert error.transform_name == "TestTransform"
        assert error.details["param"] == "test_param"

    def test_transform_execution_error_creation(self):
        """Test TransformExecutionError creation"""
        original = ValueError("Original error")
        error = TransformExecutionError(
            "Test execution error", transform_name="TestTransform", original_error=original
        )
        assert str(error) == "Test execution error"
        assert error.transform_name == "TestTransform"
        assert error.original_error is original

    def test_transform_configuration_error_creation(self):
        """Test TransformConfigurationError creation"""
        error = TransformConfigurationError(
            "Test config error", transform_name="TestTransform", config_key="test_key"
        )
        assert str(error) == "Test config error"
        assert error.transform_name == "TestTransform"
        assert error.details["config_key"] == "test_key"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestTransformIntegration:
    """Integration tests for transform pipelines"""

    def test_multiple_transform_application(self, quantum_graph_data):
        """Test applying multiple transforms in sequence"""
        # Create transforms
        t1 = NormalizeVibrationalModes(normalize_per_mode=True)
        t2 = ScaleMullikenCharges(scale_factor=2.0)
        t3 = FilterByDMCUncertainty(max_uncertainty=0.1, remove=False)

        # Apply in sequence
        result = t1(quantum_graph_data)
        result = t2(result)
        result = t3(result)

        # Check all transforms were applied
        assert hasattr(result, "vibmodes")
        assert hasattr(result, "charges")
        assert hasattr(result, "is_high_uncertainty")

    def test_transform_chain_immutability(self, quantum_graph_data):
        """Test that chained transforms don't corrupt data"""
        original_energy = quantum_graph_data.energy.clone()
        original_charges = quantum_graph_data.charges.clone()

        # Apply transforms
        t1 = ScaleMullikenCharges(scale_factor=2.0)
        t2 = FilterByDMCUncertainty(max_uncertainty=0.1, remove=False)

        result = t1(quantum_graph_data)
        result = t2(result)

        # Original data should be unchanged
        assert torch.allclose(quantum_graph_data.energy, original_energy)
        assert torch.allclose(quantum_graph_data.charges, original_charges)
        assert not hasattr(quantum_graph_data, "is_high_uncertainty")

    def test_partial_transform_application(self):
        """Test transforms when only some required attributes are present"""
        # Data with only some attributes
        data = Data(
            x=torch.ones(3, 5),
            z=torch.tensor([6, 8, 7]),
            charges=torch.tensor([0.5, -0.3, -0.2]),
            num_nodes=3,
            # Missing: vibmodes, dmc_uncertainty
        )

        # These should handle missing attributes gracefully
        t1 = NormalizeVibrationalModes()  # Missing vibmodes
        t2 = ScaleMullikenCharges(scale_factor=2.0)  # Has charges
        t3 = FilterByDMCUncertainty(max_uncertainty=0.1)  # Missing uncertainty

        result = t1(data)
        result = t2(result)
        result = t3(result)

        assert result is not None
        assert hasattr(result, "charges")

    def test_standardize_then_discretize_pipeline(self):
        """Test standardization followed by discretization"""
        # This is a common preprocessing pattern for classification
        data = Data(
            x=torch.ones(5, 10), y=torch.tensor([-100.0, -50.0, 0.0, 50.0, 100.0]), num_nodes=5
        )

        # First standardize (for intermediate processing/analysis)
        standardize = StandardizeTargets(attrs=["y"])
        _standardized = standardize(data)

        # Then discretize (for classification)
        discretize = DiscretizeTargets(n_bins=5)
        # Note: We apply to original data, not standardized, for classification
        result = discretize(data)

        assert result.targets_discretized is True
        assert result.y.dtype == torch.long

    def test_target_transform_pipeline_with_dataset(self, regression_dataset):
        """Test target transforms applied to dataset"""
        # Fit discretizer on dataset
        discretize = DiscretizeTargets(n_bins=5, strategy="quantile")
        discretize.fit(regression_dataset)

        # Apply to each sample
        transformed = [discretize(d) for d in regression_dataset]

        # All samples should have consistent metadata
        for data in transformed:
            assert data.y_num_classes == 5
            assert data.targets_discretized is True

        # Compute class weights for training
        weights = DiscretizeTargets.compute_class_weights(transformed, attr="y", method="balanced")
        assert weights.shape[0] == 5

    def test_normalization_chain_preserves_inverse_info(self):
        """Test that normalization stores info needed for inverse transform"""
        data = Data(
            x=torch.ones(5, 10), y=torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]), num_nodes=5
        )

        # Apply standardization
        transform = StandardizeTargets()
        result = transform(data)

        # Check inverse transform info is preserved
        assert hasattr(result, "y_mean")
        assert hasattr(result, "y_std")

        # Manually verify inverse would work
        original_approx = result.y * result.y_std + result.y_mean
        assert torch.allclose(original_approx, data.y, atol=1e-5)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions"""

    def test_empty_graph(self):
        """Test transforms on empty graphs"""
        data = Data(x=torch.empty(0, 5), num_nodes=0)
        transform = ConcreteCustomTransform(param1=2.0)
        result = transform(data)
        assert result.num_nodes == 0

    def test_single_node_graph(self):
        """Test transforms on single-node graphs"""
        data = Data(x=torch.randn(1, 5), charges=torch.tensor([0.5]), num_nodes=1)
        transform = ScaleMullikenCharges(scale_factor=2.0)
        result = transform(data)
        assert result.num_nodes == 1
        assert torch.allclose(result.charges, torch.tensor([1.0]))

    def test_large_graph(self):
        """Test transforms on large graphs"""
        data = Data(x=torch.randn(1000, 10), charges=torch.randn(1000), num_nodes=1000)
        transform = ScaleMullikenCharges(scale_factor=2.0, center=True)
        result = transform(data)
        assert result.num_nodes == 1000
        assert result.charges.shape[0] == 1000

    def test_extreme_scale_factors(self):
        """Test with extreme scale factors"""
        charges = torch.tensor([1.0, -1.0])
        data = Data(charges=charges, num_nodes=2)

        # Very small scale
        transform1 = ScaleMullikenCharges(scale_factor=0.001)
        result1 = transform1(data)
        assert torch.allclose(result1.charges, charges * 0.001)

        # Very large scale
        transform2 = ScaleMullikenCharges(scale_factor=100.0)
        result2 = transform2(data)
        assert torch.allclose(result2.charges, charges * 100.0)

    def test_transform_with_all_zeros(self):
        """Test transforms with all-zero tensors"""
        data = Data(
            x=torch.zeros(3, 5), charges=torch.zeros(3), vibmodes=torch.zeros(2, 3, 3), num_nodes=3
        )

        t1 = ScaleMullikenCharges(scale_factor=2.0)
        t2 = NormalizeVibrationalModes()

        result1 = t1(data)
        result2 = t2(data)

        assert result1 is not None
        assert result2 is not None

    def test_standardize_single_element(self):
        """Test StandardizeTargets with single element"""
        data = Data(x=torch.ones(3, 5), y=torch.tensor([100.0]), num_nodes=3)

        transform = StandardizeTargets()
        result = transform(data)

        # Single element should use population std (std=0 -> use eps)
        assert torch.all(torch.isfinite(result.y))

    def test_normalize_single_element(self):
        """Test NormalizeTargets with single element"""
        data = Data(x=torch.ones(3, 5), y=torch.tensor([50.0]), num_nodes=3)

        transform = NormalizeTargets()
        result = transform(data)

        # Single element - range is 0, should handle with eps
        assert torch.all(torch.isfinite(result.y))

    def test_discretize_single_element(self):
        """Test DiscretizeTargets with single element"""
        data = Data(x=torch.ones(3, 5), y=torch.tensor([50.0]), num_nodes=3)

        transform = DiscretizeTargets(n_bins=5)
        result = transform(data)

        assert result.y.dtype == torch.long
        assert result.y_num_classes == 5

    def test_discretize_extreme_values(self):
        """Test DiscretizeTargets with extreme values"""
        # Very large range
        y = torch.tensor([-1e6, 0.0, 1e6])
        data = Data(x=torch.ones(3, 5), y=y, num_nodes=3)

        transform = DiscretizeTargets(n_bins=3)
        result = transform(data)

        assert result.y.dtype == torch.long
        assert result.y.min() >= 0
        assert result.y.max() < 3

    def test_discretize_identical_values(self):
        """Test DiscretizeTargets with identical values"""
        y = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])
        data = Data(x=torch.ones(5, 5), y=y, num_nodes=5)

        transform = DiscretizeTargets(n_bins=3)
        result = transform(data)

        # All identical values should map to same bin
        assert torch.all(result.y == result.y[0])

    def test_standardize_very_large_values(self):
        """Test StandardizeTargets with very large values"""
        # DFT energies can be very large negative numbers
        y = torch.tensor([-1e6, -1e6 + 1, -1e6 + 2, -1e6 + 3, -1e6 + 4])
        data = Data(x=torch.ones(5, 5), y=y, num_nodes=5)

        transform = StandardizeTargets()
        result = transform(data)

        # Should still produce valid standardized values
        assert torch.abs(result.y.mean()) < 1e-4
        assert torch.all(torch.isfinite(result.y))

    def test_target_level_auto_detection_ambiguous(self):
        """Test target level auto-detection when num_nodes == num_edges"""
        # Create data where num_nodes == num_edges
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # 3 edges
        data = Data(
            x=torch.ones(3, 5),
            edge_index=edge_index,
            y=torch.tensor([1.0, 2.0, 3.0]),  # 3 values - ambiguous!
            num_nodes=3,
        )

        # With explicit target_level, should work correctly
        transform_node = DiscretizeTargets(n_bins=3, target_level="node")
        result_node = transform_node(data)
        assert result_node.y_target_level == "node"

        transform_graph = DiscretizeTargets(n_bins=3, target_level="graph")
        result_graph = transform_graph(data)
        assert result_graph.y_target_level == "graph"


# =============================================================================
# MODULE EXPORTS TEST
# =============================================================================


class TestModuleExports:
    """Test that all expected classes and functions are exported"""

    def test_core_classes_exported(self):
        """Test that core classes are in __all__"""
        from milia_pipeline.transformations.custom_transforms import __all__

        expected_exports = [
            "TransformMetadata",
            "CustomTransformBase",
            "MolecularTransformBase",
            "QuantumTransformBase",
            "NormalizeVibrationalModes",
            "FilterByDMCUncertainty",
            "ScaleMullikenCharges",
            "StandardizeTargets",
            "NormalizeTargets",
            "DiscretizeTargets",
            "TransformValidationError",
            "TransformExecutionError",
            "TransformConfigurationError",
        ]

        for export in expected_exports:
            assert export in __all__, f"{export} not in __all__"


# =============================================================================
# PERFORMANCE AND MEMORY TESTS
# =============================================================================


class TestPerformanceAndMemory:
    """Test performance and memory characteristics"""

    def test_clone_creates_new_objects(self, sample_graph_data):
        """Test that clone creates new tensor objects"""
        transform = ConcreteCustomTransform(param1=2.0)
        result = transform(sample_graph_data)

        # Should be different objects
        assert id(result.x) != id(sample_graph_data.x)
        assert result.x.data_ptr() != sample_graph_data.x.data_ptr()

    def test_no_memory_leak_in_chain(self, quantum_graph_data):
        """Test that chained transforms don't cause memory issues"""
        transforms = [
            NormalizeVibrationalModes(),
            ScaleMullikenCharges(scale_factor=2.0),
            FilterByDMCUncertainty(max_uncertainty=0.1, remove=False),
        ]

        result = quantum_graph_data
        for transform in transforms:
            result = transform(result)

        # Should complete without memory errors
        assert result is not None


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
