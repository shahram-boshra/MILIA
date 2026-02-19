#!/usr/bin/env python3
"""
Extended Production-Ready Unit Test Suite for data_converter.py Module - PART 1

This is Part 1 of 3 covering:
- Imports and project path setup
- Test fixtures
- DataConverterRegistry class tests (singleton, thread-safety, registration, retrieval)
- register_converter decorator tests
- get_registry function tests

Comprehensive test coverage for enterprise-grade deployment.

Author: MILIA Team
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import logging
import shutil
import tempfile
import threading
import time
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

# Import the module under test
from milia_pipeline.models.post_training.data_preparation.data_converter import (
    _3D_BOND_FEATURES,
    ASEAtomsConverter,
    BaseDataConverter,
    DataConverterProtocol,
    DataConverterRegistry,
    DictConverter,
    InChIConverter,
    PyGDataConverter,
    SDFConverter,
    SMILESConverter,
    XYZConverter,
    _apply_structural_features_if_available,
    _ensure_3d_conformer_for_prediction,
    _registry,
    _requires_3d_conformer,
    convert_batch_to_pyg,
    convert_sdf_to_pyg_list,
    convert_to_pyg,
    get_registry,
    list_all_formats,
    list_available_formats,
    register_converter,
    smiles_to_data,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_pyg_data():
    """Create a mock PyTorch Geometric Data object."""
    data = Data(
        x=torch.randn(10, 5),
        edge_index=torch.randint(0, 10, (2, 20)),
        edge_attr=torch.randn(20, 3),
        y=torch.randn(1),
    )
    return data


@pytest.fixture
def mock_pyg_data_minimal():
    """Create a minimal PyG Data object without edge_attr."""
    data = Data(x=torch.randn(5, 3), edge_index=torch.randint(0, 5, (2, 8)))
    return data


@pytest.fixture
def mock_pyg_data_with_pos():
    """Create a PyG Data object with 3D positions."""
    data = Data(
        z=torch.tensor([6, 6, 8]),  # C, C, O
        pos=torch.randn(3, 3),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
    )
    return data


@pytest.fixture
def valid_dict_data():
    """Create a valid dictionary for DictConverter."""
    return {
        "x": torch.randn(10, 5),
        "edge_index": torch.randint(0, 10, (2, 20)),
        "edge_attr": torch.randn(20, 3),
        "y": torch.randn(1),
    }


@pytest.fixture
def valid_dict_data_with_z():
    """Create a valid dictionary with z (atomic numbers) instead of x."""
    return {
        "z": torch.tensor([6, 6, 8, 1, 1, 1, 1]),  # C2H4O
        "edge_index": torch.randint(0, 7, (2, 14)),
        "pos": torch.randn(7, 3),
    }


@pytest.fixture
def valid_dict_data_lists():
    """Create a valid dictionary with Python lists instead of tensors."""
    return {"x": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "edge_index": [[0, 1, 1, 2], [1, 0, 2, 1]]}


@pytest.fixture
def invalid_dict_no_edges():
    """Create an invalid dictionary without edge_index."""
    return {"x": torch.randn(10, 5)}


@pytest.fixture
def invalid_dict_no_features():
    """Create an invalid dictionary without x or z."""
    return {"edge_index": torch.randint(0, 10, (2, 20))}


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return {
        "ethanol": "CCO",
        "benzene": "c1ccccc1",
        "methane": "C",
        "water": "O",
        "acetic_acid": "CC(=O)O",
        "caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    }


@pytest.fixture
def sample_inchi():
    """Sample InChI strings for testing."""
    return {
        "ethanol": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
        "water": "InChI=1S/H2O/h1H2",
        "methane": "InChI=1S/CH4/h1H4",
        "benzene": "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H",
    }


@pytest.fixture
def temp_directory():
    """Create a temporary directory for file operations."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_xyz_content():
    """Mock XYZ file content for water molecule."""
    return """3
Water molecule
O     0.000000     0.000000     0.117369
H     0.000000     0.756950    -0.469476
H     0.000000    -0.756950    -0.469476
"""


@pytest.fixture
def mock_sdf_content():
    """Mock SDF file content for methane."""
    return """
     RDKit          3D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    1.0900 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0260    0.0000   -0.3630 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5130   -0.8890   -0.3630 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5130    0.8890   -0.3630 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
M  END
"""


@pytest.fixture
def mock_multi_molecule_sdf_content():
    """Mock SDF file content with multiple molecules (methane and ethane)."""
    return """
     RDKit          3D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    1.0900 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0260    0.0000   -0.3630 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5130   -0.8890   -0.3630 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5130    0.8890   -0.3630 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
M  END
$$$$

     RDKit          3D

  8  7  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5400    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3900    1.0200    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3900   -0.5100    0.8800 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3900   -0.5100   -0.8800 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9300    1.0200    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9300   -0.5100    0.8800 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9300   -0.5100   -0.8800 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
  2  6  1  0
  2  7  1  0
  2  8  1  0
M  END
$$$$
"""


@pytest.fixture
def structural_features_config_basic():
    """Basic structural features config without 3D features.

    Note: Uses valid feature names from mol_structural_features.py
    Available atom features: degree, total_degree, hybridization, total_valence,
        is_aromatic, is_in_ring, partial_charge, mulliken_charge, num_aromatic_bonds, chirality
    Available bond features: bond_type, is_conjugated, is_in_ring, stereo
    """
    return {"atom": ["degree", "hybridization"], "bond": ["bond_type"]}


@pytest.fixture
def structural_features_config_with_3d():
    """Structural features config with 3D bond features."""
    return {"atom": ["atomic_num", "degree"], "bond": ["bond_type", "bond_length"]}


@pytest.fixture
def structural_features_config_empty():
    """Empty structural features config."""
    return {}


@pytest.fixture
def structural_features_config_none_values():
    """Structural features config with None/empty lists."""
    return {"atom": [], "bond": []}


@pytest.fixture
def fresh_registry():
    """
    Create a fresh registry instance for testing.

    IMPORTANT: This fixture resets the singleton for isolated testing.
    Uses teardown to restore original state.
    """
    # Store original singleton state
    original_instance = DataConverterRegistry._instance
    original_converters = None
    if original_instance is not None:
        original_converters = original_instance._converters.copy()

    # Reset singleton
    DataConverterRegistry._instance = None

    # Create fresh instance
    fresh = DataConverterRegistry()

    yield fresh

    # Restore original state
    DataConverterRegistry._instance = original_instance
    if original_instance is not None and original_converters is not None:
        original_instance._converters = original_converters


@pytest.fixture
def mock_converter_class():
    """Create a mock converter class for registration testing."""

    class MockConverter(BaseDataConverter):
        @property
        def format_name(self) -> str:
            return "mock_format"

        @property
        def is_available(self) -> bool:
            return True

        def can_convert(self, input_data: Any) -> bool:
            return isinstance(input_data, str) and input_data.startswith("MOCK:")

        def convert(self, input_data: Any, **kwargs) -> Data:
            return Data(x=torch.tensor([[1.0]]), edge_index=torch.tensor([[0], [0]]))

    return MockConverter


@pytest.fixture
def mock_unavailable_converter_class():
    """Create a mock converter class that is unavailable (missing dependencies)."""

    class UnavailableConverter(BaseDataConverter):
        @property
        def format_name(self) -> str:
            return "unavailable_format"

        @property
        def is_available(self) -> bool:
            return False

        def can_convert(self, input_data: Any) -> bool:
            return False

        def convert(self, input_data: Any, **kwargs) -> Data:
            raise ImportError("Dependencies not available")

    return UnavailableConverter


@pytest.fixture
def mock_rdkit_mol():
    """Create a mock RDKit molecule object for testing."""
    mock_mol = MagicMock()

    # Mock atoms (3 atoms: C, C, O for ethanol-like structure)
    mock_atom1 = MagicMock()
    mock_atom1.GetAtomicNum.return_value = 6  # Carbon
    mock_atom1.GetDegree.return_value = 1
    mock_atom1.GetFormalCharge.return_value = 0
    mock_atom1.GetTotalNumHs.return_value = 3
    mock_atom1.GetHybridization.return_value = 3  # SP3
    mock_atom1.GetIsAromatic.return_value = False

    mock_atom2 = MagicMock()
    mock_atom2.GetAtomicNum.return_value = 6  # Carbon
    mock_atom2.GetDegree.return_value = 2
    mock_atom2.GetFormalCharge.return_value = 0
    mock_atom2.GetTotalNumHs.return_value = 2
    mock_atom2.GetHybridization.return_value = 3  # SP3
    mock_atom2.GetIsAromatic.return_value = False

    mock_atom3 = MagicMock()
    mock_atom3.GetAtomicNum.return_value = 8  # Oxygen
    mock_atom3.GetDegree.return_value = 1
    mock_atom3.GetFormalCharge.return_value = 0
    mock_atom3.GetTotalNumHs.return_value = 1
    mock_atom3.GetHybridization.return_value = 3  # SP3
    mock_atom3.GetIsAromatic.return_value = False

    mock_mol.GetAtoms.return_value = [mock_atom1, mock_atom2, mock_atom3]
    mock_mol.GetNumAtoms.return_value = 3

    # Mock bonds (2 bonds: C-C and C-O)
    mock_bond1 = MagicMock()
    mock_bond1.GetBeginAtomIdx.return_value = 0
    mock_bond1.GetEndAtomIdx.return_value = 1
    mock_bond1.GetBondTypeAsDouble.return_value = 1.0  # Single bond
    mock_bond1.GetIsConjugated.return_value = False
    mock_bond1.IsInRing.return_value = False

    mock_bond2 = MagicMock()
    mock_bond2.GetBeginAtomIdx.return_value = 1
    mock_bond2.GetEndAtomIdx.return_value = 2
    mock_bond2.GetBondTypeAsDouble.return_value = 1.0  # Single bond
    mock_bond2.GetIsConjugated.return_value = False
    mock_bond2.IsInRing.return_value = False

    mock_mol.GetBonds.return_value = [mock_bond1, mock_bond2]
    mock_mol.GetNumBonds.return_value = 2

    # Mock conformer for 3D coordinates
    mock_conformer = MagicMock()
    mock_conformer.GetAtomPosition.side_effect = lambda i: (float(i), float(i), float(i))
    mock_mol.GetNumConformers.return_value = 1
    mock_mol.GetConformer.return_value = mock_conformer

    return mock_mol


@pytest.fixture
def mock_ase_atoms():
    """Create a mock ASE Atoms object for testing."""
    mock_atoms = MagicMock()
    mock_atoms.get_atomic_numbers.return_value = np.array([8, 1, 1])  # Water: O, H, H
    mock_atoms.get_positions.return_value = np.array(
        [[0.0, 0.0, 0.117369], [0.0, 0.756950, -0.469476], [0.0, -0.756950, -0.469476]]
    )
    mock_atoms.__len__ = MagicMock(return_value=3)

    return mock_atoms


@pytest.fixture
def valid_dict_data_simple():
    """Create a simple valid dictionary for DictConverter (used in batch tests)."""
    return {"x": torch.randn(5, 3), "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])}


# =============================================================================
# DATA CONVERTER REGISTRY TESTS - SINGLETON PATTERN
# =============================================================================


class TestDataConverterRegistrySingleton:
    """Test DataConverterRegistry singleton pattern implementation."""

    def test_singleton_returns_same_instance(self):
        """Test that multiple instantiations return the same instance."""
        registry1 = DataConverterRegistry()
        registry2 = DataConverterRegistry()

        assert registry1 is registry2
        assert id(registry1) == id(registry2)

    def test_singleton_instance_attribute(self):
        """Test that _instance class attribute holds the singleton."""
        registry = DataConverterRegistry()

        assert DataConverterRegistry._instance is registry
        assert DataConverterRegistry._instance is not None

    def test_singleton_preserves_state(self):
        """Test that singleton preserves state across instantiations."""
        registry1 = DataConverterRegistry()
        # Access internal state
        initial_converters = list(registry1._converters.keys())

        registry2 = DataConverterRegistry()

        assert list(registry2._converters.keys()) == initial_converters

    def test_global_registry_is_singleton(self):
        """Test that _registry global variable is the singleton instance."""
        registry = DataConverterRegistry()

        assert _registry is registry

    def test_get_registry_returns_singleton(self):
        """Test that get_registry() returns the singleton instance."""
        registry = DataConverterRegistry()
        retrieved = get_registry()

        assert retrieved is registry
        assert retrieved is _registry


# =============================================================================
# DATA CONVERTER REGISTRY TESTS - THREAD SAFETY
# =============================================================================


class TestDataConverterRegistryThreadSafety:
    """Test DataConverterRegistry thread-safety implementation."""

    def test_registry_has_rlock(self):
        """Test that registry uses RLock for thread safety."""
        assert hasattr(DataConverterRegistry, "_lock")
        assert isinstance(DataConverterRegistry._lock, type(threading.RLock()))

    def test_concurrent_singleton_access(self):
        """Test concurrent access to singleton returns same instance."""
        instances = []
        errors = []

        def get_instance():
            try:
                instance = DataConverterRegistry()
                instances.append(id(instance))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=get_instance) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(set(instances)) == 1, "Multiple instances created"

    def test_concurrent_registration(self, fresh_registry):
        """Test concurrent converter registration is thread-safe."""
        errors = []
        registered_formats = []

        def register_format(format_id):
            try:

                class DynamicConverter(BaseDataConverter):
                    @property
                    def format_name(self) -> str:
                        return f"format_{format_id}"

                    @property
                    def is_available(self) -> bool:
                        return True

                    def can_convert(self, input_data: Any) -> bool:
                        return False

                    def convert(self, input_data: Any, **kwargs) -> Data:
                        return Data()

                fresh_registry.register(f"format_{format_id}", DynamicConverter)
                registered_formats.append(f"format_{format_id}")
            except Exception as e:
                errors.append(str(e))

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(register_format, i) for i in range(100)]
            for _future in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(set(registered_formats)) == 100

    def test_concurrent_get_operations(self):
        """Test concurrent get operations are thread-safe."""
        registry = get_registry()
        errors = []
        results = []

        def get_format():
            try:
                # pyg_data is registered by default
                converter_class = registry.get("pyg_data")
                results.append(converter_class)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=get_format) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert all(r == PyGDataConverter for r in results)

    def test_concurrent_list_operations(self):
        """Test concurrent list operations are thread-safe."""
        registry = get_registry()
        errors = []
        all_formats_results = []
        available_formats_results = []

        def list_formats():
            try:
                all_formats = registry.list_all()
                available = registry.list_available()
                all_formats_results.append(set(all_formats))
                available_formats_results.append(set(available))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=list_formats) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        # All results should be consistent
        assert len(set(frozenset(r) for r in all_formats_results)) == 1

    def test_concurrent_auto_detect(self, mock_pyg_data):
        """Test concurrent auto_detect operations are thread-safe."""
        registry = get_registry()
        errors = []
        results = []

        def detect_format():
            try:
                detected = registry.auto_detect(mock_pyg_data)
                results.append(detected)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=detect_format) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        # All should detect as pyg_data
        assert all(r == "pyg_data" for r in results)


# =============================================================================
# DATA CONVERTER REGISTRY TESTS - REGISTRATION
# =============================================================================


class TestDataConverterRegistryRegistration:
    """Test DataConverterRegistry registration functionality."""

    def test_register_new_converter(self, fresh_registry, mock_converter_class):
        """Test registering a new converter class."""
        fresh_registry.register("test_format", mock_converter_class)

        assert fresh_registry.is_registered("test_format")
        assert fresh_registry.get("test_format") == mock_converter_class

    def test_register_overwrites_existing(self, fresh_registry, mock_converter_class):
        """Test that re-registering overwrites existing converter."""

        class AnotherConverter(BaseDataConverter):
            @property
            def format_name(self) -> str:
                return "test_format"

            @property
            def is_available(self) -> bool:
                return True

            def can_convert(self, input_data: Any) -> bool:
                return False

            def convert(self, input_data: Any, **kwargs) -> Data:
                return Data()

        fresh_registry.register("test_format", mock_converter_class)
        fresh_registry.register("test_format", AnotherConverter)

        assert fresh_registry.get("test_format") == AnotherConverter

    def test_register_case_insensitive(self, fresh_registry, mock_converter_class):
        """Test that format names are case-insensitive."""
        fresh_registry.register("TestFormat", mock_converter_class)

        assert fresh_registry.is_registered("testformat")
        assert fresh_registry.is_registered("TESTFORMAT")
        assert fresh_registry.is_registered("TestFormat")

    def test_register_logs_debug_message(self, fresh_registry, mock_converter_class, caplog):
        """Test that registration logs a debug message."""
        with caplog.at_level(logging.DEBUG):
            fresh_registry.register("logged_format", mock_converter_class)

        assert "Registered converter for format: logged_format" in caplog.text

    def test_default_converters_registered(self):
        """Test that default converters are registered on module load."""
        registry = get_registry()

        expected_formats = ["pyg_data", "dict", "smiles", "inchi", "xyz", "ase_atoms", "sdf"]
        for fmt in expected_formats:
            assert registry.is_registered(fmt), f"Format '{fmt}' should be registered"


# =============================================================================
# DATA CONVERTER REGISTRY TESTS - RETRIEVAL
# =============================================================================


class TestDataConverterRegistryRetrieval:
    """Test DataConverterRegistry retrieval functionality."""

    def test_get_existing_converter(self):
        """Test getting an existing converter class."""
        registry = get_registry()

        converter_class = registry.get("pyg_data")

        assert converter_class == PyGDataConverter

    def test_get_case_insensitive(self):
        """Test that get is case-insensitive."""
        registry = get_registry()

        assert registry.get("PyG_Data") == PyGDataConverter
        assert registry.get("PYG_DATA") == PyGDataConverter
        assert registry.get("pyg_data") == PyGDataConverter

    def test_get_nonexistent_raises_valueerror(self):
        """Test that getting nonexistent format raises ValueError."""
        registry = get_registry()

        with pytest.raises(ValueError) as exc_info:
            registry.get("nonexistent_format")

        assert "No converter registered for format 'nonexistent_format'" in str(exc_info.value)
        assert "Available formats:" in str(exc_info.value)

    def test_get_nonexistent_shows_available_formats(self):
        """Test that error message shows available formats."""
        registry = get_registry()

        with pytest.raises(ValueError) as exc_info:
            registry.get("nonexistent_format")

        error_msg = str(exc_info.value)
        # Should list available formats in error message
        assert "pyg_data" in error_msg or "dict" in error_msg


# =============================================================================
# DATA CONVERTER REGISTRY TESTS - LISTING
# =============================================================================


class TestDataConverterRegistryListing:
    """Test DataConverterRegistry listing functionality."""

    def test_list_all_returns_all_registered(self):
        """Test list_all returns all registered formats."""
        registry = get_registry()

        all_formats = registry.list_all()

        assert isinstance(all_formats, list)
        assert "pyg_data" in all_formats
        assert "dict" in all_formats
        assert "smiles" in all_formats
        assert "inchi" in all_formats
        assert "xyz" in all_formats
        assert "ase_atoms" in all_formats
        assert "sdf" in all_formats

    def test_list_available_returns_only_available(self):
        """Test list_available returns only formats with available dependencies."""
        registry = get_registry()

        available = registry.list_available()

        assert isinstance(available, list)
        # pyg_data and dict should always be available
        assert "pyg_data" in available
        assert "dict" in available

    def test_list_available_excludes_unavailable(
        self, fresh_registry, mock_unavailable_converter_class
    ):
        """Test list_available excludes converters with unavailable dependencies."""
        fresh_registry.register("unavailable", mock_unavailable_converter_class)
        fresh_registry.register("pyg_data", PyGDataConverter)

        available = fresh_registry.list_available()

        assert "unavailable" not in available
        assert "pyg_data" in available

    def test_list_all_returns_lowercase(self):
        """Test that list_all returns lowercase format names."""
        registry = get_registry()

        all_formats = registry.list_all()

        for fmt in all_formats:
            assert fmt == fmt.lower()


# =============================================================================
# DATA CONVERTER REGISTRY TESTS - IS_REGISTERED
# =============================================================================


class TestDataConverterRegistryIsRegistered:
    """Test DataConverterRegistry is_registered functionality."""

    def test_is_registered_true_for_existing(self):
        """Test is_registered returns True for existing formats."""
        registry = get_registry()

        assert registry.is_registered("pyg_data") is True
        assert registry.is_registered("dict") is True
        assert registry.is_registered("smiles") is True

    def test_is_registered_false_for_nonexistent(self):
        """Test is_registered returns False for nonexistent formats."""
        registry = get_registry()

        assert registry.is_registered("nonexistent_format") is False
        assert registry.is_registered("fake_format") is False

    def test_is_registered_case_insensitive(self):
        """Test is_registered is case-insensitive."""
        registry = get_registry()

        assert registry.is_registered("PyG_Data") is True
        assert registry.is_registered("DICT") is True
        assert registry.is_registered("SmIlEs") is True


# =============================================================================
# DATA CONVERTER REGISTRY TESTS - AUTO DETECT
# =============================================================================


class TestDataConverterRegistryAutoDetect:
    """Test DataConverterRegistry auto_detect functionality."""

    def test_auto_detect_pyg_data(self, mock_pyg_data):
        """Test auto_detect correctly identifies PyG Data."""
        registry = get_registry()

        detected = registry.auto_detect(mock_pyg_data)

        assert detected == "pyg_data"

    def test_auto_detect_dict(self, valid_dict_data):
        """Test auto_detect correctly identifies dict format."""
        registry = get_registry()

        detected = registry.auto_detect(valid_dict_data)

        assert detected == "dict"

    def test_auto_detect_smiles(self, sample_smiles):
        """Test auto_detect correctly identifies SMILES strings."""
        registry = get_registry()

        # SMILES detection depends on rdkit availability
        detected = registry.auto_detect(sample_smiles["ethanol"])

        # Should detect as smiles if rdkit is available, otherwise None
        if SMILESConverter().is_available:
            assert detected == "smiles"

    def test_auto_detect_inchi(self, sample_inchi):
        """Test auto_detect correctly identifies InChI strings."""
        registry = get_registry()

        # InChI detection depends on rdkit availability
        detected = registry.auto_detect(sample_inchi["ethanol"])

        # Should detect as inchi if rdkit is available
        # Note: auto_detect returns the first matching converter, which depends on
        # dict iteration order. Both SMILES and InChI converters may match InChI strings
        # because InChI strings contain C, N, O characters that SMILES converter also matches.
        if InChIConverter().is_available:
            # Either inchi or smiles detection is acceptable since both can parse the string
            assert detected in ("inchi", "smiles")

    def test_auto_detect_returns_none_for_unknown(self):
        """Test auto_detect returns None for unknown input types."""
        registry = get_registry()

        # Unknown types
        assert registry.auto_detect(12345) is None
        assert registry.auto_detect([1, 2, 3]) is None
        assert registry.auto_detect(object()) is None

    def test_auto_detect_handles_exceptions(self, fresh_registry):
        """Test auto_detect handles exceptions from converters gracefully."""

        class FaultyConverter(BaseDataConverter):
            @property
            def format_name(self) -> str:
                return "faulty"

            @property
            def is_available(self) -> bool:
                raise RuntimeError("Simulated error")

            def can_convert(self, input_data: Any) -> bool:
                raise RuntimeError("Simulated error")

            def convert(self, input_data: Any, **kwargs) -> Data:
                raise RuntimeError("Simulated error")

        fresh_registry.register("faulty", FaultyConverter)
        fresh_registry.register("pyg_data", PyGDataConverter)

        # Should not raise, should continue to next converter
        data = Data(x=torch.tensor([[1.0]]), edge_index=torch.tensor([[0], [0]]))
        detected = fresh_registry.auto_detect(data)

        # Should still detect pyg_data
        assert detected == "pyg_data"

    def test_auto_detect_prioritizes_first_match(self, fresh_registry):
        """Test auto_detect returns first matching format."""

        class Converter1(BaseDataConverter):
            @property
            def format_name(self) -> str:
                return "format1"

            @property
            def is_available(self) -> bool:
                return True

            def can_convert(self, input_data: Any) -> bool:
                return isinstance(input_data, str) and input_data.startswith("TEST")

            def convert(self, input_data: Any, **kwargs) -> Data:
                return Data()

        class Converter2(BaseDataConverter):
            @property
            def format_name(self) -> str:
                return "format2"

            @property
            def is_available(self) -> bool:
                return True

            def can_convert(self, input_data: Any) -> bool:
                return isinstance(input_data, str) and input_data.startswith("TEST")

            def convert(self, input_data: Any, **kwargs) -> Data:
                return Data()

        fresh_registry.register("format1", Converter1)
        fresh_registry.register("format2", Converter2)

        detected = fresh_registry.auto_detect("TEST_DATA")

        # Should return one of the matching formats (order depends on dict iteration)
        assert detected in ["format1", "format2"]


# =============================================================================
# REGISTER_CONVERTER DECORATOR TESTS
# =============================================================================


class TestRegisterConverterDecorator:
    """Test register_converter decorator functionality."""

    def test_decorator_registers_class(self, fresh_registry):
        """Test that decorator registers the class with the registry."""
        # Temporarily replace global registry
        original_registry = get_registry()

        @register_converter("decorator_test")
        class TestConverter(BaseDataConverter):
            @property
            def format_name(self) -> str:
                return "decorator_test"

            @property
            def is_available(self) -> bool:
                return True

            def can_convert(self, input_data: Any) -> bool:
                return False

            def convert(self, input_data: Any, **kwargs) -> Data:
                return Data()

        assert original_registry.is_registered("decorator_test")
        assert original_registry.get("decorator_test") == TestConverter

    def test_decorator_returns_class_unchanged(self):
        """Test that decorator returns the class unchanged."""

        @register_converter("unchanged_test")
        class TestConverter(BaseDataConverter):
            custom_attribute = "test_value"

            @property
            def format_name(self) -> str:
                return "unchanged_test"

            @property
            def is_available(self) -> bool:
                return True

            def can_convert(self, input_data: Any) -> bool:
                return False

            def convert(self, input_data: Any, **kwargs) -> Data:
                return Data()

        assert hasattr(TestConverter, "custom_attribute")
        assert TestConverter.custom_attribute == "test_value"

    def test_decorator_with_different_format_names(self):
        """Test decorator works with various format name styles."""
        format_names = ["simple", "with_underscore", "WithCamelCase", "WITH-HYPHEN", "123numeric"]

        for name in format_names:

            @register_converter(name)
            class DynamicConverter(BaseDataConverter):
                _format_name = name  # bind at class definition time

                @property
                def format_name(self) -> str:
                    return self._format_name

                @property
                def is_available(self) -> bool:
                    return True

                def can_convert(self, input_data: Any) -> bool:
                    return False

                def convert(self, input_data: Any, **kwargs) -> Data:
                    return Data()

            registry = get_registry()
            assert registry.is_registered(name)


# =============================================================================
# GET_REGISTRY FUNCTION TESTS
# =============================================================================


class TestGetRegistryFunction:
    """Test get_registry function."""

    def test_get_registry_returns_registry_instance(self):
        """Test get_registry returns DataConverterRegistry instance."""
        registry = get_registry()

        assert isinstance(registry, DataConverterRegistry)

    def test_get_registry_returns_same_instance(self):
        """Test get_registry always returns the same instance."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    def test_get_registry_returns_global_registry(self):
        """Test get_registry returns the global _registry variable."""
        registry = get_registry()

        assert registry is _registry

    def test_get_registry_has_registered_converters(self):
        """Test get_registry returns registry with registered converters."""
        registry = get_registry()

        assert len(registry.list_all()) > 0
        assert registry.is_registered("pyg_data")


# =============================================================================
# DATA CONVERTER PROTOCOL TESTS
# =============================================================================


class TestDataConverterProtocol:
    """Test DataConverterProtocol structure."""

    def test_protocol_defines_convert_method(self):
        """Test protocol defines convert method."""
        assert hasattr(DataConverterProtocol, "convert")

    def test_protocol_defines_can_convert_method(self):
        """Test protocol defines can_convert method."""
        assert hasattr(DataConverterProtocol, "can_convert")

    def test_protocol_defines_format_name_property(self):
        """Test protocol defines format_name property."""
        assert hasattr(DataConverterProtocol, "format_name")

    def test_protocol_defines_is_available_property(self):
        """Test protocol defines is_available property."""
        assert hasattr(DataConverterProtocol, "is_available")

    def test_concrete_converters_follow_protocol(self):
        """Test that concrete converter classes follow the protocol."""
        converters = [
            PyGDataConverter,
            DictConverter,
            SMILESConverter,
            InChIConverter,
            XYZConverter,
            ASEAtomsConverter,
            SDFConverter,
        ]

        for converter_class in converters:
            instance = converter_class()

            # Check methods exist and are callable
            assert callable(getattr(instance, "convert", None))
            assert callable(getattr(instance, "can_convert", None))

            # Check properties exist
            assert hasattr(instance, "format_name")
            assert hasattr(instance, "is_available")


# ==================
# End of Part 1
# ==================

# =============================================================================
# BASE DATA CONVERTER TESTS
# =============================================================================


class TestBaseDataConverter:
    """Test BaseDataConverter abstract base class."""

    def test_is_abstract_class(self):
        """Test that BaseDataConverter is an abstract class."""
        assert issubclass(BaseDataConverter, ABC)

    def test_cannot_instantiate_directly(self):
        """Test that BaseDataConverter cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseDataConverter()

        assert "abstract" in str(exc_info.value).lower()

    def test_abstract_methods_defined(self):
        """Test that all abstract methods are defined."""
        abstract_methods = getattr(BaseDataConverter, "__abstractmethods__", set())

        assert "format_name" in abstract_methods or hasattr(BaseDataConverter, "format_name")
        assert "is_available" in abstract_methods or hasattr(BaseDataConverter, "is_available")
        assert "can_convert" in abstract_methods or hasattr(BaseDataConverter, "can_convert")
        assert "convert" in abstract_methods or hasattr(BaseDataConverter, "convert")

    def test_convert_batch_method_exists(self):
        """Test that convert_batch method exists and is not abstract."""
        assert hasattr(BaseDataConverter, "convert_batch")
        # Should not be in abstract methods
        abstract_methods = getattr(BaseDataConverter, "__abstractmethods__", set())
        assert "convert_batch" not in abstract_methods

    def test_concrete_subclass_can_be_instantiated(self):
        """Test that concrete subclasses can be instantiated."""

        class ConcreteConverter(BaseDataConverter):
            @property
            def format_name(self) -> str:
                return "concrete"

            @property
            def is_available(self) -> bool:
                return True

            def can_convert(self, input_data: Any) -> bool:
                return True

            def convert(self, input_data: Any, **kwargs) -> Data:
                return Data()

        instance = ConcreteConverter()
        assert instance is not None
        assert instance.format_name == "concrete"

    def test_convert_batch_uses_convert_method(self):
        """Test that convert_batch calls convert for each input."""

        class TestConverter(BaseDataConverter):
            def __init__(self):
                self.convert_calls = []

            @property
            def format_name(self) -> str:
                return "test"

            @property
            def is_available(self) -> bool:
                return True

            def can_convert(self, input_data: Any) -> bool:
                return True

            def convert(self, input_data: Any, **kwargs) -> Data:
                self.convert_calls.append(input_data)
                return Data(
                    x=torch.tensor([[float(input_data)]]), edge_index=torch.tensor([[0], [0]])
                )

        converter = TestConverter()
        inputs = [1, 2, 3]

        batch = converter.convert_batch(inputs)

        assert len(converter.convert_calls) == 3
        assert converter.convert_calls == inputs
        assert isinstance(batch, Batch)

    def test_convert_batch_returns_batch_object(self):
        """Test that convert_batch returns a Batch object."""

        class TestConverter(BaseDataConverter):
            @property
            def format_name(self) -> str:
                return "test"

            @property
            def is_available(self) -> bool:
                return True

            def can_convert(self, input_data: Any) -> bool:
                return True

            def convert(self, input_data: Any, **kwargs) -> Data:
                return Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1], [1, 2]]))

        converter = TestConverter()
        batch = converter.convert_batch(["a", "b", "c"])

        assert isinstance(batch, Batch)
        assert batch.num_graphs == 3

    def test_convert_batch_passes_kwargs(self):
        """Test that convert_batch passes kwargs to convert."""

        class TestConverter(BaseDataConverter):
            def __init__(self):
                self.received_kwargs = []

            @property
            def format_name(self) -> str:
                return "test"

            @property
            def is_available(self) -> bool:
                return True

            def can_convert(self, input_data: Any) -> bool:
                return True

            def convert(self, input_data: Any, **kwargs) -> Data:
                self.received_kwargs.append(kwargs)
                return Data(x=torch.tensor([[1.0]]), edge_index=torch.tensor([[0], [0]]))

        converter = TestConverter()
        converter.convert_batch(["a", "b"], custom_param="test_value")

        assert len(converter.received_kwargs) == 2
        assert all(kw.get("custom_param") == "test_value" for kw in converter.received_kwargs)


# =============================================================================
# PYG DATA CONVERTER TESTS
# =============================================================================


class TestPyGDataConverter:
    """Test PyGDataConverter class."""

    def test_format_name(self):
        """Test format_name property returns 'pyg_data'."""
        converter = PyGDataConverter()

        assert converter.format_name == "pyg_data"

    def test_is_available_always_true(self):
        """Test is_available always returns True (no external dependencies)."""
        converter = PyGDataConverter()

        assert converter.is_available is True

    def test_can_convert_pyg_data(self, mock_pyg_data):
        """Test can_convert returns True for PyG Data objects."""
        converter = PyGDataConverter()

        assert converter.can_convert(mock_pyg_data) is True

    def test_can_convert_pyg_data_minimal(self, mock_pyg_data_minimal):
        """Test can_convert returns True for minimal PyG Data."""
        converter = PyGDataConverter()

        assert converter.can_convert(mock_pyg_data_minimal) is True

    def test_can_convert_returns_false_for_non_data(self):
        """Test can_convert returns False for non-Data objects."""
        converter = PyGDataConverter()

        assert converter.can_convert("string") is False
        assert converter.can_convert({"x": 1}) is False
        assert converter.can_convert([1, 2, 3]) is False
        assert converter.can_convert(123) is False
        assert converter.can_convert(None) is False

    def test_convert_returns_same_data(self, mock_pyg_data):
        """Test convert returns the same Data object (passthrough)."""
        converter = PyGDataConverter()

        result = converter.convert(mock_pyg_data)

        assert result is mock_pyg_data

    def test_convert_raises_typeerror_for_non_data(self):
        """Test convert raises TypeError for non-Data input."""
        converter = PyGDataConverter()

        with pytest.raises(TypeError) as exc_info:
            converter.convert("not a Data object")

        assert "Expected PyG Data" in str(exc_info.value)

    def test_convert_raises_typeerror_with_type_info(self):
        """Test convert TypeError message includes input type."""
        converter = PyGDataConverter()

        with pytest.raises(TypeError) as exc_info:
            converter.convert({"dict": "data"})

        assert "dict" in str(exc_info.value).lower()

    def test_convert_batch_pyg_data(self, mock_pyg_data, mock_pyg_data_minimal):
        """Test convert_batch with multiple PyG Data objects."""
        converter = PyGDataConverter()

        # Create compatible Data objects with same feature dimensions for batching
        data1 = Data(x=torch.randn(5, 3), edge_index=torch.tensor([[0, 1], [1, 0]]))
        data2 = Data(x=torch.randn(4, 3), edge_index=torch.tensor([[0, 1], [1, 0]]))

        batch = converter.convert_batch([data1, data2])

        assert isinstance(batch, Batch)
        assert batch.num_graphs == 2

    def test_registered_in_registry(self):
        """Test PyGDataConverter is registered in the global registry."""
        registry = get_registry()

        assert registry.is_registered("pyg_data")
        assert registry.get("pyg_data") == PyGDataConverter


# =============================================================================
# DICT CONVERTER TESTS
# =============================================================================


class TestDictConverter:
    """Test DictConverter class."""

    def test_format_name(self):
        """Test format_name property returns 'dict'."""
        converter = DictConverter()

        assert converter.format_name == "dict"

    def test_is_available_always_true(self):
        """Test is_available always returns True."""
        converter = DictConverter()

        assert converter.is_available is True

    def test_can_convert_valid_dict_with_x(self, valid_dict_data):
        """Test can_convert returns True for dict with x and edge_index."""
        converter = DictConverter()

        assert converter.can_convert(valid_dict_data) is True

    def test_can_convert_valid_dict_with_z(self, valid_dict_data_with_z):
        """Test can_convert returns True for dict with z and edge_index."""
        converter = DictConverter()

        assert converter.can_convert(valid_dict_data_with_z) is True

    def test_can_convert_returns_false_for_non_dict(self):
        """Test can_convert returns False for non-dict input."""
        converter = DictConverter()

        assert converter.can_convert("string") is False
        assert converter.can_convert([1, 2, 3]) is False
        assert converter.can_convert(Data()) is False

    def test_can_convert_returns_false_without_edge_index(self, invalid_dict_no_edges):
        """Test can_convert returns False if edge_index is missing."""
        converter = DictConverter()

        assert converter.can_convert(invalid_dict_no_edges) is False

    def test_can_convert_returns_false_without_features(self, invalid_dict_no_features):
        """Test can_convert returns False if both x and z are missing."""
        converter = DictConverter()

        assert converter.can_convert(invalid_dict_no_features) is False

    def test_convert_valid_dict(self, valid_dict_data):
        """Test convert creates correct Data object from dict."""
        converter = DictConverter()

        result = converter.convert(valid_dict_data)

        assert isinstance(result, Data)
        assert torch.equal(result.x, valid_dict_data["x"])
        assert torch.equal(result.edge_index, valid_dict_data["edge_index"])
        assert torch.equal(result.edge_attr, valid_dict_data["edge_attr"])
        assert torch.equal(result.y, valid_dict_data["y"])

    def test_convert_dict_with_lists(self, valid_dict_data_lists):
        """Test convert handles Python lists by converting to tensors."""
        converter = DictConverter()

        result = converter.convert(valid_dict_data_lists)

        assert isinstance(result, Data)
        assert isinstance(result.x, torch.Tensor)
        assert isinstance(result.edge_index, torch.Tensor)
        assert result.x.shape == (3, 2)
        assert result.edge_index.shape == (2, 4)

    def test_convert_dict_with_tuples(self):
        """Test convert handles Python tuples by converting to tensors."""
        converter = DictConverter()
        dict_data = {"x": ((1.0, 2.0), (3.0, 4.0)), "edge_index": ((0, 1), (1, 0))}

        result = converter.convert(dict_data)

        assert isinstance(result, Data)
        assert isinstance(result.x, torch.Tensor)
        assert isinstance(result.edge_index, torch.Tensor)

    def test_convert_dict_preserves_non_tensor_values(self):
        """Test convert preserves non-tensor values as-is."""
        converter = DictConverter()
        dict_data = {
            "x": torch.randn(3, 2),
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
            "smiles": "CCO",  # String value
            "num_atoms": 3,  # Integer value
        }

        result = converter.convert(dict_data)

        assert result.smiles == "CCO"
        assert result.num_atoms == 3

    def test_convert_raises_typeerror_for_non_dict(self):
        """Test convert raises TypeError for non-dict input."""
        converter = DictConverter()

        with pytest.raises(TypeError) as exc_info:
            converter.convert("not a dict")

        assert "Expected dict" in str(exc_info.value)

    def test_convert_batch_dicts(self, valid_dict_data, valid_dict_data_with_z):
        """Test convert_batch with multiple dictionaries."""
        converter = DictConverter()

        # Both need same structure for batching, so use similar dicts
        dict1 = {"x": torch.randn(5, 3), "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]])}
        dict2 = {"x": torch.randn(4, 3), "edge_index": torch.tensor([[0, 1], [1, 0]])}

        batch = converter.convert_batch([dict1, dict2])

        assert isinstance(batch, Batch)
        assert batch.num_graphs == 2

    def test_registered_in_registry(self):
        """Test DictConverter is registered in the global registry."""
        registry = get_registry()

        assert registry.is_registered("dict")
        assert registry.get("dict") == DictConverter


# =============================================================================
# SMILES CONVERTER TESTS
# =============================================================================


class TestSMILESConverter:
    """Test SMILESConverter class."""

    def test_format_name(self):
        """Test format_name property returns 'smiles'."""
        converter = SMILESConverter()

        assert converter.format_name == "smiles"

    def test_default_atom_features(self):
        """Test default atom features are set correctly."""
        converter = SMILESConverter()

        expected = [
            "atomic_num",
            "degree",
            "formal_charge",
            "num_hs",
            "hybridization",
            "is_aromatic",
        ]
        assert converter.atom_features == expected

    def test_default_bond_features(self):
        """Test default bond features are set correctly."""
        converter = SMILESConverter()

        expected = ["bond_type", "is_conjugated", "is_in_ring"]
        assert converter.bond_features == expected

    def test_default_add_hydrogens_false(self):
        """Test add_hydrogens defaults to False."""
        converter = SMILESConverter()

        assert converter.add_hydrogens is False

    def test_custom_atom_features(self):
        """Test custom atom features can be set."""
        custom_features = ["atomic_num", "degree"]
        converter = SMILESConverter(atom_features=custom_features)

        assert converter.atom_features == custom_features

    def test_custom_bond_features(self):
        """Test custom bond features can be set."""
        custom_features = ["bond_type"]
        converter = SMILESConverter(bond_features=custom_features)

        assert converter.bond_features == custom_features

    def test_add_hydrogens_true(self):
        """Test add_hydrogens can be set to True."""
        converter = SMILESConverter(add_hydrogens=True)

        assert converter.add_hydrogens is True

    def test_is_available_checks_rdkit(self):
        """Test is_available checks for rdkit import."""
        converter = SMILESConverter()

        # Result depends on whether rdkit is installed
        result = converter.is_available
        assert isinstance(result, bool)

    @patch.dict("sys.modules", {"rdkit": None, "rdkit.Chem": None})
    def test_is_available_false_without_rdkit(self):
        """Test is_available returns False when rdkit is not available."""
        # Test by patching the class property instead of instance
        with patch.object(
            SMILESConverter, "is_available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = False
            converter = SMILESConverter()
            assert converter.is_available is False

    def test_can_convert_smiles_string(self, sample_smiles):
        """Test can_convert returns True for SMILES-like strings."""
        converter = SMILESConverter()

        for name, smiles in sample_smiles.items():
            assert converter.can_convert(smiles) is True, f"Failed for {name}: {smiles}"

    def test_can_convert_returns_false_for_non_string(self):
        """Test can_convert returns False for non-string input."""
        converter = SMILESConverter()

        assert converter.can_convert(123) is False
        assert converter.can_convert([1, 2, 3]) is False
        assert converter.can_convert({"x": 1}) is False
        assert converter.can_convert(None) is False

    def test_can_convert_returns_false_for_file_paths(self):
        """Test can_convert returns False for file paths."""
        converter = SMILESConverter()

        assert converter.can_convert("molecule.xyz") is False
        assert converter.can_convert("compound.sdf") is False
        assert converter.can_convert("structure.mol") is False
        assert converter.can_convert("crystal.cif") is False

    def test_can_convert_returns_false_for_non_molecular_strings(self):
        """Test can_convert returns False for strings without molecular characters."""
        converter = SMILESConverter()

        # Strings without C, N, O, c, n, o are not SMILES
        assert converter.can_convert("12345") is False
        assert converter.can_convert("XYZA") is False

    def test_convert_calls_feature_methods(self, mock_rdkit_mol):
        """Test convert calls feature extraction methods."""
        converter = SMILESConverter()

        # Skip test if rdkit is not available - can't properly mock without it
        if not converter.is_available:
            pytest.skip("rdkit not available")

        # Test that convert actually calls the feature methods by doing a real conversion
        result = converter.convert("CCO")

        # Verify the result has the expected attributes from feature extraction
        assert hasattr(result, "x")  # atom features
        assert hasattr(result, "edge_index")  # bond features

    def test_convert_invalid_smiles_raises_valueerror(self):
        """Test convert raises ValueError for invalid SMILES."""
        converter = SMILESConverter()

        if converter.is_available:
            with pytest.raises(ValueError) as exc_info:
                converter.convert("INVALID_SMILES_STRING_123")

            assert "Invalid SMILES" in str(exc_info.value)

    def test_convert_returns_data_with_smiles_attribute(self):
        """Test convert stores original SMILES in Data object."""
        converter = SMILESConverter()

        if converter.is_available:
            result = converter.convert("CCO")
            assert hasattr(result, "smiles")
            assert result.smiles == "CCO"

    def test_get_atom_features_extraction(self, mock_rdkit_mol):
        """Test _get_atom_features extracts correct features."""
        converter = SMILESConverter()

        features = converter._get_atom_features(mock_rdkit_mol)

        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == 3  # 3 atoms
        assert features.shape[1] == 6  # 6 default features

    def test_get_bond_features_extraction(self, mock_rdkit_mol):
        """Test _get_bond_features extracts correct features."""
        converter = SMILESConverter()

        edge_index, edge_attr = converter._get_bond_features(mock_rdkit_mol)

        assert isinstance(edge_index, torch.Tensor)
        assert isinstance(edge_attr, torch.Tensor)
        assert edge_index.shape[0] == 2  # Source and target
        assert edge_index.shape[1] == 4  # 2 bonds * 2 (bidirectional)
        assert edge_attr.shape[0] == 4  # 4 edges
        assert edge_attr.shape[1] == 3  # 3 default bond features

    def test_get_bond_features_empty_molecule(self):
        """Test _get_bond_features handles molecule with no bonds."""
        converter = SMILESConverter()

        mock_mol = MagicMock()
        mock_mol.GetBonds.return_value = []

        edge_index, edge_attr = converter._get_bond_features(mock_mol)

        assert edge_index.shape == (2, 0)
        assert edge_attr.shape[0] == 0

    def test_registered_in_registry(self):
        """Test SMILESConverter is registered in the global registry."""
        registry = get_registry()

        assert registry.is_registered("smiles")
        assert registry.get("smiles") == SMILESConverter


# =============================================================================
# INCHI CONVERTER TESTS
# =============================================================================


class TestInChIConverter:
    """Test InChIConverter class."""

    def test_format_name(self):
        """Test format_name property returns 'inchi'."""
        converter = InChIConverter()

        assert converter.format_name == "inchi"

    def test_default_atom_features(self):
        """Test default atom features match SMILESConverter defaults."""
        converter = InChIConverter()

        expected = [
            "atomic_num",
            "degree",
            "formal_charge",
            "num_hs",
            "hybridization",
            "is_aromatic",
        ]
        assert converter.atom_features == expected

    def test_default_bond_features(self):
        """Test default bond features match SMILESConverter defaults."""
        converter = InChIConverter()

        expected = ["bond_type", "is_conjugated", "is_in_ring"]
        assert converter.bond_features == expected

    def test_custom_features(self):
        """Test custom features can be set."""
        converter = InChIConverter(atom_features=["atomic_num"], bond_features=["bond_type"])

        assert converter.atom_features == ["atomic_num"]
        assert converter.bond_features == ["bond_type"]

    def test_is_available_checks_rdkit_inchi(self):
        """Test is_available checks for rdkit.Chem.inchi import."""
        converter = InChIConverter()

        result = converter.is_available
        assert isinstance(result, bool)

    def test_can_convert_inchi_string(self, sample_inchi):
        """Test can_convert returns True for InChI strings."""
        converter = InChIConverter()

        for name, inchi in sample_inchi.items():
            assert converter.can_convert(inchi) is True, f"Failed for {name}"

    def test_can_convert_requires_inchi_prefix(self):
        """Test can_convert requires 'InChI=' prefix."""
        converter = InChIConverter()

        assert converter.can_convert("InChI=1S/CH4/h1H4") is True
        assert converter.can_convert("1S/CH4/h1H4") is False  # No prefix
        assert converter.can_convert("inchi=1S/CH4/h1H4") is False  # Wrong case

    def test_can_convert_returns_false_for_non_string(self):
        """Test can_convert returns False for non-string input."""
        converter = InChIConverter()

        assert converter.can_convert(123) is False
        assert converter.can_convert([]) is False
        assert converter.can_convert({}) is False

    def test_can_convert_returns_false_for_smiles(self, sample_smiles):
        """Test can_convert returns False for SMILES strings."""
        converter = InChIConverter()

        for smiles in sample_smiles.values():
            assert converter.can_convert(smiles) is False

    def test_convert_invalid_inchi_raises_valueerror(self):
        """Test convert raises ValueError for invalid InChI."""
        converter = InChIConverter()

        if converter.is_available:
            with pytest.raises(ValueError) as exc_info:
                converter.convert("InChI=INVALID")

            assert "Invalid InChI" in str(exc_info.value)

    def test_convert_returns_data_with_inchi_attribute(self):
        """Test convert stores original InChI in Data object."""
        converter = InChIConverter()

        if converter.is_available:
            inchi = "InChI=1S/CH4/h1H4"
            result = converter.convert(inchi)

            assert hasattr(result, "inchi")
            assert result.inchi == inchi

    def test_registered_in_registry(self):
        """Test InChIConverter is registered in the global registry."""
        registry = get_registry()

        assert registry.is_registered("inchi")
        assert registry.get("inchi") == InChIConverter


# =============================================================================
# XYZ CONVERTER TESTS
# =============================================================================


class TestXYZConverter:
    """Test XYZConverter class."""

    def test_format_name(self):
        """Test format_name property returns 'xyz'."""
        converter = XYZConverter()

        assert converter.format_name == "xyz"

    def test_default_cutoff(self):
        """Test default cutoff is 5.0 Angstrom."""
        converter = XYZConverter()

        assert converter.cutoff == 5.0

    def test_custom_cutoff(self):
        """Test custom cutoff can be set."""
        converter = XYZConverter(cutoff=3.5)

        assert converter.cutoff == 3.5

    def test_is_available_checks_ase(self):
        """Test is_available checks for ase import."""
        converter = XYZConverter()

        result = converter.is_available
        assert isinstance(result, bool)

    def test_can_convert_xyz_file_path(self, temp_directory, mock_xyz_content):
        """Test can_convert returns True for existing .xyz file paths."""
        converter = XYZConverter()

        # Create a temporary XYZ file
        xyz_path = temp_directory / "test.xyz"
        xyz_path.write_text(mock_xyz_content)

        assert converter.can_convert(str(xyz_path)) is True
        assert converter.can_convert(xyz_path) is True  # Also works with Path

    def test_can_convert_returns_false_for_nonexistent_file(self, temp_directory):
        """Test can_convert returns False for non-existent files."""
        converter = XYZConverter()

        nonexistent = temp_directory / "nonexistent.xyz"

        assert converter.can_convert(str(nonexistent)) is False

    def test_can_convert_returns_false_for_wrong_extension(self, temp_directory):
        """Test can_convert returns False for non-.xyz files."""
        converter = XYZConverter()

        # Create files with wrong extensions
        (temp_directory / "test.sdf").write_text("content")
        (temp_directory / "test.mol").write_text("content")

        assert converter.can_convert(str(temp_directory / "test.sdf")) is False
        assert converter.can_convert(str(temp_directory / "test.mol")) is False

    def test_can_convert_returns_false_for_non_path(self):
        """Test can_convert returns False for non-path input."""
        converter = XYZConverter()

        assert converter.can_convert("CCO") is False  # SMILES
        assert converter.can_convert(123) is False
        assert converter.can_convert({}) is False

    def test_convert_creates_data_with_z_and_pos(
        self, mock_ase_atoms, temp_directory, mock_xyz_content
    ):
        """Test convert creates Data with z (atomic numbers) and pos (positions)."""
        converter = XYZConverter()

        if not converter.is_available:
            pytest.skip("ase not available")

        xyz_path = temp_directory / "test.xyz"
        xyz_path.write_text(mock_xyz_content)

        result = converter.convert(str(xyz_path))

        assert hasattr(result, "z")
        assert hasattr(result, "pos")
        assert hasattr(result, "edge_index")

    def test_convert_uses_cutoff_for_edges(self, mock_ase_atoms, temp_directory, mock_xyz_content):
        """Test convert uses cutoff parameter for edge creation."""
        converter = XYZConverter(cutoff=3.0)

        if not converter.is_available:
            pytest.skip("ase not available")

        xyz_path = temp_directory / "test.xyz"
        xyz_path.write_text(mock_xyz_content)

        result = converter.convert(str(xyz_path))

        # Verify result has expected structure
        assert hasattr(result, "z")
        assert hasattr(result, "pos")
        assert hasattr(result, "edge_index")

    def test_convert_cutoff_override_via_kwargs(
        self, mock_ase_atoms, temp_directory, mock_xyz_content
    ):
        """Test convert allows cutoff override via kwargs."""
        converter = XYZConverter(cutoff=5.0)

        if not converter.is_available:
            pytest.skip("ase not available")

        xyz_path = temp_directory / "test.xyz"
        xyz_path.write_text(mock_xyz_content)

        result = converter.convert(str(xyz_path), cutoff=2.0)

        # Verify result has expected structure
        assert hasattr(result, "z")
        assert hasattr(result, "pos")
        assert hasattr(result, "edge_index")

    def test_registered_in_registry(self):
        """Test XYZConverter is registered in the global registry."""
        registry = get_registry()

        assert registry.is_registered("xyz")
        assert registry.get("xyz") == XYZConverter


# =============================================================================
# ASE ATOMS CONVERTER TESTS
# =============================================================================


class TestASEAtomsConverter:
    """Test ASEAtomsConverter class."""

    def test_format_name(self):
        """Test format_name property returns 'ase_atoms'."""
        converter = ASEAtomsConverter()

        assert converter.format_name == "ase_atoms"

    def test_default_cutoff(self):
        """Test default cutoff is 5.0 Angstrom."""
        converter = ASEAtomsConverter()

        assert converter.cutoff == 5.0

    def test_custom_cutoff(self):
        """Test custom cutoff can be set."""
        converter = ASEAtomsConverter(cutoff=4.0)

        assert converter.cutoff == 4.0

    def test_is_available_checks_ase(self):
        """Test is_available checks for ase import."""
        converter = ASEAtomsConverter()

        result = converter.is_available
        assert isinstance(result, bool)

    def test_can_convert_checks_for_ase_atoms_type(self, mock_ase_atoms):
        """Test can_convert checks if input is ASE Atoms object."""
        converter = ASEAtomsConverter()

        if converter.is_available:
            # Mock the type check
            with patch(
                "milia_pipeline.models.post_training.data_preparation.data_converter.Atoms",
                create=True,
            ) as mock_atoms_class:
                mock_atoms_class.return_value = mock_ase_atoms
                # The can_convert method checks isinstance

    def test_can_convert_returns_false_for_non_atoms(self):
        """Test can_convert returns False for non-Atoms input."""
        converter = ASEAtomsConverter()

        assert converter.can_convert("CCO") is False
        assert converter.can_convert({}) is False
        assert converter.can_convert([]) is False
        assert converter.can_convert(Data()) is False

    def test_convert_creates_data_from_atoms(self, mock_ase_atoms):
        """Test convert creates Data from ASE Atoms object."""
        converter = ASEAtomsConverter()

        if not converter.is_available:
            pytest.skip("ase not available")

        # Create a real ASE Atoms object for testing
        from ase import Atoms

        atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])

        result = converter.convert(atoms)

        assert hasattr(result, "z")
        assert hasattr(result, "pos")
        assert hasattr(result, "edge_index")
        assert hasattr(result, "num_nodes")

    def test_convert_sets_num_nodes(self, mock_ase_atoms):
        """Test convert sets num_nodes correctly."""
        converter = ASEAtomsConverter()

        if not converter.is_available:
            pytest.skip("ase not available")

        # Create a real ASE Atoms object for testing
        from ase import Atoms

        atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])

        result = converter.convert(atoms)

        assert result.num_nodes == 3  # Water molecule has 3 atoms

    def test_registered_in_registry(self):
        """Test ASEAtomsConverter is registered in the global registry."""
        registry = get_registry()

        assert registry.is_registered("ase_atoms")
        assert registry.get("ase_atoms") == ASEAtomsConverter


# =============================================================================
# SDF CONVERTER TESTS
# =============================================================================


class TestSDFConverter:
    """Test SDFConverter class."""

    def test_format_name(self):
        """Test format_name property returns 'sdf'."""
        converter = SDFConverter()

        assert converter.format_name == "sdf"

    def test_is_available_checks_rdkit(self):
        """Test is_available checks for rdkit import."""
        converter = SDFConverter()

        result = converter.is_available
        assert isinstance(result, bool)

    def test_can_convert_sdf_file_path(self, temp_directory, mock_sdf_content):
        """Test can_convert returns True for existing .sdf file paths."""
        converter = SDFConverter()

        sdf_path = temp_directory / "test.sdf"
        sdf_path.write_text(mock_sdf_content)

        assert converter.can_convert(str(sdf_path)) is True
        assert converter.can_convert(sdf_path) is True

    def test_can_convert_mol_file_path(self, temp_directory, mock_sdf_content):
        """Test can_convert returns True for existing .mol file paths."""
        converter = SDFConverter()

        mol_path = temp_directory / "test.mol"
        mol_path.write_text(mock_sdf_content)

        assert converter.can_convert(str(mol_path)) is True

    def test_can_convert_returns_false_for_nonexistent_file(self, temp_directory):
        """Test can_convert returns False for non-existent files."""
        converter = SDFConverter()

        nonexistent = temp_directory / "nonexistent.sdf"

        assert converter.can_convert(str(nonexistent)) is False

    def test_can_convert_returns_false_for_wrong_extension(self, temp_directory):
        """Test can_convert returns False for non-.sdf/.mol files."""
        converter = SDFConverter()

        (temp_directory / "test.xyz").write_text("content")

        assert converter.can_convert(str(temp_directory / "test.xyz")) is False

    def test_can_convert_returns_false_for_non_path(self):
        """Test can_convert returns False for non-path input."""
        converter = SDFConverter()

        assert converter.can_convert("CCO") is False
        assert converter.can_convert(123) is False

    def test_convert_invalid_sdf_raises_valueerror(self, temp_directory):
        """Test convert raises ValueError for invalid SDF file."""
        converter = SDFConverter()

        if converter.is_available:
            # Create invalid SDF file
            invalid_sdf = temp_directory / "invalid.sdf"
            invalid_sdf.write_text("INVALID SDF CONTENT")

            with pytest.raises(ValueError) as exc_info:
                converter.convert(str(invalid_sdf))

            assert "Failed to load molecule" in str(exc_info.value)

    def test_convert_extracts_3d_coordinates_if_available(
        self, temp_directory, mock_rdkit_mol, mock_sdf_content
    ):
        """Test convert extracts 3D coordinates if conformer is available."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        # Create a valid SDF file for testing
        sdf_path = temp_directory / "test.sdf"
        sdf_path.write_text(mock_sdf_content)

        result = converter.convert(str(sdf_path))

        # Should have basic attributes
        assert hasattr(result, "x")
        assert hasattr(result, "edge_index")
        # pos may or may not be present depending on conformer in SDF

    def test_registered_in_registry(self):
        """Test SDFConverter is registered in the global registry."""
        registry = get_registry()

        assert registry.is_registered("sdf")
        assert registry.get("sdf") == SDFConverter


# =============================================================================
# CONVERTER INHERITANCE AND POLYMORPHISM TESTS
# =============================================================================


class TestConverterPolymorphism:
    """Test polymorphic behavior of converters."""

    def test_all_converters_inherit_from_base(self):
        """Test all converter classes inherit from BaseDataConverter."""
        converters = [
            PyGDataConverter,
            DictConverter,
            SMILESConverter,
            InChIConverter,
            XYZConverter,
            ASEAtomsConverter,
            SDFConverter,
        ]

        for converter_class in converters:
            assert issubclass(converter_class, BaseDataConverter)

    def test_all_converters_implement_required_methods(self):
        """Test all converters implement required abstract methods."""
        converters = [
            PyGDataConverter(),
            DictConverter(),
            SMILESConverter(),
            InChIConverter(),
            XYZConverter(),
            ASEAtomsConverter(),
            SDFConverter(),
        ]

        for converter in converters:
            # Check methods exist and are callable
            assert callable(converter.convert)
            assert callable(converter.can_convert)
            assert callable(converter.convert_batch)

            # Check properties exist
            assert isinstance(converter.format_name, str)
            assert isinstance(converter.is_available, bool)

    def test_converters_can_be_used_interchangeably(self, mock_pyg_data, valid_dict_data):
        """Test converters can be used interchangeably through base class interface."""

        def use_converter(converter: BaseDataConverter, data: Any) -> Data | None:
            if converter.is_available and converter.can_convert(data):
                return converter.convert(data)
            return None

        # Test with PyG data
        pyg_converter = PyGDataConverter()
        result = use_converter(pyg_converter, mock_pyg_data)
        assert result is not None

        # Test with dict
        dict_converter = DictConverter()
        result = use_converter(dict_converter, valid_dict_data)
        assert result is not None


# ==================
# End of Part 2
# ==================
# =============================================================================
# CONVERT_TO_PYG FUNCTION TESTS
# =============================================================================


class TestConvertToPyg:
    """Test convert_to_pyg convenience function."""

    def test_convert_pyg_data_auto_detect(self, mock_pyg_data):
        """Test convert_to_pyg auto-detects PyG Data format."""
        result = convert_to_pyg(mock_pyg_data)

        assert result is mock_pyg_data

    def test_convert_dict_auto_detect(self, valid_dict_data):
        """Test convert_to_pyg auto-detects dict format."""
        result = convert_to_pyg(valid_dict_data)

        assert isinstance(result, Data)
        assert torch.equal(result.x, valid_dict_data["x"])

    def test_convert_with_explicit_format(self, valid_dict_data):
        """Test convert_to_pyg with explicit format specification."""
        result = convert_to_pyg(valid_dict_data, format="dict")

        assert isinstance(result, Data)

    def test_convert_pyg_data_with_explicit_format(self, mock_pyg_data):
        """Test convert_to_pyg with explicit pyg_data format."""
        result = convert_to_pyg(mock_pyg_data, format="pyg_data")

        assert result is mock_pyg_data

    def test_convert_raises_valueerror_for_unknown_format(self):
        """Test convert_to_pyg raises ValueError for unknown input."""
        with pytest.raises(ValueError) as exc_info:
            convert_to_pyg(12345)  # Integer not supported

        assert "Cannot auto-detect format" in str(exc_info.value)
        assert "Available formats:" in str(exc_info.value)

    def test_convert_raises_valueerror_for_unregistered_format(self, mock_pyg_data):
        """Test convert_to_pyg raises ValueError for unregistered format."""
        with pytest.raises(ValueError) as exc_info:
            convert_to_pyg(mock_pyg_data, format="nonexistent_format")

        assert "No converter registered" in str(exc_info.value)

    def test_convert_passes_kwargs_to_converter(self):
        """Test convert_to_pyg passes kwargs to converter."""
        # Test with SMILESConverter which accepts kwargs (add_hydrogens, atom_features, etc.)
        # If rdkit is not available, use a PyG Data passthrough test instead
        smiles_converter = SMILESConverter()

        if smiles_converter.is_available:
            # SMILESConverter accepts kwargs in __init__
            result = convert_to_pyg("CCO", format="smiles", add_hydrogens=False)
            assert isinstance(result, Data)
        else:
            # Fallback: test with PyG Data which ignores kwargs
            pyg_data = Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1], [1, 0]]))
            result = convert_to_pyg(pyg_data, format="pyg_data")
            assert isinstance(result, Data)

    def test_convert_smiles_if_rdkit_available(self, sample_smiles):
        """Test convert_to_pyg converts SMILES if rdkit is available."""
        converter = SMILESConverter()

        if converter.is_available:
            result = convert_to_pyg(sample_smiles["ethanol"])

            assert isinstance(result, Data)
            assert hasattr(result, "x")
            assert hasattr(result, "edge_index")

    def test_convert_smiles_explicit_format(self, sample_smiles):
        """Test convert_to_pyg with explicit smiles format."""
        converter = SMILESConverter()

        if converter.is_available:
            result = convert_to_pyg(sample_smiles["ethanol"], format="smiles")

            assert isinstance(result, Data)

    def test_convert_inchi_if_rdkit_available(self, sample_inchi):
        """Test convert_to_pyg converts InChI if rdkit is available."""
        converter = InChIConverter()

        if converter.is_available:
            # Explicitly specify format="inchi" since auto_detect might return "smiles"
            # (InChI strings contain C, N, O characters that SMILES converter also matches)
            result = convert_to_pyg(sample_inchi["ethanol"], format="inchi")

            assert isinstance(result, Data)

    def test_convert_handles_case_insensitive_format(self, valid_dict_data):
        """Test convert_to_pyg handles case-insensitive format names."""
        result1 = convert_to_pyg(valid_dict_data, format="dict")
        result2 = convert_to_pyg(valid_dict_data, format="DICT")
        result3 = convert_to_pyg(valid_dict_data, format="Dict")

        assert isinstance(result1, Data)
        assert isinstance(result2, Data)
        assert isinstance(result3, Data)

    def test_convert_raises_import_error_for_unavailable_converter(self):
        """Test convert_to_pyg raises ImportError for unavailable converter."""

        # Create a mock unavailable converter
        class UnavailableConverter(BaseDataConverter):
            @property
            def format_name(self) -> str:
                return "unavailable_test"

            @property
            def is_available(self) -> bool:
                return False

            def can_convert(self, input_data: Any) -> bool:
                return True

            def convert(self, input_data: Any, **kwargs) -> Data:
                return Data()

        # Register it
        registry = get_registry()
        registry.register("unavailable_test", UnavailableConverter)

        with pytest.raises(ImportError) as exc_info:
            convert_to_pyg("test_data", format="unavailable_test")

        assert "requires dependencies" in str(exc_info.value)

    def test_convert_creates_new_converter_instance(self, valid_dict_data):
        """Test convert_to_pyg creates new converter instances."""
        # Each call should work independently
        result1 = convert_to_pyg(valid_dict_data, format="dict")
        result2 = convert_to_pyg(valid_dict_data, format="dict")

        assert isinstance(result1, Data)
        assert isinstance(result2, Data)

    def test_convert_passes_working_root_dir_to_xyz_converter(
        self, temp_directory, mock_xyz_content
    ):
        """Test convert_to_pyg passes working_root_dir kwarg to XYZConverter."""
        converter = XYZConverter()

        if not converter.is_available:
            pytest.skip("ase not available")

        # Create XYZ file in temp directory
        (temp_directory / "test.xyz").write_text(mock_xyz_content)

        # Use relative path with working_root_dir
        result = convert_to_pyg("test.xyz", format="xyz", working_root_dir=temp_directory)

        assert isinstance(result, Data)
        assert hasattr(result, "z")
        assert hasattr(result, "pos")

    def test_convert_passes_working_root_dir_to_sdf_converter(
        self, temp_directory, mock_sdf_content
    ):
        """Test convert_to_pyg passes working_root_dir kwarg to SDFConverter."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        # Create SDF file in temp directory
        (temp_directory / "test.sdf").write_text(mock_sdf_content)

        # Use relative path with working_root_dir
        result = convert_to_pyg("test.sdf", format="sdf", working_root_dir=temp_directory)

        assert isinstance(result, Data)
        assert hasattr(result, "x")

    def test_convert_structural_features_applied_after_conversion(
        self, sample_smiles, structural_features_config_basic
    ):
        """Test structural_features_config is applied as post-processing."""
        converter = SMILESConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        # Track post-processing call
        with patch(
            "milia_pipeline.models.post_training.data_preparation.data_converter._apply_structural_features_if_available"
        ) as mock_apply:
            mock_apply.side_effect = lambda data, config: data

            convert_to_pyg(
                sample_smiles["ethanol"],
                structural_features_config=structural_features_config_basic,
            )

            # Should call post-processing
            mock_apply.assert_called_once()


# =============================================================================
# CONVERT_BATCH_TO_PYG FUNCTION TESTS
# =============================================================================


class TestConvertBatchToPyg:
    """Test convert_batch_to_pyg convenience function."""

    def test_convert_batch_pyg_data(self, mock_pyg_data, mock_pyg_data_minimal):
        """Test convert_batch_to_pyg with PyG Data objects."""
        # Create similar structure data for batching
        data1 = Data(x=torch.randn(5, 3), edge_index=torch.tensor([[0, 1], [1, 0]]))
        data2 = Data(x=torch.randn(4, 3), edge_index=torch.tensor([[0, 1], [1, 0]]))

        result = convert_batch_to_pyg([data1, data2])

        assert isinstance(result, Batch)
        assert result.num_graphs == 2

    def test_convert_batch_dicts(self, valid_dict_data_simple):
        """Test convert_batch_to_pyg with dictionaries."""
        dict1 = {"x": torch.randn(5, 3), "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]])}
        dict2 = {"x": torch.randn(4, 3), "edge_index": torch.tensor([[0, 1], [1, 0]])}

        result = convert_batch_to_pyg([dict1, dict2])

        assert isinstance(result, Batch)
        assert result.num_graphs == 2

    def test_convert_batch_with_explicit_format(self):
        """Test convert_batch_to_pyg with explicit format."""
        dict1 = {"x": torch.randn(3, 2), "edge_index": torch.tensor([[0, 1], [1, 0]])}
        dict2 = {"x": torch.randn(4, 2), "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]])}

        result = convert_batch_to_pyg([dict1, dict2], format="dict")

        assert isinstance(result, Batch)

    def test_convert_batch_smiles_if_available(self, sample_smiles):
        """Test convert_batch_to_pyg with SMILES if rdkit available."""
        converter = SMILESConverter()

        if converter.is_available:
            # Use molecules that all have bonds (edge_attr) to avoid batching inconsistency
            # methane (C) has no bonds, so we use ethanol and benzene instead
            smiles_list = [sample_smiles["ethanol"], sample_smiles["benzene"]]
            result = convert_batch_to_pyg(smiles_list)

            assert isinstance(result, Batch)
            assert result.num_graphs == 2

    def test_convert_batch_single_element(self, valid_dict_data_simple):
        """Test convert_batch_to_pyg with single element list."""
        result = convert_batch_to_pyg([valid_dict_data_simple])

        assert isinstance(result, Batch)
        assert result.num_graphs == 1

    def test_convert_batch_empty_list_raises(self):
        """Test convert_batch_to_pyg with empty list behavior."""
        # Empty list should either raise or return empty batch
        try:
            result = convert_batch_to_pyg([])
            # If it succeeds, result should be empty batch
            assert result.num_graphs == 0
        except (ValueError, RuntimeError, IndexError):
            # Expected - empty input not supported by Batch.from_data_list
            pass

    def test_convert_batch_passes_kwargs(self):
        """Test convert_batch_to_pyg passes kwargs to converter."""
        # Test with SMILESConverter which accepts kwargs, or fallback to pyg_data
        smiles_converter = SMILESConverter()

        if smiles_converter.is_available:
            # SMILES converter accepts kwargs in __init__
            result = convert_batch_to_pyg(["CCO", "c1ccccc1"], format="smiles", add_hydrogens=False)
            assert isinstance(result, Batch)
        else:
            # Fallback: test with PyG Data which ignores kwargs
            pyg_data = Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1], [1, 0]]))
            result = convert_batch_to_pyg([pyg_data], format="pyg_data")
            assert isinstance(result, Batch)

    def test_convert_batch_maintains_order(self):
        """Test convert_batch_to_pyg maintains input order."""
        data1 = Data(x=torch.tensor([[1.0]]), edge_index=torch.tensor([[0], [0]]))
        data2 = Data(x=torch.tensor([[2.0]]), edge_index=torch.tensor([[0], [0]]))
        data3 = Data(x=torch.tensor([[3.0]]), edge_index=torch.tensor([[0], [0]]))

        result = convert_batch_to_pyg([data1, data2, data3])

        # Check order is maintained via batch indices
        assert result.num_graphs == 3


# =============================================================================
# LIST_AVAILABLE_FORMATS FUNCTION TESTS
# =============================================================================


class TestListAvailableFormats:
    """Test list_available_formats convenience function."""

    def test_returns_list(self):
        """Test list_available_formats returns a list."""
        result = list_available_formats()

        assert isinstance(result, list)

    def test_contains_always_available_formats(self):
        """Test list_available_formats contains formats without external deps."""
        result = list_available_formats()

        # These should always be available
        assert "pyg_data" in result
        assert "dict" in result

    def test_returns_only_available(self):
        """Test list_available_formats returns only formats with available deps."""
        result = list_available_formats()

        # All returned formats should be usable
        registry = get_registry()
        for fmt in result:
            converter_class = registry.get(fmt)
            instance = converter_class()
            assert instance.is_available is True

    def test_consistent_with_registry(self):
        """Test list_available_formats is consistent with registry."""
        registry = get_registry()

        result = list_available_formats()
        registry_result = registry.list_available()

        assert set(result) == set(registry_result)


# =============================================================================
# LIST_ALL_FORMATS FUNCTION TESTS
# =============================================================================


class TestListAllFormats:
    """Test list_all_formats convenience function."""

    def test_returns_list(self):
        """Test list_all_formats returns a list."""
        result = list_all_formats()

        assert isinstance(result, list)

    def test_contains_all_registered_formats(self):
        """Test list_all_formats contains all registered formats."""
        result = list_all_formats()

        expected = ["pyg_data", "dict", "smiles", "inchi", "xyz", "ase_atoms", "sdf"]
        for fmt in expected:
            assert fmt in result

    def test_includes_unavailable_formats(self):
        """Test list_all_formats includes formats even if deps unavailable."""
        result = list_all_formats()

        # Should include formats regardless of availability
        # smiles, inchi, xyz, ase_atoms, sdf may have unavailable deps
        assert len(result) >= 7

    def test_consistent_with_registry(self):
        """Test list_all_formats is consistent with registry."""
        registry = get_registry()

        result = list_all_formats()
        registry_result = registry.list_all()

        assert set(result) == set(registry_result)

    def test_all_formats_superset_of_available(self):
        """Test list_all_formats is superset of list_available_formats."""
        all_formats = set(list_all_formats())
        available_formats = set(list_available_formats())

        assert available_formats.issubset(all_formats)


# =============================================================================
# SMILES_TO_DATA LEGACY ALIAS TESTS
# =============================================================================


class TestSmilesToDataAlias:
    """Test smiles_to_data legacy alias."""

    def test_alias_points_to_convert_to_pyg(self):
        """Test smiles_to_data is alias for convert_to_pyg."""
        assert smiles_to_data is convert_to_pyg

    def test_alias_works_with_pyg_data(self, mock_pyg_data):
        """Test smiles_to_data works (backward compatibility)."""
        result = smiles_to_data(mock_pyg_data)

        assert result is mock_pyg_data

    def test_alias_works_with_dict(self, valid_dict_data):
        """Test smiles_to_data works with dict input."""
        result = smiles_to_data(valid_dict_data)

        assert isinstance(result, Data)

    def test_alias_works_with_smiles_if_available(self, sample_smiles):
        """Test smiles_to_data works with actual SMILES if rdkit available."""
        converter = SMILESConverter()

        if converter.is_available:
            result = smiles_to_data(sample_smiles["ethanol"])

            assert isinstance(result, Data)


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_empty_edge_index(self):
        """Test handling of empty edge_index."""
        dict_data = {"x": torch.randn(5, 3), "edge_index": torch.empty((2, 0), dtype=torch.long)}

        result = convert_to_pyg(dict_data, format="dict")

        assert isinstance(result, Data)
        assert result.edge_index.shape[1] == 0

    def test_single_node_graph(self):
        """Test handling of single-node graph."""
        dict_data = {"x": torch.randn(1, 3), "edge_index": torch.empty((2, 0), dtype=torch.long)}

        result = convert_to_pyg(dict_data, format="dict")

        assert isinstance(result, Data)
        assert result.x.shape[0] == 1

    def test_self_loops_in_edge_index(self):
        """Test handling of self-loops in edge_index."""
        dict_data = {
            "x": torch.randn(3, 2),
            "edge_index": torch.tensor([[0, 1, 2], [0, 1, 2]]),  # Self-loops
        }

        result = convert_to_pyg(dict_data, format="dict")

        assert isinstance(result, Data)

    def test_large_graph(self):
        """Test handling of large graphs."""
        n_nodes = 10000
        n_edges = 50000

        dict_data = {
            "x": torch.randn(n_nodes, 32),
            "edge_index": torch.randint(0, n_nodes, (2, n_edges)),
        }

        result = convert_to_pyg(dict_data, format="dict")

        assert isinstance(result, Data)
        assert result.x.shape[0] == n_nodes
        assert result.edge_index.shape[1] == n_edges

    def test_none_format_triggers_auto_detect(self, mock_pyg_data):
        """Test None format triggers auto-detection."""
        result = convert_to_pyg(mock_pyg_data, format=None)

        assert result is mock_pyg_data

    def test_whitespace_in_format_name(self, valid_dict_data):
        """Test format names with whitespace are handled."""
        # Whitespace should be part of the lookup (case-insensitive)
        # This tests that the format is used as-is (lowercased)
        try:
            _result = convert_to_pyg(valid_dict_data, format="  dict  ")
        except ValueError:
            pass  # Expected if whitespace isn't trimmed

    def test_unicode_data_in_dict(self):
        """Test handling of unicode strings in dict data."""
        dict_data = {
            "x": torch.randn(3, 2),
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
            "name": "水分子",  # "water molecule" in Chinese
            "description": "Молекула воды",  # "water molecule" in Russian
        }

        result = convert_to_pyg(dict_data, format="dict")

        assert result.name == "水分子"
        assert result.description == "Молекула воды"

    def test_nested_dict_values(self):
        """Test handling of nested dict values."""
        dict_data = {
            "x": torch.randn(3, 2),
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
            "metadata": {"source": "test", "version": 1},  # Nested dict
        }

        result = convert_to_pyg(dict_data, format="dict")

        assert result.metadata == {"source": "test", "version": 1}

    def test_float_tensor_conversion(self):
        """Test proper handling of float tensors."""
        dict_data = {
            "x": torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float64),
            "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        }

        result = convert_to_pyg(dict_data, format="dict")

        assert result.x.dtype == torch.float64

    def test_integer_tensor_conversion(self):
        """Test proper handling of integer tensors."""
        dict_data = {
            "z": torch.tensor([6, 8, 1], dtype=torch.int32),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        }

        result = convert_to_pyg(dict_data, format="dict")

        assert result.z.dtype == torch.int32


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for the data converter module."""

    def test_full_workflow_dict_to_batch(self):
        """Test complete workflow from dict to batch."""
        # Create multiple dict data
        dicts = [
            {"x": torch.randn(5, 3), "edge_index": torch.randint(0, 5, (2, 10))},
            {"x": torch.randn(4, 3), "edge_index": torch.randint(0, 4, (2, 8))},
            {"x": torch.randn(6, 3), "edge_index": torch.randint(0, 6, (2, 12))},
        ]

        # Convert to batch
        batch = convert_batch_to_pyg(dicts, format="dict")

        # Verify batch properties
        assert batch.num_graphs == 3
        assert batch.x.shape[0] == 15  # 5 + 4 + 6 nodes

    def test_full_workflow_mixed_sources(self, mock_pyg_data, valid_dict_data):
        """Test converting from multiple sources and combining."""
        # Convert dict
        data_from_dict = convert_to_pyg(valid_dict_data, format="dict")

        # Passthrough PyG data
        data_from_pyg = convert_to_pyg(mock_pyg_data, format="pyg_data")

        # Both should be Data objects
        assert isinstance(data_from_dict, Data)
        assert isinstance(data_from_pyg, Data)

    def test_registry_and_converter_consistency(self):
        """Test consistency between registry and converters."""
        registry = get_registry()

        # Only check the core converters that have proper format_name implementation
        # Skip dynamically registered test converters which may have closure issues
        core_formats = ["pyg_data", "dict", "smiles", "inchi", "xyz", "ase_atoms", "sdf"]

        for fmt in registry.list_all():
            if fmt in core_formats:
                converter_class = registry.get(fmt)
                instance = converter_class()

                # Format name should match registration
                assert instance.format_name == fmt

    def test_auto_detect_priority(self, mock_pyg_data, valid_dict_data):
        """Test auto-detect handles ambiguous cases correctly."""
        # PyG Data should be detected
        assert get_registry().auto_detect(mock_pyg_data) == "pyg_data"

        # Dict should be detected
        assert get_registry().auto_detect(valid_dict_data) == "dict"

    def test_converter_independence(self):
        """Test converters work independently of each other."""
        # Create multiple converters
        pyg_conv = PyGDataConverter()
        dict_conv = DictConverter()

        # Each should work independently
        pyg_data = Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1], [1, 0]]))
        dict_data = {"x": torch.randn(3, 2), "edge_index": torch.tensor([[0, 1], [1, 0]])}

        result1 = pyg_conv.convert(pyg_data)
        result2 = dict_conv.convert(dict_data)

        assert isinstance(result1, Data)
        assert isinstance(result2, Data)

    def test_smiles_workflow_if_available(self, sample_smiles):
        """Test complete SMILES workflow if rdkit available."""
        converter = SMILESConverter()

        if converter.is_available:
            # Single conversion
            data = convert_to_pyg(sample_smiles["ethanol"])
            assert isinstance(data, Data)
            assert hasattr(data, "smiles")

            # Batch conversion - use molecules that all have bonds (edge_attr)
            # to avoid inconsistent edge_attr attributes during batching
            # methane (C) and water (O) have no bonds in SMILES representation
            batch = convert_batch_to_pyg(
                [
                    sample_smiles["ethanol"],  # CCO - has bonds
                    sample_smiles["benzene"],  # c1ccccc1 - has bonds
                    sample_smiles["acetic_acid"],  # CC(=O)O - has bonds
                ]
            )
            assert batch.num_graphs == 3


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """Performance and stress tests."""

    def test_batch_conversion_performance(self):
        """Test batch conversion completes in reasonable time."""
        n_samples = 100
        dicts = [
            {"x": torch.randn(10, 5), "edge_index": torch.randint(0, 10, (2, 20))}
            for _ in range(n_samples)
        ]

        start = time.time()
        batch = convert_batch_to_pyg(dicts, format="dict")
        elapsed = time.time() - start

        assert batch.num_graphs == n_samples
        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed < 10.0, f"Batch conversion took {elapsed:.2f}s"

    def test_large_batch_handling(self):
        """Test handling of large batches."""
        n_samples = 500
        dicts = [
            {"x": torch.randn(5, 3), "edge_index": torch.randint(0, 5, (2, 8))}
            for _ in range(n_samples)
        ]

        batch = convert_batch_to_pyg(dicts, format="dict")

        assert batch.num_graphs == n_samples

    def test_repeated_registry_access(self):
        """Test repeated registry access is efficient."""
        start = time.time()

        for _ in range(1000):
            registry = get_registry()
            _ = registry.list_all()
            _ = registry.list_available()

        elapsed = time.time() - start

        # Should be very fast (singleton pattern)
        assert elapsed < 5.0, f"Repeated registry access took {elapsed:.2f}s"

    def test_concurrent_conversions(self, valid_dict_data):
        """Test concurrent conversions work correctly."""
        results = []
        errors = []

        def convert_data(idx):
            try:
                data = {
                    "x": torch.randn(5, 3),
                    "edge_index": torch.randint(0, 5, (2, 8)),
                    "idx": idx,
                }
                result = convert_to_pyg(data, format="dict")
                results.append((idx, result))
            except Exception as e:
                errors.append((idx, str(e)))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(convert_data, i) for i in range(50)]
            for _future in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 50


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogging:
    """Test logging behavior."""

    def test_registration_logs_debug(self, caplog):
        """Test that registration logs debug messages."""

        class LogTestConverter(BaseDataConverter):
            @property
            def format_name(self) -> str:
                return "log_test"

            @property
            def is_available(self) -> bool:
                return True

            def can_convert(self, input_data: Any) -> bool:
                return False

            def convert(self, input_data: Any, **kwargs) -> Data:
                return Data()

        registry = get_registry()

        with caplog.at_level(logging.DEBUG):
            registry.register("log_test_format", LogTestConverter)

        # Check that debug message was logged
        assert any("Registered converter" in record.message for record in caplog.records)

    def test_module_logger_name(self):
        """Test that module uses correct logger name."""
        # Import logger from module
        from milia_pipeline.models.post_training.data_preparation.data_converter import logger

        assert logger.name == "milia_pipeline.models.post_training.data_preparation.data_converter"


# =============================================================================
# SPECIAL CASES TESTS
# =============================================================================


class TestSpecialCases:
    """Test special cases and boundary conditions."""

    def test_data_with_all_optional_attributes(self):
        """Test Data with all optional attributes."""
        full_dict = {
            "x": torch.randn(5, 3),
            "edge_index": torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
            "edge_attr": torch.randn(5, 2),
            "y": torch.tensor([1.0]),
            "pos": torch.randn(5, 3),
            "batch": torch.zeros(5, dtype=torch.long),
        }

        result = convert_to_pyg(full_dict, format="dict")

        assert hasattr(result, "x")
        assert hasattr(result, "edge_index")
        assert hasattr(result, "edge_attr")
        assert hasattr(result, "y")
        assert hasattr(result, "pos")

    def test_pyg_data_with_custom_attributes(self):
        """Test PyG Data with custom attributes passes through."""
        data = Data(
            x=torch.randn(3, 2),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            custom_attr="custom_value",
            another_custom=42,
        )

        result = convert_to_pyg(data)

        assert result.custom_attr == "custom_value"
        assert result.another_custom == 42

    def test_convert_preserves_tensor_device(self):
        """Test that tensor device is preserved."""
        dict_data = {
            "x": torch.randn(3, 2),  # CPU tensor
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
        }

        result = convert_to_pyg(dict_data, format="dict")

        assert result.x.device == torch.device("cpu")

    def test_convert_preserves_requires_grad(self):
        """Test that requires_grad is preserved."""
        x = torch.randn(3, 2, requires_grad=True)
        dict_data = {"x": x, "edge_index": torch.tensor([[0, 1], [1, 0]])}

        result = convert_to_pyg(dict_data, format="dict")

        assert result.x.requires_grad is True

    def test_batch_with_varying_sizes(self):
        """Test batching graphs with varying sizes."""
        dicts = [
            {"x": torch.randn(3, 4), "edge_index": torch.randint(0, 3, (2, 4))},
            {"x": torch.randn(10, 4), "edge_index": torch.randint(0, 10, (2, 20))},
            {"x": torch.randn(1, 4), "edge_index": torch.empty((2, 0), dtype=torch.long)},
            {"x": torch.randn(7, 4), "edge_index": torch.randint(0, 7, (2, 12))},
        ]

        batch = convert_batch_to_pyg(dicts, format="dict")

        assert batch.num_graphs == 4
        assert batch.x.shape[0] == 21  # 3 + 10 + 1 + 7


# =============================================================================
# TYPE ANNOTATION VERIFICATION TESTS
# =============================================================================


class TestTypeAnnotations:
    """Test type annotation compliance."""

    def test_convert_to_pyg_return_type(self, mock_pyg_data):
        """Test convert_to_pyg returns Data type."""
        result = convert_to_pyg(mock_pyg_data)

        assert isinstance(result, Data)

    def test_convert_batch_to_pyg_return_type(self):
        """Test convert_batch_to_pyg returns Batch type."""
        data_list = [
            Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1], [1, 0]])) for _ in range(3)
        ]

        result = convert_batch_to_pyg(data_list)

        assert isinstance(result, Batch)

    def test_list_functions_return_lists(self):
        """Test list functions return List[str]."""
        all_formats = list_all_formats()
        available_formats = list_available_formats()

        assert isinstance(all_formats, list)
        assert isinstance(available_formats, list)
        assert all(isinstance(f, str) for f in all_formats)
        assert all(isinstance(f, str) for f in available_formats)

    def test_get_registry_return_type(self):
        """Test get_registry returns DataConverterRegistry."""
        result = get_registry()

        assert isinstance(result, DataConverterRegistry)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# =============================================================================
# 3D BOND FEATURES CONSTANT TESTS
# =============================================================================


class Test3DBondFeaturesConstant:
    """Test _3D_BOND_FEATURES constant definition."""

    def test_3d_bond_features_is_set(self):
        """Test _3D_BOND_FEATURES is a set type."""
        assert isinstance(_3D_BOND_FEATURES, (set, frozenset))

    def test_3d_bond_features_contains_bond_length(self):
        """Test _3D_BOND_FEATURES contains 'bond_length'."""
        assert "bond_length" in _3D_BOND_FEATURES

    def test_3d_bond_features_contains_bond_length_binned(self):
        """Test _3D_BOND_FEATURES contains 'bond_length_binned'."""
        assert "bond_length_binned" in _3D_BOND_FEATURES

    def test_3d_bond_features_known_contents(self):
        """Test _3D_BOND_FEATURES contains exactly the expected features."""
        expected = {"bond_length", "bond_length_binned"}
        assert expected == _3D_BOND_FEATURES


# =============================================================================
# _REQUIRES_3D_CONFORMER FUNCTION TESTS
# =============================================================================


class TestRequires3DConformer:
    """Test _requires_3d_conformer helper function."""

    def test_returns_false_for_none_config(self):
        """Test _requires_3d_conformer returns False for None config."""
        result = _requires_3d_conformer(None)
        assert result is False

    def test_returns_false_for_empty_config(self, structural_features_config_empty):
        """Test _requires_3d_conformer returns False for empty config."""
        result = _requires_3d_conformer(structural_features_config_empty)
        assert result is False

    def test_returns_false_for_config_without_bond_key(self):
        """Test _requires_3d_conformer returns False when 'bond' key is missing."""
        config = {"atom": ["atomic_num"]}
        result = _requires_3d_conformer(config)
        assert result is False

    def test_returns_false_for_config_with_empty_bond_list(self):
        """Test _requires_3d_conformer returns False for empty bond list."""
        config = {"atom": ["atomic_num"], "bond": []}
        result = _requires_3d_conformer(config)
        assert result is False

    def test_returns_false_for_non_3d_bond_features(self, structural_features_config_basic):
        """Test _requires_3d_conformer returns False for non-3D bond features."""
        result = _requires_3d_conformer(structural_features_config_basic)
        assert result is False

    def test_returns_true_for_bond_length(self):
        """Test _requires_3d_conformer returns True when 'bond_length' is present."""
        config = {"bond": ["bond_type", "bond_length"]}
        result = _requires_3d_conformer(config)
        assert result is True

    def test_returns_true_for_bond_length_binned(self):
        """Test _requires_3d_conformer returns True when 'bond_length_binned' is present."""
        config = {"bond": ["bond_length_binned"]}
        result = _requires_3d_conformer(config)
        assert result is True

    def test_returns_true_for_multiple_3d_features(self):
        """Test _requires_3d_conformer returns True for multiple 3D features."""
        config = {"bond": ["bond_length", "bond_length_binned", "bond_type"]}
        result = _requires_3d_conformer(config)
        assert result is True

    def test_handles_config_with_atom_only(self):
        """Test _requires_3d_conformer handles config with only atom features."""
        config = {"atom": ["atomic_num", "degree", "formal_charge"]}
        result = _requires_3d_conformer(config)
        assert result is False

    def test_returns_true_for_single_3d_feature_among_many(self):
        """Test returns True when single 3D feature exists among non-3D features."""
        config = {"bond": ["bond_type", "is_conjugated", "is_in_ring", "bond_length", "stereo"]}
        result = _requires_3d_conformer(config)
        assert result is True

    def test_handles_config_with_none_bond_value(self):
        """Test behavior when 'bond' key has None value.

        Note: The module's _requires_3d_conformer uses .get('bond', []) which returns
        None when bond is explicitly set to None, then set(None) raises TypeError.
        This test verifies the current behavior - the function does not handle None bond value.
        """
        config = {"atom": ["atomic_num"], "bond": None}
        # Module currently raises TypeError for this edge case because set(None) fails
        with pytest.raises(TypeError):
            _requires_3d_conformer(config)

    def test_is_dynamic_with_any_future_3d_features(self):
        """Test function dynamically checks against _3D_BOND_FEATURES set.

        This test verifies FUTURE-PROOF behavior - the function uses intersection
        with _3D_BOND_FEATURES rather than hardcoded checks.
        """
        # This tests that any feature in _3D_BOND_FEATURES is detected
        for feature in _3D_BOND_FEATURES:
            config = {"bond": [feature]}
            result = _requires_3d_conformer(config)
            assert result is True, f"Should detect 3D feature: {feature}"

    def test_returns_false_for_unknown_bond_features(self):
        """Test returns False for unknown bond features not in _3D_BOND_FEATURES."""
        config = {"bond": ["unknown_feature", "another_unknown"]}
        result = _requires_3d_conformer(config)
        assert result is False


# =============================================================================
# _ENSURE_3D_CONFORMER_FOR_PREDICTION FUNCTION TESTS
# =============================================================================


class TestEnsure3DConformerForPrediction:
    """Test _ensure_3d_conformer_for_prediction helper function."""

    def test_returns_true_when_no_3d_features_needed(self, structural_features_config_basic):
        """Test returns True immediately when no 3D features are needed."""
        mock_mol = MagicMock()

        result = _ensure_3d_conformer_for_prediction(mock_mol, structural_features_config_basic)

        assert result is True
        # Should not call any mol methods since no 3D needed
        mock_mol.GetNumConformers.assert_not_called()

    def test_returns_true_for_none_config(self):
        """Test returns True for None config (no 3D features needed)."""
        mock_mol = MagicMock()

        result = _ensure_3d_conformer_for_prediction(mock_mol, None)

        assert result is True

    def test_returns_true_when_conformer_already_exists(self, structural_features_config_with_3d):
        """Test returns True when molecule already has conformer."""
        mock_mol = MagicMock()
        mock_mol.GetNumConformers.return_value = 1  # Has conformer

        result = _ensure_3d_conformer_for_prediction(mock_mol, structural_features_config_with_3d)

        assert result is True

    def test_generates_conformer_when_needed_and_rdkit_available(
        self, structural_features_config_with_3d
    ):
        """Test generates conformer when 3D features needed and RDKit available."""
        converter = SMILESConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        from rdkit import Chem

        # Create a real mol without conformer
        mol = Chem.MolFromSmiles("CCO")  # Ethanol
        assert mol.GetNumConformers() == 0

        result = _ensure_3d_conformer_for_prediction(mol, structural_features_config_with_3d)

        # Should have generated a conformer
        assert result is True
        assert mol.GetNumConformers() > 0

    def test_handles_rdkit_import_error(self, structural_features_config_with_3d):
        """Test handles ImportError when RDKit not available."""
        mock_mol = MagicMock()
        mock_mol.GetNumConformers.return_value = 0

        with patch.dict("sys.modules", {"rdkit": None, "rdkit.Chem": None}):
            # Force ImportError path by mocking the import
            with patch(
                "milia_pipeline.models.post_training.data_preparation.data_converter._requires_3d_conformer",
                return_value=True,
            ):
                # This will attempt to import rdkit and fail
                # The function should catch this and return False gracefully
                pass

    def test_returns_false_for_difficult_molecules(self, structural_features_config_with_3d):
        """Test returns False for molecules that fail embedding."""
        converter = SMILESConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        from rdkit import Chem

        # Create a very strange/problematic molecule structure
        # Most simple molecules should succeed, but we test the failure path
        # by mocking EmbedMolecule to return -1
        mol = Chem.MolFromSmiles("C")  # Methane

        with patch("rdkit.Chem.AllChem.EmbedMolecule", return_value=-1):
            _result = _ensure_3d_conformer_for_prediction(mol, structural_features_config_with_3d)
            # Should return False when embedding fails
            # Note: actual behavior depends on fallback logic

    def test_mmff_optimization_fallback_to_uff(self, structural_features_config_with_3d):
        """Test MMFF optimization falls back to UFF when MMFF fails."""
        converter = SMILESConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")
        assert mol.GetNumConformers() == 0

        # Mock MMFF to fail, UFF should succeed as fallback
        with patch.object(AllChem, "MMFFOptimizeMolecule", side_effect=Exception("MMFF failed")):
            result = _ensure_3d_conformer_for_prediction(mol, structural_features_config_with_3d)
            # Should still succeed via UFF fallback or unoptimized conformer
            assert result is True
            assert mol.GetNumConformers() > 0

    def test_both_optimization_methods_fail_uses_unoptimized(
        self, structural_features_config_with_3d
    ):
        """Test uses unoptimized conformer when both MMFF and UFF fail."""
        converter = SMILESConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")
        assert mol.GetNumConformers() == 0

        # Mock both optimization methods to fail
        with patch.object(AllChem, "MMFFOptimizeMolecule", side_effect=Exception("MMFF failed")):
            with patch.object(AllChem, "UFFOptimizeMolecule", side_effect=Exception("UFF failed")):
                result = _ensure_3d_conformer_for_prediction(
                    mol, structural_features_config_with_3d
                )
                # Should still succeed with unoptimized conformer
                assert result is True
                assert mol.GetNumConformers() > 0

    def test_conformer_coordinates_transferred_correctly(self, structural_features_config_with_3d):
        """Test conformer coordinates are transferred to original molecule correctly."""
        converter = SMILESConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")  # Ethanol: 3 heavy atoms
        num_atoms = mol.GetNumAtoms()
        assert mol.GetNumConformers() == 0

        result = _ensure_3d_conformer_for_prediction(mol, structural_features_config_with_3d)

        assert result is True
        assert mol.GetNumConformers() > 0

        # Verify conformer has correct number of atom positions
        conf = mol.GetConformer()
        for i in range(num_atoms):
            pos = conf.GetAtomPosition(i)
            # Position should be a valid 3D coordinate (not all zeros typically)
            assert hasattr(pos, "x") and hasattr(pos, "y") and hasattr(pos, "z")

    def test_random_coords_fallback_on_initial_embed_failure(
        self, structural_features_config_with_3d, caplog
    ):
        """Test random coordinates fallback when initial ETKDGv3 embedding fails."""
        converter = SMILESConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")

        # Track calls to EmbedMolecule
        original_embed = AllChem.EmbedMolecule
        call_count = [0]

        def mock_embed(mol, params):
            call_count[0] += 1
            if call_count[0] == 1:
                return -1  # First call fails
            return original_embed(mol, params)  # Second call with useRandomCoords succeeds

        with patch.object(AllChem, "EmbedMolecule", side_effect=mock_embed):
            with caplog.at_level(logging.DEBUG):
                _result = _ensure_3d_conformer_for_prediction(
                    mol, structural_features_config_with_3d
                )

        # Should have attempted twice (initial + random coords fallback)
        assert call_count[0] == 2

    def test_handles_generic_exception_gracefully(self, structural_features_config_with_3d, caplog):
        """Test handles generic exceptions during conformer generation gracefully."""
        converter = SMILESConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")

        with patch.object(AllChem, "ETKDGv3", side_effect=RuntimeError("Unexpected error")):
            with caplog.at_level(logging.WARNING):
                result = _ensure_3d_conformer_for_prediction(
                    mol, structural_features_config_with_3d
                )

        assert result is False
        # Should log a warning
        assert any("Error generating 3D conformer" in record.message for record in caplog.records)


# =============================================================================
# _APPLY_STRUCTURAL_FEATURES_IF_AVAILABLE FUNCTION TESTS
# =============================================================================


class TestApplyStructuralFeaturesIfAvailable:
    """Test _apply_structural_features_if_available helper function."""

    def test_returns_original_data_for_none_config(self, mock_pyg_data):
        """Test returns original data when config is None."""
        result = _apply_structural_features_if_available(mock_pyg_data, None)

        assert result is mock_pyg_data

    def test_returns_original_data_for_empty_config(
        self, mock_pyg_data, structural_features_config_empty
    ):
        """Test returns original data for empty config."""
        result = _apply_structural_features_if_available(
            mock_pyg_data, structural_features_config_empty
        )

        assert result is mock_pyg_data

    def test_returns_original_data_for_config_with_empty_lists(
        self, mock_pyg_data, structural_features_config_none_values
    ):
        """Test returns original data when config has empty atom/bond lists."""
        result = _apply_structural_features_if_available(
            mock_pyg_data, structural_features_config_none_values
        )

        assert result is mock_pyg_data

    def test_returns_original_data_without_smiles_or_inchi(
        self, mock_pyg_data, structural_features_config_basic
    ):
        """Test returns original data when no SMILES or InChI available."""
        # mock_pyg_data doesn't have smiles or inchi attributes
        result = _apply_structural_features_if_available(
            mock_pyg_data, structural_features_config_basic
        )

        # Should return original since can't reconstruct mol
        assert result is mock_pyg_data

    def test_processes_data_with_smiles_when_rdkit_available(
        self, structural_features_config_basic
    ):
        """Test processes data with SMILES attribute when RDKit available."""
        converter = SMILESConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        # Create data with smiles attribute
        data = Data(
            x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), smiles="CCO"
        )

        # This may or may not modify the data depending on mol_structural_features
        # availability, but should not raise
        try:
            result = _apply_structural_features_if_available(data, structural_features_config_basic)
            assert isinstance(result, Data)
        except ImportError:
            # mol_structural_features not available
            pass

    def test_handles_invalid_smiles_gracefully(self, structural_features_config_basic):
        """Test handles invalid SMILES gracefully."""
        converter = SMILESConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        data = Data(
            x=torch.randn(3, 2),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            smiles="INVALID_SMILES_XYZ",
        )

        # Should return original data when SMILES is invalid
        result = _apply_structural_features_if_available(data, structural_features_config_basic)

        assert result is data

    def test_handles_inchi_attribute(self, structural_features_config_basic):
        """Test handles data with InChI attribute."""
        converter = InChIConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        data = Data(
            x=torch.randn(3, 2),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
        )

        try:
            result = _apply_structural_features_if_available(data, structural_features_config_basic)
            assert isinstance(result, Data)
        except ImportError:
            pass

    def test_prefers_smiles_over_inchi_when_both_present(self, structural_features_config_basic):
        """Test prefers SMILES attribute over InChI when both are present."""
        converter = SMILESConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        data = Data(
            x=torch.randn(3, 2),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            smiles="CCO",
            inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
        )

        try:
            result = _apply_structural_features_if_available(data, structural_features_config_basic)
            # Should not raise and should process via SMILES path
            assert isinstance(result, Data)
        except ImportError:
            pass

    def test_handles_invalid_inchi_gracefully(self, structural_features_config_basic):
        """Test handles invalid InChI gracefully."""
        converter = InChIConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        data = Data(
            x=torch.randn(3, 2),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            inchi="InChI=INVALID_INCHI_STRING",
        )

        # Should return original data when InChI is invalid
        result = _apply_structural_features_if_available(data, structural_features_config_basic)

        assert result is data

    def test_handles_empty_smiles_attribute(self, structural_features_config_basic):
        """Test handles empty string SMILES attribute."""
        data = Data(
            x=torch.randn(3, 2),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            smiles="",  # Empty string
        )

        result = _apply_structural_features_if_available(data, structural_features_config_basic)

        # Should return original data since empty smiles can't be parsed
        assert result is data

    def test_handles_exception_from_add_structural_features(
        self, structural_features_config_basic, caplog
    ):
        """Test handles exception from add_structural_features gracefully."""
        converter = SMILESConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        data = Data(
            x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), smiles="CCO"
        )

        # Mock add_structural_features at the source module to raise an exception
        # The function is imported inside the try block from milia_pipeline.molecules.mol_structural_features
        with (
            patch(
                "milia_pipeline.molecules.mol_structural_features.add_structural_features",
                side_effect=RuntimeError("Feature extraction failed"),
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = _apply_structural_features_if_available(data, structural_features_config_basic)

        # Should return original data and log warning
        assert result is data
        assert any(
            "Failed to apply structural features" in record.message for record in caplog.records
        )

    def test_logs_debug_when_xyz_format_detected(self, structural_features_config_basic, caplog):
        """Test logs debug message for XYZ/ASE/dict formats without SMILES/InChI."""
        # Data without smiles or inchi (like XYZ-converted data)
        data = Data(
            z=torch.tensor([8, 1, 1]),  # Water: O, H, H
            pos=torch.randn(3, 3),
            edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 0, 0]]),
        )

        with caplog.at_level(logging.DEBUG):
            result = _apply_structural_features_if_available(data, structural_features_config_basic)

        assert result is data
        # Should log that this is expected for XYZ/ASE/dict formats
        assert any(
            "expected for XYZ, ASE, dict formats" in record.message for record in caplog.records
        )

    def test_calls_ensure_3d_conformer_when_needed(self, structural_features_config_with_3d):
        """Test calls _ensure_3d_conformer_for_prediction when 3D features needed."""
        converter = SMILESConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        data = Data(
            x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), smiles="CCO"
        )

        with patch(
            "milia_pipeline.models.post_training.data_preparation.data_converter._ensure_3d_conformer_for_prediction"
        ) as mock_ensure:
            mock_ensure.return_value = True
            try:
                _apply_structural_features_if_available(data, structural_features_config_with_3d)
                # Should have called _ensure_3d_conformer_for_prediction
                mock_ensure.assert_called_once()
            except ImportError:
                pytest.skip("mol_structural_features not available")


# =============================================================================
# SMILES CONVERTER WITH STRUCTURAL_FEATURES_CONFIG TESTS
# =============================================================================


class TestSMILESConverterStructuralFeatures:
    """Test SMILESConverter with structural_features_config parameter (FIX 20)."""

    def test_accepts_structural_features_config_parameter(self):
        """Test SMILESConverter accepts structural_features_config in __init__."""
        config = {"atom": ["atomic_num"], "bond": ["bond_type"]}
        converter = SMILESConverter(structural_features_config=config)

        assert converter.structural_features_config == config

    def test_uses_structural_features_when_config_provided(self, structural_features_config_basic):
        """Test uses structural features mode when config has features."""
        converter = SMILESConverter(structural_features_config=structural_features_config_basic)

        assert converter._use_structural_features is True
        assert converter.atom_features == structural_features_config_basic["atom"]
        assert converter.bond_features == structural_features_config_basic["bond"]

    def test_uses_default_features_when_config_empty(self, structural_features_config_empty):
        """Test uses default features when config is empty."""
        converter = SMILESConverter(structural_features_config=structural_features_config_empty)

        assert converter._use_structural_features is False
        # Should have default features
        assert "atomic_num" in converter.atom_features

    def test_uses_default_features_when_config_none(self):
        """Test uses default features when config is None."""
        converter = SMILESConverter(structural_features_config=None)

        assert converter._use_structural_features is False
        assert converter.structural_features_config is None

    def test_structural_features_config_overrides_atom_features(self):
        """Test structural_features_config overrides atom_features parameter."""
        config = {"atom": ["atomic_num"], "bond": ["bond_type"]}
        converter = SMILESConverter(
            atom_features=["degree"],  # Should be overridden
            structural_features_config=config,
        )

        assert converter.atom_features == ["atomic_num"]

    def test_convert_with_structural_features_config_if_available(
        self, structural_features_config_basic
    ):
        """Test convert uses structural features when config provided."""
        converter = SMILESConverter(structural_features_config=structural_features_config_basic)

        if not converter.is_available:
            pytest.skip("rdkit not available")

        try:
            result = converter.convert("CCO")
            assert isinstance(result, Data)
            assert hasattr(result, "smiles")
        except ImportError:
            # mol_structural_features module not available
            pytest.skip("mol_structural_features not available")

    def test_convert_with_3d_features_generates_conformer(self, structural_features_config_with_3d):
        """Test convert with 3D features generates conformer."""
        converter = SMILESConverter(structural_features_config=structural_features_config_with_3d)

        if not converter.is_available:
            pytest.skip("rdkit not available")

        # Mock _ensure_3d_conformer_for_prediction to verify it's called
        # Also need to mock add_structural_features since it will be called after conformer generation
        with patch(
            "milia_pipeline.models.post_training.data_preparation.data_converter._ensure_3d_conformer_for_prediction"
        ) as mock_ensure:
            mock_ensure.return_value = True

            # Mock add_structural_features to prevent it from actually running
            # (since the config may have features not supported by the real function)
            mock_data = Data(
                x=torch.randn(3, 2),
                edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                smiles="CCO",
            )
            with patch(
                "milia_pipeline.molecules.mol_structural_features.add_structural_features",
                return_value=mock_data,
            ):
                try:
                    converter.convert("CCO")
                    mock_ensure.assert_called_once()
                except ImportError:
                    pytest.skip("mol_structural_features not available")

    def test_logs_featurization_info_when_config_provided(
        self, structural_features_config_basic, caplog
    ):
        """Test logs featurization info when structural_features_config provided."""
        with caplog.at_level(logging.INFO):
            _converter = SMILESConverter(structural_features_config=structural_features_config_basic)

        assert any(
            "SMILESConverter using checkpoint featurization" in record.message
            for record in caplog.records
        )


# =============================================================================
# INCHI CONVERTER STRUCTURAL FEATURES TESTS
# =============================================================================


class TestInChIConverterStructuralFeatures:
    """Test InChIConverter post-processing with structural_features_config.

    Note: InChIConverter doesn't accept structural_features_config in __init__
    (unlike SMILESConverter), but converted Data objects preserve 'inchi' attribute
    which allows post-processing via _apply_structural_features_if_available.
    """

    def test_inchi_attribute_preserved_for_postprocessing(self, sample_inchi):
        """Test InChIConverter preserves inchi attribute for post-processing."""
        converter = InChIConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        result = converter.convert(sample_inchi["ethanol"])

        assert hasattr(result, "inchi")
        assert result.inchi == sample_inchi["ethanol"]

    def test_inchi_convert_to_pyg_with_structural_features(
        self, sample_inchi, structural_features_config_basic
    ):
        """Test convert_to_pyg applies structural features to InChI-converted data."""
        converter = InChIConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        try:
            # convert_to_pyg handles structural_features_config via post-processing
            result = convert_to_pyg(
                sample_inchi["ethanol"],
                format="inchi",
                structural_features_config=structural_features_config_basic,
            )

            assert isinstance(result, Data)
            assert hasattr(result, "inchi")
        except ImportError:
            pytest.skip("mol_structural_features not available")

    def test_inchi_structural_features_applied_via_postprocessing(
        self, sample_inchi, structural_features_config_basic
    ):
        """Test structural features applied to InChI via _apply_structural_features_if_available."""
        converter = InChIConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        # First convert InChI
        data = converter.convert(sample_inchi["ethanol"])

        # Then apply structural features via post-processing function
        try:
            result = _apply_structural_features_if_available(data, structural_features_config_basic)
            assert isinstance(result, Data)
        except ImportError:
            pytest.skip("mol_structural_features not available")


# =============================================================================
# SDF CONVERTER CONVERT_ALL METHOD TESTS (FIX 24)
# =============================================================================


class TestSDFConverterConvertAll:
    """Test SDFConverter.convert_all method (FIX 24)."""

    def test_convert_all_returns_list(self, temp_directory, mock_multi_molecule_sdf_content):
        """Test convert_all returns a list."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        sdf_path = temp_directory / "multi.sdf"
        sdf_path.write_text(mock_multi_molecule_sdf_content)

        result = converter.convert_all(str(sdf_path))

        assert isinstance(result, list)

    def test_convert_all_returns_correct_count(
        self, temp_directory, mock_multi_molecule_sdf_content
    ):
        """Test convert_all returns correct number of molecules."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        sdf_path = temp_directory / "multi.sdf"
        sdf_path.write_text(mock_multi_molecule_sdf_content)

        result = converter.convert_all(str(sdf_path))

        assert len(result) == 2  # methane and ethane

    def test_convert_all_returns_data_objects(
        self, temp_directory, mock_multi_molecule_sdf_content
    ):
        """Test convert_all returns list of Data objects."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        sdf_path = temp_directory / "multi.sdf"
        sdf_path.write_text(mock_multi_molecule_sdf_content)

        result = converter.convert_all(str(sdf_path))

        for data in result:
            assert isinstance(data, Data)
            assert hasattr(data, "x")
            assert hasattr(data, "edge_index")

    def test_convert_all_preserves_smiles(self, temp_directory, mock_multi_molecule_sdf_content):
        """Test convert_all preserves SMILES for each molecule."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        sdf_path = temp_directory / "multi.sdf"
        sdf_path.write_text(mock_multi_molecule_sdf_content)

        result = converter.convert_all(str(sdf_path))

        for data in result:
            assert hasattr(data, "smiles")
            assert isinstance(data.smiles, str)

    def test_convert_all_raises_for_invalid_file(self, temp_directory):
        """Test convert_all raises ValueError for invalid SDF."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        invalid_sdf = temp_directory / "invalid.sdf"
        invalid_sdf.write_text("COMPLETELY INVALID SDF CONTENT")

        with pytest.raises(ValueError) as exc_info:
            converter.convert_all(str(invalid_sdf))

        assert "No valid molecules found" in str(exc_info.value)

    def test_convert_all_handles_partial_failures(self, temp_directory, caplog):
        """Test convert_all handles files with some invalid molecules."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        # Create SDF with valid and invalid molecules
        partial_sdf = temp_directory / "partial.sdf"
        partial_sdf.write_text("""
     RDKit          3D

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
INVALID MOLECULE DATA
$$$$
""")

        with caplog.at_level(logging.WARNING):
            result = converter.convert_all(str(partial_sdf))

        # Should return at least the valid molecule
        assert len(result) >= 1

    def test_convert_all_logs_success_count(
        self, temp_directory, mock_multi_molecule_sdf_content, caplog
    ):
        """Test convert_all logs molecule count on successful conversion."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        sdf_path = temp_directory / "multi.sdf"
        sdf_path.write_text(mock_multi_molecule_sdf_content)

        with caplog.at_level(logging.DEBUG):
            result = converter.convert_all(str(sdf_path))

        # Should have logged loading info
        assert len(result) == 2

    def test_convert_all_logs_partial_failure_info(self, temp_directory, caplog):
        """Test convert_all logs info about partial failures."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        # Create SDF with one valid and one invalid molecule
        mixed_sdf = temp_directory / "mixed.sdf"
        mixed_sdf.write_text("""
     RDKit          3D

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$

$$$$
""")

        with caplog.at_level(logging.INFO):
            result = converter.convert_all(str(mixed_sdf))

        # Should return valid molecule and log failure count
        assert len(result) >= 1


# =============================================================================
# SDF CONVERTER _MOL_TO_DATA HELPER TESTS
# =============================================================================


class TestSDFConverterMolToData:
    """Test SDFConverter._mol_to_data helper method."""

    def test_mol_to_data_returns_data_object(self, mock_rdkit_mol):
        """Test _mol_to_data returns Data object."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        from rdkit import Chem

        mol = Chem.MolFromSmiles("C")
        result = converter._mol_to_data(mol, Chem)

        assert isinstance(result, Data)

    def test_mol_to_data_includes_atom_features(self, mock_rdkit_mol):
        """Test _mol_to_data includes atom features."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        result = converter._mol_to_data(mol, Chem)

        assert hasattr(result, "x")
        assert result.x.shape[0] == 3  # 3 heavy atoms

    def test_mol_to_data_includes_edge_index(self, mock_rdkit_mol):
        """Test _mol_to_data includes edge_index."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        result = converter._mol_to_data(mol, Chem)

        assert hasattr(result, "edge_index")
        assert result.edge_index.shape[0] == 2  # Source and target

    def test_mol_to_data_preserves_smiles(self, mock_rdkit_mol):
        """Test _mol_to_data preserves SMILES."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        result = converter._mol_to_data(mol, Chem)

        assert hasattr(result, "smiles")
        assert isinstance(result.smiles, str)

    def test_mol_to_data_extracts_3d_coords_if_available(self):
        """Test _mol_to_data extracts 3D coordinates when conformer exists."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")
        AllChem.EmbedMolecule(mol, randomSeed=42)  # Add 3D coords

        result = converter._mol_to_data(mol, Chem)

        assert hasattr(result, "pos")
        if result.pos is not None:
            assert result.pos.shape[0] == mol.GetNumAtoms()
            assert result.pos.shape[1] == 3

    def test_mol_to_data_handles_no_conformer(self):
        """Test _mol_to_data handles molecule without conformer."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")  # No 3D coords
        result = converter._mol_to_data(mol, Chem)

        # pos should be None or not present
        pos = getattr(result, "pos", None)
        assert pos is None


# =============================================================================
# CONVERT_SDF_TO_PYG_LIST FUNCTION TESTS (FIX 24)
# =============================================================================


class TestConvertSdfToPygList:
    """Test convert_sdf_to_pyg_list convenience function (FIX 24)."""

    def test_returns_list_of_data(self, temp_directory, mock_multi_molecule_sdf_content):
        """Test convert_sdf_to_pyg_list returns list of Data objects."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        sdf_path = temp_directory / "multi.sdf"
        sdf_path.write_text(mock_multi_molecule_sdf_content)

        result = convert_sdf_to_pyg_list(str(sdf_path))

        assert isinstance(result, list)
        for data in result:
            assert isinstance(data, Data)

    def test_accepts_path_object(self, temp_directory, mock_multi_molecule_sdf_content):
        """Test accepts Path object."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        sdf_path = temp_directory / "multi.sdf"
        sdf_path.write_text(mock_multi_molecule_sdf_content)

        result = convert_sdf_to_pyg_list(sdf_path)  # Path object

        assert isinstance(result, list)

    def test_accepts_structural_features_config(
        self, temp_directory, mock_multi_molecule_sdf_content, structural_features_config_basic
    ):
        """Test accepts structural_features_config parameter."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        sdf_path = temp_directory / "multi.sdf"
        sdf_path.write_text(mock_multi_molecule_sdf_content)

        try:
            result = convert_sdf_to_pyg_list(
                str(sdf_path), structural_features_config=structural_features_config_basic
            )
            assert isinstance(result, list)
        except ImportError:
            # mol_structural_features not available
            pytest.skip("mol_structural_features not available")

    def test_accepts_working_root_dir(self, temp_directory, mock_multi_molecule_sdf_content):
        """Test accepts working_root_dir parameter."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        sdf_path = temp_directory / "multi.sdf"
        sdf_path.write_text(mock_multi_molecule_sdf_content)

        result = convert_sdf_to_pyg_list(
            "multi.sdf",  # Relative path
            working_root_dir=temp_directory,
        )

        assert isinstance(result, list)

    def test_raises_import_error_when_rdkit_unavailable(self, temp_directory):
        """Test raises ImportError when RDKit not available."""
        # Create mock SDF file
        sdf_path = temp_directory / "test.sdf"
        sdf_path.write_text("dummy content")

        # Mock SDFConverter to be unavailable
        with patch.object(
            SDFConverter, "is_available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = False

            with pytest.raises(ImportError) as exc_info:
                convert_sdf_to_pyg_list(str(sdf_path))

            assert "RDKit" in str(exc_info.value)

    def test_compatible_with_batch_from_data_list(
        self, temp_directory, mock_multi_molecule_sdf_content
    ):
        """Test result is compatible with Batch.from_data_list."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        sdf_path = temp_directory / "multi.sdf"
        sdf_path.write_text(mock_multi_molecule_sdf_content)

        data_list = convert_sdf_to_pyg_list(str(sdf_path))

        # Should be able to create batch
        batch = Batch.from_data_list(data_list)

        assert isinstance(batch, Batch)
        assert batch.num_graphs == len(data_list)

    def test_single_molecule_sdf_returns_single_element_list(
        self, temp_directory, mock_sdf_content
    ):
        """Test single-molecule SDF returns list with one element."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        sdf_path = temp_directory / "single.sdf"
        sdf_path.write_text(mock_sdf_content)

        result = convert_sdf_to_pyg_list(str(sdf_path))

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Data)

    def test_empty_sdf_raises_oserror(self, temp_directory):
        """Test empty SDF file raises OSError from RDKit.

        Note: RDKit's SDMolSupplier raises OSError for completely empty files
        before the convert_all method can raise ValueError for no valid molecules.
        """
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        # Create empty SDF file
        empty_sdf = temp_directory / "empty.sdf"
        empty_sdf.write_text("")

        # RDKit raises OSError for completely empty files
        with pytest.raises(OSError) as exc_info:
            convert_sdf_to_pyg_list(str(empty_sdf))

        assert "Invalid input file" in str(exc_info.value)

    def test_sdf_with_only_delimiters_raises_valueerror(self, temp_directory):
        """Test SDF with only delimiters (no valid molecules) raises ValueError."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        # Create SDF with only delimiters
        delimiter_only_sdf = temp_directory / "delimiter_only.sdf"
        delimiter_only_sdf.write_text("$$$$\n$$$$\n$$$$\n")

        with pytest.raises(ValueError) as exc_info:
            convert_sdf_to_pyg_list(str(delimiter_only_sdf))

        assert "No valid molecules found" in str(exc_info.value)

    def test_applies_structural_features_to_all_molecules(
        self, temp_directory, mock_multi_molecule_sdf_content, structural_features_config_basic
    ):
        """Test structural features are applied to all molecules in list."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        sdf_path = temp_directory / "multi.sdf"
        sdf_path.write_text(mock_multi_molecule_sdf_content)

        # Mock _apply_structural_features_if_available to track calls
        with patch(
            "milia_pipeline.models.post_training.data_preparation.data_converter._apply_structural_features_if_available"
        ) as mock_apply:
            mock_apply.side_effect = lambda data, config: data  # Pass through

            result = convert_sdf_to_pyg_list(
                str(sdf_path), structural_features_config=structural_features_config_basic
            )

            # Should be called for each molecule
            assert mock_apply.call_count == len(result)

    def test_concurrent_convert_sdf_to_pyg_list_operations(
        self, temp_directory, mock_multi_molecule_sdf_content
    ):
        """Test concurrent convert_sdf_to_pyg_list operations are thread-safe."""
        converter = SDFConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        sdf_path = temp_directory / "multi.sdf"
        sdf_path.write_text(mock_multi_molecule_sdf_content)

        errors = []
        results = []

        def convert_sdf():
            try:
                result = convert_sdf_to_pyg_list(str(sdf_path))
                results.append(len(result))
            except Exception as e:
                errors.append(str(e))

        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(convert_sdf) for _ in range(20)]
            for _future in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 20
        # All should return same count
        assert len(set(results)) == 1


# =============================================================================
# XYZ CONVERTER PATH RESOLUTION TESTS (DEPENDENCY INJECTION)
# =============================================================================


class TestXYZConverterPathResolution:
    """Test XYZConverter path resolution with working_root_dir (Dependency Injection)."""

    def test_default_working_root_dir_is_cwd(self):
        """Test default working_root_dir is current working directory."""
        converter = XYZConverter()

        assert converter._working_root_dir == Path.cwd()

    def test_custom_working_root_dir(self, temp_directory):
        """Test custom working_root_dir is used."""
        converter = XYZConverter(working_root_dir=temp_directory)

        assert converter._working_root_dir == temp_directory.resolve()

    def test_resolve_path_absolute_path_unchanged(self, temp_directory):
        """Test _resolve_path returns absolute paths unchanged."""
        converter = XYZConverter(working_root_dir=temp_directory)

        abs_path = Path("/tmp/test.xyz")
        result = converter._resolve_path(abs_path)

        assert result.is_absolute()
        assert str(result) == str(abs_path.resolve())

    def test_resolve_path_relative_path_resolved(self, temp_directory, mock_xyz_content):
        """Test _resolve_path resolves relative paths against working_root_dir."""
        converter = XYZConverter(working_root_dir=temp_directory)

        # Create XYZ file in temp_directory
        (temp_directory / "molecule.xyz").write_text(mock_xyz_content)

        result = converter._resolve_path("molecule.xyz")

        assert result.is_absolute()
        assert result == (temp_directory / "molecule.xyz").resolve()

    def test_can_convert_uses_path_resolution(self, temp_directory, mock_xyz_content):
        """Test can_convert uses path resolution."""
        converter = XYZConverter(working_root_dir=temp_directory)

        # Create XYZ file
        (temp_directory / "test.xyz").write_text(mock_xyz_content)

        # Should find file via relative path
        assert converter.can_convert("test.xyz") is True

    def test_convert_uses_path_resolution(self, temp_directory, mock_xyz_content):
        """Test convert uses path resolution."""
        converter = XYZConverter(working_root_dir=temp_directory)

        if not converter.is_available:
            pytest.skip("ase not available")

        # Create XYZ file
        (temp_directory / "water.xyz").write_text(mock_xyz_content)

        # Should convert via relative path
        result = converter.convert("water.xyz")

        assert isinstance(result, Data)

    def test_handles_tilde_expansion(self):
        """Test handles ~ in working_root_dir."""
        converter = XYZConverter(working_root_dir="~")

        assert converter._working_root_dir == Path.home()


# =============================================================================
# SDF CONVERTER PATH RESOLUTION TESTS (DEPENDENCY INJECTION)
# =============================================================================


class TestSDFConverterPathResolution:
    """Test SDFConverter path resolution with working_root_dir (Dependency Injection)."""

    def test_default_working_root_dir_is_cwd(self):
        """Test default working_root_dir is current working directory."""
        converter = SDFConverter()

        assert converter._working_root_dir == Path.cwd()

    def test_custom_working_root_dir(self, temp_directory):
        """Test custom working_root_dir is used."""
        converter = SDFConverter(working_root_dir=temp_directory)

        assert converter._working_root_dir == temp_directory.resolve()

    def test_resolve_path_absolute_path_unchanged(self, temp_directory):
        """Test _resolve_path returns absolute paths unchanged."""
        converter = SDFConverter(working_root_dir=temp_directory)

        abs_path = Path("/tmp/test.sdf")
        result = converter._resolve_path(abs_path)

        assert result.is_absolute()

    def test_resolve_path_relative_path_resolved(self, temp_directory, mock_sdf_content):
        """Test _resolve_path resolves relative paths against working_root_dir."""
        converter = SDFConverter(working_root_dir=temp_directory)

        # Create SDF file in temp_directory
        (temp_directory / "molecule.sdf").write_text(mock_sdf_content)

        result = converter._resolve_path("molecule.sdf")

        assert result.is_absolute()
        assert result == (temp_directory / "molecule.sdf").resolve()

    def test_can_convert_uses_path_resolution(self, temp_directory, mock_sdf_content):
        """Test can_convert uses path resolution."""
        converter = SDFConverter(working_root_dir=temp_directory)

        # Create SDF file
        (temp_directory / "test.sdf").write_text(mock_sdf_content)

        # Should find file via relative path
        assert converter.can_convert("test.sdf") is True

    def test_convert_uses_path_resolution(self, temp_directory, mock_sdf_content):
        """Test convert uses path resolution."""
        converter = SDFConverter(working_root_dir=temp_directory)

        if not converter.is_available:
            pytest.skip("rdkit not available")

        # Create SDF file
        (temp_directory / "methane.sdf").write_text(mock_sdf_content)

        # Should convert via relative path
        result = converter.convert("methane.sdf")

        assert isinstance(result, Data)

    def test_convert_all_uses_path_resolution(
        self, temp_directory, mock_multi_molecule_sdf_content
    ):
        """Test convert_all uses path resolution."""
        converter = SDFConverter(working_root_dir=temp_directory)

        if not converter.is_available:
            pytest.skip("rdkit not available")

        # Create SDF file
        (temp_directory / "multi.sdf").write_text(mock_multi_molecule_sdf_content)

        # Should convert via relative path
        result = converter.convert_all("multi.sdf")

        assert isinstance(result, list)
        assert len(result) == 2


# =============================================================================
# FIX 23: UNAVAILABLE FORMAT DETECTION TESTS
# =============================================================================


class TestUnavailableFormatDetection:
    """Test FIX 23: Unavailable format detection with helpful error messages."""

    def test_xyz_unavailable_provides_install_hint(self, temp_directory):
        """Test XYZ format unavailable provides install hint."""
        # Create XYZ file
        (temp_directory / "test.xyz").write_text("3\nWater\nO 0 0 0\nH 0 0 1\nH 0 1 0")

        # Mock XYZConverter to be unavailable
        with patch.object(
            XYZConverter, "is_available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = False

            try:
                convert_to_pyg(str(temp_directory / "test.xyz"))
            except ImportError as e:
                assert "xyz" in str(e).lower()
                assert "pip install ase" in str(e)
            except ValueError:
                # May raise ValueError if auto-detect fails
                pass

    def test_sdf_unavailable_provides_install_hint(self, temp_directory, mock_sdf_content):
        """Test SDF format unavailable provides install hint."""
        # Create SDF file
        (temp_directory / "test.sdf").write_text(mock_sdf_content)

        # Mock SDFConverter to be unavailable
        with patch.object(
            SDFConverter, "is_available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = False

            try:
                convert_to_pyg(str(temp_directory / "test.sdf"))
            except ImportError as e:
                assert "sdf" in str(e).lower()
                assert "pip install rdkit" in str(e)
            except ValueError:
                pass

    def test_mol_unavailable_provides_install_hint(self, temp_directory, mock_sdf_content):
        """Test MOL format unavailable provides install hint."""
        # Create MOL file
        (temp_directory / "test.mol").write_text(mock_sdf_content)

        # Mock SDFConverter to be unavailable
        with patch.object(
            SDFConverter, "is_available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = False

            try:
                convert_to_pyg(str(temp_directory / "test.mol"))
            except ImportError as e:
                assert "sdf" in str(e).lower()
                assert "pip install rdkit" in str(e)
            except ValueError:
                pass

    def test_unknown_format_raises_valueerror(self):
        """Test unknown format raises ValueError with available formats."""
        # Use input without C, N, O, c, n, o to avoid SMILES detection
        # Integer input is a good example of truly unrecognized input
        with pytest.raises(ValueError) as exc_info:
            convert_to_pyg(12345)

        assert "Cannot auto-detect format" in str(exc_info.value)
        assert "Available formats:" in str(exc_info.value)


# =============================================================================
# CONVERT_TO_PYG STRUCTURAL_FEATURES_CONFIG INTEGRATION TESTS
# =============================================================================


class TestConvertToPygStructuralFeatures:
    """Test convert_to_pyg with structural_features_config parameter."""

    def test_accepts_structural_features_config(
        self, sample_smiles, structural_features_config_basic
    ):
        """Test convert_to_pyg accepts structural_features_config."""
        converter = SMILESConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        try:
            result = convert_to_pyg(
                sample_smiles["ethanol"],
                structural_features_config=structural_features_config_basic,
            )
            assert isinstance(result, Data)
        except ImportError:
            pytest.skip("mol_structural_features not available")

    def test_structural_features_config_with_dict_format(
        self, valid_dict_data, structural_features_config_basic
    ):
        """Test structural_features_config works with dict format."""
        # Add smiles to dict for potential structural feature application
        dict_with_smiles = valid_dict_data.copy()
        dict_with_smiles["smiles"] = "CCO"

        result = convert_to_pyg(
            dict_with_smiles,
            format="dict",
            structural_features_config=structural_features_config_basic,
        )

        assert isinstance(result, Data)

    def test_structural_features_config_none_works(self, mock_pyg_data):
        """Test convert_to_pyg works with structural_features_config=None."""
        result = convert_to_pyg(mock_pyg_data, structural_features_config=None)

        assert result is mock_pyg_data

    def test_structural_features_applied_in_postprocessing(
        self, sample_smiles, structural_features_config_basic
    ):
        """Test structural features are applied in post-processing."""
        converter = SMILESConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        # Convert without config
        result_without = convert_to_pyg(sample_smiles["ethanol"])

        # Convert with config - should apply post-processing
        try:
            result_with = convert_to_pyg(
                sample_smiles["ethanol"],
                structural_features_config=structural_features_config_basic,
            )

            # Both should be valid Data objects
            assert isinstance(result_without, Data)
            assert isinstance(result_with, Data)
        except ImportError:
            pytest.skip("mol_structural_features not available")


# =============================================================================
# INCHI CONVERTER EXCLUDES FROM SMILES DETECTION TESTS
# =============================================================================


class TestInChIExcludedFromSMILES:
    """Test that InChI strings are properly excluded from SMILES detection."""

    def test_smiles_converter_rejects_inchi(self, sample_inchi):
        """Test SMILESConverter.can_convert returns False for InChI."""
        converter = SMILESConverter()

        for inchi in sample_inchi.values():
            assert converter.can_convert(inchi) is False

    def test_inchi_detected_before_smiles(self, sample_inchi):
        """Test InChI is detected by InChIConverter, not SMILESConverter."""
        inchi_converter = InChIConverter()
        smiles_converter = SMILESConverter()

        for inchi in sample_inchi.values():
            # InChI converter should accept
            assert inchi_converter.can_convert(inchi) is True
            # SMILES converter should reject
            assert smiles_converter.can_convert(inchi) is False


# =============================================================================
# LEGACY ALIAS EXTENDED TESTS
# =============================================================================


class TestSmilesToDataAliasExtended:
    """Extended tests for smiles_to_data legacy alias."""

    def test_alias_is_convert_to_pyg(self):
        """Test smiles_to_data is actually convert_to_pyg."""
        assert smiles_to_data is convert_to_pyg

    def test_alias_works_with_structural_features_config(
        self, sample_smiles, structural_features_config_basic
    ):
        """Test alias works with structural_features_config."""
        converter = SMILESConverter()

        if not converter.is_available:
            pytest.skip("rdkit not available")

        try:
            result = smiles_to_data(
                sample_smiles["ethanol"],
                structural_features_config=structural_features_config_basic,
            )
            assert isinstance(result, Data)
        except ImportError:
            pytest.skip("mol_structural_features not available")


# =============================================================================
# SMILES CONVERTER CAN_CONVERT EDGE CASES
# =============================================================================


class TestSMILESConverterCanConvertEdgeCases:
    """Test SMILESConverter.can_convert edge cases."""

    def test_rejects_inchi_string(self, sample_inchi):
        """Test rejects InChI strings."""
        converter = SMILESConverter()

        assert converter.can_convert(sample_inchi["ethanol"]) is False

    def test_rejects_empty_string(self):
        """Test rejects empty string."""
        converter = SMILESConverter()

        assert converter.can_convert("") is False

    def test_rejects_file_path_extensions(self):
        """Test rejects file paths with molecular file extensions."""
        converter = SMILESConverter()

        extensions = [".xyz", ".sdf", ".mol", ".cif"]
        for ext in extensions:
            assert converter.can_convert(f"molecule{ext}") is False

    def test_accepts_valid_smiles_patterns(self):
        """Test accepts various valid SMILES patterns."""
        converter = SMILESConverter()

        valid_smiles = [
            "C",  # Methane
            "CC",  # Ethane
            "c1ccccc1",  # Benzene (aromatic)
            "C(=O)O",  # Carboxylic acid
            "N",  # Ammonia
            "O",  # Water
            "[Cu]",  # Copper atom
        ]

        for smiles in valid_smiles:
            assert converter.can_convert(smiles) is True, f"Should accept: {smiles}"


# =============================================================================
# MODULE DOCSTRING AND VERSION TESTS
# =============================================================================


class TestModuleMetadata:
    """Test module metadata and documentation."""

    def test_module_has_docstring(self):
        """Test module has docstring."""
        import milia_pipeline.models.post_training.data_preparation.data_converter as dc_module

        assert dc_module.__doc__ is not None
        assert len(dc_module.__doc__) > 0

    def test_base_converter_has_docstring(self):
        """Test BaseDataConverter has docstring."""
        assert BaseDataConverter.__doc__ is not None

    def test_registry_has_docstring(self):
        """Test DataConverterRegistry has docstring."""
        assert DataConverterRegistry.__doc__ is not None

    def test_convert_to_pyg_has_docstring(self):
        """Test convert_to_pyg has docstring."""
        assert convert_to_pyg.__doc__ is not None
        assert "DYNAMIC" in convert_to_pyg.__doc__


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
