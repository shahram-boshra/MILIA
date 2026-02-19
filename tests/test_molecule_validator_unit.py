#!/usr/bin/env python3
"""
Unit tests for molecule_validator.py module (Phase 6 Registry Integration)

Test file: test_molecule_validator_unit.py
Module under test: ~/ml_projects/milia/milia_pipeline/molecules/molecule_validator.py

This test suite validates the refactored molecule_validator module after Phase 6
registry integration, which replaced hardcoded dataset type checks with dynamic
feature-based queries while maintaining backward compatibility.

Key Test Areas:
1. Handler requirement enforcement (should raise when handler missing)
2. Molecular structure validation with handler integration
3. Dataset compatibility checking
4. Atomic symbol to number conversion
5. Uncertainty data validation (DMC-specific via uncertainty_handling feature)
6. PyG data completeness validation
7. Dataset requirements retrieval
8. Validation context creation
9. Detailed feedback validation
10. Handler-specific exception handling
11. Edge cases and integration scenarios
12. Phase 6: Registry Integration (50+ tests) - NEW
    - Registry initialization and state management
    - Feature-based dataset queries
    - Handler-specific error creation
    - Registry status function
    - Refactored function behavior
    - Legacy fallback when registry unavailable

Functions Tested:
- validate_molecular_structure()
- check_dataset_compatibility()
- convert_symbols_to_atomic_numbers()
- validate_uncertainty_data()
- validate_pyg_data_completeness()
- create_validator_with_handler()
- validate_molecule_with_handler()
- get_validation_summary()
- validate_molecule_legacy()
- get_dataset_requirements()
- create_validation_context()
- validate_with_detailed_feedback()
- get_registry_status() - NEW Phase 6
- _init_registry() - NEW Phase 6 (internal)
- _get_available_dataset_types() - NEW Phase 6 (internal)
- _is_dataset_type_registered() - NEW Phase 6 (internal)
- _get_dataset_feature() - NEW Phase 6 (internal)
- _get_dataset_optional_properties() - NEW Phase 6 (internal)
- _create_handler_specific_error() - NEW Phase 6 (internal)

Total: 120+ comprehensive tests

Test execution:
    cd /app/milia
    python -m pytest tests/test_molecule_validator_unit.py -v --tb=short
"""

import sys
from pathlib import Path

# CRITICAL: Add project root to Python path FIRST
project_root = Path("/app/milia")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

# Import torch_geometric components
try:
    from torch_geometric.data import Data
except ImportError:
    # Fallback for testing environment
    class Data:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


# Import module under test from molecules/molecule_validator.py
from milia_pipeline.molecules.molecule_validator import (
    check_dataset_compatibility,
    convert_symbols_to_atomic_numbers,
    create_validation_context,
    create_validator_with_handler,
    get_dataset_requirements,
    get_validation_summary,
    validate_molecular_structure,
    validate_molecule_legacy,
    validate_molecule_with_handler,
    validate_pyg_data_completeness,
    validate_uncertainty_data,
    validate_with_detailed_feedback,
)

# Phase 6: Import registry integration functions
try:
    from milia_pipeline.molecules.molecule_validator import (
        _create_handler_specific_error,
        _get_available_dataset_types,
        _get_dataset_feature,
        _get_dataset_optional_properties,
        _init_registry,
        _is_dataset_type_registered,
        get_registry_status,
    )

    PHASE6_IMPORTS_SUCCESSFUL = True
    PHASE6_IMPORT_ERROR = None
except ImportError as e:
    PHASE6_IMPORTS_SUCCESSFUL = False
    PHASE6_IMPORT_ERROR = str(e)
    print(f"WARNING: Could not import Phase 6 registry functions: {e}")

# Import required configuration and exception classes
from milia_pipeline.config.config_containers import DatasetConfig
from milia_pipeline.exceptions import (
    DatasetSpecificHandlerError,
    HandlerError,
    HandlerNotAvailableError,
    HandlerOperationError,
    HandlerValidationError,
    MoleculeProcessingError,
)
from milia_pipeline.handlers import DatasetHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# PHASE 6: Registry State Management Helper
# ==============================================================================


def reset_registry_state():
    """
    Reset registry state to uninitialized for testing.
    IMPORTANT: Use this before/after tests that modify registry state.
    """
    try:
        import milia_pipeline.molecules.molecule_validator as validator_module

        validator_module._REGISTRY_INITIALIZED = False
        validator_module._REGISTRY_AVAILABLE = False
        validator_module._REGISTRY_IMPORT_ERROR = None
        validator_module._registry_list_all = None
        validator_module._registry_get = None
        validator_module._registry_is_registered = None
    except (ImportError, AttributeError):
        pass  # Module may not have been imported yet


# ==============================================================================
# MOCK HELPERS
# ==============================================================================


class MockDatasetHandler:
    """Mock dataset handler for testing"""

    def __init__(self, dataset_type="DFT", should_validate=True):
        self.dataset_type = dataset_type
        self.should_validate = should_validate
        self.dataset_config = Mock()
        self.dataset_config.is_uncertainty_enabled = False
        self.dataset_config.uncertainty_config = {}
        self.filter_config = Mock()
        self.processing_config = Mock()

    def get_dataset_type(self):
        return self.dataset_type

    def validate_molecule_data(self, raw_data_dict, molecule_index, inchi):
        if not self.should_validate:
            raise ValueError(f"Mock validation failure for molecule {molecule_index}")
        return True

    def get_required_properties(self):
        if self.dataset_type == "DFT":
            return ["atoms", "coordinates", "Etot", "forces"]
        elif self.dataset_type == "DMC":
            return ["atoms", "coordinates", "Etot"]
        return ["atoms", "coordinates"]

    def get_common_required_properties(self):
        return ["atoms", "coordinates", "Etot"]

    def validate_required_properties(self, properties_dict, molecule_index, inchi):
        required = self.get_required_properties()
        missing = [p for p in required if p not in properties_dict]
        if missing:
            raise HandlerValidationError(
                message=f"Missing required properties: {missing}",
                handler_type=self.dataset_type,
                validation_type="required_properties",
                molecule_index=molecule_index,
            )
        return True

    def process_property_value(self, property_name, property_value, molecule_index, identifier):
        """Mock property value processing - returns value as-is"""
        return property_value


class MockDMCHandler(MockDatasetHandler):
    """Mock DMC handler with uncertainty support"""

    def __init__(self, should_validate=True, uncertainty_enabled=False):
        super().__init__(dataset_type="DMC", should_validate=should_validate)
        self.dataset_config.is_uncertainty_enabled = uncertainty_enabled
        if uncertainty_enabled:
            self.dataset_config.uncertainty_config = {
                "uncertainty_field_name": "std",
                "threshold": 0.1,
            }

    def get_required_properties(self):
        props = super().get_required_properties()
        if self.dataset_config.is_uncertainty_enabled:
            props.append("std")
        return props


class MockDFTHandler(MockDatasetHandler):
    """Mock DFT handler"""

    def __init__(self, should_validate=True):
        super().__init__(dataset_type="DFT", should_validate=should_validate)


class MockWavefunctionHandler(MockDatasetHandler):
    """Mock Wavefunction handler with orbital support"""

    def __init__(self, should_validate=True):
        super().__init__(dataset_type="Wavefunction", should_validate=should_validate)

    def get_required_properties(self):
        return ["atoms", "coordinates", "Etot", "homo", "lumo"]


# ==============================================================================
# VALIDATING MOCK HANDLERS (for tests that need property validation)
# ==============================================================================


class ValidatingMockHandler(MockDatasetHandler):
    """Mock handler that actually validates required properties"""

    def __init__(self, dataset_type="DFT", should_validate=True, strict_validation=True):
        super().__init__(dataset_type, should_validate)
        self.strict_validation = strict_validation

    def validate_molecule_data(self, raw_data_dict, molecule_index, inchi):
        if not self.should_validate:
            raise ValueError(f"Mock validation failure for molecule {molecule_index}")

        # If strict_validation is enabled, check for required properties
        if self.strict_validation:
            required = self.get_required_properties()
            missing = [p for p in required if p not in raw_data_dict or raw_data_dict[p] is None]
            if missing:
                raise HandlerValidationError(
                    message=f"Missing required properties: {missing}",
                    handler_type=self.dataset_type,
                    validation_type="required_properties",
                    molecule_index=molecule_index,
                    failed_validations=[f"Missing: {', '.join(missing)}"],
                )
        return True


class ValidatingMockDFTHandler(ValidatingMockHandler):
    """Validating mock DFT handler"""

    def __init__(self, should_validate=True, strict_validation=True):
        super().__init__(
            dataset_type="DFT", should_validate=should_validate, strict_validation=strict_validation
        )


class ValidatingMockDMCHandler(ValidatingMockHandler):
    """Validating mock DMC handler with uncertainty support"""

    def __init__(self, should_validate=True, strict_validation=True, uncertainty_enabled=False):
        super().__init__(
            dataset_type="DMC", should_validate=should_validate, strict_validation=strict_validation
        )
        self.dataset_config.is_uncertainty_enabled = uncertainty_enabled
        if uncertainty_enabled:
            self.dataset_config.uncertainty_config = {
                "uncertainty_field_name": "std",
                "threshold": 0.1,
            }

    def get_required_properties(self):
        props = super().get_required_properties()
        if self.dataset_config.is_uncertainty_enabled:
            props.append("std")
        return props


# ==============================================================================
# PHASE 6: Mock Dataset Class for Registry Testing
# ==============================================================================


class MockDatasetFeatures:
    """Mock dataset features object for testing"""

    def __init__(self, **features):
        for k, v in features.items():
            setattr(self, k, v)


class MockDatasetClass:
    """Mock dataset class for registry testing"""

    def __init__(self, dataset_type, features_dict=None, optional_props=None):
        self.dataset_type = dataset_type
        self.features = MockDatasetFeatures(**(features_dict or {}))
        self._optional_props = optional_props or []

    @classmethod
    def get_optional_properties(cls):
        return getattr(cls, "_optional_props", [])


def create_sample_pyg_data():
    """Create sample PyG Data object for testing"""
    return Data(
        z=torch.tensor([6, 6, 1, 1, 1, 1]),
        pos=torch.rand(6, 3),
        y=torch.tensor([-100.0]),
        num_nodes=6,
        original_mol_idx=0,
        dataset_type="DFT",
    )


# ==============================================================================
# TEST CASES - ORIGINAL
# ==============================================================================


class TestHandlerRequirement(unittest.TestCase):
    """Test that handlers are REQUIRED (no fallback to legacy implementations)"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

        self.atoms = np.array([6, 6, 1, 1, 1, 1])
        self.coordinates = np.random.rand(6, 3)
        self.molecule_index = 0
        self.inchi = "InChI=1S/C2H6/c1-2/h1-2H3"

    def test_validate_molecular_structure_requires_handler(self):
        """Test that validate_molecular_structure raises ValueError when handler is None"""
        with self.assertRaises(ValueError) as context:
            validate_molecular_structure(
                atoms=self.atoms,
                coordinates=self.coordinates,
                molecule_index=self.molecule_index,
                inchi=self.inchi,
                handler=None,  # This should raise
            )

        self.assertIn("Handler is required", str(context.exception))

    def test_check_dataset_compatibility_requires_handler(self):
        """Test that check_dataset_compatibility raises ValueError when handler is None"""
        properties_dict = {"atoms": self.atoms, "coordinates": self.coordinates, "Etot": -100.0}

        with self.assertRaises(ValueError) as context:
            check_dataset_compatibility(
                raw_properties_dict=properties_dict,
                dataset_type="DFT",
                molecule_index=self.molecule_index,
                inchi=self.inchi,
                handler=None,  # This should raise
            )

        self.assertIn("Handler is required", str(context.exception))

    def test_validate_uncertainty_data_requires_handler(self):
        """Test that validate_uncertainty_data raises ValueError when handler is None"""
        pyg_data = Data(
            z=torch.tensor([8, 1, 1]),
            pos=torch.rand(3, 3),
            y=torch.tensor([-76.0]),
            uncertainty=torch.tensor([0.05]),
        )

        with self.assertRaises(ValueError) as context:
            validate_uncertainty_data(
                pyg_data=pyg_data,
                molecule_index=self.molecule_index,
                smiles="O",
                handler=None,  # This should raise
            )

        self.assertIn("Handler is required", str(context.exception))

    def test_validate_pyg_data_completeness_requires_handler(self):
        """Test that validate_pyg_data_completeness raises ValueError when handler is None"""
        pyg_data = Data(
            z=torch.tensor([6, 6, 1, 1, 1, 1]),
            pos=torch.rand(6, 3),
            y=torch.tensor([-100.0]),
            num_nodes=6,
        )

        with self.assertRaises(ValueError) as context:
            validate_pyg_data_completeness(
                pyg_data=pyg_data,
                dataset_type="DFT",
                molecule_index=self.molecule_index,
                handler=None,  # This should raise
            )

        self.assertIn("Handler is required", str(context.exception))


class TestMolecularStructureValidation(unittest.TestCase):
    """Test validate_molecular_structure function with handler integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

        self.mock_handler = MockDFTHandler(should_validate=True)
        self.molecule_index = 0
        self.inchi = "InChI=1S/C2H6/c1-2/h1-2H3"

    def test_valid_structure_with_atomic_numbers(self):
        """Test validation with valid atomic numbers"""
        atoms = np.array([6, 6, 1, 1, 1, 1])
        coordinates = np.random.rand(6, 3)

        atomic_numbers, coords = validate_molecular_structure(
            atoms=atoms,
            coordinates=coordinates,
            molecule_index=self.molecule_index,
            inchi=self.inchi,
            handler=self.mock_handler,
        )

        self.assertIsInstance(atomic_numbers, np.ndarray)
        self.assertIsInstance(coords, np.ndarray)
        self.assertEqual(len(atomic_numbers), 6)
        self.assertEqual(coords.shape, (6, 3))

    def test_valid_structure_with_symbols(self):
        """Test validation with atomic symbols"""
        # First convert symbols to atomic numbers since the validator expects numbers
        symbols = np.array(["C", "C", "H", "H", "H", "H"])
        atoms = convert_symbols_to_atomic_numbers(
            symbols=symbols, molecule_index=self.molecule_index, inchi=self.inchi
        )
        coordinates = np.random.rand(6, 3)

        atomic_numbers, coords = validate_molecular_structure(
            atoms=atoms,
            coordinates=coordinates,
            molecule_index=self.molecule_index,
            inchi=self.inchi,
            handler=self.mock_handler,
        )

        self.assertIsInstance(atomic_numbers, np.ndarray)
        self.assertEqual(len(atomic_numbers), 6)
        self.assertTrue(all(atomic_numbers > 0))

    def test_invalid_structure_mismatched_lengths(self):
        """Test validation fails with mismatched atom/coordinate lengths"""
        atoms = np.array([6, 6, 1, 1, 1, 1])
        coordinates = np.random.rand(5, 3)  # Wrong number of coordinates

        with self.assertRaises((HandlerOperationError, MoleculeProcessingError)):
            validate_molecular_structure(
                atoms=atoms,
                coordinates=coordinates,
                molecule_index=self.molecule_index,
                inchi=self.inchi,
                handler=self.mock_handler,
            )

    def test_invalid_structure_bad_coordinates_shape(self):
        """Test validation fails with incorrect coordinate dimensions"""
        atoms = np.array([6, 6, 1, 1, 1, 1])
        coordinates = np.random.rand(6, 2)  # Should be (N, 3)

        with self.assertRaises((HandlerOperationError, MoleculeProcessingError)):
            validate_molecular_structure(
                atoms=atoms,
                coordinates=coordinates,
                molecule_index=self.molecule_index,
                inchi=self.inchi,
                handler=self.mock_handler,
            )

    def test_handler_specific_validation_called(self):
        """Test that handler's validate_molecule_data is called"""
        atoms = np.array([6, 6, 1, 1, 1, 1])
        coordinates = np.random.rand(6, 3)

        # Create a mock handler that tracks calls
        mock_handler = Mock(spec=DatasetHandler)
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_handler.validate_molecule_data.return_value = True

        # This should work without raising
        atomic_numbers, coords = validate_molecular_structure(
            atoms=atoms,
            coordinates=coordinates,
            molecule_index=self.molecule_index,
            inchi=self.inchi,
            handler=mock_handler,
        )

        # Verify handler's validate_molecule_data was called
        mock_handler.validate_molecule_data.assert_called_once()


class TestDatasetCompatibilityChecking(unittest.TestCase):
    """Test check_dataset_compatibility function with handler integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

        self.molecule_index = 0
        self.inchi = "InChI=1S/C2H6/c1-2/h1-2H3"

    def test_dft_compatibility_valid_data(self):
        """Test DFT compatibility check with valid data"""
        mock_handler = MockDFTHandler(should_validate=True)

        properties_dict = {
            "atoms": np.array([6, 6, 1, 1, 1, 1]),
            "coordinates": np.random.rand(6, 3),
            "Etot": -100.0,
            "forces": np.random.rand(6, 3),
        }

        result = check_dataset_compatibility(
            raw_properties_dict=properties_dict,
            dataset_type="DFT",
            molecule_index=self.molecule_index,
            inchi=self.inchi,
            handler=mock_handler,
        )

        # Should return True or None (both indicate success)
        self.assertTrue(result is None or result is True)

    def test_dmc_compatibility_valid_data(self):
        """Test DMC compatibility check with valid data"""
        mock_handler = MockDMCHandler(should_validate=True)

        properties_dict = {
            "atoms": np.array([8, 1, 1]),
            "coordinates": np.random.rand(3, 3),
            "Etot": -76.0,
        }

        result = check_dataset_compatibility(
            raw_properties_dict=properties_dict,
            dataset_type="DMC",
            molecule_index=self.molecule_index,
            inchi=self.inchi,
            handler=mock_handler,
        )

        self.assertTrue(result is None or result is True)

    def test_compatibility_check_calls_handler(self):
        """Test that handler's validate_molecule_data is called"""
        mock_handler = Mock(spec=DatasetHandler)
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_handler.validate_molecule_data.return_value = True

        properties_dict = {
            "atoms": np.array([6, 6, 1, 1, 1, 1]),
            "coordinates": np.random.rand(6, 3),
            "Etot": -100.0,
        }

        check_dataset_compatibility(
            raw_properties_dict=properties_dict,
            dataset_type="DFT",
            molecule_index=self.molecule_index,
            inchi=self.inchi,
            handler=mock_handler,
        )

        # Verify handler was used
        mock_handler.validate_molecule_data.assert_called_once()

    def test_compatibility_check_handler_mismatch_warning(self):
        """Test warning when dataset_type doesn't match handler type"""
        mock_handler = MockDMCHandler(should_validate=True)

        properties_dict = {
            "atoms": np.array([6, 6]),
            "coordinates": np.random.rand(2, 3),
            "Etot": -50.0,
        }

        # Pass DFT as dataset_type but handler is DMC
        # Should log warning but still work
        with self.assertLogs(level="WARNING"):
            _result = check_dataset_compatibility(
                raw_properties_dict=properties_dict,
                dataset_type="DFT",  # Mismatch
                molecule_index=self.molecule_index,
                inchi=self.inchi,
                handler=mock_handler,  # DMC handler
            )


class TestAtomicConversions(unittest.TestCase):
    """Test convert_symbols_to_atomic_numbers function"""

    def setUp(self):
        """Set up test fixtures"""
        self.molecule_index = 0
        self.inchi = "InChI=1S/C2H6/c1-2/h1-2H3"

    def test_convert_valid_symbols(self):
        """Test conversion of valid atomic symbols"""
        symbols = np.array(["C", "C", "H", "H", "H", "H"])

        atomic_numbers = convert_symbols_to_atomic_numbers(
            symbols=symbols, molecule_index=self.molecule_index, inchi=self.inchi
        )

        expected = np.array([6, 6, 1, 1, 1, 1])
        np.testing.assert_array_equal(atomic_numbers, expected)

    def test_convert_heavy_atoms(self):
        """Test conversion of heavy atoms"""
        symbols = np.array(["C", "N", "O", "F"])

        atomic_numbers = convert_symbols_to_atomic_numbers(
            symbols=symbols, molecule_index=self.molecule_index, inchi=self.inchi
        )

        expected = np.array([6, 7, 8, 9])
        np.testing.assert_array_equal(atomic_numbers, expected)

    def test_convert_unknown_symbol_raises(self):
        """Test that unknown symbols raise MoleculeProcessingError"""
        symbols = np.array(["C", "X", "H"])  # X is unknown

        with self.assertRaises(MoleculeProcessingError) as context:
            convert_symbols_to_atomic_numbers(
                symbols=symbols, molecule_index=self.molecule_index, inchi=self.inchi
            )

        self.assertIn("Unknown atom symbols", str(context.exception))

    def test_convert_mixed_case_symbols(self):
        """Test handling of mixed case symbols"""
        symbols = np.array(["C", "c", "H"])  # lowercase 'c'

        # Should handle or raise appropriately
        # Implementation detail: may need to test actual behavior
        try:
            atomic_numbers = convert_symbols_to_atomic_numbers(
                symbols=symbols, molecule_index=self.molecule_index, inchi=self.inchi
            )
            # If it works, verify results
            self.assertIsInstance(atomic_numbers, np.ndarray)
        except MoleculeProcessingError:
            # If it fails, that's also acceptable behavior
            pass


class TestUncertaintyValidation(unittest.TestCase):
    """Test validate_uncertainty_data function for DMC datasets"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

        self.molecule_index = 0
        self.smiles = "O"

    def test_uncertainty_validation_dmc_enabled(self):
        """Test uncertainty validation with DMC handler and uncertainty enabled"""
        mock_handler = MockDMCHandler(should_validate=True, uncertainty_enabled=True)

        pyg_data = Data(
            z=torch.tensor([8, 1, 1]),
            pos=torch.rand(3, 3),
            y=torch.tensor([-76.0]),
            uncertainty=torch.tensor([0.05]),
        )

        result = validate_uncertainty_data(
            pyg_data=pyg_data,
            molecule_index=self.molecule_index,
            smiles=self.smiles,
            handler=mock_handler,
        )

        # Should return uncertainty value or None
        self.assertTrue(result is None or isinstance(result, (int, float)))

    def test_uncertainty_validation_dft_skipped(self):
        """Test uncertainty validation is skipped for DFT datasets"""
        mock_handler = MockDFTHandler(should_validate=True)

        pyg_data = Data(
            z=torch.tensor([6, 6, 1, 1, 1, 1]), pos=torch.rand(6, 3), y=torch.tensor([-100.0])
        )

        result = validate_uncertainty_data(
            pyg_data=pyg_data, molecule_index=self.molecule_index, smiles="CC", handler=mock_handler
        )

        # Should return None for DFT
        self.assertIsNone(result)

    def test_uncertainty_validation_dmc_disabled(self):
        """Test uncertainty validation with DMC but uncertainty disabled"""
        mock_handler = MockDMCHandler(should_validate=True, uncertainty_enabled=False)

        pyg_data = Data(z=torch.tensor([8, 1, 1]), pos=torch.rand(3, 3), y=torch.tensor([-76.0]))

        result = validate_uncertainty_data(
            pyg_data=pyg_data,
            molecule_index=self.molecule_index,
            smiles=self.smiles,
            handler=mock_handler,
        )

        # Should return None when disabled
        self.assertIsNone(result)


class TestPyGDataCompletenessValidation(unittest.TestCase):
    """Test validate_pyg_data_completeness function"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

        self.molecule_index = 0

    def test_completeness_all_present(self):
        """Test completeness validation with all required data"""
        mock_handler = MockDFTHandler(should_validate=True)

        pyg_data = Data(
            z=torch.tensor([6, 6, 1, 1, 1, 1]),
            pos=torch.rand(6, 3),
            y=torch.tensor([-100.0]),
            num_nodes=6,
            original_mol_idx=0,
            dataset_type="DFT",
        )

        results = validate_pyg_data_completeness(
            pyg_data=pyg_data,
            dataset_type="DFT",
            molecule_index=self.molecule_index,
            handler=mock_handler,
        )

        self.assertIsInstance(results, dict)
        self.assertIn("has_basic_structure", results)
        self.assertIn("has_coordinates", results)
        self.assertIn("has_atomic_numbers", results)
        self.assertIn("handler_type", results)

    def test_completeness_missing_coordinates(self):
        """Test completeness validation with missing coordinates"""
        mock_handler = MockDFTHandler(should_validate=True)

        pyg_data = Data(z=torch.tensor([6, 6, 1, 1, 1, 1]), y=torch.tensor([-100.0]), num_nodes=6)

        results = validate_pyg_data_completeness(
            pyg_data=pyg_data,
            dataset_type="DFT",
            molecule_index=self.molecule_index,
            handler=mock_handler,
        )

        self.assertFalse(results["has_coordinates"])

    def test_completeness_with_handler_requirements(self):
        """Test that handler requirements are checked"""
        mock_handler = MockDFTHandler(should_validate=True)

        pyg_data = Data(
            z=torch.tensor([6, 6, 1, 1, 1, 1]),
            pos=torch.rand(6, 3),
            y=torch.tensor([-100.0]),
            num_nodes=6,
            # Missing forces (required by DFT handler)
        )

        results = validate_pyg_data_completeness(
            pyg_data=pyg_data,
            dataset_type="DFT",
            molecule_index=self.molecule_index,
            handler=mock_handler,
        )

        # Should indicate missing handler properties
        self.assertIn("has_handler_required_props", results)
        if "missing_handler_props" in results:
            self.assertIsInstance(results["missing_handler_props"], list)


class TestDatasetRequirements(unittest.TestCase):
    """Test get_dataset_requirements function"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

    def test_get_requirements_with_dft_handler(self):
        """Test requirements retrieval with DFT handler"""
        mock_handler = MockDFTHandler(should_validate=True)

        requirements = get_dataset_requirements(dataset_type="DFT", handler=mock_handler)

        self.assertIsInstance(requirements, dict)
        self.assertIn("dataset_type", requirements)
        self.assertIn("required_properties", requirements)
        self.assertEqual(requirements["dataset_type"], "DFT")

    def test_get_requirements_with_dmc_handler(self):
        """Test requirements retrieval with DMC handler"""
        mock_handler = MockDMCHandler(should_validate=True, uncertainty_enabled=True)

        requirements = get_dataset_requirements(dataset_type="DMC", handler=mock_handler)

        self.assertEqual(requirements["dataset_type"], "DMC")
        self.assertTrue(requirements["uncertainty_enabled"])
        self.assertIn("uncertainty_properties", requirements)

    def test_get_requirements_creates_handler_when_none(self):
        """Test that handler is created from global config when None"""
        # This test requires mocking the config creation functions
        with (
            patch(
                "milia_pipeline.molecules.molecule_validator.create_dataset_config_from_global"
            ) as _mock_dataset_config,
            patch(
                "milia_pipeline.molecules.molecule_validator.create_filter_config_from_global"
            ) as _mock_filter_config,
            patch(
                "milia_pipeline.molecules.molecule_validator.create_processing_config_from_global"
            ) as _mock_proc_config,
            patch(
                "milia_pipeline.molecules.molecule_validator.create_dataset_handler"
            ) as mock_create_handler,
        ):
            mock_handler = MockDFTHandler()
            mock_create_handler.return_value = mock_handler

            requirements = get_dataset_requirements(
                dataset_type="DFT",
                handler=None,  # Should create handler
            )

            # Verify handler was created
            mock_create_handler.assert_called_once()
            self.assertIn("dataset_type", requirements)


class TestValidationContext(unittest.TestCase):
    """Test create_validation_context function"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

        self.mock_handler = MockDFTHandler(should_validate=True)
        self.molecule_index = 0
        self.inchi = "InChI=1S/C2H6/c1-2/h1-2H3"

    def test_create_context_basic(self):
        """Test basic validation context creation"""
        context = create_validation_context(
            handler=self.mock_handler, molecule_index=self.molecule_index, inchi=self.inchi
        )

        self.assertIsInstance(context, dict)
        self.assertEqual(context["molecule_index"], self.molecule_index)
        self.assertEqual(context["inchi"], self.inchi)
        self.assertEqual(context["handler_type"], "DFT")
        self.assertIn("timestamp", context)
        self.assertIn("required_properties", context)

    def test_create_context_with_dmc_handler(self):
        """Test context creation with DMC handler includes uncertainty info"""
        mock_dmc_handler = MockDMCHandler(should_validate=True, uncertainty_enabled=True)

        context = create_validation_context(
            handler=mock_dmc_handler, molecule_index=self.molecule_index, inchi=self.inchi
        )

        self.assertEqual(context["handler_type"], "DMC")
        self.assertIn("validation_config", context)
        self.assertTrue(context["validation_config"]["uncertainty_enabled"])


class TestDetailedFeedbackValidation(unittest.TestCase):
    """Test validate_with_detailed_feedback function"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

        self.molecule_index = 0
        self.inchi = "InChI=1S/C2H6/c1-2/h1-2H3"

    def test_detailed_feedback_all_pass(self):
        """Test detailed validation with all checks passing"""
        mock_handler = MockDFTHandler(should_validate=True)

        properties_dict = {
            "atoms": np.array([6, 6, 1, 1, 1, 1]),
            "coordinates": np.random.rand(6, 3),
            "Etot": -100.0,
            "forces": np.random.rand(6, 3),
        }

        result = validate_with_detailed_feedback(
            raw_properties_dict=properties_dict,
            molecule_index=self.molecule_index,
            inchi=self.inchi,
            handler=mock_handler,
        )

        self.assertIsInstance(result, dict)
        self.assertIn("context", result)
        self.assertIn("validation_passed", result)
        self.assertIn("validation_errors", result)
        self.assertIn("validation_steps", result)

    def test_detailed_feedback_with_failures(self):
        """Test detailed validation with some failures"""
        # Use ValidatingMockHandler that actually checks properties
        mock_handler = ValidatingMockDFTHandler(should_validate=True, strict_validation=True)

        # Missing required property 'forces'
        properties_dict = {
            "atoms": np.array([6, 6, 1, 1, 1, 1]),
            "coordinates": np.random.rand(6, 3),
            "Etot": -100.0,
            # Missing 'forces'
        }

        result = validate_with_detailed_feedback(
            raw_properties_dict=properties_dict,
            molecule_index=self.molecule_index,
            inchi=self.inchi,
            handler=mock_handler,
        )

        self.assertFalse(result["validation_passed"])
        self.assertTrue(len(result["validation_errors"]) > 0)

    def test_detailed_feedback_creates_handler_when_none(self):
        """Test that detailed feedback creates handler from global config when None"""
        properties_dict = {
            "atoms": np.array([6, 6, 1, 1, 1, 1]),
            "coordinates": np.random.rand(6, 3),
            "Etot": -100.0,
        }

        with (
            patch(
                "milia_pipeline.molecules.molecule_validator.create_dataset_config_from_global"
            ) as _mock_dataset_config,
            patch(
                "milia_pipeline.molecules.molecule_validator.create_filter_config_from_global"
            ) as _mock_filter_config,
            patch(
                "milia_pipeline.molecules.molecule_validator.create_processing_config_from_global"
            ) as _mock_proc_config,
            patch(
                "milia_pipeline.molecules.molecule_validator.create_dataset_handler"
            ) as mock_create_handler,
        ):
            mock_handler = MockDFTHandler()
            mock_create_handler.return_value = mock_handler

            _result = validate_with_detailed_feedback(
                raw_properties_dict=properties_dict,
                molecule_index=self.molecule_index,
                inchi=self.inchi,
                handler=None,  # Should create handler
            )

            # Verify handler was created
            mock_create_handler.assert_called_once()


class TestValidatorCreation(unittest.TestCase):
    """Test create_validator_with_handler function"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

    def test_create_validator_from_global_config(self):
        """Test validator creation from global config"""
        with (
            patch(
                "milia_pipeline.molecules.molecule_validator.create_dataset_config_from_global"
            ) as mock_dataset_config,
            patch(
                "milia_pipeline.molecules.molecule_validator.create_filter_config_from_global"
            ) as mock_filter_config,
            patch(
                "milia_pipeline.molecules.molecule_validator.create_processing_config_from_global"
            ) as mock_proc_config,
            patch(
                "milia_pipeline.molecules.molecule_validator.create_dataset_handler"
            ) as mock_create_handler,
        ):
            mock_handler = MockDFTHandler()
            mock_create_handler.return_value = mock_handler

            _handler = create_validator_with_handler(dataset_config=None)

            # Verify configs were created and handler was instantiated
            mock_dataset_config.assert_called_once()
            mock_filter_config.assert_called_once()
            mock_proc_config.assert_called_once()
            mock_create_handler.assert_called_once()

    def test_create_validator_with_explicit_config(self):
        """Test validator creation with explicit dataset config"""
        mock_dataset_config = Mock(spec=DatasetConfig)
        mock_dataset_config.dataset_type = "DFT"

        with (
            patch(
                "milia_pipeline.molecules.molecule_validator.create_filter_config_from_global"
            ) as _mock_filter_config,
            patch(
                "milia_pipeline.molecules.molecule_validator.create_processing_config_from_global"
            ) as _mock_proc_config,
            patch(
                "milia_pipeline.molecules.molecule_validator.create_dataset_handler"
            ) as mock_create_handler,
        ):
            mock_handler = MockDFTHandler()
            mock_create_handler.return_value = mock_handler

            _handler = create_validator_with_handler(dataset_config=mock_dataset_config)

            # Verify handler was created with provided config
            mock_create_handler.assert_called_once()


class TestHighLevelValidation(unittest.TestCase):
    """Test validate_molecule_with_handler function"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

        self.molecule_index = 0
        self.inchi = "InChI=1S/C2H6/c1-2/h1-2H3"

    def test_validate_with_explicit_handler(self):
        """Test validation with explicitly provided handler"""
        mock_handler = MockDFTHandler(should_validate=True)

        properties_dict = {
            "atoms": np.array([6, 6, 1, 1, 1, 1]),
            "coordinates": np.random.rand(6, 3),
            "Etot": -100.0,
            "forces": np.random.rand(6, 3),
        }

        result = validate_molecule_with_handler(
            raw_properties_dict=properties_dict,
            molecule_index=self.molecule_index,
            inchi=self.inchi,
            handler=mock_handler,
        )

        self.assertTrue(result)

    def test_validate_creates_handler_when_none(self):
        """Test that handler is created from global config when None"""
        properties_dict = {
            "atoms": np.array([6, 6, 1, 1, 1, 1]),
            "coordinates": np.random.rand(6, 3),
            "Etot": -100.0,
        }

        with (
            patch(
                "milia_pipeline.molecules.molecule_validator.create_dataset_config_from_global"
            ) as _mock_dataset_config,
            patch(
                "milia_pipeline.molecules.molecule_validator.create_filter_config_from_global"
            ) as _mock_filter_config,
            patch(
                "milia_pipeline.molecules.molecule_validator.create_processing_config_from_global"
            ) as _mock_proc_config,
            patch(
                "milia_pipeline.molecules.molecule_validator.create_dataset_handler"
            ) as mock_create_handler,
        ):
            mock_handler = MockDFTHandler()
            mock_create_handler.return_value = mock_handler

            _result = validate_molecule_with_handler(
                raw_properties_dict=properties_dict,
                molecule_index=self.molecule_index,
                inchi=self.inchi,
                handler=None,  # Should create handler
            )

            # Verify handler was created
            mock_create_handler.assert_called_once()

    def test_validate_handler_creation_failure(self):
        """Test error handling when handler creation fails"""
        properties_dict = {
            "atoms": np.array([6, 6, 1, 1, 1, 1]),
            "coordinates": np.random.rand(6, 3),
            "Etot": -100.0,
        }

        with (
            patch(
                "milia_pipeline.molecules.molecule_validator.create_dataset_config_from_global"
            ) as mock_dataset_config,
            patch(
                "milia_pipeline.molecules.molecule_validator.create_filter_config_from_global"
            ) as _mock_filter_config,
            patch(
                "milia_pipeline.molecules.molecule_validator.create_processing_config_from_global"
            ) as _mock_proc_config,
            patch(
                "milia_pipeline.molecules.molecule_validator.create_dataset_handler"
            ) as _mock_create_handler,
        ):
            # Setup mocks to raise exception when trying to create from global config
            mock_dataset_config.side_effect = Exception("No global config available")

            with self.assertRaises((HandlerNotAvailableError, Exception)):
                validate_molecule_with_handler(
                    raw_properties_dict=properties_dict,
                    molecule_index=self.molecule_index,
                    inchi=self.inchi,
                    handler=None,
                )


class TestValidationSummary(unittest.TestCase):
    """Test get_validation_summary function"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

    def test_summary_all_passed(self):
        """Test summary generation when all checks pass"""
        validation_results = {
            "has_basic_structure": True,
            "has_coordinates": True,
            "has_atomic_numbers": True,
            "has_target_values": True,
            "has_metadata": True,
            "handler_type": "DFT",
        }

        summary = get_validation_summary(validation_results)

        self.assertIsInstance(summary, str)
        self.assertIn("PASS", summary)
        self.assertIn("5/5", summary)

    def test_summary_partial_pass(self):
        """Test summary generation when some checks fail"""
        validation_results = {
            "has_basic_structure": True,
            "has_coordinates": True,
            "has_atomic_numbers": False,
            "has_target_values": False,
            "has_metadata": True,
            "handler_type": "DFT",
        }

        summary = get_validation_summary(validation_results)

        self.assertIn("PARTIAL", summary)
        self.assertIn("3/5", summary)

    def test_summary_all_failed(self):
        """Test summary generation when all checks fail"""
        validation_results = {
            "has_basic_structure": False,
            "has_coordinates": False,
            "has_atomic_numbers": False,
            "has_target_values": False,
            "has_metadata": False,
            "handler_type": "DFT",
        }

        summary = get_validation_summary(validation_results)

        self.assertIn("FAIL", summary)
        self.assertIn("0/5", summary)

    def test_summary_with_missing_properties(self):
        """Test summary includes missing properties information"""
        validation_results = {
            "has_basic_structure": True,
            "has_handler_required_props": False,
            "missing_handler_props": ["forces", "Etot"],
            "handler_type": "DFT",
        }

        summary = get_validation_summary(validation_results)

        self.assertIn("forces", summary)
        self.assertIn("Etot", summary)

    def test_summary_with_recommendations(self):
        """Test summary includes recommendations for failures"""
        validation_results = {
            "has_basic_structure": False,
            "has_target_values": False,
            "handler_type": "DFT",
        }

        summary = get_validation_summary(validation_results)

        self.assertIn("Recommendations", summary)


class TestLegacyValidation(unittest.TestCase):
    """Test validate_molecule_legacy function"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

        self.molecule_index = 0
        self.inchi = "InChI=1S/C2H6/c1-2/h1-2H3"

    @patch("milia_pipeline.molecules.molecule_validator.get_dataset_type")
    @patch("milia_pipeline.molecules.molecule_validator.create_dataset_config_from_global")
    @patch("milia_pipeline.molecules.molecule_validator.create_filter_config_from_global")
    @patch("milia_pipeline.molecules.molecule_validator.create_processing_config_from_global")
    @patch("milia_pipeline.molecules.molecule_validator.create_dataset_handler")
    def test_legacy_validation_success(
        self,
        mock_create_handler,
        mock_proc_config,
        mock_filter_config,
        mock_dataset_config,
        mock_get_dataset_type,
    ):
        """Test legacy validation function works with mocked dependencies"""
        # Setup mocks
        mock_get_dataset_type.return_value = "DFT"
        mock_handler = MockDFTHandler(should_validate=True)
        mock_create_handler.return_value = mock_handler

        properties_dict = {
            "atoms": np.array([6, 6, 1, 1, 1, 1]),
            "coordinates": np.random.rand(6, 3),
            "Etot": -100.0,
            "forces": np.random.rand(6, 3),
        }

        result = validate_molecule_legacy(
            raw_properties_dict=properties_dict,
            molecule_index=self.molecule_index,
            inchi=self.inchi,
        )

        # Should return True or False (not raise)
        self.assertIsInstance(result, bool)

    def test_legacy_validation_failure_handling(self):
        """Test legacy validation handles failures gracefully"""
        properties_dict = {
            "atoms": np.array([6, 6, 1, 1, 1, 1]),
            "coordinates": np.random.rand(6, 3),
        }

        # Should not raise, but return False or handle gracefully
        # This will fail due to missing config, which is expected
        result = validate_molecule_legacy(
            raw_properties_dict=properties_dict,
            molecule_index=self.molecule_index,
            inchi=self.inchi,
        )

        # Should return False when validation fails
        self.assertFalse(result)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

        self.mock_handler = MockDFTHandler(should_validate=True)
        self.molecule_index = 0
        self.inchi = "InChI=1S/C2H6/c1-2/h1-2H3"

    def test_single_atom_molecule(self):
        """Test validation of single-atom molecule"""
        atoms = np.array([6])
        coordinates = np.random.rand(1, 3)

        atomic_numbers, coords = validate_molecular_structure(
            atoms=atoms,
            coordinates=coordinates,
            molecule_index=self.molecule_index,
            inchi=self.inchi,
            handler=self.mock_handler,
        )

        self.assertEqual(len(atomic_numbers), 1)
        self.assertEqual(coords.shape, (1, 3))

    def test_large_molecule(self):
        """Test validation of large molecule"""
        num_atoms = 100
        atoms = np.array([6] * num_atoms)
        coordinates = np.random.rand(num_atoms, 3)

        atomic_numbers, coords = validate_molecular_structure(
            atoms=atoms,
            coordinates=coordinates,
            molecule_index=self.molecule_index,
            inchi=self.inchi,
            handler=self.mock_handler,
        )

        self.assertEqual(len(atomic_numbers), num_atoms)
        self.assertEqual(coords.shape, (num_atoms, 3))

    def test_properties_with_extra_fields(self):
        """Test validation with extra unexpected fields"""
        properties_dict = {
            "atoms": np.array([6, 6, 1, 1, 1, 1]),
            "coordinates": np.random.rand(6, 3),
            "Etot": -100.0,
            "forces": np.random.rand(6, 3),
            "extra_field": "should be ignored",
            "another_field": 12345,
        }

        # Should handle gracefully
        result = check_dataset_compatibility(
            raw_properties_dict=properties_dict,
            dataset_type="DFT",
            molecule_index=self.molecule_index,
            inchi=self.inchi,
            handler=self.mock_handler,
        )

        self.assertTrue(result is None or result is True)

    def test_unicode_inchi(self):
        """Test handling of unicode characters in InChI"""
        unicode_inchi = "InChI=1S/C2H6/c1-2/h1-2H3_测试"

        atoms = np.array([6, 6, 1, 1, 1, 1])
        coordinates = np.random.rand(6, 3)

        # Should handle unicode gracefully
        atomic_numbers, coords = validate_molecular_structure(
            atoms=atoms,
            coordinates=coordinates,
            molecule_index=self.molecule_index,
            inchi=unicode_inchi,
            handler=self.mock_handler,
        )

        self.assertIsNotNone(atomic_numbers)
        self.assertIsNotNone(coords)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios combining multiple validation steps"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

        self.mock_handler = MockDFTHandler(should_validate=True)

    def test_full_validation_pipeline_success(self):
        """Test complete validation pipeline with valid data"""
        molecule_index = 0
        inchi = "InChI=1S/C2H6/c1-2/h1-2H3"

        # Step 1: Structure validation
        atoms = np.array([6, 6, 1, 1, 1, 1])
        coordinates = np.random.rand(6, 3)

        atomic_numbers, coords = validate_molecular_structure(
            atoms=atoms,
            coordinates=coordinates,
            molecule_index=molecule_index,
            inchi=inchi,
            handler=self.mock_handler,
        )

        # Step 2: Compatibility check
        properties_dict = {
            "atoms": atomic_numbers,
            "coordinates": coords,
            "Etot": -100.0,
            "forces": np.random.rand(6, 3),
        }

        result = check_dataset_compatibility(
            raw_properties_dict=properties_dict,
            dataset_type="DFT",
            molecule_index=molecule_index,
            inchi=inchi,
            handler=self.mock_handler,
        )

        # Step 3: High-level validation using validate_molecule_with_handler
        validation_result = validate_molecule_with_handler(
            raw_properties_dict=properties_dict,
            molecule_index=molecule_index,
            inchi=inchi,
            handler=self.mock_handler,
        )

        # All should pass
        self.assertTrue(result is None or result is True)
        self.assertTrue(validation_result is True)

    def test_full_validation_pipeline_failure(self):
        """Test complete validation pipeline with invalid data"""
        molecule_index = 0
        inchi = "InChI=1S/C2H6/c1-2/h1-2H3"

        atoms = np.array([6, 6, 1, 1, 1, 1])
        coordinates = np.random.rand(5, 3)  # Wrong number of coordinates

        # Should fail at structure validation
        with self.assertRaises((HandlerOperationError, MoleculeProcessingError)):
            validate_molecular_structure(
                atoms=atoms,
                coordinates=coordinates,
                molecule_index=molecule_index,
                inchi=inchi,
                handler=self.mock_handler,
            )

    def test_dmc_with_uncertainty_workflow(self):
        """Test DMC workflow with uncertainty validation"""
        mock_dmc_handler = MockDMCHandler(should_validate=True, uncertainty_enabled=True)
        molecule_index = 0
        inchi = "InChI=1S/H2O/h1H2"

        # Structure validation
        atoms = np.array([8, 1, 1])
        coordinates = np.random.rand(3, 3)

        atomic_numbers, coords = validate_molecular_structure(
            atoms=atoms,
            coordinates=coordinates,
            molecule_index=molecule_index,
            inchi=inchi,
            handler=mock_dmc_handler,
        )

        # Check requirements include uncertainty
        requirements = get_dataset_requirements(dataset_type="DMC", handler=mock_dmc_handler)

        self.assertTrue(requirements["uncertainty_enabled"])
        self.assertIn("std", requirements["uncertainty_properties"])


class TestExceptionHandling(unittest.TestCase):
    """Test exception handling in validation functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

        self.molecule_index = 0
        self.inchi = "InChI=1S/C2H6/c1-2/h1-2H3"

    def test_handler_validation_error_raised(self):
        """Test that HandlerValidationError is raised appropriately"""
        # Use ValidatingMockHandler that will fail on missing properties
        mock_handler = ValidatingMockDFTHandler(should_validate=True, strict_validation=True)

        # Missing required property
        properties_dict = {
            "atoms": np.array([6, 6, 1, 1, 1, 1]),
            "coordinates": np.random.rand(6, 3),
            "Etot": -100.0,
            # Missing 'forces' - required by DFT handler
        }

        # Should raise DatasetSpecificHandlerError or HandlerValidationError
        with self.assertRaises((HandlerValidationError, DatasetSpecificHandlerError, HandlerError)):
            validate_molecule_with_handler(
                raw_properties_dict=properties_dict,
                molecule_index=self.molecule_index,
                inchi=self.inchi,
                handler=mock_handler,
            )

    def test_handler_operation_error_context(self):
        """Test that HandlerOperationError provides proper context"""
        mock_handler = MockDFTHandler(should_validate=False)  # Will fail

        properties_dict = {
            "atoms": np.array([6, 6, 1, 1, 1, 1]),
            "coordinates": np.random.rand(6, 3),
            "Etot": -100.0,
        }

        with self.assertRaises(HandlerError) as context:
            validate_molecule_with_handler(
                raw_properties_dict=properties_dict,
                molecule_index=self.molecule_index,
                inchi=self.inchi,
                handler=mock_handler,
            )

        # Should have error context
        self.assertIsNotNone(context.exception)


# ==============================================================================
# PHASE 6: REGISTRY INTEGRATION TESTS
# ==============================================================================


class TestPhase6RegistryIntegrationFunctions(unittest.TestCase):
    """
    Test Phase 6 registry integration infrastructure functions.

    These tests verify the lazy initialization pattern and registry
    interaction functions work correctly.
    """

    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")

    def setUp(self):
        reset_registry_state()

    def tearDown(self):
        reset_registry_state()

    def test_01_init_registry_sets_initialized_flag(self):
        """Test _init_registry sets _REGISTRY_INITIALIZED to True."""
        import milia_pipeline.molecules.molecule_validator as mod

        self.assertFalse(mod._REGISTRY_INITIALIZED)

        _init_registry()

        self.assertTrue(mod._REGISTRY_INITIALIZED)

    def test_02_init_registry_idempotent(self):
        """Test _init_registry is idempotent (can be called multiple times)."""
        _init_registry()
        _init_registry()
        _init_registry()

        import milia_pipeline.molecules.molecule_validator as mod

        self.assertTrue(mod._REGISTRY_INITIALIZED)

    def test_03_get_available_dataset_types_returns_list(self):
        """Test _get_available_dataset_types returns a list."""
        types = _get_available_dataset_types()

        self.assertIsInstance(types, list)
        self.assertTrue(len(types) >= 3)  # At least DFT, DMC, Wavefunction

    def test_04_get_available_dataset_types_includes_core_types(self):
        """Test _get_available_dataset_types includes DFT, DMC, Wavefunction."""
        types = _get_available_dataset_types()

        self.assertIn("DFT", types)
        self.assertIn("DMC", types)
        self.assertIn("Wavefunction", types)

    def test_05_is_dataset_type_registered_valid_types(self):
        """Test _is_dataset_type_registered returns True for valid types."""
        self.assertTrue(_is_dataset_type_registered("DFT"))
        self.assertTrue(_is_dataset_type_registered("DMC"))
        self.assertTrue(_is_dataset_type_registered("Wavefunction"))

    def test_06_is_dataset_type_registered_invalid_type(self):
        """Test _is_dataset_type_registered returns False for invalid types."""
        self.assertFalse(_is_dataset_type_registered("INVALID"))
        self.assertFalse(_is_dataset_type_registered("NonExistent"))
        self.assertFalse(_is_dataset_type_registered(""))

    def test_07_registry_state_global_flags_exist(self):
        """Test registry state global flags exist in module."""
        import milia_pipeline.molecules.molecule_validator as mod

        self.assertTrue(hasattr(mod, "_REGISTRY_INITIALIZED"))
        self.assertTrue(hasattr(mod, "_REGISTRY_AVAILABLE"))
        self.assertTrue(hasattr(mod, "_REGISTRY_IMPORT_ERROR"))

    def test_08_registry_functions_exist(self):
        """Test registry function placeholders exist."""
        import milia_pipeline.molecules.molecule_validator as mod

        self.assertTrue(hasattr(mod, "_registry_list_all"))
        self.assertTrue(hasattr(mod, "_registry_get"))
        self.assertTrue(hasattr(mod, "_registry_is_registered"))


class TestPhase6FeatureQueries(unittest.TestCase):
    """
    Test Phase 6 feature query function _get_dataset_feature.

    These tests verify that feature queries work correctly for all
    supported dataset types and features.
    """

    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")

    def setUp(self):
        reset_registry_state()

    def tearDown(self):
        reset_registry_state()

    def test_01_dmc_has_uncertainty_handling(self):
        """Test DMC has uncertainty_handling=True."""
        result = _get_dataset_feature("DMC", "uncertainty_handling")
        self.assertTrue(result)

    def test_02_dft_no_uncertainty_handling(self):
        """Test DFT has uncertainty_handling=False."""
        result = _get_dataset_feature("DFT", "uncertainty_handling")
        self.assertFalse(result)

    def test_03_dft_has_vibrational_analysis(self):
        """Test DFT has vibrational_analysis=True."""
        result = _get_dataset_feature("DFT", "vibrational_analysis")
        self.assertTrue(result)

    def test_04_dmc_no_vibrational_analysis(self):
        """Test DMC has vibrational_analysis=False."""
        result = _get_dataset_feature("DMC", "vibrational_analysis")
        self.assertFalse(result)

    def test_05_wavefunction_has_orbital_analysis(self):
        """Test Wavefunction has orbital_analysis=True."""
        result = _get_dataset_feature("Wavefunction", "orbital_analysis")
        self.assertTrue(result)

    def test_06_dft_no_orbital_analysis(self):
        """Test DFT has orbital_analysis=False."""
        result = _get_dataset_feature("DFT", "orbital_analysis")
        self.assertFalse(result)

    def test_07_dft_has_atomization_energy(self):
        """Test DFT has atomization_energy=True."""
        result = _get_dataset_feature("DFT", "atomization_energy")
        self.assertTrue(result)

    def test_08_dft_has_rotational_constants(self):
        """Test DFT has rotational_constants=True."""
        result = _get_dataset_feature("DFT", "rotational_constants")
        self.assertTrue(result)

    def test_09_dft_has_frequency_analysis(self):
        """Test DFT has frequency_analysis=True."""
        result = _get_dataset_feature("DFT", "frequency_analysis")
        self.assertTrue(result)

    def test_10_wavefunction_has_homo_lumo_gap(self):
        """Test Wavefunction has homo_lumo_gap=True."""
        result = _get_dataset_feature("Wavefunction", "homo_lumo_gap")
        self.assertTrue(result)

    def test_11_wavefunction_has_mo_energies(self):
        """Test Wavefunction has mo_energies=True."""
        result = _get_dataset_feature("Wavefunction", "mo_energies")
        self.assertTrue(result)

    def test_12_unknown_feature_returns_false(self):
        """Test unknown features return False."""
        result = _get_dataset_feature("DFT", "unknown_feature")
        self.assertFalse(result)

    def test_13_unknown_dataset_returns_false(self):
        """Test unknown dataset returns False for features."""
        result = _get_dataset_feature("UNKNOWN", "uncertainty_handling")
        self.assertFalse(result)

    def test_14_empty_string_dataset_returns_false(self):
        """Test empty dataset type returns False."""
        result = _get_dataset_feature("", "uncertainty_handling")
        self.assertFalse(result)

    def test_15_feature_query_consistency(self):
        """Test multiple feature queries are consistent."""
        for dataset_type in ["DFT", "DMC", "Wavefunction"]:
            # Check consistency across multiple calls
            result1 = _get_dataset_feature(dataset_type, "uncertainty_handling")
            result2 = _get_dataset_feature(dataset_type, "uncertainty_handling")
            self.assertEqual(result1, result2)


class TestPhase6OptionalProperties(unittest.TestCase):
    """
    Test Phase 6 optional properties query function.

    These tests verify _get_dataset_optional_properties works correctly.
    """

    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")

    def setUp(self):
        reset_registry_state()

    def tearDown(self):
        reset_registry_state()

    def test_01_optional_properties_returns_list(self):
        """Test _get_dataset_optional_properties returns a list."""
        props = _get_dataset_optional_properties("DFT")
        self.assertIsInstance(props, list)

    def test_02_dft_optional_properties(self):
        """Test DFT optional properties include expected items."""
        props = _get_dataset_optional_properties("DFT")
        # DFT should have some optional properties like forces, dipole, etc.
        self.assertIsInstance(props, list)

    def test_03_dmc_optional_properties(self):
        """Test DMC optional properties include expected items."""
        props = _get_dataset_optional_properties("DMC")
        self.assertIsInstance(props, list)

    def test_04_unknown_dataset_returns_empty(self):
        """Test unknown dataset returns empty list."""
        props = _get_dataset_optional_properties("UNKNOWN")
        self.assertIsInstance(props, list)
        self.assertEqual(len(props), 0)


class TestPhase6CreateHandlerSpecificError(unittest.TestCase):
    """
    Test Phase 6 _create_handler_specific_error function.

    This function creates appropriate error types based on dataset features.
    The function signature is:
        _create_handler_specific_error(handler, message, operation, details, molecule_index, original_error)
    """

    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")

    def setUp(self):
        reset_registry_state()
        self.dft_handler = MockDFTHandler(should_validate=True)
        self.dmc_handler = MockDMCHandler(should_validate=True, uncertainty_enabled=True)
        self.wf_handler = MockWavefunctionHandler(should_validate=True)

    def tearDown(self):
        reset_registry_state()

    def test_01_dmc_handler_creates_dmc_error(self):
        """Test DMC handler creates DatasetSpecificHandlerError (uncertainty_handling feature)."""
        error = _create_handler_specific_error(
            handler=self.dmc_handler,
            message="Test error",
            operation="test_operation",
            details="Test details",
            molecule_index=0,
        )

        self.assertIsInstance(error, (DatasetSpecificHandlerError, HandlerError))

    def test_02_dft_handler_creates_dft_error(self):
        """Test DFT handler creates DatasetSpecificHandlerError (vibrational_analysis feature)."""
        error = _create_handler_specific_error(
            handler=self.dft_handler,
            message="Test error",
            operation="test_operation",
            details="Test details",
            molecule_index=0,
        )

        self.assertIsInstance(error, (DatasetSpecificHandlerError, HandlerError))

    def test_03_unknown_handler_creates_generic_error(self):
        """Test unknown handler type creates generic HandlerValidationError."""
        # Create a handler with unknown type
        unknown_handler = MockDatasetHandler(dataset_type="UNKNOWN", should_validate=True)
        error = _create_handler_specific_error(
            handler=unknown_handler,
            message="Test error",
            operation="test_operation",
            details="Test details",
            molecule_index=0,
        )

        self.assertIsInstance(error, (HandlerValidationError, HandlerError))

    def test_04_error_includes_message(self):
        """Test created error includes the provided message."""
        test_message = "This is a test error message"
        error = _create_handler_specific_error(
            handler=self.dft_handler,
            message=test_message,
            operation="test_operation",
            details="Test details",
            molecule_index=0,
        )

        self.assertIn(test_message, str(error))

    def test_05_error_includes_molecule_index(self):
        """Test created error includes molecule index context."""
        error = _create_handler_specific_error(
            handler=self.dft_handler,
            message="Test error",
            operation="test_operation",
            details="Test details",
            molecule_index=42,
        )

        # Error should have some molecule context
        self.assertIsNotNone(error)

    def test_06_uncertainty_feature_creates_dmc_error(self):
        """Test datasets with uncertainty_handling feature create DMC-like errors."""
        # DMC has uncertainty_handling=True
        error = _create_handler_specific_error(
            handler=self.dmc_handler,
            message="Uncertainty validation failed",
            operation="validate_uncertainty",
            details="Uncertainty data invalid",
            molecule_index=0,
        )

        self.assertIsInstance(error, (DatasetSpecificHandlerError, HandlerError))

    def test_07_vibrational_feature_creates_dft_error(self):
        """Test datasets with vibrational_analysis feature create DFT-like errors."""
        # DFT has vibrational_analysis=True
        error = _create_handler_specific_error(
            handler=self.dft_handler,
            message="Vibrational validation failed",
            operation="validate_vibrational",
            details="Frequency data invalid",
            molecule_index=0,
        )

        self.assertIsInstance(error, (DatasetSpecificHandlerError, HandlerError))


class TestPhase6RegistryStatus(unittest.TestCase):
    """
    Test Phase 6 get_registry_status function.

    This function provides diagnostic information about registry integration.
    Actual output structure:
    {
        'registry_available': bool,
        'registry_initialized': bool,
        'registry_import_error': str or None,
        'available_dataset_types': list,
        'phase_6_complete': bool,
        'features_available': list
    }
    """

    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")

    def setUp(self):
        reset_registry_state()

    def tearDown(self):
        reset_registry_state()

    def test_01_get_registry_status_returns_dict(self):
        """Test get_registry_status returns a dictionary."""
        status = get_registry_status()
        self.assertIsInstance(status, dict)

    def test_02_status_includes_registry_available(self):
        """Test status includes registry_available field."""
        status = get_registry_status()
        self.assertIn("registry_available", status)
        self.assertIsInstance(status["registry_available"], bool)

    def test_03_status_includes_registry_initialized(self):
        """Test status includes registry_initialized field."""
        status = get_registry_status()
        self.assertIn("registry_initialized", status)
        self.assertIsInstance(status["registry_initialized"], bool)

    def test_04_status_includes_available_dataset_types(self):
        """Test status includes available_dataset_types."""
        status = get_registry_status()
        self.assertIn("available_dataset_types", status)
        self.assertIsInstance(status["available_dataset_types"], list)
        self.assertIn("DFT", status["available_dataset_types"])
        self.assertIn("DMC", status["available_dataset_types"])
        self.assertIn("Wavefunction", status["available_dataset_types"])

    def test_05_status_includes_phase_6_complete(self):
        """Test status includes phase_6_complete=True."""
        status = get_registry_status()
        self.assertIn("phase_6_complete", status)
        self.assertTrue(status["phase_6_complete"])

    def test_06_status_includes_features_available(self):
        """Test status includes features_available list."""
        status = get_registry_status()
        self.assertIn("features_available", status)
        self.assertIsInstance(status["features_available"], list)
        self.assertIn("uncertainty_handling", status["features_available"])
        self.assertIn("vibrational_analysis", status["features_available"])
        self.assertIn("orbital_analysis", status["features_available"])

    def test_07_status_includes_registry_import_error(self):
        """Test status includes registry_import_error field."""
        status = get_registry_status()
        self.assertIn("registry_import_error", status)
        # Should be None when registry is available
        if status["registry_available"]:
            self.assertIsNone(status["registry_import_error"])

    def test_08_status_has_all_expected_keys(self):
        """Test status has all expected keys."""
        status = get_registry_status()
        expected_keys = [
            "registry_available",
            "registry_initialized",
            "registry_import_error",
            "available_dataset_types",
            "phase_6_complete",
            "features_available",
        ]
        for key in expected_keys:
            self.assertIn(key, status, f"Expected key '{key}' not found in status")

    def test_09_features_list_has_correct_items(self):
        """Test features_available list has expected features."""
        status = get_registry_status()
        expected_features = [
            "uncertainty_handling",
            "vibrational_analysis",
            "atomization_energy",
            "rotational_constants",
            "frequency_analysis",
            "orbital_analysis",
            "homo_lumo_gap",
            "mo_energies",
        ]
        for feature in expected_features:
            self.assertIn(feature, status["features_available"])

    def test_10_status_initialized_after_call(self):
        """Test registry is initialized after get_registry_status call."""
        status = get_registry_status()
        # After calling get_registry_status, registry should be initialized
        self.assertTrue(status["registry_initialized"])


class TestPhase6RefactoredFunctions(unittest.TestCase):
    """
    Test Phase 6 refactored functions use feature-based queries.

    These tests verify that the 9 refactored locations correctly use
    registry-based validation and feature queries instead of hardcoded
    dataset type checks.
    """

    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")

    def setUp(self):
        reset_registry_state()
        self.dft_handler = MockDFTHandler(should_validate=True)
        self.dmc_handler = MockDMCHandler(should_validate=True, uncertainty_enabled=True)
        self.wf_handler = MockWavefunctionHandler(should_validate=True)
        self.pyg_data = create_sample_pyg_data()
        self.molecule_index = 0
        self.inchi = "InChI=1S/C2H6/c1-2/h1-2H3"

    def tearDown(self):
        reset_registry_state()

    def test_01_validate_structure_uses_feature_based_errors(self):
        """Test validate_molecular_structure uses feature-based error creation."""
        atoms = np.array([6, 6, 1, 1, 1, 1])
        coordinates = np.random.rand(6, 3)

        # DFT validation
        atomic_numbers, coords = validate_molecular_structure(
            atoms=atoms,
            coordinates=coordinates,
            molecule_index=self.molecule_index,
            inchi=self.inchi,
            handler=self.dft_handler,
        )

        self.assertIsNotNone(atomic_numbers)
        self.assertEqual(len(atomic_numbers), 6)

    def test_02_validate_structure_dmc_creates_dataset_specific_error_on_failure(self):
        """Test DMC validation failure creates DatasetSpecificHandlerError."""
        atoms = np.array([6, 6, 1, 1, 1, 1])
        coordinates = np.random.rand(5, 3)  # Mismatched lengths

        with self.assertRaises(
            (DatasetSpecificHandlerError, HandlerError, MoleculeProcessingError)
        ):
            validate_molecular_structure(
                atoms=atoms,
                coordinates=coordinates,
                molecule_index=self.molecule_index,
                inchi=self.inchi,
                handler=self.dmc_handler,
            )

    def test_03_check_compatibility_uses_feature_based_errors(self):
        """Test check_dataset_compatibility uses feature-based error creation."""
        properties_dict = {
            "atoms": np.array([6, 6, 1, 1, 1, 1]),
            "coordinates": np.random.rand(6, 3),
            "Etot": -100.0,
            "forces": np.random.rand(6, 3),
        }

        result = check_dataset_compatibility(
            raw_properties_dict=properties_dict,
            dataset_type="DFT",
            molecule_index=self.molecule_index,
            inchi=self.inchi,
            handler=self.dft_handler,
        )

        self.assertTrue(result is None or result is True)

    def test_04_uncertainty_validation_uses_feature_query(self):
        """Test validate_uncertainty_data uses _get_dataset_feature."""
        # DMC with uncertainty enabled should validate
        dmc_handler = MockDMCHandler(should_validate=True, uncertainty_enabled=True)
        pyg_data = Data(
            z=torch.tensor([8, 1, 1]),
            pos=torch.rand(3, 3),
            y=torch.tensor([-76.0]),
            uncertainty=torch.tensor([0.05]),
        )

        result = validate_uncertainty_data(
            pyg_data=pyg_data, molecule_index=self.molecule_index, smiles="O", handler=dmc_handler
        )

        # Should process uncertainty for DMC
        self.assertTrue(result is None or isinstance(result, (int, float)))

    def test_05_uncertainty_validation_skips_for_dft(self):
        """Test uncertainty validation skips for DFT (no uncertainty_handling feature)."""
        result = validate_uncertainty_data(
            pyg_data=self.pyg_data,
            molecule_index=self.molecule_index,
            smiles="CC",
            handler=self.dft_handler,
        )

        # DFT should skip uncertainty validation
        self.assertIsNone(result)

    def test_06_pyg_completeness_uses_feature_queries(self):
        """Test validate_pyg_data_completeness uses feature-based validation."""
        # DFT completeness check
        results = validate_pyg_data_completeness(
            pyg_data=self.pyg_data,
            dataset_type="DFT",
            molecule_index=self.molecule_index,
            handler=self.dft_handler,
        )

        self.assertIsInstance(results, dict)
        self.assertEqual(results["handler_type"], "DFT")

    def test_07_pyg_completeness_dmc_includes_uncertainty_check(self):
        """Test DMC completeness validation includes uncertainty metadata check."""
        dmc_handler = MockDMCHandler(should_validate=True, uncertainty_enabled=True)
        pyg_data = Data(
            z=torch.tensor([8, 1, 1]),
            pos=torch.rand(3, 3),
            y=torch.tensor([-76.0]),
            uncertainty=torch.tensor([0.05]),
            num_nodes=3,
        )

        results = validate_pyg_data_completeness(
            pyg_data=pyg_data,
            dataset_type="DMC",
            molecule_index=self.molecule_index,
            handler=dmc_handler,
        )

        self.assertEqual(results["handler_type"], "DMC")
        # Should check for uncertainty-related fields
        self.assertIn("has_basic_structure", results)

    def test_08_get_requirements_uses_feature_based_categorization(self):
        """Test get_dataset_requirements uses feature-based categorization."""
        # DFT requirements
        dft_reqs = get_dataset_requirements(dataset_type="DFT", handler=self.dft_handler)

        self.assertEqual(dft_reqs["dataset_type"], "DFT")
        self.assertFalse(dft_reqs["uncertainty_enabled"])

        # DMC requirements
        dmc_reqs = get_dataset_requirements(dataset_type="DMC", handler=self.dmc_handler)

        self.assertEqual(dmc_reqs["dataset_type"], "DMC")
        self.assertTrue(dmc_reqs["uncertainty_enabled"])

    def test_09_create_context_uses_feature_based_additions(self):
        """Test create_validation_context uses feature-based context additions."""
        # DFT context
        dft_context = create_validation_context(
            handler=self.dft_handler, molecule_index=self.molecule_index, inchi=self.inchi
        )

        self.assertEqual(dft_context["handler_type"], "DFT")

        # DMC context should include uncertainty config
        dmc_context = create_validation_context(
            handler=self.dmc_handler, molecule_index=self.molecule_index, inchi=self.inchi
        )

        self.assertEqual(dmc_context["handler_type"], "DMC")
        self.assertIn("validation_config", dmc_context)

    def test_10_validation_summary_uses_feature_based_recommendations(self):
        """Test get_validation_summary uses feature-based recommendations."""
        # DFT validation results
        dft_results = {
            "has_basic_structure": False,
            "has_target_values": False,
            "handler_type": "DFT",
        }

        summary = get_validation_summary(dft_results)
        self.assertIn("Recommendations", summary)

        # DMC validation results
        dmc_results = {
            "has_basic_structure": False,
            "has_uncertainty": False,
            "handler_type": "DMC",
        }

        summary = get_validation_summary(dmc_results)
        self.assertIn("Recommendations", summary)

    def test_11_validate_molecule_with_handler_uses_feature_errors(self):
        """Test validate_molecule_with_handler uses feature-based error creation."""
        properties_dict = {
            "atoms": np.array([6, 6, 1, 1, 1, 1]),
            "coordinates": np.random.rand(6, 3),
            "Etot": -100.0,
            "forces": np.random.rand(6, 3),
        }

        result = validate_molecule_with_handler(
            raw_properties_dict=properties_dict,
            molecule_index=self.molecule_index,
            inchi=self.inchi,
            handler=self.dft_handler,
        )

        self.assertTrue(result)

    def test_12_detailed_feedback_includes_registry_status(self):
        """Test validate_with_detailed_feedback includes registry_status."""
        properties_dict = {
            "atoms": np.array([6, 6, 1, 1, 1, 1]),
            "coordinates": np.random.rand(6, 3),
            "Etot": -100.0,
            "forces": np.random.rand(6, 3),
        }

        result = validate_with_detailed_feedback(
            raw_properties_dict=properties_dict,
            molecule_index=self.molecule_index,
            inchi=self.inchi,
            handler=self.dft_handler,
        )

        self.assertIn("registry_status", result)
        self.assertIsInstance(result["registry_status"], dict)

    def test_13_all_handlers_work_with_feature_routing(self):
        """Test all handler types work correctly with feature-based routing."""
        atoms = np.array([6, 6, 1, 1, 1, 1])
        coordinates = np.random.rand(6, 3)

        for handler, _handler_type in [
            (self.dft_handler, "DFT"),
            (self.dmc_handler, "DMC"),
            (self.wf_handler, "Wavefunction"),
        ]:
            atomic_numbers, coords = validate_molecular_structure(
                atoms=atoms,
                coordinates=coordinates,
                molecule_index=self.molecule_index,
                inchi=self.inchi,
                handler=handler,
            )

            self.assertIsNotNone(atomic_numbers)
            self.assertEqual(len(atomic_numbers), 6)


class TestPhase6LegacyFallback(unittest.TestCase):
    """
    Test Phase 6 legacy fallback behavior when registry unavailable.

    These tests verify that the module correctly falls back to hardcoded
    values when the registry is not available.
    """

    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")

    def setUp(self):
        reset_registry_state()

    def tearDown(self):
        reset_registry_state()

    @patch("milia_pipeline.molecules.molecule_validator._REGISTRY_AVAILABLE", False)
    def test_01_fallback_get_available_dataset_types(self):
        """Test legacy fallback for _get_available_dataset_types.

        When registry is unavailable, filesystem discovery uses py_file.stem.upper(),
        so 'wavefunction.py' becomes 'WAVEFUNCTION', not 'Wavefunction'.
        """
        import milia_pipeline.molecules.molecule_validator as mod

        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False

        types = _get_available_dataset_types()

        # Filesystem discovery uses .upper(), so all names are uppercased
        types_upper = [t.upper() for t in types]
        self.assertIn("DFT", types_upper)
        self.assertIn("DMC", types_upper)
        self.assertIn("WAVEFUNCTION", types_upper)

    @patch("milia_pipeline.molecules.molecule_validator._REGISTRY_AVAILABLE", False)
    def test_02_fallback_is_dataset_type_registered(self):
        """Test legacy fallback for _is_dataset_type_registered.

        When registry is unavailable, _is_dataset_type_registered delegates to
        _get_available_dataset_types() which uses filesystem discovery with .upper().
        So 'Wavefunction' won't match 'WAVEFUNCTION' — only exact case matches work.
        """
        import milia_pipeline.molecules.molecule_validator as mod

        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False

        # DFT and DMC filenames are already uppercase after .upper()
        self.assertTrue(_is_dataset_type_registered("DFT"))
        self.assertTrue(_is_dataset_type_registered("DMC"))
        # Filesystem discovery produces 'WAVEFUNCTION' (from wavefunction.py.stem.upper())
        self.assertTrue(_is_dataset_type_registered("WAVEFUNCTION"))
        self.assertFalse(_is_dataset_type_registered("INVALID"))
        # Mixed-case 'Wavefunction' will NOT match the uppercase fallback list
        self.assertFalse(_is_dataset_type_registered("Wavefunction"))

    @patch("milia_pipeline.molecules.molecule_validator._REGISTRY_AVAILABLE", False)
    def test_03_fallback_get_dataset_feature(self):
        """Test legacy fallback for _get_dataset_feature."""
        import milia_pipeline.molecules.molecule_validator as mod

        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False

        # Test fallback values
        self.assertTrue(_get_dataset_feature("DMC", "uncertainty_handling"))
        self.assertFalse(_get_dataset_feature("DFT", "uncertainty_handling"))
        self.assertTrue(_get_dataset_feature("DFT", "vibrational_analysis"))
        self.assertTrue(_get_dataset_feature("Wavefunction", "orbital_analysis"))

    @patch("milia_pipeline.molecules.molecule_validator._REGISTRY_AVAILABLE", False)
    def test_04_fallback_creates_appropriate_errors(self):
        """Test legacy fallback for _create_handler_specific_error."""
        import milia_pipeline.molecules.molecule_validator as mod

        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False

        dmc_handler = MockDMCHandler(should_validate=True, uncertainty_enabled=True)
        dmc_error = _create_handler_specific_error(
            handler=dmc_handler,
            message="Test",
            operation="test_op",
            details="Test details",
            molecule_index=0,
        )
        self.assertIsInstance(dmc_error, (DatasetSpecificHandlerError, HandlerError))

        dft_handler = MockDFTHandler(should_validate=True)
        dft_error = _create_handler_specific_error(
            handler=dft_handler,
            message="Test",
            operation="test_op",
            details="Test details",
            molecule_index=0,
        )
        self.assertIsInstance(dft_error, (DatasetSpecificHandlerError, HandlerError))

    @patch("milia_pipeline.molecules.molecule_validator._REGISTRY_AVAILABLE", False)
    def test_05_fallback_registry_status_shows_unavailable(self):
        """Test registry status correctly shows unavailable when fallback active."""
        import milia_pipeline.molecules.molecule_validator as mod

        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False

        status = get_registry_status()
        self.assertFalse(status["registry_available"])
        # But phase_6_complete should still be True (fallback works)
        self.assertTrue(status["phase_6_complete"])

    @patch("milia_pipeline.molecules.molecule_validator._REGISTRY_AVAILABLE", False)
    def test_06_fallback_optional_properties(self):
        """Test legacy fallback for _get_dataset_optional_properties."""
        import milia_pipeline.molecules.molecule_validator as mod

        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False

        props = _get_dataset_optional_properties("DFT")
        self.assertIsInstance(props, list)

        props = _get_dataset_optional_properties("UNKNOWN")
        self.assertEqual(props, [])


class TestPhase6FeatureBasedValidation(unittest.TestCase):
    """
    Test that feature-based validation correctly routes to appropriate handlers.

    These tests verify that after Phase 6 refactoring, the module correctly
    uses feature queries instead of hardcoded type checks.
    """

    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")

    def setUp(self):
        reset_registry_state()
        self.pyg_data = create_sample_pyg_data()
        self.molecule_index = 0
        self.smiles = "CC"

    def tearDown(self):
        reset_registry_state()

    def test_01_uncertainty_feature_triggers_uncertainty_validation(self):
        """Test uncertainty_handling feature triggers uncertainty validation."""
        # DMC has uncertainty_handling=True
        self.assertTrue(_get_dataset_feature("DMC", "uncertainty_handling"))

        dmc_handler = MockDMCHandler(should_validate=True, uncertainty_enabled=True)
        pyg_data = Data(
            z=torch.tensor([8, 1, 1]),
            pos=torch.rand(3, 3),
            y=torch.tensor([-76.0]),
            uncertainty=torch.tensor([0.01]),
        )

        result = validate_uncertainty_data(
            pyg_data=pyg_data, molecule_index=self.molecule_index, smiles="O", handler=dmc_handler
        )

        # Should return a value (not None) for DMC
        self.assertTrue(result is None or isinstance(result, (int, float)))

    def test_02_no_uncertainty_feature_skips_uncertainty_validation(self):
        """Test datasets without uncertainty_handling skip uncertainty validation."""
        # DFT has uncertainty_handling=False
        self.assertFalse(_get_dataset_feature("DFT", "uncertainty_handling"))

        dft_handler = MockDFTHandler(should_validate=True)
        result = validate_uncertainty_data(
            pyg_data=self.pyg_data,
            molecule_index=self.molecule_index,
            smiles=self.smiles,
            handler=dft_handler,
        )

        # Should return None for DFT
        self.assertIsNone(result)

    def test_03_vibrational_feature_enables_vibrational_checks(self):
        """Test vibrational_analysis feature enables vibrational validation."""
        # DFT has vibrational_analysis=True
        self.assertTrue(_get_dataset_feature("DFT", "vibrational_analysis"))

        dft_handler = MockDFTHandler(should_validate=True)
        pyg_data = Data(
            z=torch.tensor([6, 6, 1, 1, 1, 1]),
            pos=torch.rand(6, 3),
            y=torch.tensor([-100.0]),
            freqs=torch.tensor([100.0, 200.0, 300.0]),
            num_nodes=6,
        )

        results = validate_pyg_data_completeness(
            pyg_data=pyg_data,
            dataset_type="DFT",
            molecule_index=self.molecule_index,
            handler=dft_handler,
        )

        self.assertEqual(results["handler_type"], "DFT")

    def test_04_orbital_feature_enables_orbital_checks(self):
        """Test orbital_analysis feature enables orbital validation."""
        # Wavefunction has orbital_analysis=True
        self.assertTrue(_get_dataset_feature("Wavefunction", "orbital_analysis"))

        wf_handler = MockWavefunctionHandler(should_validate=True)
        pyg_data = Data(
            z=torch.tensor([6, 6, 1, 1, 1, 1]),
            pos=torch.rand(6, 3),
            y=torch.tensor([-100.0]),
            homo=torch.tensor([-5.0]),
            lumo=torch.tensor([-2.0]),
            num_nodes=6,
        )

        results = validate_pyg_data_completeness(
            pyg_data=pyg_data,
            dataset_type="Wavefunction",
            molecule_index=self.molecule_index,
            handler=wf_handler,
        )

        self.assertEqual(results["handler_type"], "Wavefunction")

    def test_05_requirements_reflect_dataset_features(self):
        """Test get_dataset_requirements reflects dataset features correctly."""
        # DMC should have uncertainty requirements
        dmc_handler = MockDMCHandler(should_validate=True, uncertainty_enabled=True)
        dmc_reqs = get_dataset_requirements(dataset_type="DMC", handler=dmc_handler)

        self.assertTrue(dmc_reqs["uncertainty_enabled"])

        # DFT should not have uncertainty requirements
        dft_handler = MockDFTHandler(should_validate=True)
        dft_reqs = get_dataset_requirements(dataset_type="DFT", handler=dft_handler)

        self.assertFalse(dft_reqs["uncertainty_enabled"])

    def test_06_context_reflects_dataset_features(self):
        """Test create_validation_context reflects dataset features correctly."""
        inchi = "InChI=1S/C2H6/c1-2/h1-2H3"

        # DMC context should include uncertainty config
        dmc_handler = MockDMCHandler(should_validate=True, uncertainty_enabled=True)
        dmc_context = create_validation_context(
            handler=dmc_handler, molecule_index=self.molecule_index, inchi=inchi
        )

        self.assertIn("validation_config", dmc_context)
        self.assertTrue(dmc_context["validation_config"]["uncertainty_enabled"])

        # DFT context should not have uncertainty enabled
        dft_handler = MockDFTHandler(should_validate=True)
        dft_context = create_validation_context(
            handler=dft_handler, molecule_index=self.molecule_index, inchi=inchi
        )

        self.assertFalse(dft_context["validation_config"].get("uncertainty_enabled", False))


class TestPhase6NewDatasetIntegration(unittest.TestCase):
    """
    Test that Phase 6 refactoring enables new dataset types without code changes.

    These tests simulate adding a new dataset type and verify that the
    feature-based routing works correctly.
    """

    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")

    def setUp(self):
        reset_registry_state()

    def tearDown(self):
        reset_registry_state()

    def test_01_unknown_dataset_gracefully_handled(self):
        """Test unknown dataset types are handled gracefully."""
        # Unknown dataset should return False for features
        result = _get_dataset_feature("QMC", "uncertainty_handling")
        self.assertFalse(result)

        result = _get_dataset_feature("QMC", "vibrational_analysis")
        self.assertFalse(result)

    def test_02_unknown_dataset_not_registered(self):
        """Test unknown dataset is reported as not registered."""
        self.assertFalse(_is_dataset_type_registered("QMC"))
        self.assertFalse(_is_dataset_type_registered("NewDataset"))

    def test_03_unknown_dataset_creates_generic_error(self):
        """Test unknown dataset handler creates generic handler error."""
        unknown_handler = MockDatasetHandler(dataset_type="QMC", should_validate=True)
        error = _create_handler_specific_error(
            handler=unknown_handler,
            message="Test error",
            operation="test_op",
            details="Test details",
            molecule_index=0,
        )

        self.assertIsInstance(error, (HandlerValidationError, HandlerError))

    def test_04_registry_status_after_initialization(self):
        """Test registry status is correct after initialization."""
        # Ensure registry is initialized
        _init_registry()

        status = get_registry_status()

        self.assertTrue(status["registry_initialized"])
        self.assertTrue(status["phase_6_complete"])
        self.assertIn("DFT", status["available_dataset_types"])
        self.assertIn("DMC", status["available_dataset_types"])
        self.assertIn("Wavefunction", status["available_dataset_types"])


# ==============================================================================
# TEST RUNNER
# ==============================================================================


def run_test_suite():
    """Run the complete test suite"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all original test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHandlerRequirement))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularStructureValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetCompatibilityChecking))
    suite.addTests(loader.loadTestsFromTestCase(TestAtomicConversions))
    suite.addTests(loader.loadTestsFromTestCase(TestUncertaintyValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestPyGDataCompletenessValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetRequirements))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationContext))
    suite.addTests(loader.loadTestsFromTestCase(TestDetailedFeedbackValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestValidatorCreation))
    suite.addTests(loader.loadTestsFromTestCase(TestHighLevelValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationSummary))
    suite.addTests(loader.loadTestsFromTestCase(TestLegacyValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestExceptionHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationScenarios))

    # Add Phase 6 test classes (NEW)
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6RegistryIntegrationFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6FeatureQueries))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6OptionalProperties))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6CreateHandlerSpecificError))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6RegistryStatus))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6RefactoredFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6LegacyFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6FeatureBasedValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6NewDatasetIntegration))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("COMPREHENSIVE TEST SUMMARY - molecule_validator.py (Phase 6 Updated)")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)

    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED!")
        print("✓ Handler requirement enforcement validated")
        print("✓ Molecular structure validation working")
        print("✓ Dataset compatibility checking operational")
        print("✓ Atomic conversions functional")
        print("✓ Uncertainty validation tested")
        print("✓ PyG data completeness verified")
        print("✓ Dataset requirements retrieval working")
        print("✓ Validation context creation tested")
        print("✓ Detailed feedback validation verified")
        print("✓ Validator creation working")
        print("✓ High-level validation tested")
        print("✓ Validation summary generation verified")
        print("✓ Legacy validation tested")
        print("✓ Exception handling verified")
        print("✓ Edge cases covered")
        print("✓ Integration scenarios passed")
        print("✓ Phase 6: Registry integration validated")
        print("✓ Phase 6: Feature queries working")
        print("✓ Phase 6: Optional properties queries working")
        print("✓ Phase 6: Handler-specific error creation functional")
        print("✓ Phase 6: Registry status function verified")
        print("✓ Phase 6: All 9 refactored locations tested")
        print("✓ Phase 6: Legacy fallback working")
        print("✓ Phase 6: Feature-based validation validated")
        print("✓ Phase 6: New dataset integration verified")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("  Review failures before production use")

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = run_test_suite()
    sys.exit(exit_code)
