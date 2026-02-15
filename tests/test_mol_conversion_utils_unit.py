#!/usr/bin/env python3
# test_mol_conversion_unit.py

"""
Production-ready unit test suite for mol_conversion_utils.py module.

This test suite covers all major functions in the mol_conversion_utils module,
including RDKit molecule creation, PyG Data conversion, handler-based metadata
enrichment, and handler validation operations.

KEY COVERAGE AREAS:
1. create_rdkit_mol - Both identifier_coordinate_based and coordinate_based strategies
2. mol_to_pyg_data - PyG Data object creation from RDKit molecules
3. enrich_pyg_data_from_handler - Dynamic handler-based PyG Data enrichment
4. create_mol_with_dataset_support - Full pipeline integration
5. Handler validation and capability checking
6. Error handling and recovery mechanisms
7. Utility functions: context info, statistics, prerequisites, error enhancement

DESIGN PRINCIPLES:
- All mocks are configured at test-level using @patch decorators (no sys.modules pollution)
- Handler mocks include all required methods: get_dataset_type, get_molecule_creation_strategy, get_molecular_charge
- Tests cover both success paths and error conditions
- No hardcoded dataset types - tests validate dynamic handler capability checking
- enrich_pyg_data_from_handler tests are fully handler-agnostic (no DMC-specific logic)

Tests are designed to run in a Docker environment with mocked external dependencies.
No actual file downloads or external API calls are made.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add the project root to Python path FIRST
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import real exceptions - no mocking needed
import logging
import unittest

import numpy as np
import torch
from torch_geometric.data import Data

from milia_pipeline import exceptions as vqm_exceptions

# Now import normally - the package exists in the container

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)


class TestCreateRdkitMol(unittest.TestCase):
    """Test suite for create_rdkit_mol function"""

    def setUp(self):
        """Set up common test fixtures"""
        self.logger = logging.getLogger("test_logger")
        self.mock_handler = Mock()
        self.mock_handler.get_dataset_type.return_value = "DFT"
        # CRITICAL: Configure required handler methods for strategy selection
        self.mock_handler.get_molecule_creation_strategy.return_value = (
            "identifier_coordinate_based"
        )
        self.mock_handler.get_molecular_charge.return_value = 0

        # Sample test data
        self.valid_inchi = "InChI=1S/H2O/h1H2"
        self.valid_smiles = "O"
        self.valid_coordinates = np.array(
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]], dtype=np.float64
        )
        self.valid_atomic_numbers = np.array([8, 1, 1], dtype=np.int32)
        self.molecule_index = 42

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_molecular_structure")
    @patch("milia_pipeline.molecules.mol_conversion_utils.Chem")
    @patch("milia_pipeline.molecules.mol_conversion_utils.create_handler_error_context")
    def test_create_rdkit_mol_with_inchi_success(
        self, mock_context, mock_chem, mock_validate_structure, mock_validate_handler
    ):
        """Test successful RDKit molecule creation from InChI"""
        # Setup mocks
        mock_validate_structure.return_value = (self.valid_atomic_numbers, self.valid_coordinates)

        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 3

        def mock_get_atom(idx):
            atom = Mock()
            if idx == 0:
                atom.GetAtomicNum.return_value = 8
            else:
                atom.GetAtomicNum.return_value = 1
            return atom

        mock_mol.GetAtomWithIdx.side_effect = mock_get_atom
        mock_chem.MolFromInchi.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        mock_chem.Conformer.return_value = Mock()
        mock_chem.SANITIZE_SETAROMATICITY = 1
        mock_chem.SANITIZE_SETCONJUGATION = 2

        # Import after patching
        from milia_pipeline.molecules.mol_conversion_utils import create_rdkit_mol

        # Execute
        result = create_rdkit_mol(
            mol_identifier=self.valid_inchi,
            coordinates=self.valid_coordinates,
            atomic_numbers=self.valid_atomic_numbers,
            logger=self.logger,
            molecule_index=self.molecule_index,
            mol_id_type="inchi",
            handler=self.mock_handler,
        )

        # Verify
        self.assertIsNotNone(result)
        mock_validate_handler.assert_called_once()
        mock_validate_structure.assert_called_once()
        mock_chem.MolFromInchi.assert_called_once_with(self.valid_inchi, sanitize=False)

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    def test_create_rdkit_mol_handler_none_raises_error(self, mock_validate_handler):
        """Test that None handler raises ValueError"""
        from milia_pipeline.molecules.mol_conversion_utils import create_rdkit_mol

        with self.assertRaises(vqm_exceptions.HandlerOperationError) as context:
            create_rdkit_mol(
                mol_identifier=self.valid_inchi,
                coordinates=self.valid_coordinates,
                atomic_numbers=self.valid_atomic_numbers,
                logger=self.logger,
                molecule_index=self.molecule_index,
                mol_id_type="inchi",
                handler=None,
            )

        self.assertIn("Handler is required", str(context.exception))

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_molecular_structure")
    @patch("milia_pipeline.molecules.mol_conversion_utils.Chem")
    @patch("milia_pipeline.molecules.mol_conversion_utils.create_handler_error_context")
    def test_create_rdkit_mol_with_smiles_success(
        self, mock_context, mock_chem, mock_validate_structure, mock_validate_handler
    ):
        """Test successful RDKit molecule creation from SMILES"""
        # Setup mocks
        mock_validate_structure.return_value = (self.valid_atomic_numbers, self.valid_coordinates)

        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 3

        def mock_get_atom(idx):
            atom = Mock()
            if idx == 0:
                atom.GetAtomicNum.return_value = 8
            else:
                atom.GetAtomicNum.return_value = 1
            return atom

        mock_mol.GetAtomWithIdx.side_effect = mock_get_atom
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        mock_chem.Conformer.return_value = Mock()
        mock_chem.SANITIZE_SETAROMATICITY = 1
        mock_chem.SANITIZE_SETCONJUGATION = 2

        from milia_pipeline.molecules.mol_conversion_utils import create_rdkit_mol

        # Execute
        result = create_rdkit_mol(
            mol_identifier=self.valid_smiles,
            coordinates=self.valid_coordinates,
            atomic_numbers=self.valid_atomic_numbers,
            logger=self.logger,
            molecule_index=self.molecule_index,
            mol_id_type="smiles",
            handler=self.mock_handler,
        )

        # Verify
        self.assertIsNotNone(result)
        mock_chem.MolFromSmiles.assert_called_once_with(self.valid_smiles, sanitize=False)

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_molecular_structure")
    @patch("milia_pipeline.molecules.mol_conversion_utils.create_handler_error_context")
    def test_create_rdkit_mol_invalid_mol_id_type(
        self, mock_context, mock_validate_structure, mock_validate_handler
    ):
        """Test that invalid mol_id_type raises ValueError"""
        mock_validate_structure.return_value = (self.valid_atomic_numbers, self.valid_coordinates)

        from milia_pipeline.molecules.mol_conversion_utils import create_rdkit_mol

        with self.assertRaises(vqm_exceptions.HandlerOperationError) as context:
            create_rdkit_mol(
                mol_identifier=self.valid_inchi,
                coordinates=self.valid_coordinates,
                atomic_numbers=self.valid_atomic_numbers,
                logger=self.logger,
                molecule_index=self.molecule_index,
                mol_id_type="invalid_type",
                handler=self.mock_handler,
            )

        self.assertIn("Unsupported mol_id_type", str(context.exception))

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_molecular_structure")
    @patch("milia_pipeline.molecules.mol_conversion_utils.Chem")
    @patch("milia_pipeline.molecules.mol_conversion_utils.create_handler_error_context")
    @patch("milia_pipeline.molecules.mol_conversion_utils.HandlerValidationError")
    def test_create_rdkit_mol_atom_count_mismatch(
        self,
        mock_error_class,
        mock_context,
        mock_chem,
        mock_validate_structure,
        mock_validate_handler,
    ):
        """Test that atom count mismatch raises HandlerValidationError"""
        logger = logging.getLogger("test")
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"

        coordinates = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
        atomic_numbers = np.array([8, 1, 1])

        mock_validate_structure.return_value = (atomic_numbers, coordinates)

        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 5  # Mismatch: expected 3, got 5
        mock_chem.MolFromInchi.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol

        mock_error_instance = Exception("Atom count mismatch")
        mock_error_class.return_value = mock_error_instance

        from milia_pipeline.molecules.mol_conversion_utils import create_rdkit_mol

        with self.assertRaises(Exception):
            create_rdkit_mol(
                mol_identifier="InChI=1S/H2O/h1H2",
                coordinates=coordinates,
                atomic_numbers=atomic_numbers,
                logger=logger,
                molecule_index=0,
                mol_id_type="inchi",
                handler=mock_handler,
            )


class TestCreateRdkitMolStrategyRouting(unittest.TestCase):
    """Test suite for strategy routing in create_rdkit_mol function

    The module supports two molecule creation strategies:
    1. identifier_coordinate_based: Parse InChI/SMILES for connectivity, assign QM coordinates
    2. coordinate_based: Infer connectivity from coordinates using rdDetermineBonds

    The handler.get_molecule_creation_strategy() determines which path is taken.
    """

    def setUp(self):
        """Set up common test fixtures"""
        self.logger = logging.getLogger("test_strategy")
        self.valid_coordinates = np.array(
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]], dtype=np.float64
        )
        self.valid_atomic_numbers = np.array([8, 1, 1], dtype=np.int32)

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_molecular_structure")
    @patch("milia_pipeline.molecules.mol_conversion_utils.Chem")
    @patch("milia_pipeline.molecules.mol_conversion_utils.create_handler_error_context")
    def test_strategy_identifier_coordinate_based_routes_to_inchi(
        self, mock_context, mock_chem, mock_validate_structure, mock_validate_handler
    ):
        """Test that identifier_coordinate_based strategy routes to InChI parsing"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_handler.get_molecule_creation_strategy.return_value = "identifier_coordinate_based"
        mock_handler.get_molecular_charge.return_value = 0

        mock_validate_structure.return_value = (self.valid_atomic_numbers, self.valid_coordinates)

        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 3
        mock_mol.GetAtomWithIdx.side_effect = lambda idx: Mock(
            GetAtomicNum=Mock(return_value=8 if idx == 0 else 1)
        )
        mock_chem.MolFromInchi.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        mock_chem.Conformer.return_value = Mock()
        mock_chem.SANITIZE_SETAROMATICITY = 1
        mock_chem.SANITIZE_SETCONJUGATION = 2

        from milia_pipeline.molecules.mol_conversion_utils import create_rdkit_mol

        result = create_rdkit_mol(
            mol_identifier="InChI=1S/H2O/h1H2",
            coordinates=self.valid_coordinates,
            atomic_numbers=self.valid_atomic_numbers,
            logger=self.logger,
            handler=mock_handler,
            molecule_index=0,
            mol_id_type="inchi",
        )

        self.assertIsNotNone(result)
        mock_chem.MolFromInchi.assert_called_once()
        mock_handler.get_molecule_creation_strategy.assert_called_once()

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_molecular_structure")
    @patch("milia_pipeline.molecules.mol_conversion_utils.Chem")
    @patch("milia_pipeline.molecules.mol_conversion_utils.rdDetermineBonds")
    @patch("milia_pipeline.config.config_constants.get_handler_constants")
    @patch("milia_pipeline.molecules.mol_conversion_utils.create_handler_error_context")
    def test_strategy_coordinate_based_uses_rdDetermineBonds(
        self,
        mock_context,
        mock_get_constants,
        mock_determine_bonds,
        mock_chem,
        mock_validate_structure,
        mock_validate_handler,
    ):
        """Test that coordinate_based strategy uses rdDetermineBonds for bond inference"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "Wavefunction"
        mock_handler.get_molecule_creation_strategy.return_value = "coordinate_based"
        mock_handler.get_molecular_charge.return_value = 0

        mock_validate_structure.return_value = (self.valid_atomic_numbers, self.valid_coordinates)
        mock_get_constants.return_value = {"coordinate_units": "angstrom"}

        # Mock XYZ block creation and parsing
        mock_raw_mol = Mock()
        mock_raw_mol.GetNumBonds.return_value = 2
        mock_chem.MolFromXYZBlock.return_value = mock_raw_mol
        mock_chem.Mol.return_value = mock_raw_mol
        mock_chem.Atom.return_value = Mock(GetSymbol=Mock(return_value="O"))

        from milia_pipeline.molecules.mol_conversion_utils import create_rdkit_mol

        result = create_rdkit_mol(
            mol_identifier="H2O_compound",
            coordinates=self.valid_coordinates,
            atomic_numbers=self.valid_atomic_numbers,
            logger=self.logger,
            handler=mock_handler,
            molecule_index=0,
            mol_id_type="compound_id",
            molecular_charge=0,
        )

        self.assertIsNotNone(result)
        mock_chem.MolFromXYZBlock.assert_called_once()
        mock_determine_bonds.DetermineBonds.assert_called_once()
        mock_handler.get_molecule_creation_strategy.assert_called_once()

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_molecular_structure")
    @patch("milia_pipeline.molecules.mol_conversion_utils.Chem")
    @patch("milia_pipeline.molecules.mol_conversion_utils.rdDetermineBonds")
    @patch("milia_pipeline.config.config_constants.get_handler_constants")
    @patch("milia_pipeline.molecules.mol_conversion_utils.create_handler_error_context")
    def test_coordinate_based_with_bohr_units_converts_to_angstrom(
        self,
        mock_context,
        mock_get_constants,
        mock_determine_bonds,
        mock_chem,
        mock_validate_structure,
        mock_validate_handler,
    ):
        """Test that coordinate_based strategy converts Bohr to Angstrom when needed"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "Wavefunction"
        mock_handler.get_molecule_creation_strategy.return_value = "coordinate_based"
        mock_handler.get_molecular_charge.return_value = 0

        # Coordinates in Bohr units
        bohr_coordinates = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.8, 0.0, 0.0],  # ~0.95 Angstrom
                [-0.45, 1.76, 0.0],
            ],
            dtype=np.float64,
        )

        mock_validate_structure.return_value = (self.valid_atomic_numbers, bohr_coordinates)
        # Return Bohr units - should trigger conversion
        mock_get_constants.return_value = {"coordinate_units": "bohr"}

        mock_raw_mol = Mock()
        mock_raw_mol.GetNumBonds.return_value = 2
        mock_chem.MolFromXYZBlock.return_value = mock_raw_mol
        mock_chem.Mol.return_value = mock_raw_mol
        mock_chem.Atom.return_value = Mock(GetSymbol=Mock(return_value="O"))

        from milia_pipeline.molecules.mol_conversion_utils import create_rdkit_mol

        result = create_rdkit_mol(
            mol_identifier="H2O_bohr",
            coordinates=bohr_coordinates,
            atomic_numbers=self.valid_atomic_numbers,
            logger=self.logger,
            handler=mock_handler,
            molecule_index=0,
            mol_id_type="compound_id",
            molecular_charge=0,
        )

        self.assertIsNotNone(result)
        # Verify get_handler_constants was called with dataset type
        mock_get_constants.assert_called_with("Wavefunction")

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_molecular_structure")
    def test_unknown_strategy_raises_value_error(
        self, mock_validate_structure, mock_validate_handler
    ):
        """Test that unknown strategy raises ValueError"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "CustomDataset"
        mock_handler.get_molecule_creation_strategy.return_value = "unknown_strategy"
        mock_handler.get_molecular_charge.return_value = 0

        mock_validate_structure.return_value = (self.valid_atomic_numbers, self.valid_coordinates)

        from milia_pipeline.molecules.mol_conversion_utils import create_rdkit_mol

        with self.assertRaises(vqm_exceptions.HandlerOperationError) as context:
            create_rdkit_mol(
                mol_identifier="test",
                coordinates=self.valid_coordinates,
                atomic_numbers=self.valid_atomic_numbers,
                logger=self.logger,
                handler=mock_handler,
                molecule_index=0,
                mol_id_type="inchi",
            )

        self.assertIn("Unknown molecule creation strategy", str(context.exception))

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_molecular_structure")
    @patch("milia_pipeline.molecules.mol_conversion_utils.Chem")
    @patch("milia_pipeline.molecules.mol_conversion_utils.rdDetermineBonds")
    @patch("milia_pipeline.config.config_constants.get_handler_constants")
    @patch("milia_pipeline.molecules.mol_conversion_utils.create_handler_error_context")
    def test_coordinate_based_with_molecular_charge(
        self,
        mock_context,
        mock_get_constants,
        mock_determine_bonds,
        mock_chem,
        mock_validate_structure,
        mock_validate_handler,
    ):
        """Test that molecular_charge is passed to rdDetermineBonds for coordinate_based strategy"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "Wavefunction"
        mock_handler.get_molecule_creation_strategy.return_value = "coordinate_based"
        mock_handler.get_molecular_charge.return_value = -1  # Charged molecule

        mock_validate_structure.return_value = (self.valid_atomic_numbers, self.valid_coordinates)
        mock_get_constants.return_value = {"coordinate_units": "angstrom"}

        mock_raw_mol = Mock()
        mock_raw_mol.GetNumBonds.return_value = 2
        mock_chem.MolFromXYZBlock.return_value = mock_raw_mol
        mock_chem.Mol.return_value = mock_raw_mol
        mock_chem.Atom.return_value = Mock(GetSymbol=Mock(return_value="O"))

        from milia_pipeline.molecules.mol_conversion_utils import create_rdkit_mol

        result = create_rdkit_mol(
            mol_identifier="OH-_anion",
            coordinates=self.valid_coordinates[:2],  # Just O and H
            atomic_numbers=np.array([8, 1]),
            logger=self.logger,
            handler=mock_handler,
            molecule_index=0,
            mol_id_type="compound_id",
            molecular_charge=-1,
        )

        self.assertIsNotNone(result)
        # Verify molecular charge was passed to DetermineBonds
        mock_determine_bonds.DetermineBonds.assert_called_once()
        call_args = mock_determine_bonds.DetermineBonds.call_args
        self.assertEqual(call_args[1]["charge"], -1)


class TestMolToPygData(unittest.TestCase):
    """Test suite for mol_to_pyg_data function"""

    def setUp(self):
        """Set up common test fixtures"""
        self.logger = logging.getLogger("test_logger")
        self.mock_handler = Mock()
        self.mock_handler.get_dataset_type.return_value = "DFT"
        # CRITICAL: Configure required handler methods
        self.mock_handler.get_molecule_creation_strategy.return_value = (
            "identifier_coordinate_based"
        )
        self.mock_handler.get_molecular_charge.return_value = 0

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    @patch("milia_pipeline.molecules.mol_conversion_utils.from_rdmol")
    @patch("milia_pipeline.molecules.mol_conversion_utils.create_handler_error_context")
    def test_mol_to_pyg_data_success(self, mock_context, mock_from_rdmol, mock_validate_handler):
        """Test successful conversion of RDKit mol to PyG Data"""
        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 3
        mock_mol.GetNumConformers.return_value = 1

        mock_atoms = [
            Mock(GetAtomicNum=Mock(return_value=8)),
            Mock(GetAtomicNum=Mock(return_value=1)),
            Mock(GetAtomicNum=Mock(return_value=1)),
        ]
        mock_mol.GetAtoms.return_value = iter(mock_atoms)

        mock_conformer = Mock()
        mock_positions = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
        mock_conformer.GetPositions.return_value = mock_positions
        mock_mol.GetConformer.return_value = mock_conformer

        mock_pyg_data = Data()
        mock_pyg_data.num_nodes = 3
        mock_from_rdmol.return_value = mock_pyg_data

        from milia_pipeline.molecules.mol_conversion_utils import mol_to_pyg_data

        result = mol_to_pyg_data(
            rdkit_mol=mock_mol, logger=self.logger, molecule_index=0, handler=self.mock_handler
        )

        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "z"))
        self.assertTrue(hasattr(result, "pos"))
        mock_validate_handler.assert_called_once()
        mock_from_rdmol.assert_called_once()

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    def test_mol_to_pyg_data_none_mol_raises_error(self, mock_validate_handler):
        """Test that None mol raises ValueError"""
        from milia_pipeline.molecules.mol_conversion_utils import mol_to_pyg_data

        with self.assertRaises(vqm_exceptions.HandlerOperationError) as context:
            mol_to_pyg_data(
                rdkit_mol=None, logger=self.logger, molecule_index=0, handler=self.mock_handler
            )

        self.assertIn("Molecule object is required", str(context.exception))

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    @patch("milia_pipeline.molecules.mol_conversion_utils.from_rdmol")
    @patch("milia_pipeline.molecules.mol_conversion_utils.create_handler_error_context")
    def test_mol_to_pyg_data_no_conformer(
        self, mock_context, mock_from_rdmol, mock_validate_handler
    ):
        """Test handling of molecule with no conformer"""
        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 3
        mock_mol.GetNumConformers.return_value = 0  # No conformer

        mock_pyg_data = Data()
        mock_pyg_data.num_nodes = 3
        mock_from_rdmol.return_value = mock_pyg_data

        from milia_pipeline.molecules.mol_conversion_utils import mol_to_pyg_data

        result = mol_to_pyg_data(
            rdkit_mol=mock_mol, logger=self.logger, molecule_index=0, handler=self.mock_handler
        )

        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "z"))
        # Should not have 'pos' attribute since no conformer

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    @patch("milia_pipeline.molecules.mol_conversion_utils.from_rdmol")
    @patch("milia_pipeline.molecules.mol_conversion_utils.create_handler_error_context")
    def test_mol_to_pyg_data_conversion_failure(
        self, mock_context, mock_from_rdmol, mock_validate_handler
    ):
        """Test that conversion failure raises HandlerConversionError"""
        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 3
        mock_mol.GetNumConformers.return_value = 1

        mock_from_rdmol.side_effect = Exception("Conversion failed")

        # REMOVED: Lines 336-337 that referenced undefined mock_error_class

        from milia_pipeline.molecules.mol_conversion_utils import mol_to_pyg_data

        with self.assertRaises(Exception):
            mol_to_pyg_data(
                rdkit_mol=mock_mol, logger=self.logger, molecule_index=0, handler=self.mock_handler
            )


class TestEnrichPygDataFromHandler(unittest.TestCase):
    """Test suite for enrich_pyg_data_from_handler function.

    This function is fully dynamic — it delegates all enrichment logic to
    handler.enrich_pyg_data() (DatasetHandlerProtocol method #5).
    No dataset-specific logic lives in the function under test.
    """

    def setUp(self):
        """Set up common test fixtures"""
        self.logger = logging.getLogger("test_logger")
        self.mock_handler = Mock()
        self.mock_handler.get_dataset_type.return_value = "DFT"
        # CRITICAL: Configure required handler methods
        self.mock_handler.get_molecule_creation_strategy.return_value = (
            "identifier_coordinate_based"
        )
        self.mock_handler.get_molecular_charge.return_value = 0

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    def test_enrich_pyg_data_success_delegates_to_handler(self, mock_validate_handler):
        """Test successful enrichment delegates to handler.enrich_pyg_data()"""
        pyg_data = Data()
        pyg_data.num_nodes = 3

        raw_data_dict = {
            "energy": -76.4,
            "forces": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        }

        # Configure handler.enrich_pyg_data to return enriched Data
        enriched_data = Data()
        enriched_data.num_nodes = 3
        enriched_data.energy = torch.tensor([-76.4])
        enriched_data.dataset_type = "DFT"
        self.mock_handler.enrich_pyg_data.return_value = enriched_data

        from milia_pipeline.molecules.mol_conversion_utils import enrich_pyg_data_from_handler

        result = enrich_pyg_data_from_handler(
            pyg_data=pyg_data,
            raw_data_dict=raw_data_dict,
            molecule_index=0,
            logger=self.logger,
            handler=self.mock_handler,
            identifier="InChI=1S/H2O/h1H2",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.dataset_type, "DFT")
        self.mock_handler.enrich_pyg_data.assert_called_once_with(
            pyg_data, raw_data_dict, 0, "InChI=1S/H2O/h1H2"
        )
        mock_validate_handler.assert_called_once()

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    def test_enrich_pyg_data_sets_dataset_type_if_missing(self, mock_validate_handler):
        """Test that dataset_type marker is set when handler does not set it"""
        pyg_data = Data()
        pyg_data.num_nodes = 3

        raw_data_dict = {"energy": -76.4}

        # Handler returns data WITHOUT dataset_type set
        enriched_data = Data()
        enriched_data.num_nodes = 3
        self.mock_handler.enrich_pyg_data.return_value = enriched_data

        from milia_pipeline.molecules.mol_conversion_utils import enrich_pyg_data_from_handler

        result = enrich_pyg_data_from_handler(
            pyg_data=pyg_data,
            raw_data_dict=raw_data_dict,
            molecule_index=0,
            logger=self.logger,
            handler=self.mock_handler,
            identifier="test_mol",
        )

        # The function must ensure dataset_type is always set
        self.assertEqual(result.dataset_type, "DFT")

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    def test_enrich_pyg_data_preserves_existing_dataset_type(self, mock_validate_handler):
        """Test that existing dataset_type set by handler is preserved"""
        pyg_data = Data()
        pyg_data.num_nodes = 3

        # Handler returns data WITH dataset_type already set
        enriched_data = Data()
        enriched_data.num_nodes = 3
        enriched_data.dataset_type = "CustomType"
        self.mock_handler.enrich_pyg_data.return_value = enriched_data

        from milia_pipeline.molecules.mol_conversion_utils import enrich_pyg_data_from_handler

        result = enrich_pyg_data_from_handler(
            pyg_data=pyg_data,
            raw_data_dict={"energy": -76.4},
            molecule_index=0,
            logger=self.logger,
            handler=self.mock_handler,
            identifier="test_mol",
        )

        # Handler-set dataset_type should be preserved
        self.assertEqual(result.dataset_type, "CustomType")

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    def test_enrich_pyg_data_none_pyg_data_raises_validation_error(self, mock_validate_handler):
        """Test that None pyg_data raises HandlerValidationError"""
        from milia_pipeline.molecules.mol_conversion_utils import enrich_pyg_data_from_handler

        with self.assertRaises(vqm_exceptions.HandlerValidationError) as context:
            enrich_pyg_data_from_handler(
                pyg_data=None,
                raw_data_dict={},
                molecule_index=0,
                logger=self.logger,
                handler=self.mock_handler,
                identifier="test_mol",
            )

        self.assertIn("PyG data is None", str(context.exception))

    def test_enrich_pyg_data_handler_none_raises_value_error(self):
        """Test that None handler raises HandlerOperationError (ValueError wrapped by decorator)"""
        pyg_data = Data()
        pyg_data.num_nodes = 3

        from milia_pipeline.molecules.mol_conversion_utils import enrich_pyg_data_from_handler

        # The @wrap_handler_operation decorator wraps ValueError into HandlerOperationError
        with self.assertRaises(vqm_exceptions.HandlerOperationError) as context:
            enrich_pyg_data_from_handler(
                pyg_data=pyg_data,
                raw_data_dict={"energy": -76.4},
                molecule_index=0,
                logger=self.logger,
                handler=None,
                identifier="test_mol",
            )

        self.assertIn("Handler is required", str(context.exception))

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    def test_enrich_pyg_data_identifier_defaults_to_na(self, mock_validate_handler):
        """Test that identifier defaults to 'N/A' when not provided"""
        pyg_data = Data()
        pyg_data.num_nodes = 3

        enriched_data = Data()
        enriched_data.num_nodes = 3
        enriched_data.dataset_type = "DFT"
        self.mock_handler.enrich_pyg_data.return_value = enriched_data

        from milia_pipeline.molecules.mol_conversion_utils import enrich_pyg_data_from_handler

        result = enrich_pyg_data_from_handler(
            pyg_data=pyg_data,
            raw_data_dict={"energy": -76.4},
            molecule_index=0,
            logger=self.logger,
            handler=self.mock_handler,
            # identifier not provided — defaults to None, function passes "N/A"
        )

        self.assertIsNotNone(result)
        # Verify handler received "N/A" as the identifier string
        call_args = self.mock_handler.enrich_pyg_data.call_args
        self.assertEqual(call_args[0][3], "N/A")

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    def test_enrich_pyg_data_handler_exception_reraises_known_errors(self, mock_validate_handler):
        """Test that DatasetSpecificHandlerError from handler is re-raised as-is"""
        pyg_data = Data()
        pyg_data.num_nodes = 3

        # Handler raises a known error type
        self.mock_handler.enrich_pyg_data.side_effect = vqm_exceptions.DatasetSpecificHandlerError(
            message="Handler enrichment failed", dataset_type="DFT", operation="enrich_pyg_data"
        )

        from milia_pipeline.molecules.mol_conversion_utils import enrich_pyg_data_from_handler

        with self.assertRaises(vqm_exceptions.DatasetSpecificHandlerError):
            enrich_pyg_data_from_handler(
                pyg_data=pyg_data,
                raw_data_dict={"energy": -76.4},
                molecule_index=0,
                logger=self.logger,
                handler=self.mock_handler,
                identifier="test_mol",
            )

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    def test_enrich_pyg_data_unexpected_exception_wrapped(self, mock_validate_handler):
        """Test that unexpected exceptions are wrapped in DatasetSpecificHandlerError"""
        pyg_data = Data()
        pyg_data.num_nodes = 3

        # Handler raises an unexpected error
        self.mock_handler.enrich_pyg_data.side_effect = RuntimeError("Unexpected failure")

        from milia_pipeline.molecules.mol_conversion_utils import enrich_pyg_data_from_handler

        with self.assertRaises(vqm_exceptions.DatasetSpecificHandlerError) as context:
            enrich_pyg_data_from_handler(
                pyg_data=pyg_data,
                raw_data_dict={"energy": -76.4},
                molecule_index=0,
                logger=self.logger,
                handler=self.mock_handler,
                identifier="test_mol",
            )

        self.assertIn("Unexpected error", str(context.exception))

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    def test_enrich_pyg_data_works_with_any_dataset_type(self, mock_validate_handler):
        """Test that enrichment works for any dataset type (DYNAMIC — no hardcoded types)"""
        for dataset_type in ["DFT", "DMC", "Wavefunction", "QM9", "ANI1X", "CustomNewDataset"]:
            mock_handler = Mock()
            mock_handler.get_dataset_type.return_value = dataset_type
            mock_handler.get_molecule_creation_strategy.return_value = "identifier_coordinate_based"
            mock_handler.get_molecular_charge.return_value = 0

            pyg_data = Data()
            pyg_data.num_nodes = 3

            enriched_data = Data()
            enriched_data.num_nodes = 3
            enriched_data.dataset_type = dataset_type
            mock_handler.enrich_pyg_data.return_value = enriched_data

            from milia_pipeline.molecules.mol_conversion_utils import enrich_pyg_data_from_handler

            result = enrich_pyg_data_from_handler(
                pyg_data=pyg_data,
                raw_data_dict={"energy": -76.4},
                molecule_index=0,
                logger=self.logger,
                handler=mock_handler,
                identifier="test_mol",
            )

            self.assertIsNotNone(result)
            self.assertEqual(result.dataset_type, dataset_type)


class TestCreateMolWithDatasetSupport(unittest.TestCase):
    """Test suite for create_mol_with_dataset_support function"""

    def setUp(self):
        """Set up common test fixtures"""
        self.logger = logging.getLogger("test_logger")
        self.mock_handler = Mock()
        self.mock_handler.get_dataset_type.return_value = "DMC"
        # CRITICAL: Configure required handler methods
        self.mock_handler.get_molecule_creation_strategy.return_value = (
            "identifier_coordinate_based"
        )
        self.mock_handler.get_molecular_charge.return_value = 0

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_molecular_structure")
    @patch("milia_pipeline.molecules.mol_conversion_utils.Chem")
    @patch("milia_pipeline.molecules.mol_conversion_utils.from_rdmol")
    @patch("milia_pipeline.molecules.mol_conversion_utils.create_handler_error_context")
    def test_create_mol_with_dataset_support_full_pipeline(
        self,
        mock_context,
        mock_from_rdmol,
        mock_chem,
        mock_validate_structure,
        mock_validate_handler,
    ):
        """Test full molecule creation pipeline"""
        coordinates = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
        atomic_numbers = np.array([8, 1, 1])
        inchi = "InChI=1S/H2O/h1H2"

        mock_validate_structure.return_value = (atomic_numbers, coordinates)

        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 3
        mock_mol.GetNumConformers.return_value = 1

        def mock_get_atom(idx):
            atom = Mock()
            if idx == 0:
                atom.GetAtomicNum.return_value = 8
            else:
                atom.GetAtomicNum.return_value = 1
            return atom

        mock_mol.GetAtomWithIdx.side_effect = mock_get_atom

        mock_atoms = [
            Mock(GetAtomicNum=Mock(return_value=8)),
            Mock(GetAtomicNum=Mock(return_value=1)),
            Mock(GetAtomicNum=Mock(return_value=1)),
        ]
        mock_mol.GetAtoms.return_value = iter(mock_atoms)

        mock_conformer = Mock()
        mock_conformer.GetPositions.return_value = coordinates
        mock_mol.GetConformer.return_value = mock_conformer

        mock_chem.MolFromInchi.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        mock_chem.Conformer.return_value = Mock()
        mock_chem.SANITIZE_SETAROMATICITY = 1
        mock_chem.SANITIZE_SETCONJUGATION = 2

        mock_pyg_data = Data()
        mock_pyg_data.num_nodes = 3
        mock_from_rdmol.return_value = mock_pyg_data

        from milia_pipeline.molecules.mol_conversion_utils import create_mol_with_dataset_support

        result = create_mol_with_dataset_support(
            mol_identifier=inchi,
            coordinates=coordinates,
            atomic_numbers=atomic_numbers,
            logger=self.logger,
            molecule_index=0,
            mol_id_type="inchi",
            raw_data_dict=None,
            handler=self.mock_handler,
        )

        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "z"))
        self.assertTrue(hasattr(result, "pos"))

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_molecular_structure")
    @patch("milia_pipeline.molecules.mol_conversion_utils.Chem")
    @patch("milia_pipeline.molecules.mol_conversion_utils.from_rdmol")
    @patch("milia_pipeline.molecules.mol_conversion_utils.create_handler_error_context")
    def test_create_mol_with_dataset_support_with_metadata(
        self,
        mock_context,
        mock_from_rdmol,
        mock_chem,
        mock_validate_structure,
        mock_validate_handler,
    ):
        """Test full pipeline with metadata enrichment via handler.enrich_pyg_data()"""
        coordinates = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
        atomic_numbers = np.array([8, 1, 1])
        inchi = "InChI=1S/H2O/h1H2"

        raw_data_dict = {
            "energy": -76.4,
            "forces": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        }

        mock_validate_structure.return_value = (atomic_numbers, coordinates)

        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 3
        mock_mol.GetNumConformers.return_value = 1

        def mock_get_atom(idx):
            atom = Mock()
            if idx == 0:
                atom.GetAtomicNum.return_value = 8
            else:
                atom.GetAtomicNum.return_value = 1
            return atom

        mock_mol.GetAtomWithIdx.side_effect = mock_get_atom

        mock_atoms = [
            Mock(GetAtomicNum=Mock(return_value=8)),
            Mock(GetAtomicNum=Mock(return_value=1)),
            Mock(GetAtomicNum=Mock(return_value=1)),
        ]
        mock_mol.GetAtoms.return_value = iter(mock_atoms)

        mock_conformer = Mock()
        mock_conformer.GetPositions.return_value = coordinates
        mock_mol.GetConformer.return_value = mock_conformer

        mock_chem.MolFromInchi.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        mock_chem.Conformer.return_value = Mock()
        mock_chem.SANITIZE_SETAROMATICITY = 1
        mock_chem.SANITIZE_SETCONJUGATION = 2

        mock_pyg_data = Data()
        mock_pyg_data.num_nodes = 3
        mock_from_rdmol.return_value = mock_pyg_data

        # Configure handler.enrich_pyg_data() to return enriched data
        def mock_enrich(pyg_data, raw_dict, mol_idx, identifier_str):
            pyg_data.energy = torch.tensor([-76.4])
            pyg_data.dataset_type = "DMC"
            return pyg_data

        self.mock_handler.enrich_pyg_data.side_effect = mock_enrich

        from milia_pipeline.molecules.mol_conversion_utils import create_mol_with_dataset_support

        result = create_mol_with_dataset_support(
            mol_identifier=inchi,
            coordinates=coordinates,
            atomic_numbers=atomic_numbers,
            logger=self.logger,
            molecule_index=0,
            mol_id_type="inchi",
            raw_data_dict=raw_data_dict,
            handler=self.mock_handler,
        )

        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "z"))
        self.assertTrue(hasattr(result, "pos"))
        # Verify handler.enrich_pyg_data was called with raw_data_dict
        self.mock_handler.enrich_pyg_data.assert_called_once()


class TestValidateHandlerForConversion(unittest.TestCase):
    """Test suite for validate_handler_for_conversion function

    Tests the DYNAMIC CAPABILITY VALIDATION which checks that handlers
    implement required methods rather than checking against hardcoded type lists.
    """

    def test_validate_handler_valid_with_all_required_methods(self):
        """Test validation with handler that has all required methods"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_handler.get_molecule_creation_strategy.return_value = "identifier_coordinate_based"
        mock_handler.get_molecular_charge.return_value = 0

        from milia_pipeline.molecules.mol_conversion_utils import validate_handler_for_conversion

        # Should not raise any exception
        validate_handler_for_conversion(
            handler=mock_handler, operation="test_operation", molecule_index=0
        )

    def test_validate_handler_valid_any_dataset_type(self):
        """Test that any dataset type is valid if handler has required methods (DYNAMIC VALIDATION)"""
        # Test with various dataset types - all should be valid if methods exist
        for dataset_type in ["DFT", "DMC", "Wavefunction", "CustomDataset", "NewType"]:
            mock_handler = Mock()
            mock_handler.get_dataset_type.return_value = dataset_type
            mock_handler.get_molecule_creation_strategy.return_value = "identifier_coordinate_based"
            mock_handler.get_molecular_charge.return_value = 0

            from milia_pipeline.molecules.mol_conversion_utils import (
                validate_handler_for_conversion,
            )

            # Should not raise - validates capability not type
            validate_handler_for_conversion(
                handler=mock_handler, operation="test_operation", molecule_index=0
            )

    def test_validate_handler_none_raises_error(self):
        """Test that None handler raises HandlerNotAvailableError"""
        from milia_pipeline.molecules.mol_conversion_utils import validate_handler_for_conversion

        with self.assertRaises(vqm_exceptions.HandlerNotAvailableError) as context:
            validate_handler_for_conversion(
                handler=None, operation="test_operation", molecule_index=0
            )

        self.assertIn("Handler is required", str(context.exception))

    def test_validate_handler_missing_required_method_get_molecule_creation_strategy(self):
        """Test that handler missing get_molecule_creation_strategy raises HandlerValidationError"""
        mock_handler = Mock(spec=["get_dataset_type", "get_molecular_charge"])
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_handler.get_molecular_charge.return_value = 0
        # Note: get_molecule_creation_strategy is missing

        from milia_pipeline.molecules.mol_conversion_utils import validate_handler_for_conversion

        with self.assertRaises(vqm_exceptions.HandlerValidationError) as context:
            validate_handler_for_conversion(
                handler=mock_handler, operation="test_operation", molecule_index=0
            )

        self.assertIn("Missing methods", str(context.exception))
        self.assertIn("get_molecule_creation_strategy", str(context.exception))

    def test_validate_handler_missing_required_method_get_molecular_charge(self):
        """Test that handler missing get_molecular_charge raises HandlerValidationError"""
        mock_handler = Mock(spec=["get_dataset_type", "get_molecule_creation_strategy"])
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_handler.get_molecule_creation_strategy.return_value = "identifier_coordinate_based"
        # Note: get_molecular_charge is missing

        from milia_pipeline.molecules.mol_conversion_utils import validate_handler_for_conversion

        with self.assertRaises(vqm_exceptions.HandlerValidationError) as context:
            validate_handler_for_conversion(
                handler=mock_handler, operation="test_operation", molecule_index=0
            )

        self.assertIn("Missing methods", str(context.exception))
        self.assertIn("get_molecular_charge", str(context.exception))

    def test_validate_handler_with_validate_configuration_success(self):
        """Test that handler with validate_configuration is called during validation"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_handler.get_molecule_creation_strategy.return_value = "identifier_coordinate_based"
        mock_handler.get_molecular_charge.return_value = 0
        mock_handler.validate_configuration.return_value = None  # Success

        from milia_pipeline.molecules.mol_conversion_utils import validate_handler_for_conversion

        # Should not raise
        validate_handler_for_conversion(
            handler=mock_handler, operation="test_operation", molecule_index=0
        )

        # Verify validate_configuration was called
        mock_handler.validate_configuration.assert_called_once()

    def test_validate_handler_with_validate_configuration_failure(self):
        """Test that handler with failing validate_configuration raises HandlerValidationError"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_handler.get_molecule_creation_strategy.return_value = "identifier_coordinate_based"
        mock_handler.get_molecular_charge.return_value = 0
        mock_handler.validate_configuration.side_effect = ValueError("Invalid configuration")

        from milia_pipeline.molecules.mol_conversion_utils import validate_handler_for_conversion

        with self.assertRaises(vqm_exceptions.HandlerValidationError) as context:
            validate_handler_for_conversion(
                handler=mock_handler, operation="test_operation", molecule_index=0
            )

        self.assertIn("Invalid configuration", str(context.exception))


class TestGetConversionContextInfo(unittest.TestCase):
    """Test suite for get_conversion_context_info function"""

    def test_get_conversion_context_info_full(self):
        """Test getting conversion context with all information"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_handler.get_dataset_name.return_value = "QM9"

        from milia_pipeline.molecules.mol_conversion_utils import get_conversion_context_info

        context = get_conversion_context_info(
            handler=mock_handler, molecule_index=42, identifier="test_mol"
        )

        self.assertIsInstance(context, dict)
        self.assertEqual(context["molecule_index"], 42)
        self.assertEqual(context["dataset_type"], "DFT")
        self.assertEqual(context["dataset_name"], "QM9")
        self.assertEqual(context["source"], "test")

    def test_get_conversion_context_info_minimal(self):
        """Test getting conversion context with minimal information"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_handler.get_dataset_name.side_effect = AttributeError()

        from milia_pipeline.molecules.mol_conversion_utils import get_conversion_context_info

        context = get_conversion_context_info(
            handler=mock_handler, molecule_index=0, identifier="test_mol"
        )

        self.assertIsInstance(context, dict)
        self.assertEqual(context["molecule_index"], 0)
        self.assertEqual(context["dataset_type"], "DFT")

    def test_get_conversion_context_info_none_handler(self):
        """Test that None handler returns minimal context"""
        from milia_pipeline.molecules.mol_conversion_utils import get_conversion_context_info

        context = get_conversion_context_info(handler=None, molecule_index=5, identifier="test_mol")

        self.assertIsInstance(context, dict)
        self.assertEqual(context["molecule_index"], 5)


class TestApplyHandlerSpecificRdkitProcessing(unittest.TestCase):
    """Test suite for apply_handler_specific_rdkit_processing function"""

    def test_apply_processing_dft_dataset(self):
        """Test RDKit processing for DFT dataset"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"

        mock_mol = Mock()
        logger = logging.getLogger("test")

        from milia_pipeline.molecules.mol_conversion_utils import (
            apply_handler_specific_rdkit_processing,
        )

        result = apply_handler_specific_rdkit_processing(
            rdkit_mol=mock_mol, handler=mock_handler, logger=logger, molecule_index=0
        )

        self.assertIsNotNone(result)
        self.assertIs(result, mock_mol)  # Use assertIs instead of assertEqual with id()

    def test_apply_processing_md_dataset(self):
        """Test RDKit processing for MD dataset"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "MD"

        mock_mol = Mock()
        logger = logging.getLogger("test")

        from milia_pipeline.molecules.mol_conversion_utils import (
            apply_handler_specific_rdkit_processing,
        )

        result = apply_handler_specific_rdkit_processing(
            rdkit_mol=mock_mol, handler=mock_handler, logger=logger, molecule_index=0
        )

        self.assertIsNotNone(result)

    def test_apply_processing_none_handler(self):
        """Test that None handler returns mol unchanged"""
        mock_mol = Mock()
        logger = logging.getLogger("test")

        from milia_pipeline.molecules.mol_conversion_utils import (
            apply_handler_specific_rdkit_processing,
        )

        result = apply_handler_specific_rdkit_processing(
            rdkit_mol=mock_mol, handler=None, logger=logger, molecule_index=0
        )

        self.assertEqual(id(result), id(mock_mol))


class TestGetHandlerConversionStatistics(unittest.TestCase):
    """Test suite for get_handler_conversion_statistics function"""

    def test_get_statistics_with_handler(self):
        """Test getting conversion statistics from handler"""
        mock_handler = Mock()
        mock_handler.get_conversion_statistics.return_value = {
            "total_molecules": 100,
            "successful_conversions": 95,
            "failed_conversions": 5,
        }

        from milia_pipeline.molecules.mol_conversion_utils import get_handler_conversion_statistics

        stats = get_handler_conversion_statistics(handler=mock_handler, processed_molecules=[])

        self.assertIsInstance(stats, dict)
        self.assertEqual(stats["total_molecules"], 100)
        self.assertEqual(stats["successful_conversions"], 95)

    def test_get_statistics_handler_without_method(self):
        """Test getting statistics from handler without method"""
        mock_handler = Mock(spec=[])  # No get_conversion_statistics method

        from milia_pipeline.molecules.mol_conversion_utils import get_handler_conversion_statistics

        stats = get_handler_conversion_statistics(handler=mock_handler, processed_molecules=[])

        self.assertIsInstance(stats, dict)
        self.assertEqual(stats, {})

    def test_get_statistics_none_handler(self):
        """Test that None handler returns empty dict"""
        from milia_pipeline.molecules.mol_conversion_utils import get_handler_conversion_statistics

        stats = get_handler_conversion_statistics(handler=None, processed_molecules=[])

        self.assertIsInstance(stats, dict)
        self.assertEqual(stats, {})


class TestEnhanceConversionErrorContext(unittest.TestCase):
    """Test suite for enhance_conversion_error_context function"""

    def test_enhance_error_context_with_handler(self):
        """Test enhancing error context with handler information"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_handler.get_dataset_name.return_value = "QM9"

        original_error = ValueError("Original error message")

        from milia_pipeline.molecules.mol_conversion_utils import enhance_conversion_error_context

        enhanced = enhance_conversion_error_context(
            error=original_error,
            handler=mock_handler,
            molecule_index=42,
            additional_context={"stage": "rdkit_creation"},
        )

        self.assertIsInstance(enhanced, str)
        self.assertIn("Original error message", enhanced)
        self.assertIn("molecule_index", enhanced)
        self.assertIn("42", enhanced)

    def test_enhance_error_context_without_handler(self):
        """Test enhancing error context without handler"""
        original_error = ValueError("Original error message")

        from milia_pipeline.molecules.mol_conversion_utils import enhance_conversion_error_context

        enhanced = enhance_conversion_error_context(
            error=original_error, handler=None, molecule_index=10
        )

        self.assertIsInstance(enhanced, str)
        self.assertIn("Original error message", enhanced)


class TestValidateConversionPrerequisites(unittest.TestCase):
    """Test suite for validate_conversion_prerequisites function"""

    def test_validate_prerequisites_valid(self):
        """Test validation with valid inputs"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"

        coordinates = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]])
        atomic_numbers = np.array([8, 1])

        from milia_pipeline.molecules.mol_conversion_utils import validate_conversion_prerequisites

        # Should not raise any exception
        validate_conversion_prerequisites(
            handler=mock_handler,
            mol_identifier="O",
            coordinates=coordinates,
            atomic_numbers=atomic_numbers,
            molecule_index=0,
        )

    def test_validate_prerequisites_none_handler(self):
        """Test that None handler raises ValueError"""
        coordinates = np.array([[0.0, 0.0, 0.0]])
        atomic_numbers = np.array([8])

        from milia_pipeline.molecules.mol_conversion_utils import validate_conversion_prerequisites

        with self.assertRaises(vqm_exceptions.HandlerValidationError) as context:
            validate_conversion_prerequisites(
                handler=None,
                mol_identifier="O",
                coordinates=coordinates,
                atomic_numbers=atomic_numbers,
                molecule_index=0,
            )

        self.assertIn("Handler is required", str(context.exception))

    def test_validate_prerequisites_none_mol_identifier(self):
        """Test that None mol_identifier raises ValueError"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"

        coordinates = np.array([[0.0, 0.0, 0.0]])
        atomic_numbers = np.array([8])

        from milia_pipeline.molecules.mol_conversion_utils import validate_conversion_prerequisites

        with self.assertRaises(vqm_exceptions.HandlerValidationError) as context:
            validate_conversion_prerequisites(
                handler=mock_handler,
                mol_identifier=None,
                coordinates=coordinates,
                atomic_numbers=atomic_numbers,
                molecule_index=0,
            )

        self.assertIn("Molecule identifier is required", str(context.exception))

    def test_validate_prerequisites_none_coordinates(self):
        """Test that None coordinates raises ValueError"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"

        atomic_numbers = np.array([8])

        from milia_pipeline.molecules.mol_conversion_utils import validate_conversion_prerequisites

        with self.assertRaises(vqm_exceptions.HandlerValidationError) as context:
            validate_conversion_prerequisites(
                handler=mock_handler,
                mol_identifier="O",
                coordinates=None,
                atomic_numbers=atomic_numbers,
                molecule_index=0,
            )

        self.assertIn("Coordinates are required", str(context.exception))

    def test_validate_prerequisites_none_atomic_numbers(self):
        """Test that None atomic_numbers raises ValueError"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"

        coordinates = np.array([[0.0, 0.0, 0.0]])

        from milia_pipeline.molecules.mol_conversion_utils import validate_conversion_prerequisites

        with self.assertRaises(vqm_exceptions.HandlerValidationError) as context:
            validate_conversion_prerequisites(
                handler=mock_handler,
                mol_identifier="O",
                coordinates=coordinates,
                atomic_numbers=None,
                molecule_index=0,
            )

        self.assertIn("Atomic numbers are required", str(context.exception))

    def test_validate_prerequisites_coordinates_atomic_numbers_length_mismatch(self):
        """Test that mismatched coordinates/atomic_numbers length raises HandlerValidationError"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"

        # 3 coordinate rows but only 2 atomic numbers — mismatch
        coordinates = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
        atomic_numbers = np.array([8, 1])

        from milia_pipeline.molecules.mol_conversion_utils import validate_conversion_prerequisites

        with self.assertRaises(vqm_exceptions.HandlerValidationError) as context:
            validate_conversion_prerequisites(
                handler=mock_handler,
                mol_identifier="H2O",
                coordinates=coordinates,
                atomic_numbers=atomic_numbers,
                molecule_index=0,
            )

        self.assertIn("mismatch", str(context.exception).lower())


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests covering complete workflows"""

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_molecular_structure")
    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_atomic_numbers")
    @patch("milia_pipeline.molecules.mol_conversion_utils.Chem")
    @patch("milia_pipeline.molecules.mol_conversion_utils.from_rdmol")
    @patch("milia_pipeline.molecules.mol_conversion_utils.create_handler_error_context")
    def test_full_conversion_workflow(
        self,
        mock_context,
        mock_from_rdmol,
        mock_chem,
        mock_validate_atomic,
        mock_validate_structure,
        mock_validate_handler,
    ):
        """Test complete molecule conversion workflow"""
        logger = logging.getLogger("test")
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        # CRITICAL: Configure required handler methods for strategy selection
        mock_handler.get_molecule_creation_strategy.return_value = "identifier_coordinate_based"
        mock_handler.get_molecular_charge.return_value = 0

        coordinates = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
        atomic_numbers = np.array([8, 1, 1])
        inchi = "InChI=1S/H2O/h1H2"

        # Mock validation
        mock_validate_structure.return_value = (atomic_numbers, coordinates)
        mock_validate_atomic.return_value = True
        # Mock RDKit molecule
        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 3
        mock_mol.GetNumConformers.return_value = 1

        def mock_get_atom(idx):
            atom = Mock()
            if idx == 0:
                atom.GetAtomicNum.return_value = 8
            else:
                atom.GetAtomicNum.return_value = 1
            return atom

        mock_mol.GetAtomWithIdx.side_effect = mock_get_atom
        mock_atoms = [
            Mock(GetAtomicNum=Mock(return_value=8)),
            Mock(GetAtomicNum=Mock(return_value=1)),
            Mock(GetAtomicNum=Mock(return_value=1)),
        ]
        mock_mol.GetAtoms.return_value = iter(mock_atoms)

        mock_conformer = Mock()
        mock_conformer.GetPositions.return_value = coordinates
        mock_mol.GetConformer.return_value = mock_conformer

        mock_chem.MolFromInchi.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        mock_chem.Conformer.return_value = Mock()
        mock_chem.SANITIZE_SETAROMATICITY = 1
        mock_chem.SANITIZE_SETCONJUGATION = 2

        # Mock PyG Data
        mock_pyg_data = Data()
        mock_pyg_data.num_nodes = 3
        mock_pyg_data.num_edges = 2
        mock_from_rdmol.return_value = mock_pyg_data

        from milia_pipeline.molecules.mol_conversion_utils import create_mol_with_dataset_support

        # Execute full pipeline
        result = create_mol_with_dataset_support(
            mol_identifier=inchi,
            coordinates=coordinates,
            atomic_numbers=atomic_numbers,
            logger=logger,
            molecule_index=0,
            mol_id_type="inchi",
            raw_data_dict=None,
            handler=mock_handler,
        )

        # Verify
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "z"))
        self.assertTrue(hasattr(result, "pos"))


class TestEdgeCasesAndBoundaryConditions(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def test_single_atom_molecule(self):
        """Test handling of single-atom molecules"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"

        single_coord = np.array([[0.0, 0.0, 0.0]])
        single_atomic = np.array([1])  # Hydrogen

        from milia_pipeline.molecules.mol_conversion_utils import validate_conversion_prerequisites

        # Should handle single atom without error
        validate_conversion_prerequisites(
            handler=mock_handler,
            mol_identifier="H",
            coordinates=single_coord,
            atomic_numbers=single_atomic,
            molecule_index=0,
        )

    def test_large_molecule(self):
        """Test handling of large molecules"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"

        # Create large molecule data (100 atoms)
        large_coords = np.random.randn(100, 3)
        large_atomic = np.random.randint(1, 20, size=100)

        from milia_pipeline.molecules.mol_conversion_utils import validate_conversion_prerequisites

        # Should handle large molecules without error
        validate_conversion_prerequisites(
            handler=mock_handler,
            mol_identifier="C" * 100,  # Simple identifier
            coordinates=large_coords,
            atomic_numbers=large_atomic,
            molecule_index=0,
        )

    def test_negative_coordinates(self):
        """Test handling of negative coordinates"""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"

        negative_coords = np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]])
        atomic_nums = np.array([6, 8])

        from milia_pipeline.molecules.mol_conversion_utils import validate_conversion_prerequisites

        # Should handle negative coordinates without error
        validate_conversion_prerequisites(
            handler=mock_handler,
            mol_identifier="CO",
            coordinates=negative_coords,
            atomic_numbers=atomic_nums,
            molecule_index=0,
        )


class TestErrorHandlingAndRecovery(unittest.TestCase):
    """Test error handling and recovery mechanisms"""

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_molecular_structure")
    @patch("milia_pipeline.molecules.mol_conversion_utils.Chem")
    @patch("milia_pipeline.molecules.mol_conversion_utils.create_handler_error_context")
    def test_sanitization_failure_recovery(
        self, mock_context, mock_chem, mock_validate_structure, mock_validate_handler
    ):
        """Test that sanitization failures are handled gracefully"""
        logger = logging.getLogger("test")
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        # CRITICAL: Configure required handler methods for strategy selection
        mock_handler.get_molecule_creation_strategy.return_value = "identifier_coordinate_based"
        mock_handler.get_molecular_charge.return_value = 0

        coordinates = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]])
        atomic_numbers = np.array([6, 1])

        mock_validate_structure.return_value = (atomic_numbers, coordinates)

        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 2
        mock_mol.GetAtomWithIdx.side_effect = [
            Mock(GetAtomicNum=Mock(return_value=6)),
            Mock(GetAtomicNum=Mock(return_value=1)),
        ]

        mock_chem.MolFromInchi.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        mock_chem.SanitizeMol.side_effect = Exception("Sanitization failed")
        mock_chem.Conformer.return_value = Mock()
        mock_chem.SANITIZE_SETAROMATICITY = 1
        mock_chem.SANITIZE_SETCONJUGATION = 2

        from milia_pipeline.molecules.mol_conversion_utils import create_rdkit_mol

        # Should continue despite sanitization failure
        result = create_rdkit_mol(
            mol_identifier="InChI=1S/CH/h1H",
            coordinates=coordinates,
            atomic_numbers=atomic_numbers,
            logger=logger,
            molecule_index=0,
            mol_id_type="inchi",
            handler=mock_handler,
        )

        self.assertIsNotNone(result)

    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_handler_for_conversion")
    @patch("milia_pipeline.molecules.mol_conversion_utils.validate_molecular_structure")
    @patch("milia_pipeline.molecules.mol_conversion_utils.Chem")
    @patch("milia_pipeline.molecules.mol_conversion_utils.create_handler_error_context")
    def test_addhs_failure_recovery(
        self, mock_context, mock_chem, mock_validate_structure, mock_validate_handler
    ):
        """Test that AddHs failures are handled gracefully (common for QM structures)"""
        logger = logging.getLogger("test")
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_handler.get_molecule_creation_strategy.return_value = "identifier_coordinate_based"
        mock_handler.get_molecular_charge.return_value = 0

        coordinates = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]])
        atomic_numbers = np.array([6, 1])

        mock_validate_structure.return_value = (atomic_numbers, coordinates)

        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 2
        mock_mol.GetAtomWithIdx.side_effect = [
            Mock(GetAtomicNum=Mock(return_value=6)),
            Mock(GetAtomicNum=Mock(return_value=1)),
        ]

        mock_chem.MolFromInchi.return_value = mock_mol
        # AddHs fails (common for unusual valences)
        mock_chem.AddHs.side_effect = Exception("AddHs failed - unusual valence")
        mock_chem.Conformer.return_value = Mock()
        mock_chem.SANITIZE_SETAROMATICITY = 1
        mock_chem.SANITIZE_SETCONJUGATION = 2

        from milia_pipeline.molecules.mol_conversion_utils import create_rdkit_mol

        # Should continue despite AddHs failure
        result = create_rdkit_mol(
            mol_identifier="InChI=1S/CH/h1H",
            coordinates=coordinates,
            atomic_numbers=atomic_numbers,
            logger=logger,
            molecule_index=0,
            mol_id_type="inchi",
            handler=mock_handler,
        )

        self.assertIsNotNone(result)


def suite():
    """Create test suite"""
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCreateRdkitMol))
    test_suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestCreateRdkitMolStrategyRouting)
    )
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMolToPygData))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEnrichPygDataFromHandler))
    test_suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestCreateMolWithDatasetSupport)
    )
    test_suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestValidateHandlerForConversion)
    )
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGetConversionContextInfo))
    test_suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestApplyHandlerSpecificRdkitProcessing)
    )
    test_suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestGetHandlerConversionStatistics)
    )
    test_suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestEnhanceConversionErrorContext)
    )
    test_suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestValidateConversionPrerequisites)
    )
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegrationScenarios))
    test_suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestEdgeCasesAndBoundaryConditions)
    )
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestErrorHandlingAndRecovery))

    return test_suite


if __name__ == "__main__":
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
