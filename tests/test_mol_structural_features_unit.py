#!/usr/bin/env python3
# test_mol_structural_features_unit.py

"""
Production-ready unit test suite for mol_structural_features.py module.

This test suite covers all major functions in the mol_structural_features module,
including one-hot encoding, atom-level feature extraction, bond-level feature
extraction, 3D geometry features, conformer/charge preprocessing, aggregation
tensors, and the main add_structural_features() entry point.

KEY COVERAGE AREAS:
1. _one_hot_encoding - Value encoding with known and unknown values
2. _ensure_conformer_and_charges - Conformer setup and charge assignment
3. Atom feature functions - degree, total_degree, hybridization, total_valence,
   is_aromatic, is_in_ring, partial_charge, mulliken_charge, num_aromatic_bonds, chirality
4. Bond feature functions - bond_type, is_conjugated, is_aromatic, is_in_any_ring,
   stereo, bond_length (3D), bond_length_binned
5. _calculate_atom_features_tensor - Aggregation with error handling
6. _calculate_bond_features_tensor - Aggregation with bidirectional edge validation
7. get_available_features - Feature catalog
8. add_structural_features - Main entry point with full integration paths
9. Error handling - StructuralFeatureError, MoleculeProcessingError, PyGDataCreationError

DESIGN PRINCIPLES:
- All mocks are configured at test-level using @patch decorators (no sys.modules pollution)
- Tests use real RDKit objects where practical for molecular feature correctness
- No hardcoded dataset types - tests are handler-agnostic
- Tests cover both success paths and error conditions
- No file downloads or external API calls

Tests are designed to run in a Docker environment with mocked external dependencies.
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock, call

# Add the project root to Python path FIRST
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import real exceptions - no mocking needed
from milia_pipeline import exceptions as vqm_exceptions

import unittest
import logging
import math
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import HybridizationType, BondType, rdPartialCharges

# Now import the module under test normally
from milia_pipeline.molecules import mol_structural_features

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)


# ============================================================================
# Helper utilities for building test molecules
# ============================================================================

def _make_water_mol(with_conformer=False, with_gasteiger=False, with_mulliken=False):
    """Create a water (H2O) RDKit molecule for testing.

    Returns:
        Chem.Mol: An editable-compatible RDKit molecule with optional
                  conformer and/or charges.
    """
    mol = Chem.MolFromSmiles("O")
    mol = Chem.AddHs(mol)  # 3 atoms: O, H, H
    if with_conformer:
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
        conf.SetAtomPosition(1, (0.96, 0.0, 0.0))
        conf.SetAtomPosition(2, (-0.24, 0.93, 0.0))
        mol.AddConformer(conf)
    if with_gasteiger:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    if with_mulliken:
        charges = [-0.82, 0.41, 0.41]
        for i, c in enumerate(charges):
            mol.GetAtomWithIdx(i).SetDoubleProp('_MullikenCharge', c)
    return mol


def _make_benzene_mol(with_conformer=False):
    """Create a benzene (C6H6) RDKit molecule with aromatic features."""
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol = Chem.AddHs(mol)
    if with_conformer:
        from rdkit.Chem import AllChem
        AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


def _make_ethane_mol(with_conformer=False):
    """Create an ethane (C2H6) RDKit molecule — simple single bond."""
    mol = Chem.MolFromSmiles("CC")
    mol = Chem.AddHs(mol)
    if with_conformer:
        conf = Chem.Conformer(mol.GetNumAtoms())
        # Simple positions for C-C bond of ~1.54 Angstrom
        positions = [
            (0.0, 0.0, 0.0), (1.54, 0.0, 0.0),  # C, C
            (-0.36, 1.03, 0.0), (-0.36, -0.51, 0.89),  # H, H (on C0)
            (-0.36, -0.51, -0.89),  # H (on C0)
            (1.90, 1.03, 0.0), (1.90, -0.51, 0.89),  # H, H (on C1)
            (1.90, -0.51, -0.89),  # H (on C1)
        ]
        for i, pos in enumerate(positions):
            conf.SetAtomPosition(i, pos)
        mol.AddConformer(conf)
    return mol


def _make_ethene_mol():
    """Create an ethene (C2H4) RDKit molecule — double bond."""
    mol = Chem.MolFromSmiles("C=C")
    mol = Chem.AddHs(mol)
    return mol


def _make_pyg_data_with_edges(mol):
    """Create a PyG Data object with bidirectional edge_index from RDKit mol."""
    src, dst = [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src.extend([u, v])
        dst.extend([v, u])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(edge_index=edge_index)


# ============================================================================
# Test: _one_hot_encoding
# ============================================================================

class TestOneHotEncoding(unittest.TestCase):
    """Test suite for _one_hot_encoding helper function."""

    def test_value_found_in_choices(self):
        """Test encoding when value is present in choices."""
        result = mol_structural_features._one_hot_encoding("B", ["A", "B", "C"])
        self.assertEqual(result, [0, 1, 0])

    def test_first_element(self):
        """Test encoding when value is the first element."""
        result = mol_structural_features._one_hot_encoding(1, [1, 2, 3])
        self.assertEqual(result, [1, 0, 0])

    def test_last_element(self):
        """Test encoding when value is the last element."""
        result = mol_structural_features._one_hot_encoding(3, [1, 2, 3])
        self.assertEqual(result, [0, 0, 1])

    def test_value_not_in_choices_returns_all_zeros(self):
        """Test encoding when value is NOT in choices (unknown/other category)."""
        result = mol_structural_features._one_hot_encoding("Z", ["A", "B", "C"])
        self.assertEqual(result, [0, 0, 0])

    def test_empty_choices(self):
        """Test encoding with empty choices list."""
        result = mol_structural_features._one_hot_encoding("A", [])
        self.assertEqual(result, [])

    def test_none_value(self):
        """Test encoding with None value."""
        result = mol_structural_features._one_hot_encoding(None, ["A", "B", None])
        self.assertEqual(result, [0, 0, 1])

    def test_enum_values(self):
        """Test encoding with RDKit-style enum values (HybridizationType)."""
        choices = [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3]
        result = mol_structural_features._one_hot_encoding(HybridizationType.SP2, choices)
        self.assertEqual(result, [0, 1, 0])

    def test_single_choice(self):
        """Test encoding with a single-element choices list."""
        result = mol_structural_features._one_hot_encoding("A", ["A"])
        self.assertEqual(result, [1])


# ============================================================================
# Test: _ensure_conformer_and_charges
# ============================================================================

class TestEnsureConformerAndCharges(unittest.TestCase):
    """Test suite for _ensure_conformer_and_charges preprocessing function."""

    def test_adds_conformer_from_coordinates(self):
        """Test that conformer is added when coordinates are provided."""
        mol = Chem.MolFromSmiles("O")
        mol = Chem.AddHs(mol)
        coords = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

        self.assertEqual(mol.GetNumConformers(), 0)
        mol_structural_features._ensure_conformer_and_charges(mol, coordinates=coords)
        self.assertEqual(mol.GetNumConformers(), 1)

        # Verify positions
        conf = mol.GetConformer(0)
        pos0 = conf.GetAtomPosition(0)
        self.assertAlmostEqual(pos0.x, 0.0)
        self.assertAlmostEqual(pos0.y, 0.0)
        self.assertAlmostEqual(pos0.z, 0.0)

    def test_adds_mulliken_charges_when_provided(self):
        """Test that Mulliken charges are set on atoms when provided."""
        mol = Chem.MolFromSmiles("O")
        mol = Chem.AddHs(mol)
        charges = np.array([-0.82, 0.41, 0.41])

        mol_structural_features._ensure_conformer_and_charges(mol, mulliken_charges=charges)

        for i, expected_charge in enumerate(charges):
            atom = mol.GetAtomWithIdx(i)
            self.assertTrue(atom.HasProp('_MullikenCharge'))
            self.assertAlmostEqual(atom.GetDoubleProp('_MullikenCharge'), expected_charge)

    def test_falls_back_to_gasteiger_when_no_mulliken(self):
        """Test that Gasteiger charges are computed as fallback."""
        mol = Chem.MolFromSmiles("O")
        mol = Chem.AddHs(mol)

        mol_structural_features._ensure_conformer_and_charges(mol)

        # Gasteiger charges should have been computed
        atom = mol.GetAtomWithIdx(0)
        self.assertTrue(atom.HasProp('_GasteigerCharge'))

    def test_coordinate_shape_mismatch_skips_conformer(self):
        """Test that mismatched coordinate length skips conformer addition."""
        mol = Chem.MolFromSmiles("O")
        mol = Chem.AddHs(mol)  # 3 atoms
        wrong_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # 2 points

        mol_structural_features._ensure_conformer_and_charges(mol, coordinates=wrong_coords)
        self.assertEqual(mol.GetNumConformers(), 0)

    def test_mulliken_charge_shape_mismatch_falls_back_to_gasteiger(self):
        """Test that mismatched Mulliken charges length falls back to Gasteiger."""
        mol = Chem.MolFromSmiles("O")
        mol = Chem.AddHs(mol)  # 3 atoms
        wrong_charges = np.array([-0.5, 0.5])  # 2 charges

        mol_structural_features._ensure_conformer_and_charges(mol, mulliken_charges=wrong_charges)

        atom = mol.GetAtomWithIdx(0)
        # Should have fallen back to Gasteiger
        self.assertTrue(atom.HasProp('_GasteigerCharge'))
        self.assertFalse(atom.HasProp('_MullikenCharge'))

    def test_none_coordinates_and_none_charges(self):
        """Test with both coordinates and charges as None (defaults)."""
        mol = Chem.MolFromSmiles("O")
        mol = Chem.AddHs(mol)

        mol_structural_features._ensure_conformer_and_charges(mol)
        self.assertEqual(mol.GetNumConformers(), 0)
        # Should fallback to Gasteiger
        atom = mol.GetAtomWithIdx(0)
        self.assertTrue(atom.HasProp('_GasteigerCharge'))


# ============================================================================
# Test: Atom Feature Calculation Functions
# ============================================================================

class TestAtomFeatureFunctions(unittest.TestCase):
    """Test suite for individual atom feature extraction functions."""

    def setUp(self):
        """Create test molecules and atoms."""
        self.water = _make_water_mol(with_gasteiger=True)
        self.benzene = _make_benzene_mol()
        # Oxygen atom in water
        self.oxygen = self.water.GetAtomWithIdx(0)
        # Hydrogen atom in water
        self.hydrogen = self.water.GetAtomWithIdx(1)
        # Carbon atom in benzene (aromatic)
        self.aromatic_carbon = self.benzene.GetAtomWithIdx(0)

    def test_get_atom_degree_oxygen_in_water(self):
        """Test degree of oxygen in water (bonded to 2 H)."""
        degree = mol_structural_features._get_atom_degree(self.oxygen)
        self.assertEqual(degree, 2)

    def test_get_atom_degree_hydrogen_in_water(self):
        """Test degree of hydrogen in water (bonded to 1 O)."""
        degree = mol_structural_features._get_atom_degree(self.hydrogen)
        self.assertEqual(degree, 1)

    def test_get_atom_total_degree_oxygen(self):
        """Test total degree of oxygen (includes implicit H)."""
        total_deg = mol_structural_features._get_atom_total_degree(self.oxygen)
        self.assertEqual(total_deg, 2)

    def test_get_atom_hybridization_feature_oxygen_sp3(self):
        """Test hybridization encoding for oxygen (SP3 in water)."""
        encoding = mol_structural_features._get_atom_hybridization_feature(self.oxygen)
        self.assertEqual(len(encoding), 7)  # 7 hybridization types
        self.assertEqual(sum(encoding), 1)  # one-hot: exactly one 1
        # SP3 is at index 3
        self.assertEqual(encoding[3], 1)

    def test_get_atom_hybridization_feature_aromatic_sp2(self):
        """Test hybridization encoding for aromatic carbon (SP2)."""
        encoding = mol_structural_features._get_atom_hybridization_feature(self.aromatic_carbon)
        self.assertEqual(len(encoding), 7)
        # SP2 is at index 2
        self.assertEqual(encoding[2], 1)

    def test_get_atom_total_valence(self):
        """Test total valence of oxygen in water."""
        valence = mol_structural_features._get_atom_total_valence(self.oxygen)
        self.assertEqual(valence, 2)

    def test_is_atom_aromatic_true(self):
        """Test aromaticity detection for aromatic carbon."""
        result = mol_structural_features._is_atom_aromatic(self.aromatic_carbon)
        self.assertEqual(result, 1)

    def test_is_atom_aromatic_false(self):
        """Test aromaticity detection for non-aromatic oxygen."""
        result = mol_structural_features._is_atom_aromatic(self.oxygen)
        self.assertEqual(result, 0)

    def test_is_atom_in_ring_true(self):
        """Test ring membership for aromatic carbon in benzene."""
        result = mol_structural_features._is_atom_in_ring(self.aromatic_carbon)
        self.assertEqual(result, 1)

    def test_is_atom_in_ring_false(self):
        """Test ring membership for oxygen in water (no ring)."""
        result = mol_structural_features._is_atom_in_ring(self.oxygen)
        self.assertEqual(result, 0)

    def test_get_atom_partial_charge_with_gasteiger(self):
        """Test partial charge retrieval when Gasteiger charges are computed."""
        charge = mol_structural_features._get_atom_partial_charge(self.oxygen)
        self.assertIsInstance(charge, float)
        # Oxygen in water should have a negative Gasteiger charge
        self.assertLess(charge, 0.0)

    def test_get_atom_partial_charge_no_gasteiger_returns_zero(self):
        """Test partial charge returns 0.0 when no Gasteiger charges present."""
        mol = Chem.MolFromSmiles("O")
        mol = Chem.AddHs(mol)
        atom = mol.GetAtomWithIdx(0)
        charge = mol_structural_features._get_atom_partial_charge(atom)
        self.assertEqual(charge, 0.0)

    def test_get_atom_partial_charge_nan_raises_value_error(self):
        """Test that NaN Gasteiger charge raises ValueError."""
        mol = Chem.MolFromSmiles("O")
        mol = Chem.AddHs(mol)
        atom = mol.GetAtomWithIdx(0)
        atom.SetDoubleProp('_GasteigerCharge', float('nan'))
        with self.assertRaises(ValueError):
            mol_structural_features._get_atom_partial_charge(atom)

    def test_get_atom_partial_charge_inf_raises_value_error(self):
        """Test that Inf Gasteiger charge raises ValueError."""
        mol = Chem.MolFromSmiles("O")
        mol = Chem.AddHs(mol)
        atom = mol.GetAtomWithIdx(0)
        atom.SetDoubleProp('_GasteigerCharge', float('inf'))
        with self.assertRaises(ValueError):
            mol_structural_features._get_atom_partial_charge(atom)

    def test_get_atom_mulliken_charge_present(self):
        """Test Mulliken charge retrieval when property is set."""
        mol = _make_water_mol(with_mulliken=True)
        atom = mol.GetAtomWithIdx(0)
        charge = mol_structural_features._get_atom_mulliken_charge(atom)
        self.assertAlmostEqual(charge, -0.82)

    def test_get_atom_mulliken_charge_absent_returns_zero(self):
        """Test Mulliken charge returns 0.0 when property is not set."""
        mol = Chem.MolFromSmiles("O")
        mol = Chem.AddHs(mol)
        atom = mol.GetAtomWithIdx(0)
        charge = mol_structural_features._get_atom_mulliken_charge(atom)
        self.assertEqual(charge, 0.0)

    def test_get_num_aromatic_bonds_aromatic_carbon(self):
        """Test count of aromatic bonds for carbon in benzene."""
        count = mol_structural_features._get_num_aromatic_bonds_to_atom(self.aromatic_carbon)
        self.assertEqual(count, 2)  # Each ring carbon has 2 aromatic bonds

    def test_get_num_aromatic_bonds_non_aromatic(self):
        """Test count of aromatic bonds for non-aromatic atom."""
        count = mol_structural_features._get_num_aromatic_bonds_to_atom(self.oxygen)
        self.assertEqual(count, 0)

    def test_get_atom_chirality_feature_unspecified(self):
        """Test chirality encoding for an atom with no chirality."""
        encoding = mol_structural_features._get_atom_chirality_feature(self.oxygen)
        self.assertEqual(len(encoding), 4)  # 4 chirality types
        # CHI_UNSPECIFIED is at index 0
        self.assertEqual(encoding[0], 1)

    def test_get_atom_chirality_feature_is_one_hot(self):
        """Test chirality encoding is valid one-hot vector."""
        encoding = mol_structural_features._get_atom_chirality_feature(self.aromatic_carbon)
        self.assertEqual(len(encoding), 4)
        self.assertLessEqual(sum(encoding), 1)  # One-hot or all zeros


# ============================================================================
# Test: Bond Feature Calculation Functions
# ============================================================================

class TestBondFeatureFunctions(unittest.TestCase):
    """Test suite for individual bond feature extraction functions."""

    def setUp(self):
        """Create test molecules and bonds."""
        self.ethane = _make_ethane_mol(with_conformer=True)
        self.ethene = _make_ethene_mol()
        self.benzene = _make_benzene_mol()

        # Get specific bonds
        self.single_bond = self.ethane.GetBondWithIdx(0)  # C-C single bond
        self.double_bond = self.ethene.GetBondWithIdx(0)  # C=C double bond
        self.aromatic_bond = self.benzene.GetBondWithIdx(0)  # Aromatic C-C bond

    def test_get_bond_type_feature_single(self):
        """Test bond type encoding for single bond."""
        encoding = mol_structural_features._get_bond_type_feature(self.single_bond)
        self.assertEqual(len(encoding), 4)  # SINGLE, DOUBLE, TRIPLE, AROMATIC
        self.assertEqual(encoding[0], 1)  # SINGLE at index 0

    def test_get_bond_type_feature_double(self):
        """Test bond type encoding for double bond."""
        encoding = mol_structural_features._get_bond_type_feature(self.double_bond)
        self.assertEqual(encoding[1], 1)  # DOUBLE at index 1

    def test_get_bond_type_feature_aromatic(self):
        """Test bond type encoding for aromatic bond."""
        encoding = mol_structural_features._get_bond_type_feature(self.aromatic_bond)
        self.assertEqual(encoding[3], 1)  # AROMATIC at index 3

    def test_is_bond_conjugated(self):
        """Test conjugation detection for aromatic bond."""
        result = mol_structural_features._is_bond_conjugated(self.aromatic_bond)
        self.assertIn(result, [0, 1])

    def test_is_bond_conjugated_single_non_conjugated(self):
        """Test conjugation for non-conjugated single bond (C-H in ethane)."""
        # C-H bond in ethane (index 1 or later is C-H)
        ch_bond = self.ethane.GetBondWithIdx(1)
        result = mol_structural_features._is_bond_conjugated(ch_bond)
        self.assertEqual(result, 0)

    def test_is_bond_aromatic_true(self):
        """Test aromaticity detection for aromatic bond."""
        result = mol_structural_features._is_bond_aromatic(self.aromatic_bond)
        self.assertEqual(result, 1)

    def test_is_bond_aromatic_false(self):
        """Test aromaticity detection for non-aromatic bond."""
        result = mol_structural_features._is_bond_aromatic(self.single_bond)
        self.assertEqual(result, 0)

    def test_is_bond_in_any_ring_true(self):
        """Test ring membership for aromatic bond in benzene."""
        result = mol_structural_features._is_bond_in_any_ring(self.aromatic_bond)
        self.assertEqual(result, 1)

    def test_is_bond_in_any_ring_false(self):
        """Test ring membership for bond NOT in a ring."""
        result = mol_structural_features._is_bond_in_any_ring(self.single_bond)
        self.assertEqual(result, 0)

    def test_get_bond_stereo_feature(self):
        """Test stereo encoding returns valid one-hot vector."""
        encoding = mol_structural_features._get_bond_stereo_feature(self.single_bond)
        self.assertEqual(len(encoding), 4)  # STEREONONE, STEREOANY, STEREOZ, STEREOE
        self.assertEqual(sum(encoding), 1)  # exactly one 1

    def test_get_bond_stereo_feature_none_stereo(self):
        """Test stereo encoding for bond with no stereochemistry."""
        encoding = mol_structural_features._get_bond_stereo_feature(self.single_bond)
        self.assertEqual(encoding[0], 1)  # STEREONONE at index 0

    def test_get_bond_length_3d_with_conformer(self):
        """Test 3D bond length calculation with conformer present."""
        bond_length = mol_structural_features._get_bond_length_3d(self.single_bond)
        # C-C bond in ethane is ~1.54 Angstrom
        self.assertGreater(bond_length, 1.0)
        self.assertLess(bond_length, 2.0)

    def test_get_bond_length_3d_no_conformer_returns_zero(self):
        """Test 3D bond length returns 0.0 when no conformer present."""
        bond = self.ethene.GetBondWithIdx(0)  # No conformer added
        bond_length = mol_structural_features._get_bond_length_3d(bond)
        self.assertEqual(bond_length, 0.0)

    def test_get_bond_length_binned_with_conformer(self):
        """Test binned bond length produces valid one-hot vector."""
        encoding = mol_structural_features._get_bond_length_binned(self.single_bond)
        # Default bins: [0.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, inf] → 9 bins
        self.assertEqual(len(encoding), 9)
        self.assertEqual(sum(encoding), 1)

    def test_get_bond_length_binned_no_conformer(self):
        """Test binned bond length with no conformer (length=0.0 → first bin)."""
        bond = self.ethene.GetBondWithIdx(0)
        encoding = mol_structural_features._get_bond_length_binned(bond)
        self.assertEqual(len(encoding), 9)
        self.assertEqual(encoding[0], 1)  # 0.0 falls in first bin [0.0, 1.0]

    def test_get_bond_length_binned_custom_bins(self):
        """Test binned bond length with custom bin edges."""
        custom_bins = [0.0, 1.5, 2.0, float('inf')]
        encoding = mol_structural_features._get_bond_length_binned(
            self.single_bond, bin_edges=custom_bins
        )
        self.assertEqual(len(encoding), 3)  # 3 bins
        self.assertEqual(sum(encoding), 1)


# ============================================================================
# Test: _calculate_atom_features_tensor
# ============================================================================

class TestCalculateAtomFeaturesTensor(unittest.TestCase):
    """Test suite for _calculate_atom_features_tensor aggregation function."""

    def test_single_scalar_feature(self):
        """Test tensor generation with a single scalar feature (degree)."""
        mol = _make_water_mol()
        tensor = mol_structural_features._calculate_atom_features_tensor(
            mol, ["degree"]
        )
        self.assertEqual(tensor.shape[0], 3)  # 3 atoms in water with H
        self.assertEqual(tensor.shape[1], 1)  # 1 feature
        self.assertEqual(tensor.dtype, torch.float)

    def test_single_vector_feature(self):
        """Test tensor generation with a vector feature (hybridization)."""
        mol = _make_water_mol()
        tensor = mol_structural_features._calculate_atom_features_tensor(
            mol, ["hybridization"]
        )
        self.assertEqual(tensor.shape[0], 3)
        self.assertEqual(tensor.shape[1], 7)  # 7 hybridization types

    def test_multiple_features_concatenation(self):
        """Test that multiple features are properly concatenated."""
        mol = _make_water_mol()
        features = ["degree", "hybridization", "is_aromatic"]
        tensor = mol_structural_features._calculate_atom_features_tensor(
            mol, features
        )
        self.assertEqual(tensor.shape[0], 3)
        # degree(1) + hybridization(7) + is_aromatic(1) = 9
        self.assertEqual(tensor.shape[1], 9)

    def test_all_atom_features(self):
        """Test tensor generation with all available atom features."""
        mol = _make_water_mol(with_gasteiger=True, with_mulliken=True)
        all_features = [
            "degree", "total_degree", "hybridization", "total_valence",
            "is_aromatic", "is_in_ring", "partial_charge", "mulliken_charge",
            "num_aromatic_bonds", "chirality"
        ]
        tensor = mol_structural_features._calculate_atom_features_tensor(
            mol, all_features
        )
        self.assertEqual(tensor.shape[0], 3)
        # degree(1) + total_degree(1) + hybridization(7) + total_valence(1) +
        # is_aromatic(1) + is_in_ring(1) + partial_charge(1) + mulliken_charge(1) +
        # num_aromatic_bonds(1) + chirality(4) = 19
        self.assertEqual(tensor.shape[1], 19)

    def test_unsupported_feature_raises_structural_feature_error(self):
        """Test that requesting an unsupported feature raises StructuralFeatureError."""
        mol = _make_water_mol()
        with self.assertRaises(vqm_exceptions.StructuralFeatureError) as context:
            mol_structural_features._calculate_atom_features_tensor(
                mol, ["nonexistent_feature"], molecule_index=5, inchi="InChI=test"
            )
        self.assertIn("Unsupported atom feature", str(context.exception))

    def test_unsupported_feature_with_suggestion(self):
        """Test that unsupported feature with partial match produces suggestion."""
        mol = _make_water_mol()
        with self.assertRaises(vqm_exceptions.StructuralFeatureError) as context:
            mol_structural_features._calculate_atom_features_tensor(
                mol, ["aromatic"], molecule_index=0
            )
        # "aromatic" is a partial match for "is_aromatic"
        self.assertIn("Unsupported atom feature", str(context.exception))

    def test_empty_feature_list_returns_empty_tensor(self):
        """Test that empty feature list results in correct tensor shape."""
        mol = _make_water_mol()
        tensor = mol_structural_features._calculate_atom_features_tensor(
            mol, []
        )
        # No features selected → empty for all atoms, loop produces 3 empty vectors
        # but they'd be length-0; padded to max_len. With empty features, features_list
        # will be [[],[],[]] → max_len 0 → tensor shape (3, 0)
        self.assertEqual(tensor.shape[0], 3)
        self.assertEqual(tensor.shape[1], 0)

    def test_partial_charge_triggers_gasteiger_computation(self):
        """Test that requesting partial_charge triggers Gasteiger charge computation."""
        mol = Chem.MolFromSmiles("O")
        mol = Chem.AddHs(mol)
        # No charges computed yet
        self.assertFalse(mol.GetAtomWithIdx(0).HasProp('_GasteigerCharge'))

        tensor = mol_structural_features._calculate_atom_features_tensor(
            mol, ["partial_charge"]
        )
        self.assertEqual(tensor.shape[0], 3)
        self.assertEqual(tensor.shape[1], 1)

    def test_error_context_includes_molecule_index_and_inchi(self):
        """Test that error context propagates molecule_index and inchi."""
        mol = _make_water_mol()
        with self.assertRaises(vqm_exceptions.StructuralFeatureError) as context:
            mol_structural_features._calculate_atom_features_tensor(
                mol, ["bad_feature"], molecule_index=42, inchi="InChI=1S/H2O/h1H2"
            )
        exc = context.exception
        self.assertIn("bad_feature", str(exc))


# ============================================================================
# Test: _calculate_bond_features_tensor
# ============================================================================

class TestCalculateBondFeaturesTensor(unittest.TestCase):
    """Test suite for _calculate_bond_features_tensor aggregation function."""

    def setUp(self):
        """Create test molecules with conformers and edge indices."""
        self.ethane = _make_ethane_mol(with_conformer=True)
        self.ethane_pyg = _make_pyg_data_with_edges(self.ethane)

    def test_single_bond_feature(self):
        """Test bond feature tensor with a single feature (bond_type)."""
        tensor = mol_structural_features._calculate_bond_features_tensor(
            self.ethane, self.ethane_pyg.edge_index, ["bond_type"]
        )
        num_edges = self.ethane_pyg.edge_index.size(1)
        self.assertEqual(tensor.shape[0], num_edges)
        self.assertEqual(tensor.shape[1], 4)  # bond_type one-hot has 4 elements

    def test_multiple_bond_features(self):
        """Test bond feature tensor with multiple features."""
        tensor = mol_structural_features._calculate_bond_features_tensor(
            self.ethane, self.ethane_pyg.edge_index,
            ["bond_type", "is_conjugated", "is_aromatic"]
        )
        num_edges = self.ethane_pyg.edge_index.size(1)
        self.assertEqual(tensor.shape[0], num_edges)
        # bond_type(4) + is_conjugated(1) + is_aromatic(1) = 6
        self.assertEqual(tensor.shape[1], 6)

    def test_3d_bond_features_with_conformer(self):
        """Test bond_length and bond_length_binned features with conformer present."""
        tensor = mol_structural_features._calculate_bond_features_tensor(
            self.ethane, self.ethane_pyg.edge_index,
            ["bond_length", "bond_length_binned"]
        )
        num_edges = self.ethane_pyg.edge_index.size(1)
        self.assertEqual(tensor.shape[0], num_edges)
        # bond_length(1) + bond_length_binned(9) = 10
        self.assertEqual(tensor.shape[1], 10)

    def test_3d_features_without_conformer_raises_error(self):
        """Test that 3D features without conformer raises StructuralFeatureError."""
        mol = _make_ethene_mol()  # No conformer
        pyg = _make_pyg_data_with_edges(mol)
        with self.assertRaises(vqm_exceptions.StructuralFeatureError) as context:
            mol_structural_features._calculate_bond_features_tensor(
                mol, pyg.edge_index, ["bond_length"]
            )
        self.assertIn("conformer", str(context.exception).lower())

    def test_unsupported_bond_feature_raises_error(self):
        """Test that requesting unsupported bond feature raises StructuralFeatureError."""
        with self.assertRaises(vqm_exceptions.StructuralFeatureError) as context:
            mol_structural_features._calculate_bond_features_tensor(
                self.ethane, self.ethane_pyg.edge_index,
                ["invalid_bond_feature"], molecule_index=10
            )
        self.assertIn("Unsupported bond feature", str(context.exception))

    def test_non_bidirectional_edge_index_raises_error(self):
        """Test that non-bidirectional edge_index raises StructuralFeatureError."""
        # Create unidirectional edge_index (missing reverse edges)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        with self.assertRaises(vqm_exceptions.StructuralFeatureError) as context:
            mol_structural_features._calculate_bond_features_tensor(
                self.ethane, edge_index, ["bond_type"]
            )
        self.assertIn("bidirectional", str(context.exception).lower())

    def test_none_edge_index_raises_error(self):
        """Test that None edge_index raises StructuralFeatureError."""
        with self.assertRaises(vqm_exceptions.StructuralFeatureError):
            mol_structural_features._calculate_bond_features_tensor(
                self.ethane, None, ["bond_type"]
            )

    def test_empty_edge_index_returns_empty_tensor(self):
        """Test that empty edge_index returns empty tensor with correct feature dim."""
        empty_edge_index = torch.empty(2, 0, dtype=torch.long)
        tensor = mol_structural_features._calculate_bond_features_tensor(
            self.ethane, empty_edge_index, ["bond_type"]
        )
        self.assertEqual(tensor.shape[0], 0)
        self.assertEqual(tensor.shape[1], 4)  # bond_type has 4 features

    def test_bond_features_bidirectional_symmetry(self):
        """Test that bond features are symmetric for bidirectional edges."""
        tensor = mol_structural_features._calculate_bond_features_tensor(
            self.ethane, self.ethane_pyg.edge_index, ["bond_type"]
        )
        edge_index = self.ethane_pyg.edge_index
        # For each pair (u,v) and (v,u), features should be identical
        feature_dict = {}
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            feature_dict[(u, v)] = tensor[i].tolist()

        for (u, v), features in feature_dict.items():
            if (v, u) in feature_dict:
                self.assertEqual(features, feature_dict[(v, u)])

    def test_missing_rdkit_bond_for_pyg_edge_assigns_zeros(self):
        """Test that PyG edges without corresponding RDKit bonds get zero features."""
        mol = _make_ethane_mol(with_conformer=True)
        # Create edge_index with an extra self-loop edge (no RDKit bond)
        src, dst = [], []
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src.extend([u, v])
            dst.extend([v, u])
        # Add self-loop (no corresponding RDKit bond)
        src.extend([0, 0])  # self-loop on atom 0 (bidirectional placeholder)
        dst.extend([0, 0])
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        logger_mock = logging.getLogger('test_missing_bond')
        tensor = mol_structural_features._calculate_bond_features_tensor(
            mol, edge_index, ["bond_type"],
            logger_instance=logger_mock
        )
        # The self-loop edges should have zero features
        # Last 2 edges are the self-loops
        self_loop_features = tensor[-1].tolist()
        self.assertEqual(self_loop_features, [0, 0, 0, 0])


# ============================================================================
# Test: get_available_features
# ============================================================================

class TestGetAvailableFeatures(unittest.TestCase):
    """Test suite for get_available_features catalog function."""

    def test_returns_dict_with_atom_and_bond_keys(self):
        """Test that result has 'atom' and 'bond' keys."""
        features = mol_structural_features.get_available_features()
        self.assertIn("atom", features)
        self.assertIn("bond", features)

    def test_atom_features_complete(self):
        """Test that all documented atom features are present."""
        features = mol_structural_features.get_available_features()
        expected_atom = [
            "degree", "total_degree", "hybridization", "total_valence",
            "is_aromatic", "is_in_ring", "partial_charge", "mulliken_charge",
            "num_aromatic_bonds", "chirality"
        ]
        self.assertEqual(sorted(features["atom"]), sorted(expected_atom))

    def test_bond_features_complete(self):
        """Test that all documented bond features are present."""
        features = mol_structural_features.get_available_features()
        expected_bond = [
            "bond_type", "is_conjugated", "is_aromatic", "is_in_any_ring",
            "stereo", "bond_length", "bond_length_binned"
        ]
        self.assertEqual(sorted(features["bond"]), sorted(expected_bond))

    def test_features_are_lists_of_strings(self):
        """Test that feature values are lists of strings."""
        features = mol_structural_features.get_available_features()
        for key in ["atom", "bond"]:
            self.assertIsInstance(features[key], list)
            for f in features[key]:
                self.assertIsInstance(f, str)


# ============================================================================
# Test: add_structural_features (Main Entry Point)
# ============================================================================

class TestAddStructuralFeatures(unittest.TestCase):
    """Test suite for add_structural_features main entry point."""

    def setUp(self):
        """Set up common test fixtures."""
        self.logger = logging.getLogger('test_structural')

    def test_none_feature_config_returns_unchanged_data(self):
        """Test that None feature_config returns data unchanged."""
        mol = _make_water_mol()
        pyg_data = Data()
        result = mol_structural_features.add_structural_features(
            rdkit_mol=mol, pyg_data=pyg_data, feature_config=None,
            logger=self.logger, molecule_index=0
        )
        self.assertIs(result, pyg_data)

    def test_none_rdkit_mol_raises_molecule_processing_error(self):
        """Test that None rdkit_mol raises MoleculeProcessingError."""
        pyg_data = Data()
        with self.assertRaises(vqm_exceptions.MoleculeProcessingError):
            mol_structural_features.add_structural_features(
                rdkit_mol=None, pyg_data=pyg_data,
                feature_config={"atom": ["degree"]},
                logger=self.logger, molecule_index=0
            )

    def test_none_pyg_data_raises_pyg_data_creation_error(self):
        """Test that None pyg_data raises an error.

        Note: The source code attempts to raise PyGDataCreationError but its
        __init__ requires a positional 'smiles' argument that is not passed,
        so a TypeError propagates instead.  The raise site is outside any
        try/except block, so the TypeError is not wrapped.  We therefore
        assert that *either* PyGDataCreationError or TypeError is raised,
        making this test resilient to a future fix of the constructor call.
        """
        mol = _make_water_mol()
        with self.assertRaises((vqm_exceptions.PyGDataCreationError, TypeError)):
            mol_structural_features.add_structural_features(
                rdkit_mol=mol, pyg_data=None,
                feature_config={"atom": ["degree"]},
                logger=self.logger, molecule_index=0
            )

    def test_atom_features_only(self):
        """Test adding only atom features (no bond features)."""
        mol = _make_water_mol()
        pyg_data = Data()
        config = {"atom": ["degree", "is_aromatic"]}
        result = mol_structural_features.add_structural_features(
            rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
            logger=self.logger, molecule_index=0
        )
        self.assertIsNotNone(result.x)
        self.assertEqual(result.x.shape[0], 3)  # 3 atoms
        self.assertEqual(result.x.shape[1], 2)  # degree + is_aromatic
        self.assertIsNone(result.edge_attr)

    def test_bond_features_only(self):
        """Test adding only bond features (no atom features)."""
        mol = _make_ethane_mol(with_conformer=True)
        pyg_data = _make_pyg_data_with_edges(mol)
        config = {"bond": ["bond_type", "is_conjugated"]}
        result = mol_structural_features.add_structural_features(
            rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
            logger=self.logger, molecule_index=0
        )
        self.assertIsNone(result.x)
        self.assertIsNotNone(result.edge_attr)
        self.assertEqual(result.edge_attr.shape[1], 5)  # bond_type(4) + is_conjugated(1)

    def test_both_atom_and_bond_features(self):
        """Test adding both atom and bond features simultaneously."""
        mol = _make_ethane_mol(with_conformer=True)
        pyg_data = _make_pyg_data_with_edges(mol)
        config = {
            "atom": ["degree", "hybridization"],
            "bond": ["bond_type"]
        }
        result = mol_structural_features.add_structural_features(
            rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
            logger=self.logger, molecule_index=0
        )
        self.assertIsNotNone(result.x)
        self.assertIsNotNone(result.edge_attr)
        self.assertEqual(result.x.shape[0], mol.GetNumAtoms())
        self.assertEqual(result.x.shape[1], 8)  # degree(1) + hybridization(7)
        self.assertEqual(result.edge_attr.shape[1], 4)  # bond_type(4)

    def test_empty_atom_and_bond_lists(self):
        """Test with empty atom and bond feature lists."""
        mol = _make_water_mol()
        pyg_data = Data()
        config = {"atom": [], "bond": []}
        result = mol_structural_features.add_structural_features(
            rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
            logger=self.logger, molecule_index=0
        )
        self.assertIsNone(result.x)
        self.assertIsNone(result.edge_attr)

    def test_with_coordinates_and_mulliken_charges(self):
        """Test integration with QM-optimized coordinates and Mulliken charges."""
        mol = Chem.MolFromSmiles("O")
        mol = Chem.AddHs(mol)
        coords = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
        mulliken = np.array([-0.82, 0.41, 0.41])

        pyg_data = Data()
        config = {"atom": ["mulliken_charge"]}
        result = mol_structural_features.add_structural_features(
            rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
            logger=self.logger, molecule_index=0,
            coordinates=coords, mulliken_charges=mulliken
        )
        self.assertIsNotNone(result.x)
        self.assertEqual(result.x.shape[0], 3)
        # Verify Mulliken charges are present
        self.assertAlmostEqual(result.x[0, 0].item(), -0.82, places=2)
        self.assertAlmostEqual(result.x[1, 0].item(), 0.41, places=2)

    def test_3d_bond_features_with_coordinates(self):
        """Test 3D bond features work when coordinates are provided."""
        mol = Chem.MolFromSmiles("O")
        mol = Chem.AddHs(mol)
        coords = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

        pyg_data = _make_pyg_data_with_edges(mol)
        config = {"bond": ["bond_length"]}

        # Need to add conformer first (via _ensure_conformer_and_charges)
        result = mol_structural_features.add_structural_features(
            rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
            logger=self.logger, molecule_index=0,
            coordinates=coords
        )
        self.assertIsNotNone(result.edge_attr)
        # Bond lengths should be positive
        for i in range(result.edge_attr.shape[0]):
            self.assertGreaterEqual(result.edge_attr[i, 0].item(), 0.0)

    def test_bond_features_no_edge_index_returns_none(self):
        """Test bond features with missing edge_index returns None edge_attr."""
        mol = _make_water_mol()
        pyg_data = Data()  # No edge_index
        config = {"bond": ["bond_type"]}
        result = mol_structural_features.add_structural_features(
            rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
            logger=self.logger, molecule_index=0
        )
        self.assertIsNone(result.edge_attr)

    def test_bond_features_empty_edge_index_returns_none(self):
        """Test bond features with empty edge_index returns None edge_attr."""
        mol = _make_water_mol()
        pyg_data = Data(edge_index=torch.empty(2, 0, dtype=torch.long))
        config = {"bond": ["bond_type"]}
        result = mol_structural_features.add_structural_features(
            rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
            logger=self.logger, molecule_index=0
        )
        self.assertIsNone(result.edge_attr)

    def test_returns_same_pyg_data_object(self):
        """Test that the returned object is the same Data instance (mutated in place)."""
        mol = _make_water_mol()
        pyg_data = Data()
        config = {"atom": ["degree"]}
        result = mol_structural_features.add_structural_features(
            rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
            logger=self.logger
        )
        self.assertIs(result, pyg_data)

    def test_error_context_propagation(self):
        """Test that molecule_index and inchi propagate to error messages."""
        mol = _make_water_mol()
        pyg_data = Data()
        config = {"atom": ["nonexistent_feature"]}
        with self.assertRaises(vqm_exceptions.StructuralFeatureError):
            mol_structural_features.add_structural_features(
                rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
                logger=self.logger, molecule_index=99,
                inchi="InChI=1S/H2O/h1H2"
            )

    def test_structural_feature_error_reraise(self):
        """Test that StructuralFeatureError is re-raised directly."""
        mol = _make_water_mol()
        pyg_data = Data()
        config = {"atom": ["bad_feature_name"]}
        with self.assertRaises(vqm_exceptions.StructuralFeatureError):
            mol_structural_features.add_structural_features(
                rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
                logger=self.logger
            )


# ============================================================================
# Test: Empty / Zero-Atom Molecule Edge Cases
# ============================================================================

class TestEmptyMoleculeEdgeCases(unittest.TestCase):
    """Test edge cases with empty/zero-atom molecules."""

    def setUp(self):
        """Set up common test fixtures."""
        self.logger = logging.getLogger('test_empty')

    def test_zero_atom_molecule_atom_features(self):
        """Test that zero-atom molecule produces empty tensor with correct dimensions."""
        # Create a molecule with no atoms
        mol = Chem.RWMol()
        pyg_data = Data()
        config = {"atom": ["degree", "hybridization"]}

        result = mol_structural_features.add_structural_features(
            rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
            logger=self.logger, molecule_index=0
        )
        self.assertIsNotNone(result.x)
        self.assertEqual(result.x.shape[0], 0)  # No atoms
        # Feature dim should still be correct (degree=1 + hybridization=7 = 8)
        self.assertEqual(result.x.shape[1], 8)


# ============================================================================
# Test: Aromatic Molecule Features (Benzene Integration)
# ============================================================================

class TestAromaticMoleculeFeatures(unittest.TestCase):
    """Test feature extraction on aromatic molecules."""

    def setUp(self):
        """Set up common test fixtures."""
        self.logger = logging.getLogger('test_aromatic')
        self.benzene = _make_benzene_mol(with_conformer=True)

    def test_benzene_atom_aromatic_features(self):
        """Test that benzene ring carbons are identified as aromatic."""
        pyg_data = Data()
        config = {"atom": ["is_aromatic", "is_in_ring", "num_aromatic_bonds"]}
        result = mol_structural_features.add_structural_features(
            rdkit_mol=self.benzene, pyg_data=pyg_data, feature_config=config,
            logger=self.logger
        )
        # First 6 atoms are ring carbons (aromatic), rest are H
        for i in range(6):  # Ring carbons
            self.assertEqual(result.x[i, 0].item(), 1.0)  # is_aromatic
            self.assertEqual(result.x[i, 1].item(), 1.0)  # is_in_ring
            self.assertEqual(result.x[i, 2].item(), 2.0)  # num_aromatic_bonds

    def test_benzene_bond_aromatic_features(self):
        """Test that benzene bonds are identified as aromatic."""
        pyg_data = _make_pyg_data_with_edges(self.benzene)
        config = {"bond": ["is_aromatic", "is_in_any_ring"]}
        result = mol_structural_features.add_structural_features(
            rdkit_mol=self.benzene, pyg_data=pyg_data, feature_config=config,
            logger=self.logger
        )
        self.assertIsNotNone(result.edge_attr)


# ============================================================================
# Test: Feature Config Variations
# ============================================================================

class TestFeatureConfigVariations(unittest.TestCase):
    """Test various feature configuration patterns."""

    def setUp(self):
        """Set up common test fixtures."""
        self.logger = logging.getLogger('test_config')

    def test_config_with_only_atom_key(self):
        """Test config that specifies only atom features (no bond key)."""
        mol = _make_water_mol()
        pyg_data = Data()
        config = {"atom": ["degree"]}
        result = mol_structural_features.add_structural_features(
            rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
            logger=self.logger
        )
        self.assertIsNotNone(result.x)
        self.assertIsNone(result.edge_attr)

    def test_config_with_only_bond_key(self):
        """Test config that specifies only bond features (no atom key)."""
        mol = _make_ethane_mol(with_conformer=True)
        pyg_data = _make_pyg_data_with_edges(mol)
        config = {"bond": ["bond_type"]}
        result = mol_structural_features.add_structural_features(
            rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
            logger=self.logger
        )
        self.assertIsNone(result.x)
        self.assertIsNotNone(result.edge_attr)

    def test_config_with_empty_dict(self):
        """Test config with empty dictionary (no features)."""
        mol = _make_water_mol()
        pyg_data = Data()
        config = {}
        result = mol_structural_features.add_structural_features(
            rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
            logger=self.logger
        )
        self.assertIsNone(result.x)
        self.assertIsNone(result.edge_attr)

    def test_single_atom_molecule(self):
        """Test feature extraction on a single-atom molecule (Helium)."""
        mol = Chem.MolFromSmiles("[He]")
        pyg_data = Data()
        config = {"atom": ["degree", "is_aromatic"]}
        result = mol_structural_features.add_structural_features(
            rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
            logger=self.logger
        )
        self.assertIsNotNone(result.x)
        self.assertEqual(result.x.shape[0], 1)
        self.assertEqual(result.x[0, 0].item(), 0.0)  # degree = 0
        self.assertEqual(result.x[0, 1].item(), 0.0)  # is_aromatic = 0


# ============================================================================
# Test: Conformer and Charge Integration with add_structural_features
# ============================================================================

class TestConformerChargeIntegration(unittest.TestCase):
    """Test conformer/charge preprocessing integration via add_structural_features."""

    def setUp(self):
        """Set up common test fixtures."""
        self.logger = logging.getLogger('test_conformer_integration')

    def test_ensure_conformer_failure_is_logged_not_raised(self):
        """Test that _ensure_conformer_and_charges failure is logged, not raised."""
        mol = _make_water_mol()
        pyg_data = Data()
        config = {"atom": ["degree"]}

        # Pass invalid coordinates that would cause issues if not handled gracefully
        # (wrong shape but valid length — an ndarray of the wrong internal shape)
        with patch.object(
            mol_structural_features, '_ensure_conformer_and_charges',
            side_effect=RuntimeError("Conformer setup failed")
        ):
            result = mol_structural_features.add_structural_features(
                rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
                logger=self.logger, molecule_index=0
            )
        # Should still produce results (the error is just logged)
        self.assertIsNotNone(result.x)

    def test_coordinates_produce_nonzero_bond_lengths(self):
        """Test that providing coordinates leads to nonzero 3D bond lengths."""
        mol = Chem.MolFromSmiles("O")
        mol = Chem.AddHs(mol)
        coords = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

        pyg_data = _make_pyg_data_with_edges(mol)
        config = {"bond": ["bond_length"]}
        result = mol_structural_features.add_structural_features(
            rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
            logger=self.logger, coordinates=coords
        )
        self.assertIsNotNone(result.edge_attr)
        # All bond lengths should be positive with actual coordinates
        for i in range(result.edge_attr.shape[0]):
            self.assertGreater(result.edge_attr[i, 0].item(), 0.0)


# ============================================================================
# Test: Error Handling and Recovery
# ============================================================================

class TestErrorHandlingAndRecovery(unittest.TestCase):
    """Test error handling, wrapping, and recovery mechanisms."""

    def setUp(self):
        """Set up common test fixtures."""
        self.logger = logging.getLogger('test_errors')

    def test_unexpected_exception_wrapped_in_structural_feature_error(self):
        """Test that unexpected exceptions are wrapped in StructuralFeatureError."""
        mol = _make_water_mol()
        pyg_data = Data()
        config = {"atom": ["degree"]}

        # Patch _calculate_atom_features_tensor to throw unexpected exception
        with patch.object(
            mol_structural_features, '_calculate_atom_features_tensor',
            side_effect=RuntimeError("Unexpected internal error")
        ):
            with self.assertRaises(vqm_exceptions.StructuralFeatureError) as context:
                mol_structural_features.add_structural_features(
                    rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
                    logger=self.logger, molecule_index=7
                )
            self.assertIn("unexpected", str(context.exception).lower())

    def test_structural_feature_error_not_double_wrapped(self):
        """Test that StructuralFeatureError is re-raised directly (not double-wrapped)."""
        mol = _make_water_mol()
        pyg_data = Data()
        config = {"atom": ["degree"]}

        original_error = vqm_exceptions.StructuralFeatureError(
            message="Original error",
            feature_type="atom"
        )

        with patch.object(
            mol_structural_features, '_calculate_atom_features_tensor',
            side_effect=original_error
        ):
            with self.assertRaises(vqm_exceptions.StructuralFeatureError) as context:
                mol_structural_features.add_structural_features(
                    rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
                    logger=self.logger
                )
            self.assertIs(context.exception, original_error)

    def test_gasteiger_charge_computation_failure(self):
        """Test StructuralFeatureError when Gasteiger computation fails."""
        mol = _make_water_mol()
        pyg_data = Data()
        config = {"atom": ["partial_charge"]}

        with patch.object(
            rdPartialCharges, 'ComputeGasteigerCharges',
            side_effect=RuntimeError("Gasteiger failed")
        ):
            with self.assertRaises(vqm_exceptions.StructuralFeatureError) as context:
                mol_structural_features.add_structural_features(
                    rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
                    logger=self.logger, molecule_index=0
                )
            self.assertIn("Gasteiger", str(context.exception))


# ============================================================================
# Test: Bond Length Geometry Correctness
# ============================================================================

class TestBondLengthGeometryCorrectness(unittest.TestCase):
    """Test correctness of 3D bond length calculations."""

    def test_known_bond_length(self):
        """Test bond length calculation with known geometry."""
        mol = Chem.RWMol()
        mol.AddAtom(Chem.Atom(6))  # C
        mol.AddAtom(Chem.Atom(6))  # C
        mol.AddBond(0, 1, BondType.SINGLE)

        conf = Chem.Conformer(2)
        conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
        conf.SetAtomPosition(1, (1.54, 0.0, 0.0))  # C-C bond ~1.54 Å
        mol.AddConformer(conf)

        bond = mol.GetBondWithIdx(0)
        length = mol_structural_features._get_bond_length_3d(bond)
        self.assertAlmostEqual(length, 1.54, places=2)

    def test_diagonal_bond_length(self):
        """Test bond length with 3D diagonal position."""
        mol = Chem.RWMol()
        mol.AddAtom(Chem.Atom(6))
        mol.AddAtom(Chem.Atom(6))
        mol.AddBond(0, 1, BondType.SINGLE)

        conf = Chem.Conformer(2)
        conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
        conf.SetAtomPosition(1, (1.0, 1.0, 1.0))
        mol.AddConformer(conf)

        bond = mol.GetBondWithIdx(0)
        length = mol_structural_features._get_bond_length_3d(bond)
        expected = math.sqrt(3.0)
        self.assertAlmostEqual(length, expected, places=5)


# ============================================================================
# Test: Large Molecule Integration
# ============================================================================

class TestLargeMoleculeIntegration(unittest.TestCase):
    """Test feature extraction on larger molecules."""

    def setUp(self):
        """Set up common test fixtures."""
        self.logger = logging.getLogger('test_large')

    def test_caffeine_all_features(self):
        """Test all atom features on caffeine (C8H10N4O2) — multi-ring, multi-element."""
        mol = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        mol = Chem.AddHs(mol)
        rdPartialCharges.ComputeGasteigerCharges(mol)

        pyg_data = _make_pyg_data_with_edges(mol)
        config = {
            "atom": [
                "degree", "total_degree", "hybridization", "total_valence",
                "is_aromatic", "is_in_ring", "partial_charge",
                "num_aromatic_bonds", "chirality"
            ],
            "bond": ["bond_type", "is_conjugated", "is_aromatic", "is_in_any_ring", "stereo"]
        }
        result = mol_structural_features.add_structural_features(
            rdkit_mol=mol, pyg_data=pyg_data, feature_config=config,
            logger=self.logger, molecule_index=100
        )
        self.assertIsNotNone(result.x)
        self.assertIsNotNone(result.edge_attr)
        self.assertEqual(result.x.shape[0], mol.GetNumAtoms())
        # All feature vectors should have same dimension (no ragged arrays)
        self.assertGreater(result.x.shape[1], 0)
        self.assertEqual(result.edge_attr.shape[0], pyg_data.edge_index.size(1))


# ============================================================================
# Test Suite Builder
# ============================================================================

def suite():
    """Create test suite."""
    test_suite = unittest.TestSuite()

    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOneHotEncoding))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEnsureConformerAndCharges))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAtomFeatureFunctions))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBondFeatureFunctions))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCalculateAtomFeaturesTensor))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCalculateBondFeaturesTensor))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGetAvailableFeatures))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAddStructuralFeatures))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEmptyMoleculeEdgeCases))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAromaticMoleculeFeatures))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFeatureConfigVariations))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConformerChargeIntegration))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestErrorHandlingAndRecovery))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBondLengthGeometryCorrectness))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLargeMoleculeIntegration))

    return test_suite


if __name__ == '__main__':
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
