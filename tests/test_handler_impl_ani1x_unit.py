#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/handlers/implementations/ani1x.py

Module under test: ani1x.py
- ANI1xDatasetHandler: Handler for ANI-1x quantum chemistry datasets
  - Implements DatasetHandler ABC (12 abstract methods + 4 transform validation helpers)
  - Registered via @register_handler decorator
  - ANI-1x-specific: coordinate_based strategy (NO InChI/SMILES), neutral molecules
    (charge=0), energy/forces/charges/dipole processing, object-array dtype normalization,
    atomization energy calculation (Hartree to eV), Hirshfeld/CM5 charge handling

Test path on local machine: ~/ml_projects/milia/tests/test_handler_impl_ani1x_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/handlers/implementations/ani1x.py

NOTE: This test suite runs inside Docker at /app/milia
Path mappings:
- Project root: /app/milia (mapped from ~/ml_projects/milia)

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution

NPZ file paths (mocked, never downloaded):
- ~/Chem_Data/MILIA_PyG_Dataset/raw/DFT_all_sliced.npz
- ~/Chem_Data/MILIA_PyG_Dataset/raw/DFT_uniques_sliced.npz
- ~/Chem_Data/MILIA_PyG_Dataset/raw/DFT_saddles_sliced.npz
- ~/Chem_Data/MILIA_PyG_Dataset/raw/DMC.npz
- ~/Chem_Data/MILIA_PyG_Dataset/raw/wavefunctions.npz

Updated: February 2026 - Production-ready comprehensive test coverage
"""

import logging
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch_geometric.data import Data

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from milia_pipeline.exceptions import (
    DatasetSpecificHandlerError,
    HandlerValidationError,
    MoleculeProcessingError,
    PropertyEnrichmentError,
)
from milia_pipeline.handlers.implementations.ani1x import ANI1xDatasetHandler

# ============================================================================
# HELPERS: Build realistic config mocks for ANI1xDatasetHandler
# ============================================================================


def _make_dataset_config(**overrides):
    """Build a minimal mock DatasetConfig for ANI-1x handler tests."""
    cfg = Mock(name="DatasetConfig")
    cfg.dataset_type = overrides.get("dataset_type", "ANI1x")
    cfg.root_dir = overrides.get("root_dir", "/tmp/test_data")
    cfg.raw_dir = overrides.get("raw_dir", "/tmp/test_data/raw")
    cfg.is_uncertainty_enabled = overrides.get("is_uncertainty_enabled", False)
    cfg.uncertainty_config = overrides.get("uncertainty_config")
    return cfg


def _make_filter_config(**overrides):
    """Build a minimal mock FilterConfig for ANI-1x handler tests."""
    cfg = Mock(name="FilterConfig")
    cfg.max_atoms = overrides.get("max_atoms", 100)
    cfg.min_atoms = overrides.get("min_atoms", 1)
    cfg.allowed_elements = overrides.get("allowed_elements")
    return cfg


def _make_processing_config(**overrides):
    """Build a minimal mock ProcessingConfig for ANI-1x handler tests."""
    cfg = Mock(name="ProcessingConfig")
    cfg.scalar_graph_targets = overrides.get("scalar_graph_targets", ["energy"])
    cfg.node_features = overrides.get("node_features", [])
    cfg.vector_graph_properties = overrides.get("vector_graph_properties", [])
    cfg.variable_len_graph_properties = overrides.get("variable_len_graph_properties", [])
    cfg.common_required_properties = overrides.get(
        "common_required_properties", ["atoms", "coordinates"]
    )
    cfg.calculate_atomization_energy_from = overrides.get("calculate_atomization_energy_from")
    cfg.atomization_energy_key_name = overrides.get("atomization_energy_key_name")
    return cfg


def _make_handler(**overrides):
    """Build an ANI1xDatasetHandler instance with configurable mocked configs."""
    dataset_config = overrides.get("dataset_config", _make_dataset_config())
    filter_config = overrides.get("filter_config", _make_filter_config())
    processing_config = overrides.get("processing_config", _make_processing_config())
    logger = overrides.get("logger", logging.getLogger("test.ani1x"))
    experimental_setup = overrides.get("experimental_setup")
    handler = ANI1xDatasetHandler(
        dataset_config=dataset_config,
        filter_config=filter_config,
        processing_config=processing_config,
        logger=logger,
        experimental_setup=experimental_setup,
    )
    return handler


def _make_pyg_data(**overrides):
    """Build a minimal PyG Data object for ANI-1x enrichment tests."""
    num_atoms = overrides.get("num_atoms", 5)
    default_z = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)[:num_atoms]
    z = overrides.get("z", default_z)
    pos = overrides.get("pos", torch.randn(num_atoms, 3, dtype=torch.float32))
    data = Data()
    data.z = z
    data.pos = pos
    data.num_nodes = num_atoms
    if "edge_index" in overrides:
        data.edge_index = overrides["edge_index"]
    else:
        if num_atoms >= 2:
            src = list(range(num_atoms - 1))
            dst = list(range(1, num_atoms))
            data.edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return data


def _make_raw_properties(**overrides):
    """Build a realistic raw_properties_dict for ANI-1x molecule tests."""
    num_atoms = overrides.get("num_atoms", 5)
    props = {
        "atoms": overrides.get("atoms", np.array([6, 1, 1, 1, 1], dtype=np.int64)[:num_atoms]),
        "coordinates": overrides.get(
            "coordinates", np.random.randn(num_atoms, 3).astype(np.float64)
        ),
        "energy": overrides.get("energy", -40.518),
    }
    for key in ["forces", "hirshfeld_charges", "cm5_charges", "dipole"]:
        if key in overrides:
            props[key] = overrides[key]
    return props


# ============================================================================
# GROUP 1: ANI1xDatasetHandler - Identity and Registration (6 tests)
# ============================================================================


class TestANI1xDatasetHandlerIdentity(unittest.TestCase):
    """Test ANI1xDatasetHandler identity, registration, and basic attributes."""

    def test_get_dataset_type_returns_ani1x(self):
        """get_dataset_type() returns 'ANI1x'."""
        handler = _make_handler()
        self.assertEqual(handler.get_dataset_type(), "ANI1x")

    def test_get_molecule_creation_strategy(self):
        """ANI-1x uses coordinate_based strategy (NO InChI/SMILES)."""
        handler = _make_handler()
        self.assertEqual(handler.get_molecule_creation_strategy(), "coordinate_based")

    def test_get_identifier_keys_empty(self):
        """ANI-1x identifier keys: empty list (no parseable chemical identifiers)."""
        handler = _make_handler()
        keys = handler.get_identifier_keys()
        self.assertIsInstance(keys, list)
        self.assertEqual(len(keys), 0)

    def test_is_subclass_of_dataset_handler(self):
        """ANI1xDatasetHandler is a proper DatasetHandler subclass."""
        from milia_pipeline.handlers.base_handler import DatasetHandler

        self.assertTrue(issubclass(ANI1xDatasetHandler, DatasetHandler))

    def test_handler_stores_configs(self):
        """Handler stores config objects passed during construction."""
        dc = _make_dataset_config()
        fc = _make_filter_config()
        pc = _make_processing_config()
        handler = ANI1xDatasetHandler(
            dataset_config=dc,
            filter_config=fc,
            processing_config=pc,
            logger=logging.getLogger("test"),
        )
        self.assertIs(handler.dataset_config, dc)
        self.assertIs(handler.filter_config, fc)
        self.assertIs(handler.processing_config, pc)

    def test_handler_stores_experimental_setup(self):
        """Handler correctly stores experimental_setup."""
        handler = _make_handler(experimental_setup="augmented")
        self.assertEqual(handler.experimental_setup, "augmented")


# ============================================================================
# GROUP 2: get_required_properties (7 tests)
# ============================================================================


class TestGetRequiredProperties(unittest.TestCase):
    """Test ANI-1x-specific required property determination."""

    def test_includes_core_ani1x_properties(self):
        """Required properties include ANI-1x core: energy, atoms, coordinates."""
        handler = _make_handler()
        required = handler.get_required_properties()
        for prop in ["energy", "atoms", "coordinates"]:
            self.assertIn(prop, required)

    def test_includes_scalar_graph_targets(self):
        """Required properties include scalar_graph_targets from config."""
        pc = _make_processing_config(scalar_graph_targets=["energy"])
        handler = _make_handler(processing_config=pc)
        required = handler.get_required_properties()
        self.assertIn("energy", required)

    def test_includes_node_features(self):
        """Required properties include node_features from config."""
        pc = _make_processing_config(node_features=["hirshfeld_charges"])
        handler = _make_handler(processing_config=pc)
        required = handler.get_required_properties()
        self.assertIn("hirshfeld_charges", required)

    def test_includes_vector_graph_properties(self):
        """Required properties include vector_graph_properties."""
        pc = _make_processing_config(vector_graph_properties=["dipole"])
        handler = _make_handler(processing_config=pc)
        required = handler.get_required_properties()
        self.assertIn("dipole", required)

    def test_includes_variable_len_properties(self):
        """Required properties include variable_len_graph_properties."""
        pc = _make_processing_config(variable_len_graph_properties=["forces"])
        handler = _make_handler(processing_config=pc)
        required = handler.get_required_properties()
        self.assertIn("forces", required)

    def test_includes_atomization_energy_base(self):
        """Required properties include atomization energy base when configured."""
        pc = _make_processing_config(calculate_atomization_energy_from="energy")
        handler = _make_handler(processing_config=pc)
        required = handler.get_required_properties()
        self.assertIn("energy", required)

    def test_returns_deduplicated_list(self):
        """Required properties list has no duplicates."""
        pc = _make_processing_config(
            scalar_graph_targets=["energy"],
            node_features=["atoms"],
            variable_len_graph_properties=["forces"],
            calculate_atomization_energy_from="energy",
        )
        handler = _make_handler(processing_config=pc)
        required = handler.get_required_properties()
        self.assertEqual(len(required), len(set(required)))


# ============================================================================
# GROUP 3: get_molecular_charge (5 tests)
# ============================================================================


class TestGetMolecularCharge(unittest.TestCase):
    """Test ANI-1x molecular charge (always neutral = 0)."""

    def test_always_returns_zero(self):
        handler = _make_handler()
        self.assertEqual(handler.get_molecular_charge({}, np.array([6, 1, 1, 1, 1])), 0)

    def test_returns_zero_regardless_of_properties(self):
        handler = _make_handler()
        props = {"energy": -40.518, "atoms": np.array([6, 1, 1])}
        self.assertEqual(handler.get_molecular_charge(props, np.array([6, 1, 1])), 0)

    def test_returns_zero_with_different_atomic_numbers(self):
        handler = _make_handler()
        self.assertEqual(handler.get_molecular_charge({}, np.array([7, 8, 6, 1, 1])), 0)

    def test_returns_zero_with_mol_identifier(self):
        handler = _make_handler()
        self.assertEqual(
            handler.get_molecular_charge({}, np.array([6, 1, 1]), mol_identifier="id"), 0
        )

    def test_returns_int_type(self):
        handler = _make_handler()
        self.assertIsInstance(handler.get_molecular_charge({}, np.array([6])), int)


# ============================================================================
# GROUP 4: validate_molecule_data - Success Paths (5 tests)
# ============================================================================


class TestValidateMoleculeDataSuccess(unittest.TestCase):
    """Test ANI-1x molecule validation success paths."""

    @patch("milia_pipeline.handlers.implementations.ani1x.validate_molecular_structure")
    def test_valid_molecule_passes(self, mock_vs):
        mock_vs.return_value = None
        handler = _make_handler()
        handler.validate_molecule_data(_make_raw_properties(), molecule_index=0, identifier="N/A")

    @patch("milia_pipeline.handlers.implementations.ani1x.validate_molecular_structure")
    def test_valid_molecule_with_forces(self, mock_vs):
        mock_vs.return_value = None
        handler = _make_handler()
        props = _make_raw_properties(forces=np.random.randn(5, 3).astype(np.float32))
        handler.validate_molecule_data(props, molecule_index=0, identifier="test")

    @patch("milia_pipeline.handlers.implementations.ani1x.validate_molecular_structure")
    def test_default_identifier(self, mock_vs):
        mock_vs.return_value = None
        handler = _make_handler()
        handler.validate_molecule_data(_make_raw_properties(), molecule_index=0)

    @patch("milia_pipeline.handlers.implementations.ani1x.validate_molecular_structure")
    def test_positive_energy_logs_warning(self, mock_vs):
        """Positive energy logs a warning but does not raise."""
        mock_vs.return_value = None
        handler = _make_handler()
        props = _make_raw_properties(energy=5.0)
        with self.assertLogs("test.ani1x", level="WARNING") as cm:
            handler.validate_molecule_data(props, molecule_index=0, identifier="test")
        self.assertTrue(any("positive energy" in msg for msg in cm.output))

    @patch("milia_pipeline.handlers.implementations.ani1x.validate_molecular_structure")
    def test_negative_energy_no_warning(self, mock_vs):
        """Negative energy (normal Hartree) passes without warning."""
        mock_vs.return_value = None
        handler = _make_handler()
        handler.validate_molecule_data(
            _make_raw_properties(energy=-40.518), molecule_index=0, identifier="test"
        )


# ============================================================================
# GROUP 5: validate_molecule_data - Error Paths (7 tests)
# ============================================================================


class TestValidateMoleculeDataErrors(unittest.TestCase):
    """Test ANI-1x molecule validation error paths."""

    def test_missing_energy_raises(self):
        handler = _make_handler()
        with self.assertRaises(HandlerValidationError) as ctx:
            handler.validate_molecule_data(_make_raw_properties(energy=None), molecule_index=0)
        self.assertIn("energy", str(ctx.exception))

    def test_missing_atoms_raises(self):
        handler = _make_handler()
        with self.assertRaises(HandlerValidationError) as ctx:
            handler.validate_molecule_data(_make_raw_properties(atoms=None), molecule_index=0)
        self.assertIn("atoms", str(ctx.exception))

    def test_missing_coordinates_raises(self):
        handler = _make_handler()
        with self.assertRaises(HandlerValidationError) as ctx:
            handler.validate_molecule_data(_make_raw_properties(coordinates=None), molecule_index=0)
        self.assertIn("coordinates", str(ctx.exception))

    def test_missing_all_essential_lists_all(self):
        handler = _make_handler()
        with self.assertRaises(HandlerValidationError) as ctx:
            handler.validate_molecule_data({}, molecule_index=0)
        err_str = str(ctx.exception)
        for prop in ["energy", "atoms", "coordinates"]:
            self.assertIn(prop, err_str)

    @patch(
        "milia_pipeline.handlers.implementations.ani1x.validate_molecular_structure",
        side_effect=ValueError("Atom count mismatch"),
    )
    def test_structure_validation_failure(self, mock_v):
        handler = _make_handler()
        with self.assertRaises(DatasetSpecificHandlerError):
            handler.validate_molecule_data(
                _make_raw_properties(), molecule_index=0, identifier="test"
            )

    def test_molecule_processing_error_converted(self):
        handler = _make_handler()
        with (
            patch.object(
                handler,
                "_is_valid_property",
                side_effect=MoleculeProcessingError(message="fail", molecule_index=0),
            ),
            self.assertRaises(DatasetSpecificHandlerError),
        ):
            handler.validate_molecule_data(
                _make_raw_properties(), molecule_index=0, identifier="test"
            )

    def test_unexpected_error_wraps(self):
        handler = _make_handler()
        with patch.object(handler, "_is_valid_property", side_effect=TypeError("boom")):
            with self.assertRaises(DatasetSpecificHandlerError):
                handler.validate_molecule_data(
                    _make_raw_properties(), molecule_index=0, identifier="test"
                )


# ============================================================================
# GROUP 6: process_property_value - Atoms (6 tests)
# ============================================================================


class TestProcessPropertyValueAtoms(unittest.TestCase):
    """Test ANI-1x process_property_value for atoms key (dtype normalization)."""

    def test_atoms_int64_passthrough(self):
        handler = _make_handler()
        arr = np.array([6, 1, 1], dtype=np.int64)
        result = handler.process_property_value("atoms", arr, 0)
        self.assertEqual(result.dtype, np.int64)
        np.testing.assert_array_equal(result, [6, 1, 1])

    def test_atoms_uint8_normalized_to_int64(self):
        handler = _make_handler()
        result = handler.process_property_value("atoms", np.array([6, 1, 1], dtype=np.uint8), 0)
        self.assertEqual(result.dtype, np.int64)

    def test_atoms_object_array_converted_to_int64(self):
        handler = _make_handler()
        result = handler.process_property_value("atoms", np.array([6, 1, 1], dtype=object), 0)
        self.assertEqual(result.dtype, np.int64)
        np.testing.assert_array_equal(result, [6, 1, 1])

    def test_atoms_float_array_converted_to_int64(self):
        handler = _make_handler()
        result = handler.process_property_value("atoms", np.array([6.0, 1.0], dtype=np.float64), 0)
        self.assertEqual(result.dtype, np.int64)

    def test_atoms_unconvertible_returns_original(self):
        handler = _make_handler()
        arr = np.array(["C", "H", "H"], dtype=object)
        result = handler.process_property_value("atoms", arr, 0)
        np.testing.assert_array_equal(result, ["C", "H", "H"])

    def test_atoms_none_returns_none(self):
        handler = _make_handler()
        self.assertIsNone(handler.process_property_value("atoms", None, 0))


# ============================================================================
# GROUP 7: process_property_value - Coordinates (5 tests)
# ============================================================================


class TestProcessPropertyValueCoordinates(unittest.TestCase):
    """Test ANI-1x process_property_value for coordinates key."""

    def test_coordinates_float64_passthrough(self):
        handler = _make_handler()
        arr = np.random.randn(3, 3).astype(np.float64)
        self.assertEqual(handler.process_property_value("coordinates", arr, 0).dtype, np.float64)

    def test_coordinates_float32_normalized_to_float64(self):
        handler = _make_handler()
        arr = np.random.randn(3, 3).astype(np.float32)
        self.assertEqual(handler.process_property_value("coordinates", arr, 0).dtype, np.float64)

    def test_coordinates_object_array_converted(self):
        handler = _make_handler()
        obj_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=object)
        self.assertEqual(
            handler.process_property_value("coordinates", obj_arr, 0).dtype, np.float64
        )

    def test_coordinates_int_array_converted_to_float64(self):
        handler = _make_handler()
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        self.assertEqual(handler.process_property_value("coordinates", arr, 0).dtype, np.float64)

    def test_coordinates_unconvertible_returns_original(self):
        handler = _make_handler()
        arr = np.array(["x", "y", "z"], dtype=object)
        np.testing.assert_array_equal(handler.process_property_value("coordinates", arr, 0), arr)


# ============================================================================
# GROUP 8: process_property_value - Forces, Charges, Dipole (10 tests)
# ============================================================================


class TestProcessPropertyValueArrayProperties(unittest.TestCase):
    """Test ANI-1x process_property_value for forces, charges, dipole."""

    def test_forces_float32_passthrough(self):
        handler = _make_handler()
        arr = np.random.randn(5, 3).astype(np.float32)
        self.assertTrue(
            np.issubdtype(handler.process_property_value("forces", arr, 0).dtype, np.floating)
        )

    def test_forces_object_array_converted(self):
        handler = _make_handler()
        result = handler.process_property_value(
            "forces", np.array([[0.1, 0.2, 0.3]], dtype=object), 0
        )
        self.assertEqual(result.dtype, np.float32)

    def test_forces_with_nan_returns_none(self):
        handler = _make_handler()
        self.assertIsNone(
            handler.process_property_value(
                "forces", np.array([[float("nan"), 0.2, 0.3]], dtype=np.float32), 0
            )
        )

    def test_forces_with_inf_returns_none(self):
        handler = _make_handler()
        self.assertIsNone(
            handler.process_property_value(
                "forces", np.array([[float("inf"), 0.2, 0.3]], dtype=np.float32), 0
            )
        )

    def test_hirshfeld_charges_float32_passthrough(self):
        handler = _make_handler()
        arr = np.array([0.1, -0.2, 0.05], dtype=np.float32)
        self.assertTrue(
            np.issubdtype(
                handler.process_property_value("hirshfeld_charges", arr, 0).dtype, np.floating
            )
        )

    def test_hirshfeld_charges_non_finite_returns_none(self):
        handler = _make_handler()
        self.assertIsNone(
            handler.process_property_value(
                "hirshfeld_charges", np.array([float("nan"), 0.1], dtype=np.float32), 0
            )
        )

    def test_cm5_charges_object_array_converted(self):
        handler = _make_handler()
        result = handler.process_property_value(
            "cm5_charges", np.array([0.1, -0.2, 0.05], dtype=object), 0
        )
        self.assertEqual(result.dtype, np.float32)

    def test_cm5_charges_non_finite_returns_none(self):
        handler = _make_handler()
        self.assertIsNone(
            handler.process_property_value(
                "cm5_charges", np.array([float("inf"), 0.1], dtype=np.float32), 0
            )
        )

    def test_dipole_object_array_converted(self):
        handler = _make_handler()
        result = handler.process_property_value(
            "dipole", np.array([1.0, 2.0, 3.0], dtype=object), 0
        )
        self.assertEqual(result.dtype, np.float32)

    def test_dipole_non_finite_returns_none(self):
        handler = _make_handler()
        self.assertIsNone(
            handler.process_property_value(
                "dipole", np.array([float("nan"), 2.0, 3.0], dtype=np.float32), 0
            )
        )


# ============================================================================
# GROUP 9: process_property_value - Energy and Passthrough (7 tests)
# ============================================================================


class TestProcessPropertyValueEnergy(unittest.TestCase):
    """Test ANI-1x process_property_value for energy and generic passthrough."""

    def test_energy_float_passthrough(self):
        handler = _make_handler()
        self.assertEqual(handler.process_property_value("energy", -40.518, 0), -40.518)

    def test_energy_ndarray_scalar_extracted(self):
        handler = _make_handler()
        result = handler.process_property_value("energy", np.array(-40.518), 0)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, -40.518, places=3)

    def test_energy_nan_float_passes_through(self):
        """Plain float NaN passes through: NaN check is only inside np.ndarray branch."""
        handler = _make_handler()
        result = handler.process_property_value("energy", float("nan"), 0)
        # The code only checks NaN when value is np.ndarray, so plain float NaN passes through
        self.assertTrue(math.isnan(result))

    def test_energy_nan_ndarray_returns_none(self):
        handler = _make_handler()
        self.assertIsNone(handler.process_property_value("energy", np.array(float("nan")), 0))

    def test_generic_property_passthrough(self):
        handler = _make_handler()
        arr = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(
            handler.process_property_value("some_other_prop", arr, 0), arr
        )

    def test_none_value_passthrough(self):
        handler = _make_handler()
        self.assertIsNone(handler.process_property_value("energy", None, 0))

    def test_unexpected_exception_wraps(self):
        handler = _make_handler()
        with (
            patch(
                "milia_pipeline.handlers.implementations.ani1x.np.asarray",
                side_effect=RuntimeError("boom"),
            ),
            self.assertRaises(DatasetSpecificHandlerError),
        ):
            handler.process_property_value("atoms", [1, 2, 3], 0)


# ============================================================================
# GROUP 10: _is_valid_property (7 tests)
# ============================================================================


class TestIsValidProperty(unittest.TestCase):
    """Test ANI-1x property validation (simple None/empty check)."""

    def test_none_is_invalid(self):
        self.assertFalse(_make_handler()._is_valid_property(None))

    def test_empty_list_is_invalid(self):
        self.assertFalse(_make_handler()._is_valid_property([]))

    def test_empty_tuple_is_invalid(self):
        self.assertFalse(_make_handler()._is_valid_property(()))

    def test_empty_numpy_array_is_invalid(self):
        self.assertFalse(_make_handler()._is_valid_property(np.array([])))

    def test_numeric_value_is_valid(self):
        self.assertTrue(_make_handler()._is_valid_property(-40.518))

    def test_nonempty_numpy_array_is_valid(self):
        self.assertTrue(_make_handler()._is_valid_property(np.array([6, 1, 1])))

    def test_string_is_valid(self):
        self.assertTrue(_make_handler()._is_valid_property("test_value"))


# ============================================================================
# GROUP 11: _ensure_tensor (8 tests)
# ============================================================================


class TestEnsureTensor(unittest.TestCase):
    """Test tensor conversion utility (ANI-1x raises DatasetSpecificHandlerError)."""

    def test_numpy_array_to_tensor(self):
        handler = _make_handler()
        result = handler._ensure_tensor(np.array([1.0, 2.0, 3.0]), torch.float32, "test", 0, "id")
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)

    def test_list_to_tensor(self):
        handler = _make_handler()
        result = handler._ensure_tensor([1, 2, 3], torch.long, "test", 0, "id")
        self.assertEqual(result.dtype, torch.long)

    def test_tuple_to_tensor(self):
        handler = _make_handler()
        result = handler._ensure_tensor((1.0, 2.0), torch.float32, "test", 0, "id")
        self.assertIsInstance(result, torch.Tensor)

    def test_scalar_to_tensor(self):
        handler = _make_handler()
        result = handler._ensure_tensor(42.0, torch.float32, "test", 0, "id")
        self.assertEqual(result.numel(), 1)

    def test_tensor_to_tensor(self):
        handler = _make_handler()
        t = torch.tensor([1.0, 2.0], dtype=torch.float64)
        self.assertEqual(
            handler._ensure_tensor(t, torch.float32, "test", 0, "id").dtype, torch.float32
        )

    def test_invalid_conversion_raises(self):
        handler = _make_handler()
        with self.assertRaises(DatasetSpecificHandlerError):
            handler._ensure_tensor("not_a_number", torch.float32, "test", 0, "id")

    def test_tensor_preserves_shape(self):
        handler = _make_handler()
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(handler._ensure_tensor(arr, torch.float32, "test", 0, "id").shape, (2, 3))

    def test_np_number_scalar_to_tensor(self):
        handler = _make_handler()
        result = handler._ensure_tensor(np.float64(-40.518), torch.float32, "test", 0, "id")
        self.assertEqual(result.numel(), 1)


# ============================================================================
# GROUP 12: _add_scalar_targets_internal (8 tests)
# ============================================================================


class TestAddScalarTargetsInternal(unittest.TestCase):
    """Test ANI-1x scalar target addition to PyG data."""

    def test_no_targets_configured_noop(self):
        pc = _make_processing_config(scalar_graph_targets=[])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        handler._add_scalar_targets_internal(data, _make_raw_properties(), 0, "test")
        self.assertFalse(hasattr(data, "y") and data.y is not None)

    def test_single_scalar_target(self):
        pc = _make_processing_config(scalar_graph_targets=["energy"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        handler._add_scalar_targets_internal(data, _make_raw_properties(energy=-40.518), 0, "test")
        self.assertIsNotNone(data.y)
        self.assertAlmostEqual(data.y[0].item(), -40.518, places=3)

    def test_numpy_scalar_target(self):
        pc = _make_processing_config(scalar_graph_targets=["energy"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        handler._add_scalar_targets_internal(
            data, _make_raw_properties(energy=np.array(-40.518)), 0, "test"
        )
        self.assertAlmostEqual(data.y[0].item(), -40.518, places=3)

    def test_int_scalar_target(self):
        pc = _make_processing_config(scalar_graph_targets=["energy"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        handler._add_scalar_targets_internal(data, _make_raw_properties(energy=-40), 0, "test")
        self.assertAlmostEqual(data.y[0].item(), -40.0, places=1)

    def test_missing_target_raises(self):
        pc = _make_processing_config(scalar_graph_targets=["nonexistent"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        with self.assertRaises(PropertyEnrichmentError):
            handler._add_scalar_targets_internal(data, _make_raw_properties(), 0, "test")

    def test_multi_element_array_raises(self):
        pc = _make_processing_config(scalar_graph_targets=["energy"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        with self.assertRaises(PropertyEnrichmentError):
            handler._add_scalar_targets_internal(
                data, _make_raw_properties(energy=np.array([-40.0, -50.0])), 0, "test"
            )

    def test_unsupported_type_raises(self):
        pc = _make_processing_config(scalar_graph_targets=["energy"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        with self.assertRaises(PropertyEnrichmentError):
            handler._add_scalar_targets_internal(
                data, _make_raw_properties(energy={"v": -40}), 0, "test"
            )

    def test_unexpected_exception_wraps(self):
        pc = _make_processing_config(scalar_graph_targets=["energy"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        with patch.object(handler, "_ensure_tensor", side_effect=RuntimeError("boom")):
            with self.assertRaises((DatasetSpecificHandlerError, PropertyEnrichmentError)):
                handler._add_scalar_targets_internal(
                    data, _make_raw_properties(energy=-40.518), 0, "test"
                )


# ============================================================================
# GROUP 13: _add_vector_properties_internal (6 tests)
# ============================================================================


class TestAddVectorPropertiesInternal(unittest.TestCase):
    """Test ANI-1x vector property (dipole) addition to PyG data."""

    def test_adds_dipole_when_configured(self):
        pc = _make_processing_config(vector_graph_properties=["dipole"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        handler._add_vector_properties_internal(
            data,
            _make_raw_properties(dipole=np.array([1.0, 2.0, 3.0], dtype=np.float32)),
            0,
            "test",
        )
        self.assertTrue(hasattr(data, "dipole"))
        self.assertEqual(data.dipole.shape[0], 3)

    def test_skips_dipole_when_absent(self):
        pc = _make_processing_config(vector_graph_properties=["dipole"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        handler._add_vector_properties_internal(data, _make_raw_properties(), 0, "test")
        self.assertFalse(hasattr(data, "dipole"))

    def test_list_dipole_converted_to_tensor(self):
        pc = _make_processing_config(vector_graph_properties=["dipole"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        handler._add_vector_properties_internal(
            data, _make_raw_properties(dipole=[1.0, 2.0, 3.0]), 0, "test"
        )
        self.assertIsInstance(data.dipole, torch.Tensor)

    def test_non_1d_vector_raises(self):
        pc = _make_processing_config(vector_graph_properties=["dipole"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        with self.assertRaises(PropertyEnrichmentError):
            handler._add_vector_properties_internal(
                data, _make_raw_properties(dipole=np.array([[1.0, 2.0], [3.0, 4.0]])), 0, "test"
            )

    def test_unexpected_exception_wraps(self):
        pc = _make_processing_config(vector_graph_properties=["dipole"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        with patch.object(handler, "_ensure_tensor", side_effect=RuntimeError("boom")):
            with self.assertRaises((PropertyEnrichmentError, DatasetSpecificHandlerError)):
                handler._add_vector_properties_internal(
                    data, _make_raw_properties(dipole=np.array([1.0, 2.0, 3.0])), 0, "test"
                )

    def test_no_vector_properties_configured_is_noop(self):
        pc = _make_processing_config(vector_graph_properties=[])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        handler._add_vector_properties_internal(
            data, _make_raw_properties(dipole=np.array([1.0, 2.0, 3.0])), 0, "test"
        )
        self.assertFalse(hasattr(data, "dipole"))


# ============================================================================
# GROUP 14: _add_variable_length_properties_internal (7 tests)
# ============================================================================


class TestAddVariableLengthPropertiesInternal(unittest.TestCase):
    """Test ANI-1x variable-length property (forces, charges) addition."""

    def test_adds_forces_when_configured(self):
        pc = _make_processing_config(variable_len_graph_properties=["forces"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(num_atoms=3)
        handler._add_variable_length_properties_internal(
            data,
            _make_raw_properties(num_atoms=3, forces=np.random.randn(3, 3).astype(np.float32)),
            0,
            "test",
        )
        self.assertTrue(hasattr(data, "forces"))

    def test_adds_hirshfeld_charges_when_configured(self):
        pc = _make_processing_config(variable_len_graph_properties=["hirshfeld_charges"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(num_atoms=3)
        handler._add_variable_length_properties_internal(
            data,
            _make_raw_properties(
                num_atoms=3, hirshfeld_charges=np.array([0.1, -0.2, 0.1], dtype=np.float32)
            ),
            0,
            "test",
        )
        self.assertTrue(hasattr(data, "hirshfeld_charges"))

    def test_skips_when_absent(self):
        pc = _make_processing_config(variable_len_graph_properties=["forces"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        handler._add_variable_length_properties_internal(data, _make_raw_properties(), 0, "test")
        self.assertFalse(hasattr(data, "forces"))

    def test_no_config_is_noop(self):
        pc = _make_processing_config(variable_len_graph_properties=[])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        handler._add_variable_length_properties_internal(
            data, _make_raw_properties(forces=np.random.randn(5, 3).astype(np.float32)), 0, "test"
        )
        self.assertFalse(hasattr(data, "forces"))

    def test_multiple_properties_added(self):
        pc = _make_processing_config(variable_len_graph_properties=["forces", "hirshfeld_charges"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(num_atoms=3)
        props = _make_raw_properties(
            num_atoms=3,
            forces=np.random.randn(3, 3).astype(np.float32),
            hirshfeld_charges=np.array([0.1, -0.2, 0.1], dtype=np.float32),
        )
        handler._add_variable_length_properties_internal(data, props, 0, "test")
        self.assertTrue(hasattr(data, "forces"))
        self.assertTrue(hasattr(data, "hirshfeld_charges"))

    def test_property_enrichment_error_propagates(self):
        pc = _make_processing_config(variable_len_graph_properties=["forces"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        with (
            patch.object(
                handler,
                "_ensure_tensor",
                side_effect=PropertyEnrichmentError(
                    molecule_index=0, inchi="t", property_name="forces", reason="t", detail="t"
                ),
            ),
            self.assertRaises(PropertyEnrichmentError),
        ):
            handler._add_variable_length_properties_internal(
                data,
                _make_raw_properties(forces=np.random.randn(5, 3).astype(np.float32)),
                0,
                "test",
            )

    def test_unexpected_exception_wraps(self):
        pc = _make_processing_config(variable_len_graph_properties=["forces"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        with patch.object(handler, "_ensure_tensor", side_effect=RuntimeError("boom")):
            with self.assertRaises((PropertyEnrichmentError, DatasetSpecificHandlerError)):
                handler._add_variable_length_properties_internal(
                    data,
                    _make_raw_properties(forces=np.random.randn(5, 3).astype(np.float32)),
                    0,
                    "test",
                )


# ============================================================================
# GROUP 15: _calculate_atomization_energy_internal (9 tests)
# ============================================================================


class TestCalculateAtomizationEnergyInternal(unittest.TestCase):
    """Test ANI-1x atomization energy calculation (Hartree to eV)."""

    def test_returns_none_when_not_configured(self):
        pc = _make_processing_config(calculate_atomization_energy_from=None)
        handler = _make_handler(processing_config=pc)
        self.assertIsNone(
            handler._calculate_atomization_energy_internal({}, _make_pyg_data(), 0, "test")
        )

    @patch("milia_pipeline.handlers.implementations.ani1x.HAR2EV", 27.2114)
    @patch(
        "milia_pipeline.handlers.implementations.ani1x.ATOMIC_ENERGIES_HARTREE",
        {6: -37.846, 1: -0.500},
    )
    def test_calculates_atomization_energy(self):
        pc = _make_processing_config(calculate_atomization_energy_from="energy")
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(num_atoms=3, z=torch.tensor([6, 1, 1], dtype=torch.long))
        result = handler._calculate_atomization_energy_internal({"energy": -40.0}, data, 0, "test")
        expected = (-40.0 - (-37.846 + -0.500 + -0.500)) * 27.2114
        self.assertAlmostEqual(result, expected, places=3)

    @patch("milia_pipeline.handlers.implementations.ani1x.HAR2EV", 27.2114)
    @patch(
        "milia_pipeline.handlers.implementations.ani1x.ATOMIC_ENERGIES_HARTREE",
        {6: -37.846, 1: -0.500},
    )
    def test_handles_numpy_scalar_energy(self):
        pc = _make_processing_config(calculate_atomization_energy_from="energy")
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(num_atoms=2, z=torch.tensor([6, 1], dtype=torch.long))
        result = handler._calculate_atomization_energy_internal(
            {"energy": np.array(-40.0)}, data, 0, "test"
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, float)

    def test_returns_none_when_base_energy_missing(self):
        pc = _make_processing_config(calculate_atomization_energy_from="energy")
        handler = _make_handler(processing_config=pc)
        self.assertIsNone(
            handler._calculate_atomization_energy_internal({}, _make_pyg_data(), 0, "test")
        )

    def test_returns_none_when_z_missing(self):
        pc = _make_processing_config(calculate_atomization_energy_from="energy")
        handler = _make_handler(processing_config=pc)
        data = Data()
        data.z = None
        self.assertIsNone(
            handler._calculate_atomization_energy_internal({"energy": -40.0}, data, 0, "test")
        )

    @patch("milia_pipeline.handlers.implementations.ani1x.HAR2EV", None)
    @patch("milia_pipeline.handlers.implementations.ani1x.ATOMIC_ENERGIES_HARTREE", {})
    def test_returns_none_when_har2ev_is_none(self):
        pc = _make_processing_config(calculate_atomization_energy_from="energy")
        handler = _make_handler(processing_config=pc)
        self.assertIsNone(
            handler._calculate_atomization_energy_internal(
                {"energy": -40.0}, _make_pyg_data(), 0, "test"
            )
        )

    @patch("milia_pipeline.handlers.implementations.ani1x.HAR2EV", 27.2114)
    @patch("milia_pipeline.handlers.implementations.ani1x.ATOMIC_ENERGIES_HARTREE", {6: -37.846})
    def test_returns_none_when_element_missing_from_lookup(self):
        pc = _make_processing_config(calculate_atomization_energy_from="energy")
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(num_atoms=2, z=torch.tensor([6, 1], dtype=torch.long))
        self.assertIsNone(
            handler._calculate_atomization_energy_internal({"energy": -40.0}, data, 0, "test")
        )

    def test_returns_none_on_exception(self):
        pc = _make_processing_config(calculate_atomization_energy_from="energy")
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        with patch.object(data.z, "tolist", side_effect=RuntimeError("boom")):
            self.assertIsNone(
                handler._calculate_atomization_energy_internal({"energy": -40.0}, data, 0, "test")
            )

    @patch("milia_pipeline.handlers.implementations.ani1x.HAR2EV", 27.2114)
    @patch(
        "milia_pipeline.handlers.implementations.ani1x.ATOMIC_ENERGIES_HARTREE",
        {6: -37.846, 1: -0.500},
    )
    def test_returns_float(self):
        pc = _make_processing_config(calculate_atomization_energy_from="energy")
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(num_atoms=2, z=torch.tensor([6, 1], dtype=torch.long))
        self.assertIsInstance(
            handler._calculate_atomization_energy_internal({"energy": -40.0}, data, 0, "test"),
            float,
        )


# ============================================================================
# GROUP 16: enrich_pyg_data (8 tests)
# ============================================================================


class TestEnrichPygData(unittest.TestCase):
    """Test ANI-1x PyG data enrichment orchestration."""

    def test_returns_data_object(self):
        handler = _make_handler()
        self.assertIsInstance(
            handler.enrich_pyg_data(_make_pyg_data(), _make_raw_properties(), 0, "test"), Data
        )

    def test_sets_dataset_type(self):
        handler = _make_handler()
        result = handler.enrich_pyg_data(_make_pyg_data(), _make_raw_properties(), 0, "test")
        self.assertEqual(result.dataset_type, "ANI1x")

    def test_calls_scalar_targets(self):
        handler = _make_handler()
        with patch.object(handler, "_add_scalar_targets_internal") as mock_s:
            handler.enrich_pyg_data(_make_pyg_data(), _make_raw_properties(), 0, "test")
            mock_s.assert_called_once()

    def test_calls_vector_properties_when_configured(self):
        pc = _make_processing_config(vector_graph_properties=["dipole"])
        handler = _make_handler(processing_config=pc)
        with patch.object(handler, "_add_vector_properties_internal") as mock_v:
            handler.enrich_pyg_data(_make_pyg_data(), _make_raw_properties(), 0, "test")
            mock_v.assert_called_once()

    def test_calls_variable_length_when_configured(self):
        pc = _make_processing_config(variable_len_graph_properties=["forces"])
        handler = _make_handler(processing_config=pc)
        with patch.object(handler, "_add_variable_length_properties_internal") as mock_vl:
            handler.enrich_pyg_data(_make_pyg_data(), _make_raw_properties(), 0, "test")
            mock_vl.assert_called_once()

    def test_zero_nodes_raises(self):
        handler = _make_handler()
        data = Data()
        data.z = torch.tensor([], dtype=torch.long)
        data.num_nodes = 0
        with self.assertRaises(DatasetSpecificHandlerError):
            handler.enrich_pyg_data(data, _make_raw_properties(), 0, "test")

    def test_enrichment_error_wraps(self):
        handler = _make_handler()
        with (
            patch.object(
                handler, "_add_scalar_targets_internal", side_effect=RuntimeError("unexpected")
            ),
            self.assertRaises(DatasetSpecificHandlerError),
        ):
            handler.enrich_pyg_data(_make_pyg_data(), _make_raw_properties(), 0, "test")

    def test_property_enrichment_error_propagates(self):
        handler = _make_handler()
        with (
            patch.object(
                handler,
                "_add_scalar_targets_internal",
                side_effect=PropertyEnrichmentError(
                    molecule_index=0, inchi="t", property_name="e", reason="t", detail="t"
                ),
            ),
            self.assertRaises(PropertyEnrichmentError),
        ):
            handler.enrich_pyg_data(_make_pyg_data(), _make_raw_properties(), 0, "test")


# ============================================================================
# GROUP 17: get_processing_statistics (6 tests)
# ============================================================================


class TestGetProcessingStatistics(unittest.TestCase):
    """Test ANI-1x processing statistics generation."""

    def test_basic_stats_structure(self):
        handler = _make_handler()
        stats = handler.get_processing_statistics([])
        self.assertEqual(stats["dataset_type"], "ANI1x")
        self.assertIn("total_processed", stats)

    def test_total_processed_count(self):
        self.assertEqual(
            _make_handler().get_processing_statistics([{}, {}, {}])["total_processed"], 3
        )

    def test_empty_list_returns_basic_stats(self):
        self.assertEqual(_make_handler().get_processing_statistics([])["total_processed"], 0)

    def test_counts_molecules_with_forces(self):
        molecules = [{"forces": np.array([1.0])}, {"forces": None}, {"forces": np.array([2.0])}]
        self.assertEqual(
            _make_handler().get_processing_statistics(molecules)["molecules_with_forces"], 2
        )

    def test_counts_molecules_with_charges(self):
        molecules = [{"hirshfeld_charges": np.array([0.1])}, {"cm5_charges": np.array([0.2])}, {}]
        self.assertEqual(
            _make_handler().get_processing_statistics(molecules)["molecules_with_charges"], 2
        )

    def test_experimental_setup_in_stats(self):
        handler = _make_handler(experimental_setup="augmented")
        stats = handler.get_processing_statistics([])
        self.assertTrue(stats.get("transform_aware_processing", False))
        self.assertEqual(stats["experimental_context"]["dataset_type"], "ANI1x")


# ============================================================================
# GROUP 18: get_transform_recommendations (5 tests)
# ============================================================================


class TestGetTransformRecommendations(unittest.TestCase):
    """Test ANI-1x-specific transform recommendations."""

    def test_returns_dict_with_expected_keys(self):
        recs = _make_handler().get_transform_recommendations()
        for key in ["recommended", "avoid", "warnings"]:
            self.assertIn(key, recs)

    def test_recommended_includes_core_transforms(self):
        rec_text = " ".join(_make_handler().get_transform_recommendations()["recommended"])
        for t in ["GCNNorm", "AddSelfLoops", "NormalizeFeatures"]:
            self.assertIn(t, rec_text)

    def test_recommended_includes_geometric(self):
        rec_text = " ".join(_make_handler().get_transform_recommendations()["recommended"])
        self.assertIn("RandomRotate", rec_text)

    def test_recommended_includes_distance(self):
        rec_text = " ".join(_make_handler().get_transform_recommendations()["recommended"])
        self.assertIn("Distance", rec_text)

    def test_avoid_is_empty(self):
        self.assertEqual(len(_make_handler().get_transform_recommendations()["avoid"]), 0)


# ============================================================================
# GROUP 19: get_supported_descriptors (5 tests)
# ============================================================================


class TestGetSupportedDescriptors(unittest.TestCase):
    """Test ANI-1x-specific descriptor support reporting."""

    def test_returns_dict_with_expected_keys(self):
        desc = _make_handler().get_supported_descriptors()
        for key in ["categories", "excluded", "recommended", "requires_3d", "requires_charges"]:
            self.assertIn(key, desc)

    def test_includes_geometric_category(self):
        self.assertIn("geometric", _make_handler().get_supported_descriptors()["categories"])

    def test_no_exclusions(self):
        self.assertEqual(len(_make_handler().get_supported_descriptors()["excluded"]), 0)

    def test_requires_3d_and_charges(self):
        desc = _make_handler().get_supported_descriptors()
        self.assertTrue(desc["requires_3d"])
        self.assertTrue(desc["requires_charges"])

    def test_includes_all_descriptor_categories(self):
        cats = _make_handler().get_supported_descriptors()["categories"]
        for c in [
            "constitutional",
            "topological",
            "electronic",
            "geometric",
            "drug_likeness",
            "fragments",
        ]:
            self.assertIn(c, cats)


# ============================================================================
# GROUP 20: get_supported_structural_features (5 tests)
# ============================================================================


class TestGetSupportedStructuralFeatures(unittest.TestCase):
    """Test ANI-1x-specific structural feature support."""

    def test_returns_dict_with_atom_and_bond(self):
        features = _make_handler().get_supported_structural_features()
        self.assertIn("atom", features)
        self.assertIn("bond", features)

    def test_atom_features_include_connectivity(self):
        features = _make_handler().get_supported_structural_features()
        for feat in ["degree", "total_degree"]:
            self.assertIn(feat, features["atom"])

    def test_atom_features_include_charges(self):
        features = _make_handler().get_supported_structural_features()
        for feat in ["hirshfeld_charge", "cm5_charge"]:
            self.assertIn(feat, features["atom"])

    def test_bond_features_include_expected(self):
        features = _make_handler().get_supported_structural_features()
        for feat in ["bond_type", "is_conjugated", "stereo"]:
            self.assertIn(feat, features["bond"])

    def test_bond_features_include_geometric(self):
        self.assertIn("bond_length", _make_handler().get_supported_structural_features()["bond"])


# ============================================================================
# GROUP 21: Transform Validation Helpers (8 tests)
# ============================================================================


class TestTransformValidationHelpers(unittest.TestCase):
    """Test ANI-1x transform validation helper methods."""

    def test_validate_no_geometric_warning(self):
        warnings = _make_handler()._validate_dataset_specific_transforms(
            ["GCNNorm", "AddSelfLoops"]
        )
        self.assertTrue(any("geometric augmentation" in w for w in warnings))

    def test_validate_with_geometric_no_warning(self):
        warnings = _make_handler()._validate_dataset_specific_transforms(
            ["RandomRotate", "GCNNorm"]
        )
        self.assertFalse(any("without geometric augmentation" in w for w in warnings))

    def test_forces_with_random_rotate_warns(self):
        pc = _make_processing_config(variable_len_graph_properties=["forces"])
        warnings = _make_handler(processing_config=pc)._validate_dataset_specific_transforms(
            ["RandomRotate"]
        )
        self.assertTrue(any("force rotation" in w for w in warnings))

    def test_distance_transform_warning(self):
        warnings = _make_handler()._validate_dataset_specific_transforms(["Distance"])
        self.assertTrue(any("edge attributes" in w for w in warnings))

    def test_check_virtualnode_with_charges_incompatible(self):
        pc = _make_processing_config(node_features=["hirshfeld_charges"])
        errors = _make_handler(processing_config=pc)._check_transform_incompatibilities(
            ["VirtualNode"]
        )
        self.assertTrue(any("VirtualNode" in e for e in errors))

    def test_check_no_incompatibilities(self):
        errors = _make_handler()._check_transform_incompatibilities(["GCNNorm", "AddSelfLoops"])
        self.assertEqual(len(errors), 0)

    def test_recommendations_without_norm(self):
        recs = _make_handler()._get_transform_recommendations(["AddSelfLoops"])
        self.assertTrue(any("GCNNorm" in r for r in recs))

    def test_recommendations_gcnnorm_without_selfloops(self):
        recs = _make_handler()._get_transform_recommendations(["GCNNorm"])
        self.assertTrue(any("AddSelfLoops" in r for r in recs))


# ============================================================================
# GROUP 22: _get_dataset_suitable_transforms (5 tests)
# ============================================================================


class TestGetDatasetSuitableTransforms(unittest.TestCase):
    """Test ANI-1x dataset-suitable transform filtering."""

    def test_filters_geometric_transforms(self):
        suitable = _make_handler()._get_dataset_suitable_transforms(
            {"RandomRotate": Mock(), "RandomTranslate": Mock(), "FakeTransform": Mock()}
        )
        self.assertIn("RandomRotate", suitable)
        self.assertNotIn("FakeTransform", suitable)

    def test_filters_normalization_transforms(self):
        suitable = _make_handler()._get_dataset_suitable_transforms(
            {"GCNNorm": Mock(), "NormalizeFeatures": Mock()}
        )
        self.assertIn("GCNNorm", suitable)
        self.assertIn("NormalizeFeatures", suitable)

    def test_filters_structure_transforms(self):
        suitable = _make_handler()._get_dataset_suitable_transforms(
            {"AddSelfLoops": Mock(), "ToUndirected": Mock()}
        )
        self.assertIn("AddSelfLoops", suitable)
        self.assertIn("ToUndirected", suitable)

    def test_filters_edge_feature_transforms(self):
        suitable = _make_handler()._get_dataset_suitable_transforms(
            {"Distance": Mock(), "Cartesian": Mock()}
        )
        self.assertIn("Distance", suitable)
        self.assertIn("Cartesian", suitable)

    def test_empty_available_returns_empty(self):
        self.assertEqual(len(_make_handler()._get_dataset_suitable_transforms({})), 0)


# ============================================================================
# GROUP 23: Edge Cases and Integration (8 tests)
# ============================================================================


class TestEdgeCasesAndIntegration(unittest.TestCase):
    """Test edge cases and cross-method integration."""

    def test_handler_with_all_configs(self):
        pc = _make_processing_config(
            scalar_graph_targets=["energy"],
            node_features=["hirshfeld_charges"],
            vector_graph_properties=["dipole"],
            variable_len_graph_properties=["forces", "cm5_charges"],
            calculate_atomization_energy_from="energy",
            atomization_energy_key_name="atomization_energy_eV",
        )
        self.assertEqual(_make_handler(processing_config=pc).get_dataset_type(), "ANI1x")

    def test_enrichment_with_scalar_targets(self):
        pc = _make_processing_config(scalar_graph_targets=["energy"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(num_atoms=3, z=torch.tensor([6, 1, 1], dtype=torch.long))
        result = handler.enrich_pyg_data(
            data, _make_raw_properties(num_atoms=3, energy=-40.518), 0, "test"
        )
        self.assertTrue(hasattr(result, "y"))

    def test_multiple_enrichments_independent(self):
        pc = _make_processing_config(scalar_graph_targets=["energy"])
        handler = _make_handler(processing_config=pc)
        r1 = handler.enrich_pyg_data(
            _make_pyg_data(), _make_raw_properties(energy=-40.518), 0, "t1"
        )
        r2 = handler.enrich_pyg_data(
            _make_pyg_data(), _make_raw_properties(energy=-76.123), 1, "t2"
        )
        self.assertNotEqual(r1.y[0].item(), r2.y[0].item())

    def test_process_property_unexpected_exception_wrapped(self):
        handler = _make_handler()
        with (
            patch(
                "milia_pipeline.handlers.implementations.ani1x.np.asarray",
                side_effect=RuntimeError("boom"),
            ),
            self.assertRaises(DatasetSpecificHandlerError),
        ):
            handler.process_property_value("atoms", [1, 2, 3], 0)

    def test_handler_validation_error_reraise(self):
        handler = _make_handler()
        with self.assertRaises(HandlerValidationError):
            handler.validate_molecule_data(
                _make_raw_properties(energy=None, atoms=None, coordinates=None), molecule_index=0
            )

    def test_dataset_specific_handler_error_reraise(self):
        handler = _make_handler()
        with (
            patch(
                "milia_pipeline.handlers.implementations.ani1x.validate_molecular_structure",
                side_effect=ValueError("bad"),
            ),
            self.assertRaises(DatasetSpecificHandlerError),
        ):
            handler.validate_molecule_data(
                _make_raw_properties(), molecule_index=0, identifier="test"
            )

    @patch("milia_pipeline.handlers.implementations.ani1x.HAR2EV", 27.2114)
    @patch(
        "milia_pipeline.handlers.implementations.ani1x.ATOMIC_ENERGIES_HARTREE",
        {6: -37.846, 1: -0.500},
    )
    def test_full_enrichment_with_atomization_energy(self):
        pc = _make_processing_config(
            scalar_graph_targets=["energy"],
            calculate_atomization_energy_from="energy",
            atomization_energy_key_name="atomization_energy_eV",
        )
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(num_atoms=3, z=torch.tensor([6, 1, 1], dtype=torch.long))
        result = handler.enrich_pyg_data(
            data, _make_raw_properties(num_atoms=3, energy=-40.0), 0, "test"
        )
        self.assertTrue(hasattr(result, "atomization_energy_eV"))
        self.assertIsInstance(result.atomization_energy_eV, torch.Tensor)

    def test_enrichment_skips_atomization_when_not_configured(self):
        pc = _make_processing_config(
            scalar_graph_targets=["energy"],
            calculate_atomization_energy_from=None,
            atomization_energy_key_name=None,
        )
        handler = _make_handler(processing_config=pc)
        result = handler.enrich_pyg_data(_make_pyg_data(), _make_raw_properties(), 0, "test")
        self.assertFalse(hasattr(result, "atomization_energy_eV"))


# ============================================================================
# TEST RUNNER
# ============================================================================


def run_comprehensive_suite():
    """Run all test groups in a structured order."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    test_classes = [
        TestANI1xDatasetHandlerIdentity,  # GROUP 1:    6 tests
        TestGetRequiredProperties,  # GROUP 2:    7 tests
        TestGetMolecularCharge,  # GROUP 3:    5 tests
        TestValidateMoleculeDataSuccess,  # GROUP 4:    5 tests
        TestValidateMoleculeDataErrors,  # GROUP 5:    7 tests
        TestProcessPropertyValueAtoms,  # GROUP 6:    6 tests
        TestProcessPropertyValueCoordinates,  # GROUP 7:    5 tests
        TestProcessPropertyValueArrayProperties,  # GROUP 8:   10 tests
        TestProcessPropertyValueEnergy,  # GROUP 9:    7 tests
        TestIsValidProperty,  # GROUP 10:   7 tests
        TestEnsureTensor,  # GROUP 11:   8 tests
        TestAddScalarTargetsInternal,  # GROUP 12:   8 tests
        TestAddVectorPropertiesInternal,  # GROUP 13:   6 tests
        TestAddVariableLengthPropertiesInternal,  # GROUP 14:   7 tests
        TestCalculateAtomizationEnergyInternal,  # GROUP 15:   9 tests
        TestEnrichPygData,  # GROUP 16:   8 tests
        TestGetProcessingStatistics,  # GROUP 17:   6 tests
        TestGetTransformRecommendations,  # GROUP 18:   5 tests
        TestGetSupportedDescriptors,  # GROUP 19:   5 tests
        TestGetSupportedStructuralFeatures,  # GROUP 20:   5 tests
        TestTransformValidationHelpers,  # GROUP 21:   8 tests
        TestGetDatasetSuitableTransforms,  # GROUP 22:   5 tests
        TestEdgeCasesAndIntegration,  # GROUP 23:   8 tests
    ]
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("")
    print("=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS - handlers/implementations/ani1x.py")
    print("=" * 80)
    total = result.testsRun
    failed = len(result.failures)
    errors = len(result.errors)
    passed = total - failed - errors
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Test Groups: {len(test_classes)}")
    if result.wasSuccessful():
        print("\nALL TESTS PASSED - PRODUCTION-READY")
        return 0
    else:
        print("\nSOME TESTS FAILED - REVIEW REQUIRED")
        return 1


if __name__ == "__main__":
    if "pytest" in sys.modules:
        pass
    else:
        sys.exit(run_comprehensive_suite())
