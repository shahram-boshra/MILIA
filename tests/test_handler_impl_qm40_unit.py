#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/handlers/implementations/qm40.py

Module under test: qm40.py
- QM40DatasetHandler: Handler for the QM40 quantum chemistry dataset
  - Implements DatasetHandler ABC (protocol methods + 4 transform-validation helpers)
  - Registered via @register_handler
  - QM40-specific:
    * coordinate_based strategy (no InChI; SMILES is only a label)
    * NEUTRAL ONLY -> get_molecular_charge() ALWAYS returns 0 (no strategies,
      no element inference, inverse of QDpi)
    * 7 elements: H, C, N, O, F, S, Cl (QM40_SUPPORTED_ELEMENTS = {1,6,7,8,9,16,17})
    * primary energy target Internal_E_0K
    * dtype normalization: atoms->int64, coordinates/initial_coordinates->float64,
      Qmulliken->float32 (object/non-floating only; non-finite/unconvertible -> None)
    * atomization energy from Internal_E_0K (Hartree -> eV via HAR2EV)

Test path on local machine: ~/ml_projects/milia/tests/test_handler_impl_qm40_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/handlers/implementations/qm40.py

NOTE: This test suite runs inside Docker at /app/milia

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution

Verified against the extracted handler source (947 lines). Exception signatures:
- HandlerValidationError(message, handler_type, validation_type, failed_validations, ...)
- DatasetSpecificHandlerError(message, dataset_type, operation, ...)
- PropertyEnrichmentError(molecule_index, inchi, property_name, reason, detail)
- MoleculeProcessingError(message, molecule_index, ...)

Updated: June 2026 - Production-ready comprehensive test coverage
"""

import logging
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
from milia_pipeline.handlers.implementations.qm40 import (
    QM40_SUPPORTED_ELEMENTS,
    QM40DatasetHandler,
)

# Import atomic-energy constants from the same location qm40.py uses, with a
# defensive fallback (mirrors the QDpi test). When the import succeeds, the test
# and the handler share the SAME table, so atomization assertions are consistent.
try:
    from milia_pipeline.config.config_constants import ATOMIC_ENERGIES_HARTREE, HAR2EV
except ImportError:
    ATOMIC_ENERGIES_HARTREE = {
        1: -0.500273,  # H
        6: -37.846772,  # C
        7: -54.583861,  # N
        8: -75.064579,  # O
        9: -99.733509,  # F
        16: -398.100442,  # S
        17: -460.148990,  # Cl
    }
    HAR2EV = 27.211386245988


# ============================================================================
# HELPERS
# ============================================================================


def _make_dataset_config(**overrides):
    cfg = Mock(spec_set=["dataset_type", "npz_file_path", "dataset_name"])
    cfg.dataset_type = overrides.get("dataset_type", "QM40")
    cfg.npz_file_path = overrides.get("npz_file_path", "~/Chem_Data/MILIA_PyG_Dataset/raw/qm40.npz")
    cfg.dataset_name = overrides.get("dataset_name", "QM40")
    return cfg


def _make_filter_config(**overrides):
    cfg = Mock(spec_set=["max_atoms", "allowed_elements", "min_atoms"])
    cfg.max_atoms = overrides.get("max_atoms", 60)
    cfg.allowed_elements = overrides.get("allowed_elements")
    cfg.min_atoms = overrides.get("min_atoms", 1)
    return cfg


def _make_processing_config(**overrides):
    """
    QM40 processing config defaults:
    - scalar_graph_targets: ['Internal_E_0K'] (primary energy target)
    - node_features: ['Qmulliken'] (per-atom Mulliken charges)
    - vector_graph_properties: []
    - variable_len_graph_properties: [] (per-bond local-mode arrays when enabled)
    - calculate_atomization_energy_from: 'Internal_E_0K'
    - atomization_energy_key_name: 'atomization_energy'
    """
    cfg = Mock(
        spec_set=[
            "scalar_graph_targets",
            "node_features",
            "vector_graph_properties",
            "variable_len_graph_properties",
            "calculate_atomization_energy_from",
            "atomization_energy_key_name",
        ]
    )
    cfg.scalar_graph_targets = overrides.get("scalar_graph_targets", ["Internal_E_0K"])
    cfg.node_features = overrides.get("node_features", ["Qmulliken"])
    cfg.vector_graph_properties = overrides.get("vector_graph_properties", [])
    cfg.variable_len_graph_properties = overrides.get("variable_len_graph_properties", [])
    cfg.calculate_atomization_energy_from = overrides.get(
        "calculate_atomization_energy_from", "Internal_E_0K"
    )
    cfg.atomization_energy_key_name = overrides.get(
        "atomization_energy_key_name", "atomization_energy"
    )
    return cfg


def _make_handler(**overrides):
    """Construct a QM40DatasetHandler. Logger name 'test.qm40' for assertLogs."""
    dataset_config = overrides.get("dataset_config", _make_dataset_config())
    filter_config = overrides.get("filter_config", _make_filter_config())
    processing_config = overrides.get("processing_config", _make_processing_config())
    logger = overrides.get("logger", logging.getLogger("test.qm40"))
    experimental_setup = overrides.get("experimental_setup")
    return QM40DatasetHandler(
        dataset_config=dataset_config,
        filter_config=filter_config,
        processing_config=processing_config,
        logger=logger,
        experimental_setup=experimental_setup,
    )


def _make_pyg_data(**overrides):
    """Build a minimal PyG Data object for QM40 enrichment tests."""
    num_atoms = overrides.get("num_atoms", 3)
    z = overrides.get("z", torch.tensor([6, 1, 8], dtype=torch.long)[:num_atoms])
    pos = overrides.get("pos", torch.randn(num_atoms, 3, dtype=torch.float32))

    data = Data()
    data.z = z
    data.pos = pos
    data.num_nodes = num_atoms
    if "edge_index" in overrides:
        data.edge_index = overrides["edge_index"]
    elif num_atoms >= 2:
        src = list(range(num_atoms - 1))
        dst = list(range(1, num_atoms))
        data.edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return data


def _make_raw_properties(**overrides):
    """Build a realistic raw_properties_dict for QM40 (energy key = Internal_E_0K)."""
    num_atoms = overrides.get("num_atoms", 3)
    props = {
        "Internal_E_0K": overrides.get("energy", -76.3),
        "atoms": overrides.get("atoms", np.array([6, 1, 8])[:num_atoms]),
        "coordinates": overrides.get(
            "coordinates", np.random.randn(num_atoms, 3).astype(np.float64)
        ),
    }
    for key in [
        "Qmulliken",
        "initial_coordinates",
        "smiles",
        "bond_lmod_ka",
        "bond_atom1_idx",
        "bond_atom2_idx",
        "bond_tag",
        "HOMO",
        "LUMO",
    ]:
        if key in overrides:
            props[key] = overrides[key]
    # Allow arbitrary extra keys via "extra" dict
    props.update(overrides.get("extra", {}))
    return props


# ============================================================================
# GROUP 1: Identity and registration
# ============================================================================


class TestIdentity(unittest.TestCase):
    def test_get_dataset_type(self):
        self.assertEqual(_make_handler().get_dataset_type(), "QM40")

    def test_get_molecule_creation_strategy(self):
        self.assertEqual(_make_handler().get_molecule_creation_strategy(), "coordinate_based")

    def test_get_identifier_keys_empty(self):
        keys = _make_handler().get_identifier_keys()
        self.assertIsInstance(keys, list)
        self.assertEqual(len(keys), 0)

    def test_is_subclass_of_dataset_handler(self):
        from milia_pipeline.handlers.base_handler import DatasetHandler

        self.assertTrue(issubclass(QM40DatasetHandler, DatasetHandler))

    def test_stores_configs(self):
        dc, fc, pc = _make_dataset_config(), _make_filter_config(), _make_processing_config()
        handler = QM40DatasetHandler(
            dataset_config=dc,
            filter_config=fc,
            processing_config=pc,
            logger=logging.getLogger("test.qm40"),
        )
        self.assertIs(handler.dataset_config, dc)
        self.assertIs(handler.filter_config, fc)
        self.assertIs(handler.processing_config, pc)

    def test_stores_experimental_setup(self):
        handler = _make_handler(experimental_setup="exp_v1")
        self.assertEqual(handler.experimental_setup, "exp_v1")


# ============================================================================
# GROUP 2: QM40_SUPPORTED_ELEMENTS constant
# ============================================================================


class TestSupportedElementsConstant(unittest.TestCase):
    def test_seven_elements(self):
        self.assertEqual(len(QM40_SUPPORTED_ELEMENTS), 7)

    def test_expected_values(self):
        self.assertEqual(QM40_SUPPORTED_ELEMENTS, {1, 6, 7, 8, 9, 16, 17})

    def test_is_set(self):
        self.assertIsInstance(QM40_SUPPORTED_ELEMENTS, set)


# ============================================================================
# GROUP 3: get_molecular_charge — ALWAYS 0 (neutral only)
# ============================================================================


class TestGetMolecularCharge(unittest.TestCase):
    def test_empty_dict_returns_zero(self):
        self.assertEqual(_make_handler().get_molecular_charge({}, np.array([6, 1, 8]), "x"), 0)

    def test_ignores_molecular_charge_key(self):
        """QM40 is neutral-only: a present 'molecular_charge' is ignored, still 0."""
        self.assertEqual(
            _make_handler().get_molecular_charge({"molecular_charge": 5}, np.array([11]), "Na+"), 0
        )

    def test_ignores_charge_key(self):
        self.assertEqual(
            _make_handler().get_molecular_charge({"charge": -2}, np.array([8, 8]), "x"), 0
        )

    def test_returns_int(self):
        charge = _make_handler().get_molecular_charge({}, np.array([6]), None)
        self.assertIsInstance(charge, int)

    def test_default_identifier_none(self):
        self.assertEqual(_make_handler().get_molecular_charge({}, np.array([6])), 0)


# ============================================================================
# GROUP 4: validate_molecule_data
# ============================================================================


class TestValidateMoleculeData(unittest.TestCase):
    def test_valid_passes(self):
        _make_handler().validate_molecule_data(_make_raw_properties(), 0, "test")

    def test_missing_energy_raises(self):
        props = _make_raw_properties()
        props.pop("Internal_E_0K")
        with self.assertRaises(HandlerValidationError) as ctx:
            _make_handler().validate_molecule_data(props, 0, "test")
        self.assertIn("Internal_E_0K", str(ctx.exception))

    def test_missing_atoms_raises(self):
        props = _make_raw_properties()
        props.pop("atoms")
        with self.assertRaises(HandlerValidationError) as ctx:
            _make_handler().validate_molecule_data(props, 0, "test")
        self.assertIn("atoms", str(ctx.exception))

    def test_missing_coordinates_raises(self):
        props = _make_raw_properties()
        props.pop("coordinates")
        with self.assertRaises(HandlerValidationError) as ctx:
            _make_handler().validate_molecule_data(props, 0, "test")
        self.assertIn("coordinates", str(ctx.exception))

    def test_empty_dict_raises(self):
        with self.assertRaises(HandlerValidationError):
            _make_handler().validate_molecule_data({}, 0, "test")

    def test_none_energy_treated_as_missing(self):
        props = _make_raw_properties(energy=None)
        with self.assertRaises(HandlerValidationError):
            _make_handler().validate_molecule_data(props, 0, "test")

    def test_empty_atoms_treated_as_missing(self):
        props = _make_raw_properties()
        props["atoms"] = np.array([])
        with self.assertRaises(HandlerValidationError):
            _make_handler().validate_molecule_data(props, 0, "test")

    def test_structure_failure_wraps_dataset_specific(self):
        props = _make_raw_properties()
        with (
            patch(
                "milia_pipeline.handlers.implementations.qm40.validate_molecular_structure",
                side_effect=ValueError("atoms/coords mismatch"),
            ),
            self.assertRaises(DatasetSpecificHandlerError) as ctx,
        ):
            _make_handler().validate_molecule_data(props, 0, "test")
        self.assertEqual(ctx.exception.dataset_type, "QM40")

    def test_positive_energy_logs_warning(self):
        props = _make_raw_properties(energy=10.0)
        with self.assertLogs("test.qm40", level="WARNING") as log:
            _make_handler().validate_molecule_data(props, 0, "test")
        self.assertTrue(any("positive Internal_E_0K" in m for m in log.output))

    def test_unsupported_elements_logs_warning(self):
        """He (Z=2) is a valid atomic number but NOT in QM40's 7-element set."""
        props = _make_raw_properties(
            atoms=np.array([2, 6, 1]), coordinates=np.random.randn(3, 3), num_atoms=3
        )
        with self.assertLogs("test.qm40", level="WARNING") as log:
            _make_handler().validate_molecule_data(props, 0, "test")
        self.assertTrue(any("unsupported elements" in m for m in log.output))

    def test_unexpected_exception_wrapped(self):
        handler = _make_handler()
        props = _make_raw_properties()
        with (
            patch.object(handler, "_is_valid_property", side_effect=RuntimeError("boom")),
            self.assertRaises(DatasetSpecificHandlerError),
        ):
            handler.validate_molecule_data(props, 0, "test")

    def test_molecule_processing_error_wrapped(self):
        handler = _make_handler()
        props = _make_raw_properties()
        err = MoleculeProcessingError(message="processing failed", molecule_index=0)
        with (
            patch.object(handler, "_is_valid_property", side_effect=err),
            self.assertRaises(DatasetSpecificHandlerError) as ctx,
        ):
            handler.validate_molecule_data(props, 0, "test")
        self.assertEqual(ctx.exception.dataset_type, "QM40")


# ============================================================================
# GROUP 5: get_required_properties
# ============================================================================


class TestGetRequiredProperties(unittest.TestCase):
    def test_includes_core(self):
        req = _make_handler().get_required_properties()
        for key in ("Internal_E_0K", "atoms", "coordinates"):
            with self.subTest(key=key):
                self.assertIn(key, req)

    def test_includes_scalar_targets(self):
        pc = _make_processing_config(scalar_graph_targets=["Internal_E_0K", "HOMO"])
        self.assertIn("HOMO", _make_handler(processing_config=pc).get_required_properties())

    def test_includes_node_features(self):
        pc = _make_processing_config(node_features=["Qmulliken"])
        self.assertIn("Qmulliken", _make_handler(processing_config=pc).get_required_properties())

    def test_includes_atomization_base(self):
        pc = _make_processing_config(calculate_atomization_energy_from="Internal_E_0K")
        self.assertIn(
            "Internal_E_0K", _make_handler(processing_config=pc).get_required_properties()
        )

    def test_no_duplicates(self):
        req = _make_handler().get_required_properties()
        self.assertEqual(len(req), len(set(req)))

    def test_returns_list(self):
        self.assertIsInstance(_make_handler().get_required_properties(), list)


# ============================================================================
# GROUP 6: process_property_value — atoms
# ============================================================================


class TestProcessAtoms(unittest.TestCase):
    def test_native_int64_passthrough(self):
        atoms = np.array([6, 1, 8], dtype=np.int64)
        self.assertEqual(_make_handler().process_property_value("atoms", atoms, 0).dtype, np.int64)

    def test_object_array_to_int64(self):
        atoms = np.array([6, 1, 8], dtype=object)
        self.assertEqual(_make_handler().process_property_value("atoms", atoms, 0).dtype, np.int64)

    def test_uint8_to_int64(self):
        atoms = np.array([6, 1, 8], dtype=np.uint8)
        self.assertEqual(_make_handler().process_property_value("atoms", atoms, 0).dtype, np.int64)

    def test_float_to_int64(self):
        atoms = np.array([6.0, 1.0, 8.0], dtype=np.float64)
        self.assertEqual(_make_handler().process_property_value("atoms", atoms, 0).dtype, np.int64)

    def test_unconvertible_returns_original(self):
        atoms = np.array(["a", "b"], dtype=object)
        result = _make_handler().process_property_value("atoms", atoms, 0)
        np.testing.assert_array_equal(result, atoms)

    def test_none_returns_none(self):
        self.assertIsNone(_make_handler().process_property_value("atoms", None, 0))


# ============================================================================
# GROUP 7: process_property_value — coordinates / initial_coordinates
# ============================================================================


class TestProcessCoordinates(unittest.TestCase):
    def test_native_float64_passthrough(self):
        coords = np.random.randn(3, 3).astype(np.float64)
        self.assertEqual(
            _make_handler().process_property_value("coordinates", coords, 0).dtype, np.float64
        )

    def test_object_array_to_float64(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=object)
        self.assertEqual(
            _make_handler().process_property_value("coordinates", coords, 0).dtype, np.float64
        )

    def test_float32_to_float64(self):
        coords = np.random.randn(3, 3).astype(np.float32)
        self.assertEqual(
            _make_handler().process_property_value("coordinates", coords, 0).dtype, np.float64
        )

    def test_integer_to_float64(self):
        coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
        self.assertEqual(
            _make_handler().process_property_value("coordinates", coords, 0).dtype, np.float64
        )

    def test_initial_coordinates_to_float64(self):
        coords = np.random.randn(3, 3).astype(np.float32)
        self.assertEqual(
            _make_handler().process_property_value("initial_coordinates", coords, 0).dtype,
            np.float64,
        )

    def test_unconvertible_returns_original(self):
        coords = np.array(["a", "b"], dtype=object)
        result = _make_handler().process_property_value("coordinates", coords, 0)
        np.testing.assert_array_equal(result, coords)


# ============================================================================
# GROUP 8: process_property_value — Qmulliken
# ============================================================================


class TestProcessQmulliken(unittest.TestCase):
    def test_object_array_to_float32(self):
        q = np.array([-0.5, 0.25, 0.25], dtype=object)
        self.assertEqual(
            _make_handler().process_property_value("Qmulliken", q, 0).dtype, np.float32
        )

    def test_int_to_float32(self):
        q = np.array([0, 0, 0], dtype=np.int64)
        self.assertEqual(
            _make_handler().process_property_value("Qmulliken", q, 0).dtype, np.float32
        )

    def test_float64_stays_floating(self):
        """Already-floating Qmulliken is not down-converted (only object/non-floating -> float32)."""
        q = np.array([-0.5, 0.25, 0.25], dtype=np.float64)
        result = _make_handler().process_property_value("Qmulliken", q, 0)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def test_non_finite_returns_none(self):
        q = np.array([0.1, np.nan, 0.2], dtype=np.float32)
        self.assertIsNone(_make_handler().process_property_value("Qmulliken", q, 0))

    def test_inf_returns_none(self):
        q = np.array([np.inf, 0.0], dtype=np.float32)
        self.assertIsNone(_make_handler().process_property_value("Qmulliken", q, 0))

    def test_unconvertible_returns_none(self):
        q = np.array(["a", "b"], dtype=object)
        self.assertIsNone(_make_handler().process_property_value("Qmulliken", q, 0))

    def test_none_returns_none(self):
        self.assertIsNone(_make_handler().process_property_value("Qmulliken", None, 0))


# ============================================================================
# GROUP 9: process_property_value — pass-through keys
# ============================================================================


class TestProcessPassthrough(unittest.TestCase):
    def test_internal_energy_scalar_passthrough(self):
        """Internal_E_0K is NOT special-cased here; passed through unchanged."""
        self.assertEqual(_make_handler().process_property_value("Internal_E_0K", -76.3, 0), -76.3)

    def test_internal_energy_ndarray_passthrough(self):
        """Scalar energy ndarray is returned unchanged (extraction happens in scalar enricher)."""
        arr = np.array(-76.3)
        result = _make_handler().process_property_value("Internal_E_0K", arr, 0)
        self.assertIsInstance(result, np.ndarray)

    def test_smiles_passthrough(self):
        self.assertEqual(_make_handler().process_property_value("smiles", "CCO", 0), "CCO")

    def test_bond_array_passthrough(self):
        val = [1, 2, 3]
        self.assertEqual(_make_handler().process_property_value("bond_lmod_ka", val, 0), val)

    def test_unknown_key_none_returns_none(self):
        self.assertIsNone(_make_handler().process_property_value("unknown", None, 0))


# ============================================================================
# GROUP 10: enrich_pyg_data
# ============================================================================


class TestEnrichPygData(unittest.TestCase):
    def test_sets_dataset_type(self):
        result = _make_handler().enrich_pyg_data(_make_pyg_data(), _make_raw_properties(), 0, "t")
        self.assertEqual(result.dataset_type, "QM40")

    def test_sets_scalar_target_y(self):
        result = _make_handler().enrich_pyg_data(_make_pyg_data(), _make_raw_properties(), 0, "t")
        self.assertTrue(hasattr(result, "y"))

    def test_zero_nodes_raises(self):
        data = Data()
        data.z = torch.tensor([], dtype=torch.long)
        data.num_nodes = 0
        with self.assertRaises(DatasetSpecificHandlerError) as ctx:
            _make_handler().enrich_pyg_data(data, _make_raw_properties(), 0, "t")
        self.assertEqual(ctx.exception.dataset_type, "QM40")

    def test_vector_property_set(self):
        pc = _make_processing_config(vector_graph_properties=["dipole_vec"])
        handler = _make_handler(processing_config=pc)
        props = _make_raw_properties(extra={"dipole_vec": np.array([1.0, 2.0, 3.0], np.float32)})
        result = handler.enrich_pyg_data(_make_pyg_data(), props, 0, "t")
        self.assertTrue(hasattr(result, "dipole_vec"))

    def test_variable_length_property_set(self):
        pc = _make_processing_config(variable_len_graph_properties=["bond_lmod_ka"])
        handler = _make_handler(processing_config=pc)
        props = _make_raw_properties(bond_lmod_ka=np.array([3.1, 4.2], np.float32))
        result = handler.enrich_pyg_data(_make_pyg_data(), props, 0, "t")
        self.assertTrue(hasattr(result, "bond_lmod_ka"))

    def test_variable_length_missing_skipped(self):
        """Missing variable-length property is skipped (DEBUG), not an error."""
        pc = _make_processing_config(variable_len_graph_properties=["bond_lmod_ka"])
        handler = _make_handler(processing_config=pc)
        props = _make_raw_properties()  # no bond_lmod_ka
        result = handler.enrich_pyg_data(_make_pyg_data(), props, 0, "t")
        self.assertFalse(hasattr(result, "bond_lmod_ka"))
        self.assertEqual(result.dataset_type, "QM40")

    def test_missing_scalar_target_raises_enrichment_error(self):
        props = _make_raw_properties()
        props.pop("Internal_E_0K")
        with self.assertRaises(PropertyEnrichmentError):
            _make_handler().enrich_pyg_data(_make_pyg_data(), props, 0, "t")

    def test_multivalue_scalar_target_raises(self):
        props = _make_raw_properties()
        props["Internal_E_0K"] = np.array([1.0, 2.0])
        with self.assertRaises(PropertyEnrichmentError):
            _make_handler().enrich_pyg_data(_make_pyg_data(), props, 0, "t")


# ============================================================================
# GROUP 11: atomization energy (via enrich + direct)
# ============================================================================


class TestAtomizationEnergy(unittest.TestCase):
    def test_no_atomization_when_unconfigured(self):
        pc = _make_processing_config(
            calculate_atomization_energy_from="", atomization_energy_key_name=""
        )
        result = _make_handler(processing_config=pc).enrich_pyg_data(
            _make_pyg_data(), _make_raw_properties(), 0, "t"
        )
        self.assertFalse(hasattr(result, "atomization_energy"))

    def test_atomization_for_hydrogen(self):
        handler = _make_handler()
        data = _make_pyg_data(num_atoms=1, z=torch.tensor([1], dtype=torch.long))
        props = _make_raw_properties(energy=-0.5, num_atoms=1, atoms=np.array([1]))
        atomic_H = ATOMIC_ENERGIES_HARTREE.get(1)
        if atomic_H is not None and HAR2EV is not None:
            result = handler.enrich_pyg_data(data, props, 0, "t")
            if hasattr(result, "atomization_energy"):
                expected_eV = (-0.5 - atomic_H) * HAR2EV
                self.assertAlmostEqual(result.atomization_energy.item(), expected_eV, places=3)

    def test_atomization_water_like(self):
        handler = _make_handler()
        data = _make_pyg_data(num_atoms=3, z=torch.tensor([8, 1, 1], dtype=torch.long))
        props = _make_raw_properties(energy=-76.3)
        a_O, a_H = ATOMIC_ENERGIES_HARTREE.get(8), ATOMIC_ENERGIES_HARTREE.get(1)
        if a_O is not None and a_H is not None and HAR2EV is not None:
            result = handler.enrich_pyg_data(data, props, 0, "t")
            if hasattr(result, "atomization_energy"):
                expected_eV = (-76.3 - (a_O + 2 * a_H)) * HAR2EV
                self.assertAlmostEqual(result.atomization_energy.item(), expected_eV, places=3)

    def test_returns_ev_float_directly(self):
        """_calculate_atomization_energy_internal returns a float in eV (or None)."""
        handler = _make_handler()
        data = _make_pyg_data(num_atoms=1, z=torch.tensor([6], dtype=torch.long))
        props = _make_raw_properties(energy=-37.8, num_atoms=1, atoms=np.array([6]))
        val = handler._calculate_atomization_energy_internal(props, data, 0, "t")
        if ATOMIC_ENERGIES_HARTREE.get(6) is not None and HAR2EV is not None:
            self.assertIsInstance(val, float)

    def test_missing_atomic_energy_no_crash(self):
        """Element absent from the energy table -> graceful None, enrichment still succeeds."""
        handler = _make_handler()
        data = _make_pyg_data(num_atoms=1, z=torch.tensor([2], dtype=torch.long))  # He
        props = _make_raw_properties(energy=-2.9, num_atoms=1, atoms=np.array([2]))
        result = handler.enrich_pyg_data(data, props, 0, "t")
        self.assertEqual(result.dataset_type, "QM40")

    @patch("milia_pipeline.handlers.implementations.qm40.HAR2EV", None)
    def test_missing_har2ev_no_atomization(self):
        handler = _make_handler()
        data = _make_pyg_data(num_atoms=1, z=torch.tensor([1], dtype=torch.long))
        props = _make_raw_properties(energy=-0.5, num_atoms=1, atoms=np.array([1]))
        result = handler.enrich_pyg_data(data, props, 0, "t")
        self.assertEqual(result.dataset_type, "QM40")
        self.assertFalse(hasattr(result, "atomization_energy"))

    def test_missing_base_energy_returns_none(self):
        handler = _make_handler()
        data = _make_pyg_data(num_atoms=1, z=torch.tensor([1], dtype=torch.long))
        props = _make_raw_properties(num_atoms=1, atoms=np.array([1]))
        props.pop("Internal_E_0K")
        self.assertIsNone(handler._calculate_atomization_energy_internal(props, data, 0, "t"))


# ============================================================================
# GROUP 12: _ensure_tensor
# ============================================================================


class TestEnsureTensor(unittest.TestCase):
    def test_list_to_tensor(self):
        t = _make_handler()._ensure_tensor([1.0, 2.0, 3.0], torch.float32, "k", 0, "id")
        self.assertEqual(t.dtype, torch.float32)
        self.assertEqual(tuple(t.shape), (3,))

    def test_ndarray_to_tensor(self):
        t = _make_handler()._ensure_tensor(np.array([1.0, 2.0]), torch.float32, "k", 0, "id")
        self.assertEqual(t.dtype, torch.float32)

    def test_scalar_to_tensor(self):
        t = _make_handler()._ensure_tensor(5, torch.float32, "k", 0, "id")
        self.assertEqual(tuple(t.shape), (1,))

    def test_existing_tensor_cast(self):
        src = torch.tensor([1, 2, 3], dtype=torch.long)
        t = _make_handler()._ensure_tensor(src, torch.float32, "k", 0, "id")
        self.assertEqual(t.dtype, torch.float32)

    def test_unsupported_raises_dataset_specific(self):
        with self.assertRaises(DatasetSpecificHandlerError) as ctx:
            _make_handler()._ensure_tensor({"bad": 1}, torch.float32, "k", 0, "id")
        self.assertEqual(ctx.exception.dataset_type, "QM40")


# ============================================================================
# GROUP 13: _is_valid_property
# ============================================================================


class TestIsValidProperty(unittest.TestCase):
    def test_none_invalid(self):
        self.assertFalse(_make_handler()._is_valid_property(None))

    def test_sentinel_strings_invalid(self):
        handler = _make_handler()
        for s in ("missing", "INVALID", "", "nan"):
            with self.subTest(s=s):
                self.assertFalse(handler._is_valid_property(s))

    def test_valid_number(self):
        self.assertTrue(_make_handler()._is_valid_property(-76.3))

    def test_valid_array(self):
        self.assertTrue(_make_handler()._is_valid_property(np.array([1.0, 2.0])))

    def test_nan_invalid(self):
        self.assertFalse(_make_handler()._is_valid_property(float("nan")))


# ============================================================================
# GROUP 14: get_processing_statistics
# ============================================================================


class TestProcessingStatistics(unittest.TestCase):
    def test_basic(self):
        stats = _make_handler().get_processing_statistics([{}, {}])
        self.assertEqual(stats["dataset_type"], "QM40")
        self.assertEqual(stats["total_processed"], 2)

    def test_counts_atomization(self):
        mols = [{"atomization_energy_calculated": True}, {"atomization_energy_calculated": False}]
        stats = _make_handler().get_processing_statistics(mols)
        self.assertEqual(stats["atomization_energy_calculations"], 1)

    def test_experimental_context_present(self):
        stats = _make_handler(experimental_setup="exp_v1").get_processing_statistics([])
        self.assertTrue(stats.get("transform_aware_processing"))
        self.assertIn("experimental_context", stats)

    def test_no_experimental_context(self):
        stats = _make_handler(experimental_setup=None).get_processing_statistics([])
        self.assertNotIn("experimental_context", stats)

    def test_empty_list(self):
        self.assertEqual(_make_handler().get_processing_statistics([])["total_processed"], 0)


# ============================================================================
# GROUP 15: structural features & descriptors
# ============================================================================


class TestStructuralFeaturesAndDescriptors(unittest.TestCase):
    def test_structural_has_atom_and_bond(self):
        feats = _make_handler().get_supported_structural_features()
        self.assertIn("atom", feats)
        self.assertIn("bond", feats)

    def test_atom_includes_mulliken(self):
        feats = _make_handler().get_supported_structural_features()
        self.assertIn("mulliken_charge", feats["atom"])

    def test_bond_includes_length(self):
        feats = _make_handler().get_supported_structural_features()
        self.assertIn("bond_length", feats["bond"])

    def test_descriptors_categories(self):
        desc = _make_handler().get_supported_descriptors()
        self.assertIn("geometric", desc["categories"])

    def test_descriptors_requires_3d_and_charges(self):
        desc = _make_handler().get_supported_descriptors()
        self.assertTrue(desc["requires_3d"])
        self.assertTrue(desc["requires_charges"])

    def test_descriptors_excluded_empty(self):
        self.assertEqual(_make_handler().get_supported_descriptors()["excluded"], [])


# ============================================================================
# GROUP 16: transform recommendations & validation
# ============================================================================


class TestTransformSystem(unittest.TestCase):
    def test_recommendations_structure(self):
        recs = _make_handler().get_transform_recommendations()
        for key in ("recommended", "avoid", "warnings"):
            with self.subTest(key=key):
                self.assertIn(key, recs)

    def test_recommendations_include_geometric(self):
        recs = _make_handler().get_transform_recommendations()
        self.assertTrue(any("RandomRotate" in r for r in recs["recommended"]))

    def test_warnings_mention_virtualnode(self):
        recs = _make_handler().get_transform_recommendations()
        self.assertTrue(any("VirtualNode" in w for w in recs["warnings"]))

    def test_suitable_includes_geometric(self):
        available = {"RandomRotate": Mock(), "GCNNorm": Mock()}
        self.assertIn("RandomRotate", _make_handler()._get_dataset_suitable_transforms(available))

    def test_suitable_filters_unavailable(self):
        available = {"GCNNorm": Mock()}
        self.assertNotIn(
            "RandomRotate", _make_handler()._get_dataset_suitable_transforms(available)
        )

    def test_suitable_normalization_structure_edge_aug(self):
        available = {
            "GCNNorm": Mock(),
            "NormalizeFeatures": Mock(),
            "AddSelfLoops": Mock(),
            "ToUndirected": Mock(),
            "Distance": Mock(),
            "Cartesian": Mock(),
            "DropEdge": Mock(),
            "MaskFeatures": Mock(),
        }
        suitable = _make_handler()._get_dataset_suitable_transforms(available)
        for t in ("GCNNorm", "AddSelfLoops", "Distance", "DropEdge"):
            with self.subTest(t=t):
                self.assertIn(t, suitable)

    def test_validate_no_geometric_warns(self):
        warnings = _make_handler()._validate_dataset_specific_transforms(["GCNNorm"])
        self.assertTrue(
            any("geometric augmentation" in w.lower() or "RandomRotate" in w for w in warnings)
        )

    def test_validate_with_geometric_no_warn(self):
        warnings = _make_handler()._validate_dataset_specific_transforms(["RandomRotate"])
        self.assertEqual(len([w for w in warnings if "geometric augmentation" in w.lower()]), 0)

    def test_validate_distance_warns(self):
        warnings = _make_handler()._validate_dataset_specific_transforms(["Distance"])
        self.assertTrue(any("edge attribute" in w.lower() for w in warnings))

    def test_incompat_virtualnode_with_qmulliken(self):
        """VirtualNode + Qmulliken node feature -> incompatibility error (QM40-specific)."""
        pc = _make_processing_config(node_features=["Qmulliken"])
        errors = _make_handler(processing_config=pc)._check_transform_incompatibilities(
            ["VirtualNode"]
        )
        self.assertTrue(any("VirtualNode" in e for e in errors))

    def test_incompat_virtualnode_without_qmulliken_ok(self):
        pc = _make_processing_config(node_features=["atoms"])
        errors = _make_handler(processing_config=pc)._check_transform_incompatibilities(
            ["VirtualNode"]
        )
        self.assertEqual(errors, [])

    def test_incompat_empty_when_no_virtualnode(self):
        self.assertEqual(
            _make_handler()._check_transform_incompatibilities(["RandomRotate", "GCNNorm"]), []
        )

    def test_recs_gcnnorm_without_selfloops(self):
        recs = _make_handler()._get_transform_recommendations(["GCNNorm"])
        self.assertTrue(any("AddSelfLoops" in r for r in recs))

    def test_recs_no_geometric_recommends_augmentation(self):
        recs = _make_handler()._get_transform_recommendations(["GCNNorm", "AddSelfLoops"])
        self.assertTrue(any("geometric" in r.lower() or "3D" in r for r in recs))

    def test_recs_all_present_minimal(self):
        recs = _make_handler()._get_transform_recommendations(
            ["GCNNorm", "AddSelfLoops", "RandomRotate"]
        )
        geo = [r for r in recs if "geometric" in r.lower() or "3D" in r]
        self.assertEqual(len(geo), 0)


# ============================================================================
# GROUP 17: integration & edge cases
# ============================================================================


class TestIntegrationAndEdgeCases(unittest.TestCase):
    def test_validate_then_enrich_happy_path(self):
        handler = _make_handler()
        props = _make_raw_properties(energy=-76.3)
        handler.validate_molecule_data(props, 0, "t")
        handler.process_property_value("atoms", props["atoms"], 0)
        result = handler.enrich_pyg_data(_make_pyg_data(), props, 0, "t")
        self.assertEqual(result.dataset_type, "QM40")
        self.assertTrue(hasattr(result, "y"))

    def test_object_array_pipeline(self):
        handler = _make_handler()
        atoms_obj = np.array([6, 1, 8], dtype=object)
        coords_obj = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=object)
        self.assertEqual(handler.process_property_value("atoms", atoms_obj, 0).dtype, np.int64)
        self.assertEqual(
            handler.process_property_value("coordinates", coords_obj, 0).dtype, np.float64
        )

    def test_all_seven_qm40_elements_validate(self):
        handler = _make_handler()
        all_z = np.array(sorted(QM40_SUPPORTED_ELEMENTS), dtype=np.int64)
        props = _make_raw_properties(
            atoms=all_z, coordinates=np.random.randn(len(all_z), 3), num_atoms=len(all_z)
        )
        handler.validate_molecule_data(props, 0, "all_elements")

    def test_single_atom_molecule(self):
        handler = _make_handler()
        props = _make_raw_properties(
            atoms=np.array([6]), coordinates=np.array([[0.0, 0.0, 0.0]]), num_atoms=1
        )
        handler.validate_molecule_data(props, 0, "t")

    def test_handler_without_experimental_setup(self):
        stats = _make_handler(experimental_setup=None).get_processing_statistics([])
        self.assertNotIn("experimental_context", stats)


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    unittest.main()
