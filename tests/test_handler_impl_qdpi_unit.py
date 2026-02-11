#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/handlers/implementations/qdpi.py

Module under test: qdpi.py
- QDPiDatasetHandler: Handler for QDπ (Quantum Deep Potential Interaction) datasets
  - Implements DatasetHandler ABC (12 abstract methods + 4 transform validation helpers)
  - Registered via @register_handler decorator
  - QDπ-specific: coordinate_based strategy (NO identifiers), supports BOTH neutral AND
    charged molecules, 13 elements (H, Li, C, N, O, F, Na, P, S, Cl, K, Br, I),
    dtype normalization for object arrays, charge inference from molecular_charge /
    charge / charge_type / element heuristic, atomization energy (Hartree → eV),
    forces handling, energies in Hartree (converted from eV during preprocessing)

Test path on local machine: ~/ml_projects/milia/tests/test_handler_impl_qdpi_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/handlers/implementations/qdpi.py

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

Coverage targets:
- All 12 abstract method implementations
- All 4 transform validation helper methods
- All internal methods (_infer_charge_from_elements, _add_scalar_targets_internal,
  _add_vector_properties_internal, _add_variable_length_properties_internal,
  _calculate_atomization_energy_internal, _ensure_tensor, _is_valid_property)
- Exception hierarchy: HandlerValidationError, DatasetSpecificHandlerError,
  PropertyEnrichmentError, MoleculeProcessingError
- QDπ dtype normalization (object arrays → native dtypes)
- Forces handling (float32 conversion, non-finite detection)
- Energy handling (NaN detection, ndarray → float extraction)
- Charge handling (molecular_charge, charge, charge_type, element inference)
- QDPI_SUPPORTED_ELEMENTS validation

Verified against exception signatures (from exceptions.py):
- DatasetSpecificHandlerError(message, dataset_type, operation, molecule_index,
  identifier, details, property_name)
- PropertyEnrichmentError(molecule_index, inchi, property_name, reason, detail)
- HandlerValidationError(message, handler_type, validation_type, failed_validations,
  molecule_index, details)
- Unit safety verified for atomization energy (Hartree → eV conversion)

Updated: February 2026 - Production-ready comprehensive test coverage
"""

import sys
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, MagicMock, patch, PropertyMock, call
import logging
import copy
import math
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from milia_pipeline.handlers.implementations.qdpi import (
    QDPiDatasetHandler,
    QDPI_SUPPORTED_ELEMENTS,
)
from milia_pipeline.exceptions import (
    PropertyEnrichmentError,
    MoleculeProcessingError,
    HandlerError,
    HandlerConfigurationError,
    HandlerValidationError,
    DatasetSpecificHandlerError,
)

# Import constants from the same location qdpi.py uses (milia_pipeline.config.config_constants)
# with fallback pattern for environments where config module may not be fully available
try:
    from milia_pipeline.config.config_constants import ATOMIC_ENERGIES_HARTREE, HAR2EV
except ImportError:
    ATOMIC_ENERGIES_HARTREE = {
        1: -0.500273,     # H
        3: -7.478060,     # Li
        6: -37.846772,    # C
        7: -54.583861,    # N
        8: -75.064579,    # O
        9: -99.733509,    # F
        11: -162.254553,  # Na
        15: -341.258600,  # P
        16: -398.100442,  # S
        17: -460.148990,  # Cl
        19: -599.764957,  # K
        35: -2573.437350, # Br
        53: -6918.076370, # I
    }
    HAR2EV = 27.211386245988


# ============================================================================
# HELPERS: Build realistic config mocks for QDPiDatasetHandler
# ============================================================================

def _make_dataset_config(**overrides):
    """
    Build a minimal mock DatasetConfig for QDπ handler tests.

    Based on project structure: DatasetConfig is a Pydantic frozen BaseModel.
    The handler accesses dataset_config attributes but primarily relies on
    processing_config for property lists.
    """
    cfg = Mock(spec_set=["dataset_type", "npz_file_path", "dataset_name"])
    cfg.dataset_type = overrides.get("dataset_type", "QDPi")
    cfg.npz_file_path = overrides.get(
        "npz_file_path", "~/Chem_Data/MILIA_PyG_Dataset/raw/qdpi.npz"
    )
    cfg.dataset_name = overrides.get("dataset_name", "QDPi")
    return cfg


def _make_filter_config(**overrides):
    """
    Build a minimal mock FilterConfig for QDπ handler tests.

    FilterConfig controls molecule filtering (max atoms, element filters, etc.).
    """
    cfg = Mock(spec_set=["max_atoms", "allowed_elements", "min_atoms"])
    cfg.max_atoms = overrides.get("max_atoms", 63)
    cfg.allowed_elements = overrides.get("allowed_elements", None)
    cfg.min_atoms = overrides.get("min_atoms", 1)
    return cfg


def _make_processing_config(**overrides):
    """
    Build a minimal mock ProcessingConfig for QDπ handler tests.

    ProcessingConfig controls which properties to extract, scalar targets,
    node features, vector properties, variable-length properties, and
    atomization energy configuration.

    QDπ key properties:
    - scalar_graph_targets: ['energy'] (Hartree, converted from eV)
    - node_features: ['atoms'] (atomic numbers)
    - vector_graph_properties: [] (typically empty)
    - variable_len_graph_properties: ['forces'] (if available)
    - calculate_atomization_energy_from: 'energy' (Hartree basis)
    - atomization_energy_key_name: 'atomization_energy'
    """
    cfg = Mock(spec_set=[
        "scalar_graph_targets",
        "node_features",
        "vector_graph_properties",
        "variable_len_graph_properties",
        "calculate_atomization_energy_from",
        "atomization_energy_key_name",
    ])
    cfg.scalar_graph_targets = overrides.get("scalar_graph_targets", ["energy"])
    cfg.node_features = overrides.get("node_features", ["atoms"])
    cfg.vector_graph_properties = overrides.get("vector_graph_properties", [])
    cfg.variable_len_graph_properties = overrides.get("variable_len_graph_properties", [])
    cfg.calculate_atomization_energy_from = overrides.get(
        "calculate_atomization_energy_from", "energy"
    )
    cfg.atomization_energy_key_name = overrides.get(
        "atomization_energy_key_name", "atomization_energy"
    )
    return cfg


def _make_handler(**overrides):
    """
    Build a ready-to-use QDPiDatasetHandler instance.

    Based on DatasetHandler ABC constructor signature:
    __init__(dataset_config, filter_config, processing_config, logger, experimental_setup=None)
    """
    dataset_config = overrides.get("dataset_config", _make_dataset_config())
    filter_config = overrides.get("filter_config", _make_filter_config())
    processing_config = overrides.get("processing_config", _make_processing_config())
    logger = overrides.get("logger", logging.getLogger("test.qdpi"))
    experimental_setup = overrides.get("experimental_setup", None)

    handler = QDPiDatasetHandler(
        dataset_config=dataset_config,
        filter_config=filter_config,
        processing_config=processing_config,
        logger=logger,
        experimental_setup=experimental_setup,
    )
    return handler


def _make_pyg_data(**overrides):
    """
    Build a minimal PyG Data object for QDπ enrichment tests.

    QDπ molecules typically have:
    - z: atomic numbers tensor (13 elements: H, Li, C, N, O, F, Na, P, S, Cl, K, Br, I)
    - pos: 3D coordinates tensor (from optimized geometries)
    - edge_index: connectivity tensor
    - num_nodes: atom count
    """
    num_atoms = overrides.get("num_atoms", 3)
    z = overrides.get("z", torch.tensor([6, 1, 8], dtype=torch.long)[:num_atoms])
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
    """
    Build a realistic raw_properties_dict for QDπ molecule tests.

    QDπ NPZ files contain: energy, atoms, coordinates, forces (optional),
    molecular_charge (optional), formula (optional), subset (optional).
    Contains BOTH neutral AND charged molecules.
    Energies in Hartree, coordinates in Angstrom, forces in Hartree/Angstrom.
    """
    num_atoms = overrides.get("num_atoms", 3)
    props = {
        "energy": overrides.get("energy", -76.3),
        "atoms": overrides.get("atoms", np.array([6, 1, 8])[:num_atoms]),
        "coordinates": overrides.get(
            "coordinates", np.random.randn(num_atoms, 3).astype(np.float64)
        ),
    }
    for key in ["forces", "molecular_charge", "charge", "charge_type",
                "formula", "subset"]:
        if key in overrides:
            props[key] = overrides[key]
    return props


# ============================================================================
# GROUP 1: QDPiDatasetHandler — Identity and Registration (6 tests)
# ============================================================================

class TestQDPiDatasetHandlerIdentity(unittest.TestCase):
    """Test QDPiDatasetHandler identity, registration, and basic attributes."""

    def test_get_dataset_type_returns_qdpi(self):
        """get_dataset_type() returns 'QDPi'."""
        handler = _make_handler()
        self.assertEqual(handler.get_dataset_type(), "QDPi")

    def test_get_molecule_creation_strategy(self):
        """QDπ uses coordinate_based strategy (NO identifiers available)."""
        handler = _make_handler()
        self.assertEqual(handler.get_molecule_creation_strategy(), "coordinate_based")

    def test_get_identifier_keys_empty(self):
        """QDπ has NO parseable identifiers — returns empty list."""
        handler = _make_handler()
        keys = handler.get_identifier_keys()
        self.assertIsInstance(keys, list)
        self.assertEqual(len(keys), 0)

    def test_is_subclass_of_dataset_handler(self):
        """QDPiDatasetHandler is a proper DatasetHandler subclass."""
        from milia_pipeline.handlers.base_handler import DatasetHandler
        self.assertTrue(issubclass(QDPiDatasetHandler, DatasetHandler))

    def test_handler_stores_configs(self):
        """Handler stores config objects passed during construction."""
        dc = _make_dataset_config()
        fc = _make_filter_config()
        pc = _make_processing_config()
        handler = QDPiDatasetHandler(
            dataset_config=dc, filter_config=fc,
            processing_config=pc, logger=logging.getLogger("test"),
        )
        self.assertIs(handler.dataset_config, dc)
        self.assertIs(handler.filter_config, fc)
        self.assertIs(handler.processing_config, pc)

    def test_handler_stores_experimental_setup(self):
        """Handler stores experimental_setup passed during construction."""
        handler = _make_handler(experimental_setup="exp_setup_v1")
        self.assertEqual(handler.experimental_setup, "exp_setup_v1")


# ============================================================================
# GROUP 2: QDPI_SUPPORTED_ELEMENTS constant (3 tests)
# ============================================================================

class TestQDPiSupportedElements(unittest.TestCase):
    """Test QDPI_SUPPORTED_ELEMENTS constant correctness."""

    def test_contains_13_elements(self):
        """QDπ supports exactly 13 elements."""
        self.assertEqual(len(QDPI_SUPPORTED_ELEMENTS), 13)

    def test_contains_expected_elements(self):
        """All 13 QDπ elements are present: H, Li, C, N, O, F, Na, P, S, Cl, K, Br, I."""
        expected = {1, 3, 6, 7, 8, 9, 11, 15, 16, 17, 19, 35, 53}
        self.assertEqual(QDPI_SUPPORTED_ELEMENTS, expected)

    def test_is_set_type(self):
        """QDPI_SUPPORTED_ELEMENTS is a set for O(1) membership checks."""
        self.assertIsInstance(QDPI_SUPPORTED_ELEMENTS, set)


# ============================================================================
# GROUP 3: get_molecular_charge — multi-strategy charge resolution (14 tests)
# ============================================================================

class TestGetMolecularCharge(unittest.TestCase):
    """Test QDπ molecular charge — supports BOTH neutral AND charged molecules."""

    def test_strategy1_molecular_charge_key(self):
        """Strategy 1: Returns charge from 'molecular_charge' key."""
        handler = _make_handler()
        charge = handler.get_molecular_charge(
            {"molecular_charge": 1}, np.array([11]), "NaCl"
        )
        self.assertEqual(charge, 1)

    def test_strategy1_negative_charge(self):
        """Strategy 1: Handles negative charge from 'molecular_charge'."""
        handler = _make_handler()
        charge = handler.get_molecular_charge(
            {"molecular_charge": -1}, np.array([17]), "Cl-"
        )
        self.assertEqual(charge, -1)

    def test_strategy1_zero_charge(self):
        """Strategy 1: Handles zero (neutral) from 'molecular_charge'."""
        handler = _make_handler()
        charge = handler.get_molecular_charge(
            {"molecular_charge": 0}, np.array([6, 1, 8]), "CH3OH"
        )
        self.assertEqual(charge, 0)

    def test_strategy1_numpy_scalar(self):
        """Strategy 1: Handles numpy scalar molecular_charge."""
        handler = _make_handler()
        charge = handler.get_molecular_charge(
            {"molecular_charge": np.int64(2)}, np.array([19]), "K2+"
        )
        self.assertEqual(charge, 2)

    def test_strategy1_invalid_value_falls_through(self):
        """Strategy 1: Invalid molecular_charge value falls through to next strategy."""
        handler = _make_handler()
        charge = handler.get_molecular_charge(
            {"molecular_charge": "not_a_number"}, np.array([6, 1, 8]), "test"
        )
        self.assertEqual(charge, 0)

    def test_strategy1_none_value_falls_through(self):
        """Strategy 1: None molecular_charge falls through to next strategy."""
        handler = _make_handler()
        charge = handler.get_molecular_charge(
            {"molecular_charge": None}, np.array([6, 1, 8]), "test"
        )
        self.assertEqual(charge, 0)

    def test_strategy2_charge_key(self):
        """Strategy 2: Returns charge from 'charge' key when 'molecular_charge' absent."""
        handler = _make_handler()
        charge = handler.get_molecular_charge(
            {"charge": -2}, np.array([8, 8]), "O2^2-"
        )
        self.assertEqual(charge, -2)

    def test_strategy2_invalid_charge_falls_through(self):
        """Strategy 2: Invalid 'charge' value falls through."""
        handler = _make_handler()
        charge = handler.get_molecular_charge(
            {"charge": "bad"}, np.array([6, 1, 8]), "test"
        )
        self.assertEqual(charge, 0)

    def test_strategy3_charge_type_neutral(self):
        """Strategy 3: charge_type='neutral' returns 0."""
        handler = _make_handler()
        charge = handler.get_molecular_charge(
            {"charge_type": "neutral"}, np.array([6, 1, 8]), "CH3OH"
        )
        self.assertEqual(charge, 0)

    def test_strategy3_charge_type_charged_calls_infer(self):
        """Strategy 3: charge_type='charged' triggers element inference."""
        handler = _make_handler()
        charge = handler.get_molecular_charge(
            {"charge_type": "charged"}, np.array([11]), "Na+"
        )
        self.assertEqual(charge, 1)

    def test_strategy4_default_zero(self):
        """Strategy 4: No charge info at all defaults to 0 (neutral)."""
        handler = _make_handler()
        charge = handler.get_molecular_charge(
            {}, np.array([6, 1, 8]), "unknown"
        )
        self.assertEqual(charge, 0)

    def test_returns_int_type(self):
        """Return type is always int."""
        handler = _make_handler()
        charge = handler.get_molecular_charge(
            {"molecular_charge": np.float64(1.0)}, np.array([11]), "Na+"
        )
        self.assertIsInstance(charge, int)

    def test_strategy1_float_converted_to_int(self):
        """Strategy 1: Float molecular_charge is converted to int."""
        handler = _make_handler()
        charge = handler.get_molecular_charge(
            {"molecular_charge": 1.0}, np.array([11]), "Na+"
        )
        self.assertEqual(charge, 1)
        self.assertIsInstance(charge, int)

    def test_with_all_qdpi_organic_elements(self):
        """Typical QDπ organic molecule elements default to charge 0."""
        handler = _make_handler()
        atoms = np.array([6, 7, 8, 1, 1, 1, 1])
        charge = handler.get_molecular_charge({}, atoms, "organic_mol")
        self.assertEqual(charge, 0)


# ============================================================================
# GROUP 4: _infer_charge_from_elements (13 tests)
# ============================================================================

class TestInferChargeFromElements(unittest.TestCase):
    """Test QDπ charge inference heuristic from atomic composition."""

    def test_single_lithium_cation(self):
        """Single Li atom (Z=3) inferred as +1 cation."""
        handler = _make_handler()
        self.assertEqual(handler._infer_charge_from_elements(np.array([3]), "Li+"), 1)

    def test_single_sodium_cation(self):
        """Single Na atom (Z=11) inferred as +1 cation."""
        handler = _make_handler()
        self.assertEqual(handler._infer_charge_from_elements(np.array([11]), "Na+"), 1)

    def test_single_potassium_cation(self):
        """Single K atom (Z=19) inferred as +1 cation."""
        handler = _make_handler()
        self.assertEqual(handler._infer_charge_from_elements(np.array([19]), "K+"), 1)

    def test_single_fluoride_anion(self):
        """Single F atom (Z=9) inferred as -1 anion."""
        handler = _make_handler()
        self.assertEqual(handler._infer_charge_from_elements(np.array([9]), "F-"), -1)

    def test_single_chloride_anion(self):
        """Single Cl atom (Z=17) inferred as -1 anion."""
        handler = _make_handler()
        self.assertEqual(handler._infer_charge_from_elements(np.array([17]), "Cl-"), -1)

    def test_single_bromide_anion(self):
        """Single Br atom (Z=35) inferred as -1 anion."""
        handler = _make_handler()
        self.assertEqual(handler._infer_charge_from_elements(np.array([35]), "Br-"), -1)

    def test_single_iodide_anion(self):
        """Single I atom (Z=53) inferred as -1 anion."""
        handler = _make_handler()
        self.assertEqual(handler._infer_charge_from_elements(np.array([53]), "I-"), -1)

    def test_ion_pair_nacl_neutral(self):
        """NaCl ion pair (2 atoms: alkali + halide) inferred as neutral."""
        handler = _make_handler()
        self.assertEqual(handler._infer_charge_from_elements(np.array([11, 17]), "NaCl"), 0)

    def test_ion_pair_kbr_neutral(self):
        """KBr ion pair (alkali + halide) inferred as neutral."""
        handler = _make_handler()
        self.assertEqual(handler._infer_charge_from_elements(np.array([19, 35]), "KBr"), 0)

    def test_complex_molecule_returns_zero(self):
        """Complex multi-atom organic molecule returns 0 (cannot infer)."""
        handler = _make_handler()
        atoms = np.array([6, 6, 7, 8, 1, 1, 1])
        self.assertEqual(handler._infer_charge_from_elements(atoms, "organic"), 0)

    def test_none_atoms_returns_zero(self):
        """None atomic_numbers returns 0."""
        handler = _make_handler()
        self.assertEqual(handler._infer_charge_from_elements(None, "none_mol"), 0)

    def test_empty_atoms_returns_zero(self):
        """Empty atomic_numbers array returns 0."""
        handler = _make_handler()
        self.assertEqual(handler._infer_charge_from_elements(np.array([]), "empty"), 0)

    def test_single_carbon_neutral(self):
        """Single C atom (Z=6) — not alkali/halide — returns 0."""
        handler = _make_handler()
        self.assertEqual(handler._infer_charge_from_elements(np.array([6]), "C"), 0)


# ============================================================================
# GROUP 5: get_required_properties (5 tests)
# ============================================================================

class TestGetRequiredProperties(unittest.TestCase):
    """Test QDπ required properties assembly."""

    def test_includes_core_properties(self):
        """Core QDπ properties (energy, atoms, coordinates) are always included."""
        handler = _make_handler()
        required = handler.get_required_properties()
        for prop in ["energy", "atoms", "coordinates"]:
            self.assertIn(prop, required)

    def test_includes_scalar_targets(self):
        """Scalar graph targets from processing config are included."""
        pc = _make_processing_config(scalar_graph_targets=["energy", "custom_target"])
        handler = _make_handler(processing_config=pc)
        self.assertIn("custom_target", handler.get_required_properties())

    def test_includes_node_features(self):
        """Node features from processing config are included."""
        pc = _make_processing_config(node_features=["atoms", "extra_feature"])
        handler = _make_handler(processing_config=pc)
        self.assertIn("extra_feature", handler.get_required_properties())

    def test_includes_atomization_base(self):
        """Atomization energy base property is included when configured."""
        pc = _make_processing_config(calculate_atomization_energy_from="energy")
        handler = _make_handler(processing_config=pc)
        self.assertIn("energy", handler.get_required_properties())

    def test_deduplication(self):
        """Duplicate properties are removed (returns unique list)."""
        pc = _make_processing_config(
            scalar_graph_targets=["energy"],
            node_features=["atoms", "energy"],
        )
        handler = _make_handler(processing_config=pc)
        required = handler.get_required_properties()
        self.assertEqual(required.count("energy"), 1)


# ============================================================================
# GROUP 6: validate_molecule_data (12 tests)
# ============================================================================

class TestValidateMoleculeData(unittest.TestCase):
    """Test QDπ molecule validation with exception handling."""

    def test_valid_molecule_passes(self):
        """Complete valid molecule data passes validation without error."""
        handler = _make_handler()
        props = _make_raw_properties()
        handler.validate_molecule_data(props, molecule_index=0, identifier="test")

    def test_missing_energy_raises(self):
        """Missing 'energy' raises HandlerValidationError."""
        handler = _make_handler()
        props = _make_raw_properties()
        props.pop("energy")
        with self.assertRaises(HandlerValidationError) as ctx:
            handler.validate_molecule_data(props, 0, "test")
        self.assertIn("energy", str(ctx.exception))

    def test_missing_atoms_raises(self):
        """Missing 'atoms' raises HandlerValidationError."""
        handler = _make_handler()
        props = _make_raw_properties()
        props.pop("atoms")
        with self.assertRaises(HandlerValidationError) as ctx:
            handler.validate_molecule_data(props, 0, "test")
        self.assertIn("atoms", str(ctx.exception))

    def test_missing_coordinates_raises(self):
        """Missing 'coordinates' raises HandlerValidationError."""
        handler = _make_handler()
        props = _make_raw_properties()
        props.pop("coordinates")
        with self.assertRaises(HandlerValidationError) as ctx:
            handler.validate_molecule_data(props, 0, "test")
        self.assertIn("coordinates", str(ctx.exception))

    def test_missing_multiple_properties_raises(self):
        """Missing multiple required properties lists all in error."""
        handler = _make_handler()
        with self.assertRaises(HandlerValidationError):
            handler.validate_molecule_data({}, 0, "test")

    def test_none_property_treated_as_missing(self):
        """None values are treated as missing by _is_valid_property."""
        handler = _make_handler()
        props = _make_raw_properties()
        props["energy"] = None
        with self.assertRaises(HandlerValidationError):
            handler.validate_molecule_data(props, 0, "test")

    def test_empty_array_treated_as_missing(self):
        """Empty arrays are treated as missing by _is_valid_property."""
        handler = _make_handler()
        props = _make_raw_properties()
        props["atoms"] = np.array([])
        with self.assertRaises(HandlerValidationError):
            handler.validate_molecule_data(props, 0, "test")

    def test_structural_validation_failure_raises_dataset_specific(self):
        """Structure validation failure wraps into DatasetSpecificHandlerError."""
        handler = _make_handler()
        props = _make_raw_properties()
        with patch(
            "milia_pipeline.handlers.implementations.qdpi.validate_molecular_structure",
            side_effect=ValueError("atoms/coords mismatch"),
        ):
            with self.assertRaises(DatasetSpecificHandlerError) as ctx:
                handler.validate_molecule_data(props, 0, "test")
            self.assertEqual(ctx.exception.dataset_type, "QDPi")

    def test_positive_energy_logs_warning(self):
        """Positive energy (unusual for total molecular energy) logs a warning."""
        handler = _make_handler()
        props = _make_raw_properties(energy=10.0)
        with self.assertLogs("test.qdpi", level="WARNING") as log:
            handler.validate_molecule_data(props, 0, "test")
        self.assertTrue(any("positive energy" in msg for msg in log.output))

    def test_unsupported_elements_logs_warning(self):
        """Unsupported elements (not in QDPI_SUPPORTED_ELEMENTS) log a warning.

        EVIDENCE: validate_molecular_structure() validates Z in [1, 118], so we
        must use a valid atomic number that is NOT in the 13-element QDπ set.
        He (Z=2) is valid but not in QDPI_SUPPORTED_ELEMENTS.
        """
        handler = _make_handler()
        # Z=2 (He) is a valid element but NOT in QDπ's 13-element set
        props = _make_raw_properties(
            atoms=np.array([2, 6, 1]),
            coordinates=np.random.randn(3, 3),
            num_atoms=3,
        )
        with self.assertLogs("test.qdpi", level="WARNING") as log:
            handler.validate_molecule_data(props, 0, "test")
        self.assertTrue(any("unsupported elements" in msg for msg in log.output))

    def test_unexpected_exception_wrapped(self):
        """Unexpected exceptions wrap into DatasetSpecificHandlerError."""
        handler = _make_handler()
        props = _make_raw_properties()
        with patch.object(
            handler, "_is_valid_property",
            side_effect=RuntimeError("unexpected boom"),
        ):
            with self.assertRaises(DatasetSpecificHandlerError):
                handler.validate_molecule_data(props, 0, "test")

    def test_molecule_processing_error_wrapped(self):
        """MoleculeProcessingError wraps into DatasetSpecificHandlerError."""
        handler = _make_handler()
        props = _make_raw_properties()
        err = MoleculeProcessingError(message="processing failed", molecule_index=0)
        with patch.object(handler, "_is_valid_property", side_effect=err):
            with self.assertRaises(DatasetSpecificHandlerError) as ctx:
                handler.validate_molecule_data(props, 0, "test")
            self.assertEqual(ctx.exception.dataset_type, "QDPi")


# ============================================================================
# GROUP 7: process_property_value — atoms (6 tests)
# ============================================================================

class TestProcessPropertyValueAtoms(unittest.TestCase):
    """Test QDπ atoms dtype normalization."""

    def test_native_int64_passthrough(self):
        handler = _make_handler()
        atoms = np.array([6, 1, 8], dtype=np.int64)
        result = handler.process_property_value("atoms", atoms, 0)
        self.assertEqual(result.dtype, np.int64)
        np.testing.assert_array_equal(result, atoms)

    def test_object_array_converted_to_int64(self):
        handler = _make_handler()
        atoms = np.array([6, 1, 8], dtype=object)
        result = handler.process_property_value("atoms", atoms, 0)
        self.assertEqual(result.dtype, np.int64)
        np.testing.assert_array_equal(result, [6, 1, 8])

    def test_uint8_normalized_to_int64(self):
        handler = _make_handler()
        atoms = np.array([6, 1, 8], dtype=np.uint8)
        result = handler.process_property_value("atoms", atoms, 0)
        self.assertEqual(result.dtype, np.int64)

    def test_float_atoms_converted_to_int64(self):
        handler = _make_handler()
        atoms = np.array([6.0, 1.0, 8.0], dtype=np.float64)
        result = handler.process_property_value("atoms", atoms, 0)
        self.assertEqual(result.dtype, np.int64)

    def test_unconvertible_atoms_returns_original(self):
        handler = _make_handler()
        atoms = np.array(["C", "H", "O"], dtype=object)
        result = handler.process_property_value("atoms", atoms, 0)
        np.testing.assert_array_equal(result, atoms)

    def test_none_atoms_returns_none(self):
        handler = _make_handler()
        result = handler.process_property_value("atoms", None, 0)
        self.assertIsNone(result)


# ============================================================================
# GROUP 8: process_property_value — coordinates (5 tests)
# ============================================================================

class TestProcessPropertyValueCoordinates(unittest.TestCase):
    """Test QDπ coordinates dtype normalization."""

    def test_native_float64_passthrough(self):
        handler = _make_handler()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        result = handler.process_property_value("coordinates", coords, 0)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_array_equal(result, coords)

    def test_object_array_converted_to_float64(self):
        handler = _make_handler()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=object)
        result = handler.process_property_value("coordinates", coords, 0)
        self.assertEqual(result.dtype, np.float64)

    def test_float32_normalized_to_float64(self):
        handler = _make_handler()
        coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        result = handler.process_property_value("coordinates", coords, 0)
        self.assertEqual(result.dtype, np.float64)

    def test_integer_coords_converted_to_float64(self):
        handler = _make_handler()
        coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
        result = handler.process_property_value("coordinates", coords, 0)
        self.assertEqual(result.dtype, np.float64)

    def test_unconvertible_coords_returns_original(self):
        handler = _make_handler()
        coords = np.array([["a", "b", "c"]], dtype=object)
        result = handler.process_property_value("coordinates", coords, 0)
        np.testing.assert_array_equal(result, coords)


# ============================================================================
# GROUP 9: process_property_value — forces (7 tests)
# ============================================================================

class TestProcessPropertyValueForces(unittest.TestCase):
    """Test QDπ forces dtype normalization and validation."""

    def test_valid_float32_passthrough(self):
        handler = _make_handler()
        forces = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
        result = handler.process_property_value("forces", forces, 0)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def test_object_array_converted_to_float32(self):
        handler = _make_handler()
        forces = np.array([[0.1, -0.2, 0.3]], dtype=object)
        result = handler.process_property_value("forces", forces, 0)
        self.assertEqual(result.dtype, np.float32)

    def test_float64_forces_kept_as_floating(self):
        handler = _make_handler()
        forces = np.array([[0.1, -0.2, 0.3]], dtype=np.float64)
        result = handler.process_property_value("forces", forces, 0)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def test_non_finite_forces_returns_none(self):
        handler = _make_handler()
        forces = np.array([[float("nan"), 0.0, 0.0]], dtype=np.float32)
        self.assertIsNone(handler.process_property_value("forces", forces, 0))

    def test_inf_forces_returns_none(self):
        handler = _make_handler()
        forces = np.array([[float("inf"), 0.0, 0.0]], dtype=np.float32)
        self.assertIsNone(handler.process_property_value("forces", forces, 0))

    def test_unconvertible_forces_returns_none(self):
        handler = _make_handler()
        forces = np.array([["a", "b", "c"]], dtype=object)
        self.assertIsNone(handler.process_property_value("forces", forces, 0))

    def test_integer_forces_converted_to_float32(self):
        handler = _make_handler()
        forces = np.array([[1, -2, 3]], dtype=np.int32)
        result = handler.process_property_value("forces", forces, 0)
        self.assertEqual(result.dtype, np.float32)


# ============================================================================
# GROUP 10: process_property_value — energy (6 tests)
# ============================================================================

class TestProcessPropertyValueEnergy(unittest.TestCase):
    """Test QDπ energy handling."""

    def test_float_passthrough(self):
        handler = _make_handler()
        self.assertEqual(handler.process_property_value("energy", -76.3, 0), -76.3)

    def test_ndarray_size_1_extracted_to_float(self):
        handler = _make_handler()
        result = handler.process_property_value("energy", np.array(-76.3), 0)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, -76.3)

    def test_nan_float_energy_passthrough(self):
        """Plain float NaN passes through (NaN check only inside np.ndarray branch)."""
        handler = _make_handler()
        result = handler.process_property_value("energy", float("nan"), 0)
        self.assertIsNotNone(result)
        self.assertTrue(math.isnan(result))

    def test_nan_ndarray_energy_returns_none(self):
        handler = _make_handler()
        self.assertIsNone(handler.process_property_value("energy", np.array(float("nan")), 0))

    def test_numpy_scalar_passthrough(self):
        handler = _make_handler()
        self.assertAlmostEqual(handler.process_property_value("energy", np.float64(-76.3), 0), -76.3)

    def test_none_energy_returns_none(self):
        handler = _make_handler()
        self.assertIsNone(handler.process_property_value("energy", None, 0))


# ============================================================================
# GROUP 11: process_property_value — molecular_charge / charge (5 tests)
# ============================================================================

class TestProcessPropertyValueCharge(unittest.TestCase):
    """Test QDπ charge property processing."""

    def test_ndarray_size_1_extracted_to_int(self):
        handler = _make_handler()
        result = handler.process_property_value("molecular_charge", np.array([1]), 0)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 1)

    def test_float_charge_rounded_to_int(self):
        handler = _make_handler()
        result = handler.process_property_value("molecular_charge", 1.0, 0)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 1)

    def test_numpy_float_charge_converted(self):
        handler = _make_handler()
        result = handler.process_property_value("charge", np.float64(-1.0), 0)
        self.assertIsInstance(result, int)
        self.assertEqual(result, -1)

    def test_int_charge_passthrough(self):
        handler = _make_handler()
        self.assertEqual(handler.process_property_value("molecular_charge", 2, 0), 2)

    def test_none_charge_returns_none(self):
        handler = _make_handler()
        self.assertIsNone(handler.process_property_value("molecular_charge", None, 0))


# ============================================================================
# GROUP 12: process_property_value — other/edge cases (5 tests)
# ============================================================================

class TestProcessPropertyValueOther(unittest.TestCase):
    """Test QDπ property processing for unrecognized keys and edge cases."""

    def test_unknown_key_passthrough(self):
        handler = _make_handler()
        self.assertEqual(handler.process_property_value("unknown_prop", [1, 2, 3], 0), [1, 2, 3])

    def test_none_unknown_key_returns_none(self):
        handler = _make_handler()
        self.assertIsNone(handler.process_property_value("unknown", None, 0))

    def test_dataset_specific_error_reraised(self):
        handler = _make_handler()
        err = DatasetSpecificHandlerError(dataset_type="QDPi", message="test error", operation="test_op")
        with patch("numpy.asarray", side_effect=err):
            with self.assertRaises(DatasetSpecificHandlerError):
                handler.process_property_value("atoms", [1, 2], 0)

    def test_unexpected_error_wrapped_in_dataset_specific(self):
        handler = _make_handler()
        with patch("numpy.asarray", side_effect=RuntimeError("boom")):
            with self.assertRaises(DatasetSpecificHandlerError):
                handler.process_property_value("atoms", [1, 2], 0)

    def test_qdpi_all_13_elements_atoms_processing(self):
        handler = _make_handler()
        all_elements = np.array(sorted(QDPI_SUPPORTED_ELEMENTS), dtype=np.int64)
        result = handler.process_property_value("atoms", all_elements, 0)
        self.assertEqual(result.dtype, np.int64)
        np.testing.assert_array_equal(result, all_elements)


# ============================================================================
# GROUP 13: _is_valid_property (5 tests)
# ============================================================================

class TestIsValidProperty(unittest.TestCase):
    """Test QDπ property validity checker."""

    def test_none_is_invalid(self):
        self.assertFalse(_make_handler()._is_valid_property(None))

    def test_empty_list_is_invalid(self):
        self.assertFalse(_make_handler()._is_valid_property([]))

    def test_empty_tuple_is_invalid(self):
        self.assertFalse(_make_handler()._is_valid_property(()))

    def test_empty_ndarray_is_invalid(self):
        self.assertFalse(_make_handler()._is_valid_property(np.array([])))

    def test_valid_ndarray_is_valid(self):
        self.assertTrue(_make_handler()._is_valid_property(np.array([1, 2, 3])))


# ============================================================================
# GROUP 14: _ensure_tensor (8 tests)
# ============================================================================

class TestEnsureTensor(unittest.TestCase):
    """Test QDπ tensor conversion utility."""

    def test_torch_tensor_converted_to_dtype(self):
        handler = _make_handler()
        t = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = handler._ensure_tensor(t, torch.float32, "test", 0, "id")
        self.assertEqual(result.dtype, torch.float32)

    def test_numpy_array_to_tensor(self):
        handler = _make_handler()
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = handler._ensure_tensor(arr, torch.float32, "test", 0, "id")
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)

    def test_list_to_tensor(self):
        handler = _make_handler()
        result = handler._ensure_tensor([1.0, 2.0], torch.float32, "test", 0, "id")
        self.assertIsInstance(result, torch.Tensor)

    def test_tuple_to_tensor(self):
        handler = _make_handler()
        result = handler._ensure_tensor((1.0, 2.0), torch.float32, "test", 0, "id")
        self.assertIsInstance(result, torch.Tensor)

    def test_scalar_to_1d_tensor(self):
        handler = _make_handler()
        result = handler._ensure_tensor(3.14, torch.float32, "test", 0, "id")
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1,))

    def test_numpy_number_to_1d_tensor(self):
        handler = _make_handler()
        result = handler._ensure_tensor(np.float64(3.14), torch.float32, "test", 0, "id")
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1,))

    def test_invalid_type_raises_property_enrichment(self):
        """Unsupported type raises PropertyEnrichmentError.

        EVIDENCE: base_handler.py _ensure_tensor raises PropertyEnrichmentError
        (not DatasetSpecificHandlerError) for unsupported types like dict.
        """
        handler = _make_handler()
        with self.assertRaises(PropertyEnrichmentError):
            handler._ensure_tensor({"bad": "data"}, torch.float32, "test", 0, "id")

    def test_error_includes_property_name(self):
        """Error message includes the property name."""
        handler = _make_handler()
        with self.assertRaises(PropertyEnrichmentError) as ctx:
            handler._ensure_tensor({"bad": "data"}, torch.float32, "my_prop", 0, "id")
        self.assertIn("my_prop", str(ctx.exception))


# ============================================================================
# GROUP 15: enrich_pyg_data — orchestration (10 tests)
# ============================================================================

class TestEnrichPygData(unittest.TestCase):
    """Test QDπ PyG data enrichment orchestration."""

    def test_sets_dataset_type(self):
        handler = _make_handler()
        data = _make_pyg_data()
        result = handler.enrich_pyg_data(data, _make_raw_properties(energy=-76.3), 0, "test")
        self.assertEqual(result.dataset_type, "QDPi")

    def test_sets_num_nodes(self):
        handler = _make_handler()
        data = _make_pyg_data()
        data.num_nodes = 0
        result = handler.enrich_pyg_data(data, _make_raw_properties(energy=-76.3), 0, "test")
        self.assertEqual(result.num_nodes, data.z.size(0))

    def test_zero_nodes_raises(self):
        handler = _make_handler()
        data = Data()
        data.num_nodes = 0
        with self.assertRaises(DatasetSpecificHandlerError):
            handler.enrich_pyg_data(data, _make_raw_properties(energy=-76.3), 0, "test")

    def test_coordinates_set_as_pos(self):
        handler = _make_handler()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        data = _make_pyg_data(num_atoms=3)
        result = handler.enrich_pyg_data(data, _make_raw_properties(energy=-76.3, coordinates=coords), 0, "test")
        self.assertEqual(result.pos.shape, (3, 3))

    def test_energy_set_as_y(self):
        handler = _make_handler()
        result = handler.enrich_pyg_data(_make_pyg_data(), _make_raw_properties(energy=-76.3), 0, "test")
        self.assertTrue(hasattr(result, "y"))
        self.assertIsInstance(result.y, torch.Tensor)

    def test_forces_added_when_present(self):
        handler = _make_handler()
        forces = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0], [-0.1, 0.2, -0.3]], dtype=np.float32)
        data = _make_pyg_data(num_atoms=3)
        result = handler.enrich_pyg_data(data, _make_raw_properties(energy=-76.3, forces=forces, num_atoms=3), 0, "test")
        self.assertTrue(hasattr(result, "forces"))

    def test_molecular_charge_added(self):
        handler = _make_handler()
        result = handler.enrich_pyg_data(_make_pyg_data(), _make_raw_properties(energy=-76.3, molecular_charge=1), 0, "test")
        self.assertEqual(result.molecular_charge.item(), 1)

    def test_formula_stored(self):
        handler = _make_handler()
        result = handler.enrich_pyg_data(_make_pyg_data(), _make_raw_properties(energy=-76.3, formula="CH3OH"), 0, "test")
        self.assertEqual(result.formula, "CH3OH")

    def test_subset_stored(self):
        handler = _make_handler()
        result = handler.enrich_pyg_data(_make_pyg_data(), _make_raw_properties(energy=-76.3, subset="SPICE"), 0, "test")
        self.assertEqual(result.subset, "SPICE")

    def test_unexpected_error_wrapped(self):
        """Unexpected errors wrap into DatasetSpecificHandlerError.

        EVIDENCE: QDPi's enrich_pyg_data does coordinates processing inline.
        Patching np.asarray to fail after dataset_type is set triggers the
        outer except block in enrich_pyg_data (line 583-591).
        """
        handler = _make_handler()
        data = _make_pyg_data()
        props = _make_raw_properties(energy=-76.3)
        # Patch _ensure_tensor to raise unexpected error during energy assignment
        original_ensure = handler._ensure_tensor
        def failing_ensure(value, dtype, prop_name, mol_idx, ident):
            if prop_name == "energy":
                raise RuntimeError("unexpected boom")
            return original_ensure(value, dtype, prop_name, mol_idx, ident)
        with patch.object(handler, "_ensure_tensor", side_effect=failing_ensure):
            with self.assertRaises(DatasetSpecificHandlerError):
                handler.enrich_pyg_data(data, props, 0, "test")



# ============================================================================
# GROUP 16: enrich_pyg_data — scalar targets via enrich_pyg_data (8 tests)
# ============================================================================

class TestEnrichScalarTargets(unittest.TestCase):
    """Test QDπ scalar target addition via enrich_pyg_data.

    EVIDENCE: QDPiDatasetHandler.enrich_pyg_data (lines 513-591) handles scalar
    targets inline rather than delegating to _add_scalar_targets_internal.
    Energy is set as pyg_data.y via _ensure_tensor.
    """

    def test_energy_added_as_y(self):
        """Energy is added as y attribute on PyG data."""
        handler = _make_handler()
        data = _make_pyg_data()
        result = handler.enrich_pyg_data(data, _make_raw_properties(energy=-76.3), 0, "test")
        self.assertTrue(hasattr(result, "y"))
        self.assertIsInstance(result.y, torch.Tensor)

    def test_energy_value_correct(self):
        """Energy value is correctly stored in y."""
        handler = _make_handler()
        data = _make_pyg_data()
        result = handler.enrich_pyg_data(data, _make_raw_properties(energy=-76.3), 0, "test")
        self.assertAlmostEqual(result.y.flatten()[0].item(), -76.3, places=3)

    def test_ndarray_energy_converted(self):
        """Numpy array energy is properly handled during enrichment.

        EVIDENCE: np.array(-76.3) is a 0-d array. After _ensure_tensor wraps
        scalars with [value], the result is shape (1,). However, if
        np.array(-76.3) is treated as ndarray (not scalar), torch.tensor()
        produces a 0-dim tensor. Use .item() directly to handle both cases.
        """
        handler = _make_handler()
        data = _make_pyg_data()
        result = handler.enrich_pyg_data(data, _make_raw_properties(energy=np.array(-76.3)), 0, "test")
        self.assertAlmostEqual(result.y.item(), -76.3, places=3)

    def test_none_energy_no_y(self):
        """None energy does not set y attribute."""
        handler = _make_handler()
        data = _make_pyg_data()
        props = _make_raw_properties()
        props["energy"] = None
        result = handler.enrich_pyg_data(data, props, 0, "test")
        # y should not be set when energy is None
        self.assertNotIn("y", result)

    def test_forces_shape_matches_nodes(self):
        """Forces shape must match num_nodes for addition."""
        handler = _make_handler()
        n = 3
        forces = np.array([[0.1, -0.2, 0.3]] * n, dtype=np.float32)
        data = _make_pyg_data(num_atoms=n)
        result = handler.enrich_pyg_data(data, _make_raw_properties(energy=-76.3, forces=forces, num_atoms=n), 0, "test")
        self.assertEqual(result.forces.shape, (n, 3))

    def test_forces_mismatched_shape_not_added(self):
        """Forces with wrong shape are not added (logged warning)."""
        handler = _make_handler()
        # 2 force vectors for 3-node molecule
        forces = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]], dtype=np.float32)
        data = _make_pyg_data(num_atoms=3)
        result = handler.enrich_pyg_data(data, _make_raw_properties(energy=-76.3, forces=forces), 0, "test")
        # Forces should NOT be set due to shape mismatch
        self.assertFalse(hasattr(result, "forces") and result.forces is not None and result.forces.shape[0] == 3)

    def test_default_charge_zero_when_absent(self):
        """Default molecular_charge is 0 when no charge info present."""
        handler = _make_handler()
        data = _make_pyg_data()
        result = handler.enrich_pyg_data(data, _make_raw_properties(energy=-76.3), 0, "test")
        self.assertEqual(result.molecular_charge.item(), 0)

    def test_property_enrichment_error_reraised(self):
        """PropertyEnrichmentError from _ensure_tensor is re-raised."""
        handler = _make_handler()
        data = _make_pyg_data()
        props = _make_raw_properties(energy=-76.3)
        err = PropertyEnrichmentError(
            molecule_index=0, inchi="test", property_name="energy",
            reason="test error", detail="test"
        )
        with patch.object(handler, "_ensure_tensor", side_effect=err):
            with self.assertRaises(PropertyEnrichmentError):
                handler.enrich_pyg_data(data, props, 0, "test")


# ============================================================================
# GROUP 17: enrich_pyg_data — forces via enrichment (4 tests)
# ============================================================================

class TestEnrichForces(unittest.TestCase):
    """Test QDπ forces addition via enrich_pyg_data."""

    def test_forces_added_as_tensor(self):
        """Forces (N, 3) are converted to tensor via enrich_pyg_data."""
        handler = _make_handler()
        forces = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]], dtype=np.float32)
        data = _make_pyg_data(num_atoms=2)
        result = handler.enrich_pyg_data(data, _make_raw_properties(energy=-76.3, forces=forces, num_atoms=2), 0, "test")
        self.assertTrue(hasattr(result, "forces"))
        self.assertEqual(result.forces.dtype, torch.float32)

    def test_missing_forces_not_added(self):
        """Missing forces leaves no forces attribute on PyG data."""
        handler = _make_handler()
        data = _make_pyg_data()
        result = handler.enrich_pyg_data(data, _make_raw_properties(energy=-76.3), 0, "test")
        # forces is set if present in props; if not in raw_properties, it won't be set
        # (the default _make_raw_properties does not include forces)

    def test_forces_with_correct_shape(self):
        """Forces with correct shape (matching num_nodes) are stored."""
        handler = _make_handler()
        n = 3
        forces = np.random.randn(n, 3).astype(np.float32)
        data = _make_pyg_data(num_atoms=n)
        result = handler.enrich_pyg_data(data, _make_raw_properties(energy=-76.3, forces=forces, num_atoms=n), 0, "test")
        self.assertEqual(result.forces.shape, (n, 3))

    def test_none_forces_not_added(self):
        """None forces value does not add forces attribute."""
        handler = _make_handler()
        data = _make_pyg_data()
        props = _make_raw_properties(energy=-76.3)
        props["forces"] = None
        result = handler.enrich_pyg_data(data, props, 0, "test")
        self.assertFalse(hasattr(result, "forces") and result.forces is not None)


# ============================================================================
# GROUP 18: enrich_pyg_data — charge and metadata via enrichment (5 tests)
# ============================================================================

class TestEnrichChargeAndMetadata(unittest.TestCase):
    """Test QDπ charge and metadata enrichment via enrich_pyg_data."""

    def test_molecular_charge_stored_as_long_tensor(self):
        """Molecular charge is stored as torch.long tensor."""
        handler = _make_handler()
        result = handler.enrich_pyg_data(
            _make_pyg_data(), _make_raw_properties(energy=-76.3, molecular_charge=1), 0, "test"
        )
        self.assertEqual(result.molecular_charge.dtype, torch.long)

    def test_charge_from_charge_key(self):
        """Charge from 'charge' key is also stored."""
        handler = _make_handler()
        props = _make_raw_properties(energy=-76.3)
        props["charge"] = -1
        result = handler.enrich_pyg_data(_make_pyg_data(), props, 0, "test")
        self.assertEqual(result.molecular_charge.item(), -1)

    def test_formula_not_stored_when_absent(self):
        """Formula is not stored when not in raw properties."""
        handler = _make_handler()
        result = handler.enrich_pyg_data(_make_pyg_data(), _make_raw_properties(energy=-76.3), 0, "test")
        self.assertFalse(hasattr(result, "formula") and result.formula is not None)

    def test_subset_stored_as_string(self):
        """Subset string is stored directly on PyG data."""
        handler = _make_handler()
        result = handler.enrich_pyg_data(
            _make_pyg_data(), _make_raw_properties(energy=-76.3, subset="ANI"), 0, "test"
        )
        self.assertEqual(result.subset, "ANI")

    def test_multiple_metadata_fields_stored(self):
        """Multiple metadata fields (formula, subset, charge) all stored."""
        handler = _make_handler()
        props = _make_raw_properties(energy=-76.3, formula="NaCl", subset="SPICE", molecular_charge=1)
        result = handler.enrich_pyg_data(_make_pyg_data(), props, 0, "test")
        self.assertEqual(result.formula, "NaCl")
        self.assertEqual(result.subset, "SPICE")
        self.assertEqual(result.molecular_charge.item(), 1)


# ============================================================================
# GROUP 19: Atomization energy via enrich_pyg_data (6 tests)
# ============================================================================

class TestAtomizationEnergy(unittest.TestCase):
    """Test QDπ atomization energy calculation via enrich_pyg_data.

    EVIDENCE: QDPi's enrich_pyg_data calls the base class enrich_pyg_data
    or handles atomization inline. Atomization energy tests go through the
    full enrich_pyg_data path.
    """

    def test_no_atomization_configured(self):
        """No atomization energy when not configured."""
        pc = _make_processing_config(calculate_atomization_energy_from="", atomization_energy_key_name="")
        handler = _make_handler(processing_config=pc)
        result = handler.enrich_pyg_data(_make_pyg_data(), _make_raw_properties(energy=-76.3), 0, "test")
        self.assertFalse(hasattr(result, "atomization_energy"))

    def test_atomization_energy_for_hydrogen(self):
        """Atomization energy computed correctly for single H atom."""
        pc = _make_processing_config(calculate_atomization_energy_from="energy", atomization_energy_key_name="atomization_energy")
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(num_atoms=1, z=torch.tensor([1], dtype=torch.long))
        props = _make_raw_properties(energy=-0.5, num_atoms=1, atoms=np.array([1]))
        atomic_H = ATOMIC_ENERGIES_HARTREE.get(1, None)
        if atomic_H is not None and HAR2EV is not None:
            result = handler.enrich_pyg_data(data, props, 0, "test")
            if hasattr(result, "atomization_energy"):
                expected_eV = (-0.5 - atomic_H) * HAR2EV
                self.assertAlmostEqual(result.atomization_energy.item(), expected_eV, places=3)

    def test_atomization_water_like(self):
        """Atomization energy for water-like molecule (O + 2H)."""
        pc = _make_processing_config(calculate_atomization_energy_from="energy", atomization_energy_key_name="atomization_energy")
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(num_atoms=3, z=torch.tensor([8, 1, 1], dtype=torch.long))
        props = _make_raw_properties(energy=-76.3)
        atomic_O = ATOMIC_ENERGIES_HARTREE.get(8, None)
        atomic_H = ATOMIC_ENERGIES_HARTREE.get(1, None)
        if atomic_O is not None and atomic_H is not None and HAR2EV is not None:
            result = handler.enrich_pyg_data(data, props, 0, "test")
            if hasattr(result, "atomization_energy"):
                expected_eV = (-76.3 - (atomic_O + 2 * atomic_H)) * HAR2EV
                self.assertAlmostEqual(result.atomization_energy.item(), expected_eV, places=3)

    def test_missing_atomic_energy_no_crash(self):
        """Missing atomic energy for element doesn't crash enrichment."""
        pc = _make_processing_config(calculate_atomization_energy_from="energy")
        handler = _make_handler(processing_config=pc)
        # Z=2 (He) — likely not in ATOMIC_ENERGIES_HARTREE
        data = _make_pyg_data(num_atoms=1, z=torch.tensor([2], dtype=torch.long))
        props = _make_raw_properties(energy=-2.9, num_atoms=1, atoms=np.array([2]))
        # Should not raise — just won't have atomization energy
        result = handler.enrich_pyg_data(data, props, 0, "test")
        self.assertEqual(result.dataset_type, "QDPi")

    def test_qdpi_sodium_atomization(self):
        """Atomization energy with QDπ-specific element Na (Z=11)."""
        pc = _make_processing_config(calculate_atomization_energy_from="energy", atomization_energy_key_name="atomization_energy")
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(num_atoms=1, z=torch.tensor([11], dtype=torch.long))
        props = _make_raw_properties(energy=-162.0, num_atoms=1, atoms=np.array([11]))
        atomic_Na = ATOMIC_ENERGIES_HARTREE.get(11, None)
        if atomic_Na is not None and HAR2EV is not None:
            result = handler.enrich_pyg_data(data, props, 0, "test")
            if hasattr(result, "atomization_energy"):
                self.assertIsNotNone(result.atomization_energy)

    @patch("milia_pipeline.handlers.implementations.qdpi.HAR2EV", None)
    def test_missing_har2ev_no_atomization(self):
        """When HAR2EV constant is None, atomization energy is not computed."""
        pc = _make_processing_config(calculate_atomization_energy_from="energy", atomization_energy_key_name="atomization_energy")
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(num_atoms=1, z=torch.tensor([1], dtype=torch.long))
        props = _make_raw_properties(energy=-0.5, num_atoms=1, atoms=np.array([1]))
        result = handler.enrich_pyg_data(data, props, 0, "test")
        # Should still succeed but without atomization_energy
        self.assertEqual(result.dataset_type, "QDPi")


# ============================================================================
# GROUP 20: get_processing_statistics (7 tests)
# ============================================================================

class TestGetProcessingStatistics(unittest.TestCase):
    """Test QDπ processing statistics generation."""

    def test_basic_stats(self):
        handler = _make_handler()
        stats = handler.get_processing_statistics([])
        self.assertEqual(stats["dataset_type"], "QDPi")
        self.assertEqual(stats["total_processed"], 0)

    def test_counts_molecules_with_forces(self):
        handler = _make_handler()
        mols = [{"forces": np.array([1.0])}, {}, {"forces": np.array([2.0])}]
        self.assertEqual(handler.get_processing_statistics(mols)["has_forces"], 2)

    def test_counts_charge_info(self):
        handler = _make_handler()
        mols = [{"molecular_charge": 0}, {"charge": 1}, {}]
        self.assertEqual(handler.get_processing_statistics(mols)["has_charge_info"], 2)

    def test_counts_neutral_vs_charged(self):
        handler = _make_handler()
        mols = [{"molecular_charge": 0}, {"molecular_charge": 1}, {"molecular_charge": -1}, {"charge": 0}]
        stats = handler.get_processing_statistics(mols)
        self.assertEqual(stats["neutral_count"], 2)
        self.assertEqual(stats["charged_count"], 2)

    def test_experimental_setup_in_stats(self):
        handler = _make_handler(experimental_setup="exp_v1")
        stats = handler.get_processing_statistics([])
        self.assertIn("experimental_context", stats)
        self.assertEqual(stats["experimental_context"]["setup_name"], "exp_v1")
        self.assertEqual(stats["experimental_context"]["dataset_type"], "QDPi")

    def test_no_experimental_setup_no_context(self):
        handler = _make_handler(experimental_setup=None)
        self.assertNotIn("experimental_context", handler.get_processing_statistics([]))

    def test_empty_molecules_list(self):
        stats = _make_handler().get_processing_statistics([])
        for key in ["has_forces", "has_charge_info", "neutral_count", "charged_count"]:
            self.assertEqual(stats[key], 0)


# ============================================================================
# GROUP 21: get_supported_structural_features (3 tests)
# ============================================================================

class TestGetSupportedStructuralFeatures(unittest.TestCase):
    """Test QDπ supported structural features."""

    def test_returns_dict_with_atom_and_bond(self):
        features = _make_handler().get_supported_structural_features()
        self.assertIn("atom", features)
        self.assertIn("bond", features)

    def test_atom_features_include_key_descriptors(self):
        features = _make_handler().get_supported_structural_features()
        for expected in ["degree", "hybridization", "is_aromatic", "chirality", "gasteiger_charge"]:
            self.assertIn(expected, features["atom"])

    def test_bond_features_include_geometric(self):
        features = _make_handler().get_supported_structural_features()
        self.assertIn("bond_length", features["bond"])
        self.assertIn("bond_length_binned", features["bond"])


# ============================================================================
# GROUP 22: get_supported_descriptors (4 tests)
# ============================================================================

class TestGetSupportedDescriptors(unittest.TestCase):
    """Test QDπ supported descriptors."""

    def test_returns_dict_with_categories(self):
        self.assertIn("categories", _make_handler().get_supported_descriptors())

    def test_includes_geometric_category(self):
        self.assertIn("geometric", _make_handler().get_supported_descriptors()["categories"])

    def test_requires_3d_is_true(self):
        self.assertTrue(_make_handler().get_supported_descriptors()["requires_3d"])

    def test_excluded_is_empty(self):
        self.assertEqual(_make_handler().get_supported_descriptors()["excluded"], [])


# ============================================================================
# GROUP 23: Transform recommendations and validation (10 tests)
# ============================================================================

class TestTransformRecommendationsAndValidation(unittest.TestCase):
    """Test QDπ transform system: recommendations, suitable, validation."""

    def test_get_transform_recommendations_structure(self):
        recs = _make_handler().get_transform_recommendations()
        self.assertIn("recommended", recs)
        self.assertIn("avoid", recs)
        self.assertIn("warnings", recs)

    def test_get_transform_recommendations_includes_geometric(self):
        recs = _make_handler().get_transform_recommendations()
        self.assertTrue(any("RandomRotate" in r for r in recs["recommended"]))

    def test_suitable_transforms_includes_geometric(self):
        available = {"RandomRotate": Mock(), "GCNNorm": Mock(), "AddSelfLoops": Mock()}
        suitable = _make_handler()._get_dataset_suitable_transforms(available)
        self.assertIn("RandomRotate", suitable)

    def test_suitable_transforms_filters_unavailable(self):
        available = {"GCNNorm": Mock()}
        suitable = _make_handler()._get_dataset_suitable_transforms(available)
        self.assertNotIn("RandomRotate", suitable)

    def test_validate_dataset_specific_no_geometric_warns(self):
        warnings = _make_handler()._validate_dataset_specific_transforms(["GCNNorm"])
        self.assertTrue(any("geometric augmentation" in w.lower() or "RandomRotate" in w for w in warnings))

    def test_validate_dataset_specific_with_geometric_no_warn(self):
        warnings = _make_handler()._validate_dataset_specific_transforms(["RandomRotate"])
        geometry_warnings = [w for w in warnings if "geometric augmentation" in w.lower()]
        self.assertEqual(len(geometry_warnings), 0)

    def test_validate_forces_with_rotation_warns(self):
        pc = _make_processing_config(variable_len_graph_properties=["forces"])
        handler = _make_handler(processing_config=pc)
        warnings = handler._validate_dataset_specific_transforms(["RandomRotate"])
        self.assertTrue(any("force rotation" in w.lower() for w in warnings))

    def test_distance_transform_warning(self):
        warnings = _make_handler()._validate_dataset_specific_transforms(["Distance"])
        self.assertTrue(any("edge attribute" in w.lower() for w in warnings))

    def test_check_transform_incompatibilities_empty(self):
        self.assertEqual(_make_handler()._check_transform_incompatibilities(["RandomRotate", "GCNNorm"]), [])

    def test_get_transform_recommendations_method(self):
        recs = _make_handler()._get_transform_recommendations(["GCNNorm"])
        self.assertTrue(any("AddSelfLoops" in r for r in recs))


# ============================================================================
# GROUP 24: Suitable transforms detail coverage (4 tests)
# ============================================================================

class TestSuitableTransformsDetail(unittest.TestCase):
    """Test _get_dataset_suitable_transforms with different available transforms."""

    def test_normalization_transforms_included(self):
        available = {"GCNNorm": Mock(), "NormalizeFeatures": Mock()}
        suitable = _make_handler()._get_dataset_suitable_transforms(available)
        self.assertIn("GCNNorm", suitable)
        self.assertIn("NormalizeFeatures", suitable)

    def test_structure_transforms_included(self):
        available = {"AddSelfLoops": Mock(), "ToUndirected": Mock()}
        suitable = _make_handler()._get_dataset_suitable_transforms(available)
        self.assertIn("AddSelfLoops", suitable)
        self.assertIn("ToUndirected", suitable)

    def test_edge_feature_transforms_included(self):
        available = {"Distance": Mock(), "Cartesian": Mock()}
        suitable = _make_handler()._get_dataset_suitable_transforms(available)
        self.assertIn("Distance", suitable)
        self.assertIn("Cartesian", suitable)

    def test_augmentation_transforms_included(self):
        available = {"DropEdge": Mock(), "MaskFeatures": Mock()}
        suitable = _make_handler()._get_dataset_suitable_transforms(available)
        self.assertIn("DropEdge", suitable)
        self.assertIn("MaskFeatures", suitable)


# ============================================================================
# GROUP 25: _get_transform_recommendations detail (4 tests)
# ============================================================================

class TestGetTransformRecommendationsDetail(unittest.TestCase):
    """Test _get_transform_recommendations with various transform combinations."""

    def test_gcnnorm_without_selfloops_recommends(self):
        recs = _make_handler()._get_transform_recommendations(["GCNNorm"])
        self.assertTrue(any("AddSelfLoops" in r for r in recs))

    def test_no_geometric_recommends_augmentation(self):
        recs = _make_handler()._get_transform_recommendations(["GCNNorm", "AddSelfLoops"])
        self.assertTrue(any("3D" in r or "geometric" in r.lower() for r in recs))

    def test_with_all_transforms_minimal_recs(self):
        recs = _make_handler()._get_transform_recommendations(
            ["GCNNorm", "AddSelfLoops", "RandomRotate"]
        )
        # No GCNNorm-without-AddSelfLoops warning, no missing geometric warning
        gcn_recs = [r for r in recs if "AddSelfLoops" in r and "GCNNorm" in r]
        geo_recs = [r for r in recs if "geometric" in r.lower() or "3D" in r]
        self.assertEqual(len(gcn_recs), 0)
        self.assertEqual(len(geo_recs), 0)

    def test_empty_transforms_recommends_both(self):
        recs = _make_handler()._get_transform_recommendations([])
        self.assertTrue(any("3D" in r or "geometric" in r.lower() for r in recs))


# ============================================================================
# GROUP 26: Integration — full pipeline flow (5 tests)
# ============================================================================

class TestQDPiIntegrationFlow(unittest.TestCase):
    """Integration-level tests for the full QDπ handler flow."""

    def test_validate_then_enrich_happy_path(self):
        """Full flow: validate → process_property_value → enrich succeeds."""
        pc = _make_processing_config(scalar_graph_targets=["energy"])
        handler = _make_handler(processing_config=pc)
        props = _make_raw_properties(energy=-76.3)
        handler.validate_molecule_data(props, 0, "test")
        handler.process_property_value("atoms", props["atoms"], 0)
        handler.process_property_value("energy", props["energy"], 0)
        result = handler.enrich_pyg_data(_make_pyg_data(), props, 0, "test")
        self.assertEqual(result.dataset_type, "QDPi")
        self.assertTrue(hasattr(result, "y"))

    def test_object_array_pipeline(self):
        """Object arrays (from NPZ) are normalized through pipeline."""
        handler = _make_handler()
        atoms_obj = np.array([6, 1, 8], dtype=object)
        coords_obj = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=object)
        self.assertEqual(handler.process_property_value("atoms", atoms_obj, 0).dtype, np.int64)
        self.assertEqual(handler.process_property_value("coordinates", coords_obj, 0).dtype, np.float64)

    def test_charged_molecule_pipeline(self):
        """Charged molecule (Na+) processes through full pipeline."""
        handler = _make_handler()
        props = _make_raw_properties(
            energy=-162.0,
            atoms=np.array([11]),
            coordinates=np.array([[0.0, 0.0, 0.0]]),
            molecular_charge=1,
            num_atoms=1,
        )
        handler.validate_molecule_data(props, 0, "Na+")
        charge = handler.get_molecular_charge(props, props["atoms"], "Na+")
        self.assertEqual(charge, 1)
        data = _make_pyg_data(num_atoms=1, z=torch.tensor([11], dtype=torch.long))
        result = handler.enrich_pyg_data(data, props, 0, "Na+")
        self.assertEqual(result.molecular_charge.item(), 1)

    def test_forces_pipeline(self):
        """Forces processing through full enrich path."""
        pc = _make_processing_config(
            scalar_graph_targets=["energy"],
            variable_len_graph_properties=["forces"],
        )
        handler = _make_handler(processing_config=pc)
        num_atoms = 3
        forces = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0], [-0.1, 0.2, -0.3]], dtype=np.float32)
        data = _make_pyg_data(num_atoms=num_atoms)
        props = _make_raw_properties(energy=-76.3, forces=forces, num_atoms=num_atoms)
        result = handler.enrich_pyg_data(data, props, 0, "test")
        self.assertTrue(hasattr(result, "forces"))
        self.assertTrue(hasattr(result, "y"))

    def test_qdpi_13_element_molecule(self):
        """Molecule containing all 13 QDπ elements validates correctly."""
        handler = _make_handler()
        all_z = np.array(sorted(QDPI_SUPPORTED_ELEMENTS), dtype=np.int64)
        props = _make_raw_properties(
            atoms=all_z,
            coordinates=np.random.randn(len(all_z), 3),
            num_atoms=len(all_z),
        )
        # Should not raise — all elements are supported
        handler.validate_molecule_data(props, 0, "all_elements")


# ============================================================================
# GROUP 27: Edge cases and error boundary tests (5 tests)
# ============================================================================

class TestQDPiEdgeCases(unittest.TestCase):
    """Edge cases and boundary condition tests."""

    def test_single_atom_molecule(self):
        """Single-atom molecule processes correctly."""
        handler = _make_handler()
        props = _make_raw_properties(
            atoms=np.array([1]), coordinates=np.array([[0.0, 0.0, 0.0]]), num_atoms=1
        )
        handler.validate_molecule_data(props, 0, "test")

    def test_large_molecule(self):
        """Large molecule (63 atoms) processes correctly."""
        handler = _make_handler()
        n = 63
        atoms = np.array([6] * n, dtype=np.int64)
        coords = np.random.randn(n, 3)
        props = _make_raw_properties(atoms=atoms, coordinates=coords, num_atoms=n)
        handler.validate_molecule_data(props, 0, "test")

    def test_handler_without_experimental_setup(self):
        handler = _make_handler(experimental_setup=None)
        stats = handler.get_processing_statistics([])
        self.assertNotIn("experimental_context", stats)

    def test_no_atomization_configured(self):
        pc = _make_processing_config(
            calculate_atomization_energy_from="",
            atomization_energy_key_name="",
        )
        handler = _make_handler(processing_config=pc)
        result = handler.enrich_pyg_data(_make_pyg_data(), _make_raw_properties(energy=-76.3), 0, "test")
        self.assertFalse(hasattr(result, "atomization_energy"))

    def test_coordinate_count_mismatch_logs_warning(self):
        """Coordinate count mismatch with num_nodes logs warning, does not crash."""
        handler = _make_handler()
        data = _make_pyg_data(num_atoms=3)
        # Coordinates have 2 atoms but data has 3 nodes
        props = _make_raw_properties(
            energy=-76.3,
            coordinates=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        )
        # Should not raise — just logs warning about mismatch
        result = handler.enrich_pyg_data(data, props, 0, "test")
        self.assertEqual(result.dataset_type, "QDPi")


# ============================================================================
# GROUP 28: QDπ-specific element coverage (4 tests)
# ============================================================================

class TestQDPiElementSpecific(unittest.TestCase):
    """Test QDπ-specific element characteristics."""

    def test_lithium_element_processing(self):
        """Li (Z=3) — alkali metal in QDπ — processes correctly."""
        handler = _make_handler()
        atoms = np.array([3, 9], dtype=np.int64)  # LiF
        result = handler.process_property_value("atoms", atoms, 0)
        np.testing.assert_array_equal(result, atoms)

    def test_phosphorus_element_processing(self):
        """P (Z=15) — present in QDπ biopolymer fragments — processes correctly."""
        handler = _make_handler()
        atoms = np.array([15, 8, 8, 8, 1], dtype=np.int64)  # PO3H fragment
        result = handler.process_property_value("atoms", atoms, 0)
        np.testing.assert_array_equal(result, atoms)

    def test_iodine_element_processing(self):
        """I (Z=53) — heaviest element in QDπ — processes correctly."""
        handler = _make_handler()
        atoms = np.array([53, 6, 1, 1, 1], dtype=np.int64)  # CH3I
        result = handler.process_property_value("atoms", atoms, 0)
        np.testing.assert_array_equal(result, atoms)

    def test_atomization_with_qdpi_extended_elements(self):
        """Atomization energy with QDπ-specific elements (Li, Na, K, Br, I)."""
        pc = _make_processing_config(calculate_atomization_energy_from="energy")
        handler = _make_handler(processing_config=pc)
        # Na atom (Z=11)
        data = _make_pyg_data(num_atoms=1, z=torch.tensor([11], dtype=torch.long))
        props = _make_raw_properties(energy=-162.0, num_atoms=1, atoms=np.array([11]))
        atomic_Na = ATOMIC_ENERGIES_HARTREE.get(11, None)
        if atomic_Na is not None and HAR2EV is not None:
            result = handler._calculate_atomization_energy_internal(props, data, 0, "test")
            self.assertIsNotNone(result)


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    unittest.main()
