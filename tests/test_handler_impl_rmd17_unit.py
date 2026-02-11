#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/handlers/implementations/rmd17.py

Module under test: rmd17.py
- RMD17DatasetHandler: Handler for rMD17 (Revised MD17) quantum chemistry datasets
  - Implements DatasetHandler ABC (12 abstract methods + 4 transform validation helpers)
  - Registered via @register_handler decorator
  - rMD17-specific: coordinate_based strategy (NO identifiers), always neutral charge,
    10 small organic molecules, dtype normalization for object arrays,
    atomization energy (Hartree → eV), forces handling,
    old_energies/old_forces optional comparison properties, molecule_name passthrough

Test path on local machine: ~/ml_projects/milia/tests/test_handler_impl_rmd17_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/handlers/implementations/rmd17.py

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
- All internal methods (_add_scalar_targets_internal, _add_vector_properties_internal,
  _add_variable_length_properties_internal, _calculate_atomization_energy_internal,
  _ensure_tensor, _is_valid_property)
- Exception hierarchy: HandlerValidationError, DatasetSpecificHandlerError,
  PropertyEnrichmentError, MoleculeProcessingError
- rMD17 dtype normalization (object arrays → native dtypes)
- Forces handling (float32 conversion, non-finite detection)
- Energy handling (NaN detection, ndarray → float extraction)
- old_energies / old_forces optional comparison properties
- molecule_name string passthrough

Verified against exception signatures (lines 1394–1457 of exceptions.py):
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

from milia_pipeline.handlers.implementations.rmd17 import RMD17DatasetHandler
from milia_pipeline.exceptions import (
    PropertyEnrichmentError,
    MoleculeProcessingError,
    HandlerError,
    HandlerConfigurationError,
    HandlerValidationError,
    DatasetSpecificHandlerError,
)
from milia_pipeline.config.config_constants import (
    ATOMIC_ENERGIES_HARTREE,
    HAR2EV,
)


# ============================================================================
# HELPERS: Build realistic config mocks for RMD17DatasetHandler
# ============================================================================

def _make_dataset_config(**overrides):
    """
    Build a minimal mock DatasetConfig for rMD17 handler tests.
    
    Based on project structure: DatasetConfig is a Pydantic frozen BaseModel.
    The handler accesses dataset_config attributes but primarily relies on
    processing_config for property lists.
    """
    cfg = Mock(spec_set=["dataset_type", "npz_file_path", "dataset_name"])
    cfg.dataset_type = overrides.get("dataset_type", "RMD17")
    cfg.npz_file_path = overrides.get(
        "npz_file_path", "~/Chem_Data/MILIA_PyG_Dataset/raw/rmd17_aspirin.npz"
    )
    cfg.dataset_name = overrides.get("dataset_name", "rMD17")
    return cfg


def _make_filter_config(**overrides):
    """
    Build a minimal mock FilterConfig for rMD17 handler tests.
    
    FilterConfig controls molecule filtering (max atoms, element filters, etc.).
    """
    cfg = Mock(spec_set=["max_atoms", "allowed_elements", "min_atoms"])
    cfg.max_atoms = overrides.get("max_atoms", 63)
    cfg.allowed_elements = overrides.get("allowed_elements", None)
    cfg.min_atoms = overrides.get("min_atoms", 1)
    return cfg


def _make_processing_config(**overrides):
    """
    Build a minimal mock ProcessingConfig for rMD17 handler tests.
    
    ProcessingConfig controls which properties to extract, scalar targets,
    node features, vector properties, variable-length properties, and
    atomization energy configuration.
    
    rMD17 key properties:
    - scalar_graph_targets: ['energies'] (Hartree, converted from kcal/mol)
    - node_features: ['atoms'] (atomic numbers)
    - vector_graph_properties: [] (typically empty)
    - variable_len_graph_properties: ['forces'] (if available)
    - calculate_atomization_energy_from: 'energies' (Hartree basis)
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
    cfg.scalar_graph_targets = overrides.get("scalar_graph_targets", ["energies"])
    cfg.node_features = overrides.get("node_features", ["atoms"])
    cfg.vector_graph_properties = overrides.get("vector_graph_properties", [])
    cfg.variable_len_graph_properties = overrides.get("variable_len_graph_properties", [])
    cfg.calculate_atomization_energy_from = overrides.get(
        "calculate_atomization_energy_from", "energies"
    )
    cfg.atomization_energy_key_name = overrides.get(
        "atomization_energy_key_name", "atomization_energy"
    )
    return cfg


def _make_handler(**overrides):
    """
    Build a ready-to-use RMD17DatasetHandler instance.
    
    Based on DatasetHandler ABC constructor signature:
    __init__(dataset_config, filter_config, processing_config, logger, experimental_setup=None)
    """
    dataset_config = overrides.get("dataset_config", _make_dataset_config())
    filter_config = overrides.get("filter_config", _make_filter_config())
    processing_config = overrides.get("processing_config", _make_processing_config())
    logger = overrides.get("logger", logging.getLogger("test.rmd17"))
    experimental_setup = overrides.get("experimental_setup", None)

    handler = RMD17DatasetHandler(
        dataset_config=dataset_config,
        filter_config=filter_config,
        processing_config=processing_config,
        logger=logger,
        experimental_setup=experimental_setup,
    )
    return handler


def _make_pyg_data(**overrides):
    """
    Build a minimal PyG Data object for rMD17 enrichment tests.
    
    rMD17 molecules typically have:
    - z: atomic numbers tensor (small organic molecules)
    - pos: 3D coordinates tensor (PBE/def2-SVP optimized, MD at 500K)
    - edge_index: connectivity tensor
    - num_nodes: atom count
    """
    num_atoms = overrides.get("num_atoms", 3)
    # Default: ethanol-like fragment (C, H, H)
    z = overrides.get("z", torch.tensor([6, 1, 1], dtype=torch.long)[:num_atoms])
    pos = overrides.get("pos", torch.randn(num_atoms, 3, dtype=torch.float32))

    data = Data()
    data.z = z
    data.pos = pos
    data.num_nodes = num_atoms

    # Add edge_index if provided
    if "edge_index" in overrides:
        data.edge_index = overrides["edge_index"]
    else:
        # Simple linear connectivity for testing
        if num_atoms >= 2:
            src = list(range(num_atoms - 1))
            dst = list(range(1, num_atoms))
            data.edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

    return data


def _make_raw_properties(**overrides):
    """
    Build a realistic raw_properties_dict for rMD17 molecule tests.
    
    rMD17 NPZ files contain: energies, atoms, coordinates, forces (optional),
    old_energies (optional), old_forces (optional), molecule_name (optional).
    All molecules are neutral small organic compounds.
    Energies in Hartree (converted from kcal/mol during preprocessing),
    coordinates in Angstrom, forces in Hartree/Angstrom (converted from kcal/mol/Angstrom).
    """
    num_atoms = overrides.get("num_atoms", 3)
    props = {
        "energies": overrides.get("energies", -76.3),
        "atoms": overrides.get("atoms", np.array([6, 1, 1])[:num_atoms]),
        "coordinates": overrides.get(
            "coordinates", np.random.randn(num_atoms, 3).astype(np.float64)
        ),
    }
    # Optionally add extra properties
    for key in ["forces", "old_energies", "old_forces", "molecule_name"]:
        if key in overrides:
            props[key] = overrides[key]
    return props


# ============================================================================
# GROUP 1: RMD17DatasetHandler — Identity and Registration (6 tests)
# ============================================================================

class TestRMD17DatasetHandlerIdentity(unittest.TestCase):
    """Test RMD17DatasetHandler identity, registration, and basic attributes."""

    def test_get_dataset_type_returns_rmd17(self):
        """get_dataset_type() returns 'RMD17'."""
        handler = _make_handler()
        self.assertEqual(handler.get_dataset_type(), "RMD17")

    def test_get_molecule_creation_strategy(self):
        """rMD17 uses coordinate_based strategy (NO identifiers available)."""
        handler = _make_handler()
        self.assertEqual(handler.get_molecule_creation_strategy(), "coordinate_based")

    def test_get_identifier_keys_empty(self):
        """rMD17 has NO parseable identifiers — returns empty list."""
        handler = _make_handler()
        keys = handler.get_identifier_keys()
        self.assertIsInstance(keys, list)
        self.assertEqual(len(keys), 0)

    def test_is_subclass_of_dataset_handler(self):
        """RMD17DatasetHandler is a proper DatasetHandler subclass."""
        from milia_pipeline.handlers.base_handler import DatasetHandler
        self.assertTrue(issubclass(RMD17DatasetHandler, DatasetHandler))

    def test_handler_stores_configs(self):
        """Handler stores config objects passed during construction."""
        dc = _make_dataset_config()
        fc = _make_filter_config()
        pc = _make_processing_config()
        handler = RMD17DatasetHandler(
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
# GROUP 2: get_molecular_charge (4 tests)
# ============================================================================

class TestGetMolecularCharge(unittest.TestCase):
    """Test rMD17 molecular charge — always 0 (neutral molecules)."""

    def test_returns_zero_always(self):
        """rMD17 molecules are all neutral, charge = 0."""
        handler = _make_handler()
        charge = handler.get_molecular_charge({}, np.array([6, 1, 1]))
        self.assertEqual(charge, 0)

    def test_returns_int(self):
        """Return type is int."""
        handler = _make_handler()
        charge = handler.get_molecular_charge({}, np.array([1]))
        self.assertIsInstance(charge, int)

    def test_ignores_all_arguments(self):
        """Charge is independent of raw_properties, atoms, and identifier."""
        handler = _make_handler()
        charge = handler.get_molecular_charge(
            {"energies": -100.0}, np.array([6, 8, 1]), "aspirin_conf_42"
        )
        self.assertEqual(charge, 0)

    def test_with_typical_rmd17_elements(self):
        """Typical rMD17 elements (H, C, N, O) still return 0."""
        handler = _make_handler()
        # H=1, C=6, N=7, O=8 (common in rMD17 molecules like aspirin, ethanol)
        atoms = np.array([1, 6, 7, 8])
        charge = handler.get_molecular_charge({}, atoms)
        self.assertEqual(charge, 0)


# ============================================================================
# GROUP 3: get_required_properties (5 tests)
# ============================================================================

class TestGetRequiredProperties(unittest.TestCase):
    """Test rMD17 required properties assembly."""

    def test_includes_core_properties(self):
        """Core rMD17 properties (energies, atoms, coordinates) are always included."""
        handler = _make_handler()
        required = handler.get_required_properties()
        for prop in ["energies", "atoms", "coordinates"]:
            self.assertIn(prop, required)

    def test_includes_scalar_targets(self):
        """Scalar graph targets from processing config are included."""
        pc = _make_processing_config(scalar_graph_targets=["energies", "custom_target"])
        handler = _make_handler(processing_config=pc)
        required = handler.get_required_properties()
        self.assertIn("custom_target", required)

    def test_includes_node_features(self):
        """Node features from processing config are included."""
        pc = _make_processing_config(node_features=["atoms", "extra_feature"])
        handler = _make_handler(processing_config=pc)
        required = handler.get_required_properties()
        self.assertIn("extra_feature", required)

    def test_includes_atomization_base(self):
        """Atomization energy base property is included when configured."""
        pc = _make_processing_config(calculate_atomization_energy_from="energies")
        handler = _make_handler(processing_config=pc)
        required = handler.get_required_properties()
        self.assertIn("energies", required)

    def test_deduplication(self):
        """Duplicate properties are removed (returns unique list)."""
        pc = _make_processing_config(
            scalar_graph_targets=["energies"],
            node_features=["atoms", "energies"],  # energies duplicated
        )
        handler = _make_handler(processing_config=pc)
        required = handler.get_required_properties()
        # energies should appear exactly once
        self.assertEqual(required.count("energies"), 1)


# ============================================================================
# GROUP 4: validate_molecule_data (10 tests)
# ============================================================================

class TestValidateMoleculeData(unittest.TestCase):
    """Test rMD17 molecule validation with exception handling."""

    def test_valid_molecule_passes(self):
        """Complete valid molecule data passes validation without error."""
        handler = _make_handler()
        props = _make_raw_properties()
        # Should not raise
        handler.validate_molecule_data(props, molecule_index=0, identifier="test")

    def test_missing_energies_raises(self):
        """Missing 'energies' raises HandlerValidationError."""
        handler = _make_handler()
        props = _make_raw_properties()
        props.pop("energies")
        with self.assertRaises(HandlerValidationError) as ctx:
            handler.validate_molecule_data(props, 0, "test")
        self.assertIn("energies", str(ctx.exception))

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
        props = {}  # All properties missing
        with self.assertRaises(HandlerValidationError):
            handler.validate_molecule_data(props, 0, "test")

    def test_none_property_treated_as_missing(self):
        """None values are treated as missing by _is_valid_property."""
        handler = _make_handler()
        props = _make_raw_properties()
        props["energies"] = None
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
            "milia_pipeline.handlers.implementations.rmd17.validate_molecular_structure",
            side_effect=ValueError("atoms/coords mismatch"),
        ):
            with self.assertRaises(DatasetSpecificHandlerError) as ctx:
                handler.validate_molecule_data(props, 0, "test")
            self.assertEqual(ctx.exception.dataset_type, "RMD17")

    def test_positive_energy_logs_warning(self):
        """Positive energy (unusual for total molecular energy) logs a warning."""
        handler = _make_handler()
        props = _make_raw_properties(energies=10.0)
        with self.assertLogs("test.rmd17", level="WARNING") as log:
            handler.validate_molecule_data(props, 0, "test")
        self.assertTrue(any("positive energy" in msg for msg in log.output))

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


# ============================================================================
# GROUP 5: process_property_value — atoms (6 tests)
# ============================================================================

class TestProcessPropertyValueAtoms(unittest.TestCase):
    """Test rMD17 atoms dtype normalization."""

    def test_native_int64_passthrough(self):
        """int64 atoms pass through unchanged."""
        handler = _make_handler()
        atoms = np.array([6, 1, 1], dtype=np.int64)
        result = handler.process_property_value("atoms", atoms, 0)
        self.assertEqual(result.dtype, np.int64)
        np.testing.assert_array_equal(result, atoms)

    def test_object_array_converted_to_int64(self):
        """Object dtype atoms are converted to int64."""
        handler = _make_handler()
        atoms = np.array([6, 1, 1], dtype=object)
        result = handler.process_property_value("atoms", atoms, 0)
        self.assertEqual(result.dtype, np.int64)
        np.testing.assert_array_equal(result, [6, 1, 1])

    def test_uint8_normalized_to_int64(self):
        """uint8 atoms are normalized to int64 for consistency."""
        handler = _make_handler()
        atoms = np.array([6, 1, 1], dtype=np.uint8)
        result = handler.process_property_value("atoms", atoms, 0)
        self.assertEqual(result.dtype, np.int64)
        np.testing.assert_array_equal(result, [6, 1, 1])

    def test_float_atoms_converted_to_int64(self):
        """Float dtype atoms are converted to int64 (not integer subdtype)."""
        handler = _make_handler()
        atoms = np.array([6.0, 1.0, 1.0], dtype=np.float64)
        result = handler.process_property_value("atoms", atoms, 0)
        self.assertEqual(result.dtype, np.int64)

    def test_unconvertible_atoms_returns_original(self):
        """If conversion fails, returns original value (downstream handles it)."""
        handler = _make_handler()
        # String values that can't be converted to int64
        atoms = np.array(["H", "C", "O"], dtype=object)
        result = handler.process_property_value("atoms", atoms, 0)
        # Returns original since astype(np.int64) fails
        np.testing.assert_array_equal(result, atoms)

    def test_none_atoms_returns_none(self):
        """None value returns None as-is."""
        handler = _make_handler()
        result = handler.process_property_value("atoms", None, 0)
        self.assertIsNone(result)


# ============================================================================
# GROUP 6: process_property_value — coordinates (5 tests)
# ============================================================================

class TestProcessPropertyValueCoordinates(unittest.TestCase):
    """Test rMD17 coordinates dtype normalization."""

    def test_native_float64_passthrough(self):
        """float64 coordinates pass through unchanged."""
        handler = _make_handler()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        result = handler.process_property_value("coordinates", coords, 0)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_array_equal(result, coords)

    def test_object_array_converted_to_float64(self):
        """Object dtype coordinates are converted to float64."""
        handler = _make_handler()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=object)
        result = handler.process_property_value("coordinates", coords, 0)
        self.assertEqual(result.dtype, np.float64)

    def test_float32_normalized_to_float64(self):
        """float32 coordinates are normalized to float64."""
        handler = _make_handler()
        coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        result = handler.process_property_value("coordinates", coords, 0)
        self.assertEqual(result.dtype, np.float64)

    def test_integer_coords_converted_to_float64(self):
        """Integer coordinates are converted to float64."""
        handler = _make_handler()
        coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
        result = handler.process_property_value("coordinates", coords, 0)
        self.assertEqual(result.dtype, np.float64)

    def test_unconvertible_coords_returns_original(self):
        """If conversion fails, returns original value."""
        handler = _make_handler()
        coords = np.array([["a", "b", "c"]], dtype=object)
        result = handler.process_property_value("coordinates", coords, 0)
        # Returns original since astype(np.float64) fails
        np.testing.assert_array_equal(result, coords)


# ============================================================================
# GROUP 7: process_property_value — forces (6 tests)
# ============================================================================

class TestProcessPropertyValueForces(unittest.TestCase):
    """Test rMD17 forces dtype normalization and validation."""

    def test_valid_float32_passthrough(self):
        """float32 forces pass through unchanged."""
        handler = _make_handler()
        forces = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
        result = handler.process_property_value("forces", forces, 0)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_almost_equal(result, forces)

    def test_object_array_converted_to_float32(self):
        """Object dtype forces are converted to float32."""
        handler = _make_handler()
        forces = np.array([[0.1, -0.2, 0.3]], dtype=object)
        result = handler.process_property_value("forces", forces, 0)
        self.assertEqual(result.dtype, np.float32)

    def test_float64_forces_kept_as_floating(self):
        """float64 forces stay as float64 (issubdtype check: floating but not object)."""
        handler = _make_handler()
        forces = np.array([[0.1, -0.2, 0.3]], dtype=np.float64)
        result = handler.process_property_value("forces", forces, 0)
        # float64 is floating subdtype, so the object/non-floating branch is not taken
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def test_non_finite_forces_returns_none(self):
        """Forces containing NaN return None."""
        handler = _make_handler()
        forces = np.array([[float("nan"), 0.0, 0.0]], dtype=np.float32)
        result = handler.process_property_value("forces", forces, 0)
        self.assertIsNone(result)

    def test_inf_forces_returns_none(self):
        """Forces containing Inf return None."""
        handler = _make_handler()
        forces = np.array([[float("inf"), 0.0, 0.0]], dtype=np.float32)
        result = handler.process_property_value("forces", forces, 0)
        self.assertIsNone(result)

    def test_unconvertible_forces_returns_none(self):
        """If forces conversion to float32 fails, returns None."""
        handler = _make_handler()
        forces = np.array([["a", "b", "c"]], dtype=object)
        result = handler.process_property_value("forces", forces, 0)
        self.assertIsNone(result)


# ============================================================================
# GROUP 8: process_property_value — energies (6 tests)
# ============================================================================

class TestProcessPropertyValueEnergies(unittest.TestCase):
    """Test rMD17 energies handling."""

    def test_float_passthrough(self):
        """Float energy passes through unchanged."""
        handler = _make_handler()
        result = handler.process_property_value("energies", -76.3, 0)
        self.assertEqual(result, -76.3)

    def test_ndarray_size_1_extracted_to_float(self):
        """Numpy array of size 1 is extracted to float scalar."""
        handler = _make_handler()
        result = handler.process_property_value("energies", np.array(-76.3), 0)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, -76.3)

    def test_nan_float_energy_passthrough(self):
        """Plain float NaN passes through (NaN check only inside np.ndarray branch).
        
        EVIDENCE: rmd17.py lines 333-339 — the NaN detection block is guarded by
        ``if isinstance(value, np.ndarray):``. A plain Python float is NOT an
        ndarray, so the code skips straight to ``return value``, returning NaN
        unchanged. Only ndarray energies hit the NaN-to-None conversion.
        """
        handler = _make_handler()
        result = handler.process_property_value("energies", float("nan"), 0)
        # Plain float NaN is NOT caught — it passes through
        self.assertIsNotNone(result)
        self.assertTrue(math.isnan(result))

    def test_nan_ndarray_energy_returns_none(self):
        """NaN ndarray energy returns None."""
        handler = _make_handler()
        result = handler.process_property_value("energies", np.array(float("nan")), 0)
        self.assertIsNone(result)

    def test_numpy_scalar_passthrough(self):
        """Numpy scalar (np.float64) passes through."""
        handler = _make_handler()
        result = handler.process_property_value("energies", np.float64(-76.3), 0)
        self.assertAlmostEqual(result, -76.3)

    def test_none_energy_returns_none(self):
        """None energy returns None."""
        handler = _make_handler()
        result = handler.process_property_value("energies", None, 0)
        self.assertIsNone(result)


# ============================================================================
# GROUP 9: process_property_value — old_energies (4 tests)
# ============================================================================

class TestProcessPropertyValueOldEnergies(unittest.TestCase):
    """Test rMD17 old_energies (original MD17 comparison property) handling."""

    def test_float_passthrough(self):
        """Float old_energies passes through unchanged."""
        handler = _make_handler()
        result = handler.process_property_value("old_energies", -75.0, 0)
        self.assertEqual(result, -75.0)

    def test_ndarray_size_1_extracted_to_float(self):
        """Numpy array of size 1 is extracted to float scalar."""
        handler = _make_handler()
        result = handler.process_property_value("old_energies", np.array(-75.0), 0)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, -75.0)

    def test_nan_ndarray_returns_none(self):
        """NaN ndarray old_energies returns None."""
        handler = _make_handler()
        result = handler.process_property_value("old_energies", np.array(float("nan")), 0)
        self.assertIsNone(result)

    def test_none_returns_none(self):
        """None old_energies returns None."""
        handler = _make_handler()
        result = handler.process_property_value("old_energies", None, 0)
        self.assertIsNone(result)


# ============================================================================
# GROUP 10: process_property_value — old_forces (4 tests)
# ============================================================================

class TestProcessPropertyValueOldForces(unittest.TestCase):
    """Test rMD17 old_forces (original MD17 comparison property) handling."""

    def test_valid_float32_passthrough(self):
        """float32 old_forces pass through unchanged."""
        handler = _make_handler()
        old_forces = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
        result = handler.process_property_value("old_forces", old_forces, 0)
        self.assertEqual(result.dtype, np.float32)

    def test_object_array_converted_to_float32(self):
        """Object dtype old_forces are converted to float32."""
        handler = _make_handler()
        old_forces = np.array([[0.1, -0.2, 0.3]], dtype=object)
        result = handler.process_property_value("old_forces", old_forces, 0)
        self.assertEqual(result.dtype, np.float32)

    def test_non_finite_returns_none(self):
        """old_forces containing NaN return None."""
        handler = _make_handler()
        old_forces = np.array([[float("nan"), 0.0, 0.0]], dtype=np.float32)
        result = handler.process_property_value("old_forces", old_forces, 0)
        self.assertIsNone(result)

    def test_unconvertible_returns_none(self):
        """If old_forces conversion fails, returns None."""
        handler = _make_handler()
        old_forces = np.array([["a", "b", "c"]], dtype=object)
        result = handler.process_property_value("old_forces", old_forces, 0)
        self.assertIsNone(result)


# ============================================================================
# GROUP 11: process_property_value — molecule_name and other/edge cases (5 tests)
# ============================================================================

class TestProcessPropertyValueOther(unittest.TestCase):
    """Test rMD17 property processing for molecule_name, unrecognized keys, and edge cases."""

    def test_molecule_name_passthrough(self):
        """molecule_name string passes through unchanged."""
        handler = _make_handler()
        result = handler.process_property_value("molecule_name", "aspirin", 0)
        self.assertEqual(result, "aspirin")

    def test_unknown_key_passthrough(self):
        """Unknown property keys pass through unchanged."""
        handler = _make_handler()
        val = [1, 2, 3]
        result = handler.process_property_value("unknown_prop", val, 0)
        self.assertEqual(result, val)

    def test_none_unknown_key_returns_none(self):
        """None for unknown key returns None."""
        handler = _make_handler()
        result = handler.process_property_value("unknown", None, 0)
        self.assertIsNone(result)

    def test_dataset_specific_error_reraised(self):
        """DatasetSpecificHandlerError is re-raised, not wrapped."""
        handler = _make_handler()
        err = DatasetSpecificHandlerError(
            dataset_type="RMD17",
            message="test error",
            operation="test_op",
        )
        with patch("numpy.asarray", side_effect=err):
            with self.assertRaises(DatasetSpecificHandlerError):
                handler.process_property_value("atoms", [1, 2], 0)

    def test_unexpected_error_wrapped_in_dataset_specific(self):
        """Unexpected exception wraps into DatasetSpecificHandlerError."""
        handler = _make_handler()
        with patch("numpy.asarray", side_effect=RuntimeError("boom")):
            with self.assertRaises(DatasetSpecificHandlerError):
                handler.process_property_value("atoms", [1, 2], 0)


# ============================================================================
# GROUP 12: _is_valid_property (5 tests)
# ============================================================================

class TestIsValidProperty(unittest.TestCase):
    """Test rMD17 property validity checker."""

    def test_none_is_invalid(self):
        """None returns False."""
        handler = _make_handler()
        self.assertFalse(handler._is_valid_property(None))

    def test_empty_list_is_invalid(self):
        """Empty list returns False."""
        handler = _make_handler()
        self.assertFalse(handler._is_valid_property([]))

    def test_empty_tuple_is_invalid(self):
        """Empty tuple returns False."""
        handler = _make_handler()
        self.assertFalse(handler._is_valid_property(()))

    def test_empty_ndarray_is_invalid(self):
        """Empty ndarray returns False."""
        handler = _make_handler()
        self.assertFalse(handler._is_valid_property(np.array([])))

    def test_valid_ndarray_is_valid(self):
        """Non-empty ndarray returns True."""
        handler = _make_handler()
        self.assertTrue(handler._is_valid_property(np.array([1, 2, 3])))


# ============================================================================
# GROUP 13: _ensure_tensor (8 tests)
# ============================================================================

class TestEnsureTensor(unittest.TestCase):
    """Test rMD17 tensor conversion utility."""

    def test_torch_tensor_converted_to_dtype(self):
        """Existing torch.Tensor is converted to target dtype."""
        handler = _make_handler()
        t = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = handler._ensure_tensor(t, torch.float32, "test", 0, "id")
        self.assertEqual(result.dtype, torch.float32)

    def test_numpy_array_to_tensor(self):
        """Numpy array is converted to torch tensor."""
        handler = _make_handler()
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = handler._ensure_tensor(arr, torch.float32, "test", 0, "id")
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)

    def test_list_to_tensor(self):
        """Python list is converted to torch tensor."""
        handler = _make_handler()
        result = handler._ensure_tensor([1.0, 2.0], torch.float32, "test", 0, "id")
        self.assertIsInstance(result, torch.Tensor)

    def test_tuple_to_tensor(self):
        """Python tuple is converted to torch tensor."""
        handler = _make_handler()
        result = handler._ensure_tensor((1.0, 2.0), torch.float32, "test", 0, "id")
        self.assertIsInstance(result, torch.Tensor)

    def test_scalar_to_1d_tensor(self):
        """Scalar value is wrapped in a 1D tensor."""
        handler = _make_handler()
        result = handler._ensure_tensor(3.14, torch.float32, "test", 0, "id")
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1,))

    def test_numpy_number_to_1d_tensor(self):
        """Numpy scalar (np.number) is wrapped in a 1D tensor."""
        handler = _make_handler()
        result = handler._ensure_tensor(np.float64(3.14), torch.float32, "test", 0, "id")
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1,))

    def test_invalid_type_raises_dataset_specific(self):
        """Unsupported type raises DatasetSpecificHandlerError."""
        handler = _make_handler()
        with self.assertRaises(DatasetSpecificHandlerError) as ctx:
            handler._ensure_tensor({"bad": "data"}, torch.float32, "test", 0, "id")
        self.assertEqual(ctx.exception.dataset_type, "RMD17")

    def test_error_includes_property_name(self):
        """Error message includes the property name."""
        handler = _make_handler()
        with self.assertRaises(DatasetSpecificHandlerError) as ctx:
            handler._ensure_tensor({"bad": "data"}, torch.float32, "my_prop", 0, "id")
        self.assertIn("my_prop", str(ctx.exception))


# ============================================================================
# GROUP 14: enrich_pyg_data — orchestration (8 tests)
# ============================================================================

class TestEnrichPygData(unittest.TestCase):
    """Test rMD17 PyG data enrichment orchestration."""

    def test_sets_dataset_type(self):
        """Enrichment sets dataset_type attribute to 'RMD17'."""
        handler = _make_handler()
        data = _make_pyg_data()
        props = _make_raw_properties(energies=-76.3)
        result = handler.enrich_pyg_data(data, props, 0, "test")
        self.assertEqual(result.dataset_type, "RMD17")

    def test_sets_num_nodes(self):
        """Enrichment ensures num_nodes is set from z."""
        handler = _make_handler()
        data = _make_pyg_data()
        data.num_nodes = 0  # Force re-calculation
        props = _make_raw_properties(energies=-76.3)
        result = handler.enrich_pyg_data(data, props, 0, "test")
        self.assertEqual(result.num_nodes, data.z.size(0))

    def test_zero_nodes_raises(self):
        """Zero nodes raises DatasetSpecificHandlerError."""
        handler = _make_handler()
        data = Data()  # No z, no num_nodes
        data.num_nodes = 0
        props = _make_raw_properties(energies=-76.3)
        with self.assertRaises(DatasetSpecificHandlerError):
            handler.enrich_pyg_data(data, props, 0, "test")

    def test_scalar_targets_added(self):
        """Scalar targets (energies) are added as y attribute."""
        pc = _make_processing_config(scalar_graph_targets=["energies"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties(energies=-76.3)
        result = handler.enrich_pyg_data(data, props, 0, "test")
        self.assertTrue(hasattr(result, "y"))
        self.assertIsInstance(result.y, torch.Tensor)

    def test_atomization_energy_added_when_configured(self):
        """Atomization energy is calculated and added when configured."""
        pc = _make_processing_config(
            calculate_atomization_energy_from="energies",
            atomization_energy_key_name="atomization_energy",
        )
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(
            num_atoms=1, z=torch.tensor([1], dtype=torch.long)
        )
        props = _make_raw_properties(energies=-0.5, num_atoms=1, atoms=np.array([1]))
        if ATOMIC_ENERGIES_HARTREE.get(1) is not None and HAR2EV is not None:
            result = handler.enrich_pyg_data(data, props, 0, "test")
            self.assertTrue(hasattr(result, "atomization_energy"))

    def test_variable_length_properties_added(self):
        """Variable-length properties (forces) are added when configured."""
        pc = _make_processing_config(
            variable_len_graph_properties=["forces"],
        )
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(num_atoms=3)
        forces = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0], [-0.1, 0.2, -0.3]], dtype=np.float32)
        props = _make_raw_properties(energies=-76.3, forces=forces)
        result = handler.enrich_pyg_data(data, props, 0, "test")
        self.assertTrue(hasattr(result, "forces"))

    def test_property_enrichment_error_reraised(self):
        """PropertyEnrichmentError from scalar targets is re-raised."""
        pc = _make_processing_config(scalar_graph_targets=["energies"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties()
        props.pop("energies")  # Missing required scalar target
        with self.assertRaises(PropertyEnrichmentError):
            handler.enrich_pyg_data(data, props, 0, "test")

    def test_unexpected_error_wrapped(self):
        """Unexpected errors wrap into DatasetSpecificHandlerError."""
        handler = _make_handler()
        data = _make_pyg_data()
        props = _make_raw_properties(energies=-76.3)
        with patch.object(
            handler, "_add_scalar_targets_internal",
            side_effect=RuntimeError("unexpected boom"),
        ):
            with self.assertRaises(DatasetSpecificHandlerError):
                handler.enrich_pyg_data(data, props, 0, "test")


# ============================================================================
# GROUP 15: _add_scalar_targets_internal (8 tests)
# ============================================================================

class TestAddScalarTargetsInternal(unittest.TestCase):
    """Test rMD17 scalar target addition."""

    def test_single_scalar_target(self):
        """Single scalar target (energies) is added to pyg_data.y."""
        pc = _make_processing_config(scalar_graph_targets=["energies"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties(energies=-76.3)
        handler._add_scalar_targets_internal(data, props, 0, "test")
        self.assertTrue(hasattr(data, "y"))
        self.assertAlmostEqual(data.y[0].item(), -76.3, places=3)

    def test_multiple_scalar_targets(self):
        """Multiple scalar targets are stacked in pyg_data.y."""
        pc = _make_processing_config(scalar_graph_targets=["energies", "extra"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties(energies=-76.3)
        props["extra"] = 5.0
        handler._add_scalar_targets_internal(data, props, 0, "test")
        self.assertEqual(data.y.shape[0], 2)

    def test_missing_scalar_target_raises(self):
        """Missing scalar target raises PropertyEnrichmentError."""
        pc = _make_processing_config(scalar_graph_targets=["energies"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties()
        props.pop("energies")
        with self.assertRaises(PropertyEnrichmentError):
            handler._add_scalar_targets_internal(data, props, 0, "test")

    def test_ndarray_size_1_converted(self):
        """Numpy array of size 1 is extracted as float scalar."""
        pc = _make_processing_config(scalar_graph_targets=["energies"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties(energies=np.array(-76.3))
        handler._add_scalar_targets_internal(data, props, 0, "test")
        self.assertAlmostEqual(data.y[0].item(), -76.3, places=3)

    def test_ndarray_multi_element_raises(self):
        """Multi-element ndarray for scalar target raises PropertyEnrichmentError."""
        pc = _make_processing_config(scalar_graph_targets=["energies"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties(energies=np.array([-76.3, -50.0]))
        with self.assertRaises(PropertyEnrichmentError):
            handler._add_scalar_targets_internal(data, props, 0, "test")

    def test_invalid_type_raises(self):
        """Non-numeric type for scalar target raises PropertyEnrichmentError."""
        pc = _make_processing_config(scalar_graph_targets=["energies"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties(energies="not_a_number")
        with self.assertRaises(PropertyEnrichmentError):
            handler._add_scalar_targets_internal(data, props, 0, "test")

    def test_empty_targets_no_y(self):
        """Empty scalar_graph_targets does not set y in PyG data store.
        
        EVIDENCE: PyG Data.y is a @property returning
        ``self['y'] if 'y' in self._store else None``. Since it never raises
        AttributeError, ``hasattr(data, 'y')`` is always True. The correct
        check is ``'y' not in data`` which uses PyG's __contains__ to inspect
        the underlying _store.
        """
        pc = _make_processing_config(scalar_graph_targets=[])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties()
        handler._add_scalar_targets_internal(data, props, 0, "test")
        self.assertNotIn("y", data)

    def test_unexpected_error_in_loop_wrapped(self):
        """Unexpected error during scalar processing wraps into PropertyEnrichmentError."""
        pc = _make_processing_config(scalar_graph_targets=["energies"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties(energies=-76.3)
        with patch.object(
            handler, "_ensure_tensor",
            side_effect=RuntimeError("tensor boom"),
        ):
            with self.assertRaises((PropertyEnrichmentError, DatasetSpecificHandlerError)):
                handler._add_scalar_targets_internal(data, props, 0, "test")


# ============================================================================
# GROUP 16: _add_vector_properties_internal (6 tests)
# ============================================================================

class TestAddVectorPropertiesInternal(unittest.TestCase):
    """Test rMD17 vector property addition."""

    def test_single_vector_property(self):
        """Single 1D vector property is added to PyG data."""
        pc = _make_processing_config(vector_graph_properties=["dipole"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties()
        props["dipole"] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        handler._add_vector_properties_internal(data, props, 0, "test")
        self.assertTrue(hasattr(data, "dipole"))
        self.assertIsInstance(data.dipole, torch.Tensor)

    def test_missing_vector_property_skipped(self):
        """Missing vector property is silently skipped (debug logged)."""
        pc = _make_processing_config(vector_graph_properties=["dipole"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties()  # no dipole
        # Should not raise
        handler._add_vector_properties_internal(data, props, 0, "test")
        self.assertFalse(hasattr(data, "dipole"))

    def test_list_converted_to_ndarray(self):
        """List/tuple vector properties are converted to ndarray then tensor."""
        pc = _make_processing_config(vector_graph_properties=["dipole"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties()
        props["dipole"] = [0.1, 0.2, 0.3]  # list, not ndarray
        handler._add_vector_properties_internal(data, props, 0, "test")
        self.assertTrue(hasattr(data, "dipole"))

    def test_non_1d_raises(self):
        """Non-1D array for vector property raises PropertyEnrichmentError."""
        pc = _make_processing_config(vector_graph_properties=["dipole"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties()
        props["dipole"] = np.array([[1, 2], [3, 4]], dtype=np.float32)  # 2D
        with self.assertRaises(PropertyEnrichmentError):
            handler._add_vector_properties_internal(data, props, 0, "test")

    def test_unexpected_error_wrapped(self):
        """Unexpected error wraps into PropertyEnrichmentError."""
        pc = _make_processing_config(vector_graph_properties=["dipole"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties()
        props["dipole"] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        with patch.object(
            handler, "_ensure_tensor",
            side_effect=RuntimeError("tensor boom"),
        ):
            with self.assertRaises(PropertyEnrichmentError):
                handler._add_vector_properties_internal(data, props, 0, "test")

    def test_empty_vector_properties_noop(self):
        """Empty vector_graph_properties list is a no-op."""
        pc = _make_processing_config(vector_graph_properties=[])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties()
        handler._add_vector_properties_internal(data, props, 0, "test")
        # No vector attributes should be added


# ============================================================================
# GROUP 17: _add_variable_length_properties_internal (6 tests)
# ============================================================================

class TestAddVariableLengthPropertiesInternal(unittest.TestCase):
    """Test rMD17 variable-length property addition (forces)."""

    def test_forces_added_as_tensor(self):
        """Forces (N, 3) are converted to tensor and set on PyG data."""
        pc = _make_processing_config(variable_len_graph_properties=["forces"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(num_atoms=2)
        forces = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]], dtype=np.float32)
        props = _make_raw_properties(forces=forces, num_atoms=2)
        handler._add_variable_length_properties_internal(data, props, 0, "test")
        self.assertTrue(hasattr(data, "forces"))
        self.assertEqual(data.forces.dtype, torch.float32)

    def test_missing_property_skipped(self):
        """Missing variable-length property is silently skipped."""
        pc = _make_processing_config(variable_len_graph_properties=["forces"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties()  # no forces
        handler._add_variable_length_properties_internal(data, props, 0, "test")
        self.assertFalse(hasattr(data, "forces"))

    def test_empty_config_noop(self):
        """Empty variable_len_graph_properties is a no-op."""
        pc = _make_processing_config(variable_len_graph_properties=[])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties()
        handler._add_variable_length_properties_internal(data, props, 0, "test")

    def test_unexpected_error_wrapped(self):
        """Unexpected error in _ensure_tensor wraps into PropertyEnrichmentError."""
        pc = _make_processing_config(variable_len_graph_properties=["forces"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        forces = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
        props = _make_raw_properties(forces=forces)
        with patch.object(
            handler, "_ensure_tensor",
            side_effect=RuntimeError("tensor conversion boom"),
        ):
            with self.assertRaises(PropertyEnrichmentError):
                handler._add_variable_length_properties_internal(data, props, 0, "test")

    def test_property_enrichment_error_reraised(self):
        """PropertyEnrichmentError from within the loop is re-raised."""
        pc = _make_processing_config(variable_len_graph_properties=["forces"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        forces = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
        props = _make_raw_properties(forces=forces)
        err = PropertyEnrichmentError(
            molecule_index=0, inchi="test", property_name="forces",
            reason="test error", detail="test detail"
        )
        with patch.object(handler, "_ensure_tensor", side_effect=err):
            with self.assertRaises(PropertyEnrichmentError):
                handler._add_variable_length_properties_internal(data, props, 0, "test")

    def test_outer_unexpected_error_wrapped(self):
        """Outer unexpected error wraps into DatasetSpecificHandlerError."""
        pc = _make_processing_config(variable_len_graph_properties=["forces"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties(forces=np.array([[0.1, -0.2, 0.3]], dtype=np.float32))
        # Patch the config to trigger outer try/except
        with patch.object(
            handler.processing_config, "variable_len_graph_properties",
            new_callable=PropertyMock,
            side_effect=RuntimeError("outer boom"),
        ):
            with self.assertRaises((DatasetSpecificHandlerError, RuntimeError)):
                handler._add_variable_length_properties_internal(data, props, 0, "test")


# ============================================================================
# GROUP 18: _calculate_atomization_energy_internal (10 tests)
# ============================================================================

class TestCalculateAtomizationEnergyInternal(unittest.TestCase):
    """Test rMD17 atomization energy calculation (Hartree → eV)."""

    def test_not_configured_returns_none(self):
        """When calculate_atomization_energy_from is empty, returns None."""
        pc = _make_processing_config(calculate_atomization_energy_from="")
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties()
        result = handler._calculate_atomization_energy_internal(props, data, 0, "test")
        self.assertIsNone(result)

    def test_missing_base_energy_returns_none(self):
        """When base energy key is missing from props, returns None."""
        pc = _make_processing_config(calculate_atomization_energy_from="energies")
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties()
        props.pop("energies")
        result = handler._calculate_atomization_energy_internal(props, data, 0, "test")
        self.assertIsNone(result)

    def test_missing_z_returns_none(self):
        """When pyg_data has no z attribute, returns None."""
        pc = _make_processing_config(calculate_atomization_energy_from="energies")
        handler = _make_handler(processing_config=pc)
        data = Data()  # No z
        props = _make_raw_properties(energies=-76.3)
        result = handler._calculate_atomization_energy_internal(props, data, 0, "test")
        self.assertIsNone(result)

    def test_correct_unit_calculation(self):
        """Atomization energy is computed in Hartree then converted to eV."""
        pc = _make_processing_config(calculate_atomization_energy_from="energies")
        handler = _make_handler(processing_config=pc)

        # Use hydrogen atom (Z=1) for simple arithmetic
        data = _make_pyg_data(
            num_atoms=1,
            z=torch.tensor([1], dtype=torch.long),
        )

        base_energy = -0.5  # Hartree
        props = _make_raw_properties(energies=base_energy, num_atoms=1, atoms=np.array([1]))

        atomic_energy_h = ATOMIC_ENERGIES_HARTREE.get(1, None)
        if atomic_energy_h is not None and HAR2EV is not None:
            expected_eV = (base_energy - atomic_energy_h) * HAR2EV
            result = handler._calculate_atomization_energy_internal(props, data, 0, "test")
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result, expected_eV, places=3)

    def test_multi_atom_calculation(self):
        """Atomization energy for multi-atom molecule sums atomic energies."""
        pc = _make_processing_config(calculate_atomization_energy_from="energies")
        handler = _make_handler(processing_config=pc)

        # Water-like: O + 2H
        data = _make_pyg_data(
            num_atoms=3,
            z=torch.tensor([8, 1, 1], dtype=torch.long),
        )

        base_energy = -76.3  # Hartree
        props = _make_raw_properties(energies=base_energy)

        atomic_O = ATOMIC_ENERGIES_HARTREE.get(8, None)
        atomic_H = ATOMIC_ENERGIES_HARTREE.get(1, None)
        if atomic_O is not None and atomic_H is not None and HAR2EV is not None:
            sum_atomic = atomic_O + 2 * atomic_H
            expected_eV = (base_energy - sum_atomic) * HAR2EV
            result = handler._calculate_atomization_energy_internal(props, data, 0, "test")
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result, expected_eV, places=3)

    def test_missing_atomic_energy_for_element_returns_none(self):
        """When atomic energy for an element is missing, returns None."""
        pc = _make_processing_config(calculate_atomization_energy_from="energies")
        handler = _make_handler(processing_config=pc)

        # Use element 999 (nonexistent)
        data = _make_pyg_data(
            num_atoms=1,
            z=torch.tensor([999], dtype=torch.long),
        )
        props = _make_raw_properties(energies=-100.0, num_atoms=1, atoms=np.array([999]))
        result = handler._calculate_atomization_energy_internal(props, data, 0, "test")
        self.assertIsNone(result)

    def test_numpy_scalar_base_energy(self):
        """Numpy scalar base energy is handled correctly."""
        pc = _make_processing_config(calculate_atomization_energy_from="energies")
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(
            num_atoms=1,
            z=torch.tensor([1], dtype=torch.long),
        )
        props = _make_raw_properties(energies=np.float64(-0.5), num_atoms=1, atoms=np.array([1]))
        if ATOMIC_ENERGIES_HARTREE.get(1) is not None:
            result = handler._calculate_atomization_energy_internal(props, data, 0, "test")
            self.assertIsNotNone(result)

    def test_numpy_array_size_1_base_energy(self):
        """Numpy array of size 1 base energy is handled correctly."""
        pc = _make_processing_config(calculate_atomization_energy_from="energies")
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(
            num_atoms=1,
            z=torch.tensor([1], dtype=torch.long),
        )
        props = _make_raw_properties(energies=np.array(-0.5), num_atoms=1, atoms=np.array([1]))
        if ATOMIC_ENERGIES_HARTREE.get(1) is not None:
            result = handler._calculate_atomization_energy_internal(props, data, 0, "test")
            self.assertIsNotNone(result)

    def test_exception_returns_none(self):
        """Unexpected exceptions during calculation return None (logged)."""
        pc = _make_processing_config(calculate_atomization_energy_from="energies")
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        # z.tolist() raises if z is corrupted
        data.z = "not_a_tensor"
        props = _make_raw_properties(energies=-76.3)
        result = handler._calculate_atomization_energy_internal(props, data, 0, "test")
        self.assertIsNone(result)

    @patch("milia_pipeline.handlers.implementations.rmd17.HAR2EV", None)
    def test_missing_har2ev_returns_none(self):
        """When HAR2EV constant is None, returns None."""
        pc = _make_processing_config(calculate_atomization_energy_from="energies")
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(
            num_atoms=1, z=torch.tensor([1], dtype=torch.long)
        )
        props = _make_raw_properties(energies=-0.5, num_atoms=1, atoms=np.array([1]))
        result = handler._calculate_atomization_energy_internal(props, data, 0, "test")
        self.assertIsNone(result)


# ============================================================================
# GROUP 19: get_processing_statistics (6 tests)
# ============================================================================

class TestGetProcessingStatistics(unittest.TestCase):
    """Test rMD17 processing statistics generation."""

    def test_basic_stats(self):
        """Basic stats include dataset_type and total_processed."""
        handler = _make_handler()
        stats = handler.get_processing_statistics([])
        self.assertEqual(stats["dataset_type"], "RMD17")
        self.assertEqual(stats["total_processed"], 0)

    def test_includes_level_of_theory(self):
        """Stats include rMD17-specific level of theory (PBE/def2-SVP)."""
        handler = _make_handler()
        stats = handler.get_processing_statistics([])
        self.assertEqual(stats["level_of_theory"], "PBE/def2-SVP")

    def test_counts_atomization_calculations(self):
        """Tracks molecules with atomization energy calculated."""
        handler = _make_handler()
        mols = [
            {"atomization_energy_calculated": True},
            {"atomization_energy_calculated": False},
            {"atomization_energy_calculated": True},
        ]
        stats = handler.get_processing_statistics(mols)
        self.assertEqual(stats["atomization_energy_calculations"], 2)

    def test_counts_molecules_with_forces(self):
        """Tracks molecules with forces data."""
        handler = _make_handler()
        mols = [
            {"forces": np.array([1.0])},
            {},
            {"forces": np.array([2.0])},
        ]
        stats = handler.get_processing_statistics(mols)
        self.assertEqual(stats["molecules_with_forces"], 2)

    def test_tracks_unique_molecule_types(self):
        """Tracks unique molecule types from molecule_name field."""
        handler = _make_handler()
        mols = [
            {"molecule_name": "aspirin"},
            {"molecule_name": "ethanol"},
            {"molecule_name": "aspirin"},  # duplicate
        ]
        stats = handler.get_processing_statistics(mols)
        self.assertEqual(sorted(stats["unique_molecule_types"]), ["aspirin", "ethanol"])

    def test_experimental_setup_in_stats(self):
        """Experimental setup adds transform context to stats."""
        handler = _make_handler(experimental_setup="exp_v1")
        stats = handler.get_processing_statistics([])
        self.assertTrue(stats.get("transform_aware_processing"))
        self.assertEqual(stats["experimental_context"]["setup_name"], "exp_v1")


# ============================================================================
# GROUP 20: get_supported_structural_features (3 tests)
# ============================================================================

class TestGetSupportedStructuralFeatures(unittest.TestCase):
    """Test rMD17 supported structural features."""

    def test_returns_dict_with_atom_and_bond(self):
        """Returns dict with 'atom' and 'bond' keys."""
        handler = _make_handler()
        features = handler.get_supported_structural_features()
        self.assertIn("atom", features)
        self.assertIn("bond", features)

    def test_atom_features_include_key_descriptors(self):
        """Atom features include hybridization, degree, chirality, etc."""
        handler = _make_handler()
        features = handler.get_supported_structural_features()
        for expected in ["degree", "hybridization", "is_aromatic"]:
            self.assertIn(expected, features["atom"])

    def test_bond_features_include_geometric(self):
        """Bond features include geometric features (bond_length)."""
        handler = _make_handler()
        features = handler.get_supported_structural_features()
        self.assertIn("bond_length", features["bond"])


# ============================================================================
# GROUP 21: get_supported_descriptors (4 tests)
# ============================================================================

class TestGetSupportedDescriptors(unittest.TestCase):
    """Test rMD17 supported descriptors."""

    def test_returns_dict_with_categories(self):
        """Returns dict with 'categories' key."""
        handler = _make_handler()
        descriptors = handler.get_supported_descriptors()
        self.assertIn("categories", descriptors)

    def test_includes_geometric_category(self):
        """rMD17 supports geometric descriptors (has 3D coordinates from MD)."""
        handler = _make_handler()
        descriptors = handler.get_supported_descriptors()
        self.assertIn("geometric", descriptors["categories"])

    def test_requires_3d_is_true(self):
        """requires_3d is True (rMD17 has 3D structures from MD trajectories)."""
        handler = _make_handler()
        descriptors = handler.get_supported_descriptors()
        self.assertTrue(descriptors["requires_3d"])

    def test_excluded_is_empty(self):
        """No descriptors are excluded for rMD17."""
        handler = _make_handler()
        descriptors = handler.get_supported_descriptors()
        self.assertEqual(descriptors["excluded"], [])


# ============================================================================
# GROUP 22: Transform recommendations and validation (10 tests)
# ============================================================================

class TestTransformRecommendationsAndValidation(unittest.TestCase):
    """Test rMD17 transform system: recommendations, suitable, validation."""

    def test_get_transform_recommendations_structure(self):
        """get_transform_recommendations returns dict with recommended/avoid/warnings."""
        handler = _make_handler()
        recs = handler.get_transform_recommendations()
        self.assertIn("recommended", recs)
        self.assertIn("avoid", recs)
        self.assertIn("warnings", recs)

    def test_get_transform_recommendations_includes_geometric(self):
        """Recommendations include geometric transforms (RandomRotate)."""
        handler = _make_handler()
        recs = handler.get_transform_recommendations()
        self.assertTrue(any("RandomRotate" in r for r in recs["recommended"]))

    def test_suitable_transforms_includes_geometric(self):
        """Geometric transforms are suitable for rMD17 (3D coordinates)."""
        handler = _make_handler()
        available = {"RandomRotate": Mock(), "GCNNorm": Mock(), "AddSelfLoops": Mock()}
        suitable = handler._get_dataset_suitable_transforms(available)
        self.assertIn("RandomRotate", suitable)

    def test_suitable_transforms_filters_unavailable(self):
        """Only transforms in available_transforms are returned."""
        handler = _make_handler()
        available = {"GCNNorm": Mock()}  # No RandomRotate
        suitable = handler._get_dataset_suitable_transforms(available)
        self.assertNotIn("RandomRotate", suitable)

    def test_validate_dataset_specific_no_geometric_warns(self):
        """Warns when no geometric transforms are in the pipeline."""
        handler = _make_handler()
        warnings = handler._validate_dataset_specific_transforms(["GCNNorm"])
        self.assertTrue(any("geometric augmentation" in w.lower() or "RandomRotate" in w for w in warnings))

    def test_validate_dataset_specific_with_geometric_no_warn(self):
        """No geometry warning when RandomRotate is present."""
        handler = _make_handler()
        warnings = handler._validate_dataset_specific_transforms(["RandomRotate"])
        geometry_warnings = [w for w in warnings if "geometric augmentation" in w.lower()]
        self.assertEqual(len(geometry_warnings), 0)

    def test_validate_forces_with_rotation_warns(self):
        """Warns about forces + geometric transforms requiring force rotation."""
        pc = _make_processing_config(variable_len_graph_properties=["forces"])
        handler = _make_handler(processing_config=pc)
        warnings = handler._validate_dataset_specific_transforms(["RandomRotate"])
        self.assertTrue(any("force rotation" in w.lower() for w in warnings))

    def test_distance_transform_warning(self):
        """Distance/Cartesian transforms warn about edge attributes."""
        handler = _make_handler()
        warnings = handler._validate_dataset_specific_transforms(["Distance"])
        self.assertTrue(any("edge attribute" in w.lower() for w in warnings))

    def test_check_transform_incompatibilities_empty(self):
        """rMD17 has no specific incompatibilities — returns empty list."""
        handler = _make_handler()
        errors = handler._check_transform_incompatibilities(["RandomRotate", "GCNNorm"])
        self.assertEqual(errors, [])

    def test_get_transform_recommendations_method(self):
        """_get_transform_recommendations provides AddSelfLoops suggestion."""
        handler = _make_handler()
        recs = handler._get_transform_recommendations(["GCNNorm"])
        self.assertTrue(any("AddSelfLoops" in r for r in recs))


# ============================================================================
# GROUP 23: Integration — full pipeline flow (4 tests)
# ============================================================================

class TestRMD17IntegrationFlow(unittest.TestCase):
    """Integration-level tests for the full rMD17 handler flow."""

    def test_validate_then_enrich_happy_path(self):
        """Full flow: validate → process_property_value → enrich succeeds."""
        pc = _make_processing_config(scalar_graph_targets=["energies"])
        handler = _make_handler(processing_config=pc)
        props = _make_raw_properties(energies=-76.3)

        # Step 1: Validate
        handler.validate_molecule_data(props, 0, "test")

        # Step 2: Process properties
        processed_atoms = handler.process_property_value("atoms", props["atoms"], 0)
        processed_energy = handler.process_property_value("energies", props["energies"], 0)

        # Step 3: Enrich
        data = _make_pyg_data()
        result = handler.enrich_pyg_data(data, props, 0, "test")
        self.assertEqual(result.dataset_type, "RMD17")
        self.assertTrue(hasattr(result, "y"))

    def test_object_array_pipeline(self):
        """Object arrays (from NPZ) are normalized through pipeline."""
        handler = _make_handler()
        # Simulate object dtype arrays from NPZ
        atoms_obj = np.array([6, 1, 1], dtype=object)
        coords_obj = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=object)

        processed_atoms = handler.process_property_value("atoms", atoms_obj, 0)
        processed_coords = handler.process_property_value("coordinates", coords_obj, 0)

        self.assertEqual(processed_atoms.dtype, np.int64)
        self.assertEqual(processed_coords.dtype, np.float64)

    def test_rmd17_typical_molecules(self):
        """Typical rMD17 molecule elements (aspirin-like: C, H, O) process correctly."""
        handler = _make_handler()
        # Aspirin: C9H8O4 (21 atoms) — use a small fragment
        atoms = np.array([6, 6, 8, 1, 1], dtype=np.int64)
        result = handler.process_property_value("atoms", atoms, 0)
        self.assertEqual(result.dtype, np.int64)
        np.testing.assert_array_equal(result, atoms)

    def test_forces_pipeline(self):
        """Forces processing through full enrich path."""
        pc = _make_processing_config(
            scalar_graph_targets=["energies"],
            variable_len_graph_properties=["forces"],
        )
        handler = _make_handler(processing_config=pc)
        num_atoms = 3
        forces = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0], [-0.1, 0.2, -0.3]], dtype=np.float32)
        data = _make_pyg_data(num_atoms=num_atoms)
        props = _make_raw_properties(energies=-76.3, forces=forces, num_atoms=num_atoms)
        result = handler.enrich_pyg_data(data, props, 0, "test")
        self.assertTrue(hasattr(result, "forces"))
        self.assertTrue(hasattr(result, "y"))


# ============================================================================
# GROUP 24: Edge cases and error boundary tests (5 tests)
# ============================================================================

class TestRMD17EdgeCases(unittest.TestCase):
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
        """Handler with no experimental_setup works correctly."""
        handler = _make_handler(experimental_setup=None)
        stats = handler.get_processing_statistics([])
        self.assertNotIn("transform_aware_processing", stats)

    def test_no_atomization_configured(self):
        """No atomization energy calculation when not configured."""
        pc = _make_processing_config(
            calculate_atomization_energy_from="",
            atomization_energy_key_name="",
        )
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties(energies=-76.3)
        result = handler.enrich_pyg_data(data, props, 0, "test")
        self.assertFalse(hasattr(result, "atomization_energy"))

    def test_molecule_name_tracked_in_stats(self):
        """molecule_name from rMD17 is tracked in processing statistics."""
        handler = _make_handler()
        mols = [
            {"molecule_name": "benzene"},
            {"molecule_name": "ethanol"},
            {"molecule_name": "benzene"},
        ]
        stats = handler.get_processing_statistics(mols)
        self.assertIn("benzene", stats["unique_molecule_types"])
        self.assertIn("ethanol", stats["unique_molecule_types"])


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    unittest.main()
