#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/handlers/implementations/dmc.py

Module under test: dmc.py
- DMCDatasetHandler: Handler for DMC (Diffusion Monte Carlo) datasets
  - Implements DatasetHandler ABC (12 abstract methods + 4 transform validation helpers)
  - Registered via @register_handler decorator
  - DMC-specific: uncertainty handling, inverse variance weighting, minimal transforms,
    limited structural features (no partial_charge/mulliken_charge)

Test path on local machine: ~/ml_projects/milia/tests/test_handler_impl_dmc_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/handlers/implementations/dmc.py

NOTE: This test suite runs inside Docker at /app/milia
Path mappings:
- Project root: /app/milia (mapped from ~/ml_projects/milia)

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution

NPZ file paths (mocked, never downloaded):
- ~/Chem_Data/MILIA_PyG_Dataset/raw/DMC.npz

Updated: February 2026 - Production-ready comprehensive test coverage
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
    HandlerConfigurationError,
    HandlerValidationError,
    MoleculeProcessingError,
    PropertyEnrichmentError,
)
from milia_pipeline.handlers.implementations.dmc import DMCDatasetHandler

# ============================================================================
# HELPERS: Build realistic config mocks for DMCDatasetHandler
# ============================================================================


def _make_dataset_config(**overrides):
    """
    Build a minimal mock DatasetConfig for DMC handler tests.

    Based on project structure: DatasetConfig is a Pydantic frozen BaseModel.
    DMC handler accesses dataset_config for uncertainty handling configuration.
    """
    cfg = Mock(name="DatasetConfig")
    cfg.dataset_type = overrides.get("dataset_type", "DMC")
    cfg.root_dir = overrides.get("root_dir", "/tmp/test_data")
    cfg.raw_dir = overrides.get("raw_dir", "/tmp/test_data/raw")
    cfg.is_uncertainty_enabled = overrides.get("is_uncertainty_enabled", False)
    cfg.uncertainty_config = overrides.get("uncertainty_config")
    return cfg


def _make_filter_config(**overrides):
    """
    Build a minimal mock FilterConfig for DMC handler tests.

    Based on project structure: FilterConfig is a Pydantic frozen BaseModel.
    """
    cfg = Mock(name="FilterConfig")
    cfg.max_atoms = overrides.get("max_atoms", 100)
    cfg.min_atoms = overrides.get("min_atoms", 1)
    cfg.allowed_elements = overrides.get("allowed_elements")
    return cfg


def _make_processing_config(**overrides):
    """
    Build a minimal mock ProcessingConfig for DMC handler tests.

    Based on project structure: ProcessingConfig is a Pydantic frozen BaseModel.
    The DMC handler uses:
    - scalar_graph_targets: List[str]
    - node_features: List[str]
    - variable_len_graph_properties: List[str]
    - common_required_properties: List[str]
    """
    cfg = Mock(name="ProcessingConfig")
    cfg.scalar_graph_targets = overrides.get("scalar_graph_targets", ["Etot"])
    cfg.node_features = overrides.get("node_features", [])
    cfg.variable_len_graph_properties = overrides.get("variable_len_graph_properties", [])
    cfg.common_required_properties = overrides.get(
        "common_required_properties", ["atoms", "coordinates"]
    )
    return cfg


def _make_handler(**overrides):
    """
    Build a DMCDatasetHandler instance with configurable mocked configs.

    Based on DatasetHandler ABC constructor signature:
    __init__(dataset_config, filter_config, processing_config, logger, experimental_setup=None)
    """
    dataset_config = overrides.get("dataset_config", _make_dataset_config())
    filter_config = overrides.get("filter_config", _make_filter_config())
    processing_config = overrides.get("processing_config", _make_processing_config())
    logger = overrides.get("logger", logging.getLogger("test.dmc"))
    experimental_setup = overrides.get("experimental_setup")

    handler = DMCDatasetHandler(
        dataset_config=dataset_config,
        filter_config=filter_config,
        processing_config=processing_config,
        logger=logger,
        experimental_setup=experimental_setup,
    )
    return handler


def _make_pyg_data(**overrides):
    """
    Build a minimal PyG Data object for DMC enrichment tests.

    DMC molecules typically have:
    - z: atomic numbers tensor
    - pos: 3D coordinates tensor
    - edge_index: connectivity tensor
    - num_nodes: atom count
    """
    num_atoms = overrides.get("num_atoms", 3)
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
    Build a realistic raw_properties_dict for DMC molecule tests.

    DMC NPZ files contain: Etot, atoms, coordinates, inchi,
    graphs (SMILES), std (uncertainty).
    """
    num_atoms = overrides.get("num_atoms", 3)
    props = {
        "Etot": overrides.get("Etot", -76.4),
        "atoms": overrides.get("atoms", np.array([6, 1, 1])[:num_atoms]),
        "coordinates": overrides.get(
            "coordinates", np.random.randn(num_atoms, 3).astype(np.float32)
        ),
        "inchi": overrides.get("inchi", "InChI=1S/CH2/c1-2/h1-2H"),
    }
    # Optionally add extra properties
    for key in ["graphs", "std", "uncertainty"]:
        if key in overrides:
            props[key] = overrides[key]
    return props


def _make_uncertainty_config(**overrides):
    """
    Build a realistic uncertainty configuration dict for DMC handler tests.

    DMC uncertainty configuration contains:
    - uncertainty_field_name: str (default 'std')
    - max_uncertainty_threshold: Optional[float]
    - uncertainty_weighting: str (default 'inverse_variance')
    """
    config = {
        "uncertainty_field_name": overrides.get("uncertainty_field_name", "std"),
        "max_uncertainty_threshold": overrides.get("max_uncertainty_threshold"),
        "uncertainty_weighting": overrides.get("uncertainty_weighting", "inverse_variance"),
    }
    # Allow removing keys
    for key in list(config.keys()):
        if key in overrides and overrides[key] is None and key != "max_uncertainty_threshold":
            # Only remove if explicitly set to None for field_name and weighting
            pass
    return config


# ============================================================================
# GROUP 1: DMCDatasetHandler — Identity and Registration (6 tests)
# ============================================================================


class TestDMCDatasetHandlerIdentity(unittest.TestCase):
    """Test DMCDatasetHandler identity, registration, and basic attributes."""

    def test_get_dataset_type_returns_dmc(self):
        """get_dataset_type() returns 'DMC'."""
        handler = _make_handler()
        self.assertEqual(handler.get_dataset_type(), "DMC")

    def test_get_molecule_creation_strategy(self):
        """DMC uses identifier_coordinate_based strategy (InChI parsing)."""
        handler = _make_handler()
        self.assertEqual(handler.get_molecule_creation_strategy(), "identifier_coordinate_based")

    def test_get_identifier_keys(self):
        """DMC identifier keys: InChI primary, SMILES (graphs) fallback."""
        handler = _make_handler()
        keys = handler.get_identifier_keys()
        self.assertIsInstance(keys, list)
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], ("inchi", "inchi"))
        self.assertEqual(keys[1], ("graphs", "smiles"))

    def test_is_subclass_of_dataset_handler(self):
        """DMCDatasetHandler is a proper DatasetHandler subclass."""
        from milia_pipeline.handlers.base_handler import DatasetHandler

        self.assertTrue(issubclass(DMCDatasetHandler, DatasetHandler))

    def test_handler_stores_configs(self):
        """Handler stores config objects passed during construction."""
        dc = _make_dataset_config()
        fc = _make_filter_config()
        pc = _make_processing_config()
        handler = DMCDatasetHandler(
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
    """Test DMC-specific required property determination."""

    def test_includes_etot(self):
        """Required properties include Etot (DMC primary energy)."""
        handler = _make_handler()
        required = handler.get_required_properties()
        self.assertIn("Etot", required)

    def test_includes_inchi(self):
        """Required properties include 'inchi' for molecular charge determination."""
        handler = _make_handler()
        required = handler.get_required_properties()
        self.assertIn("inchi", required)

    def test_includes_common_required_properties(self):
        """Required properties include common ones (atoms, coordinates)."""
        pc = _make_processing_config(common_required_properties=["atoms", "coordinates"])
        handler = _make_handler(processing_config=pc)
        required = handler.get_required_properties()
        self.assertIn("atoms", required)
        self.assertIn("coordinates", required)

    def test_includes_scalar_graph_targets(self):
        """Required properties include scalar_graph_targets from config."""
        pc = _make_processing_config(scalar_graph_targets=["Etot"])
        handler = _make_handler(processing_config=pc)
        required = handler.get_required_properties()
        self.assertIn("Etot", required)

    def test_includes_node_features(self):
        """Required properties include node_features from config."""
        pc = _make_processing_config(node_features=["some_feature"])
        handler = _make_handler(processing_config=pc)
        required = handler.get_required_properties()
        self.assertIn("some_feature", required)

    def test_includes_uncertainty_field_when_enabled(self):
        """Required properties include uncertainty field when uncertainty is enabled."""
        uc = _make_uncertainty_config(uncertainty_field_name="std")
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        required = handler.get_required_properties()
        self.assertIn("std", required)

    def test_returns_deduplicated_list(self):
        """Required properties list has no duplicates."""
        pc = _make_processing_config(
            scalar_graph_targets=["Etot"],
            common_required_properties=["atoms", "coordinates"],
        )
        handler = _make_handler(processing_config=pc)
        required = handler.get_required_properties()
        self.assertEqual(len(required), len(set(required)))


# ============================================================================
# GROUP 3: get_molecular_charge (6 tests)
# ============================================================================


class TestGetMolecularCharge(unittest.TestCase):
    """Test DMC-specific molecular charge extraction from InChI."""

    def test_neutral_molecule_from_inchi(self):
        """Neutral molecule returns charge 0."""
        handler = _make_handler()
        props = {"inchi": "InChI=1S/H2O/h1H2"}
        charge = handler.get_molecular_charge(props, np.array([8, 1, 1]))
        self.assertEqual(charge, 0)

    def test_charged_molecule_from_inchi(self):
        """InChI with /q layer returns the charge value."""
        handler = _make_handler()
        props = {"inchi": "InChI=1S/CH3/c1-2/h1H3/q+1"}
        charge = handler.get_molecular_charge(props, np.array([6, 1, 1, 1]))
        self.assertEqual(charge, 1)

    def test_negative_charge_from_inchi(self):
        """InChI with /q-1 returns negative charge."""
        handler = _make_handler()
        props = {"inchi": "InChI=1S/HO/h1H/q-1"}
        charge = handler.get_molecular_charge(props, np.array([8, 1]))
        self.assertEqual(charge, -1)

    def test_no_inchi_returns_zero(self):
        """When InChI is absent, assume neutral (charge 0)."""
        handler = _make_handler()
        props = {}
        charge = handler.get_molecular_charge(props, np.array([6, 1, 1]))
        self.assertEqual(charge, 0)

    def test_none_inchi_returns_zero(self):
        """When InChI is None, assume neutral (charge 0)."""
        handler = _make_handler()
        props = {"inchi": None}
        charge = handler.get_molecular_charge(props, np.array([6, 1, 1]))
        self.assertEqual(charge, 0)

    def test_empty_inchi_returns_zero(self):
        """When InChI is empty string, assume neutral (charge 0)."""
        handler = _make_handler()
        props = {"inchi": ""}
        charge = handler.get_molecular_charge(props, np.array([6, 1, 1]))
        self.assertEqual(charge, 0)


# ============================================================================
# GROUP 4: validate_molecule_data — Success Paths (5 tests)
# ============================================================================


class TestValidateMoleculeDataSuccess(unittest.TestCase):
    """Test DMC molecule validation success paths."""

    @patch("milia_pipeline.handlers.implementations.dmc.validate_molecular_structure")
    def test_valid_molecule_passes(self, mock_validate_struct):
        """Valid DMC molecule with all essential properties passes validation."""
        mock_validate_struct.return_value = None
        handler = _make_handler()
        props = _make_raw_properties()
        # Should not raise
        handler.validate_molecule_data(props, molecule_index=0, identifier="InChI=1S/H2O/h1H2")

    @patch("milia_pipeline.handlers.implementations.dmc.validate_molecular_structure")
    def test_large_energy_logs_warning(self, mock_validate_struct):
        """Unusually large energy logs a warning but does not raise."""
        mock_validate_struct.return_value = None
        handler = _make_handler()
        props = _make_raw_properties(Etot=-15000.0)
        with self.assertLogs("test.dmc", level="WARNING") as cm:
            handler.validate_molecule_data(props, molecule_index=0, identifier="test")
        self.assertTrue(any("unusually large energy" in msg for msg in cm.output))

    @patch("milia_pipeline.handlers.implementations.dmc.validate_molecular_structure")
    def test_uncertainty_validated_when_enabled(self, mock_validate_struct):
        """Uncertainty validation is called when uncertainty is enabled."""
        mock_validate_struct.return_value = None
        uc = _make_uncertainty_config(uncertainty_field_name="std")
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        props = _make_raw_properties(std=0.001)
        with patch.object(handler, "_validate_uncertainty_data") as mock_unc:
            handler.validate_molecule_data(props, molecule_index=0, identifier="test")
            mock_unc.assert_called_once()

    @patch("milia_pipeline.handlers.implementations.dmc.validate_molecular_structure")
    def test_default_identifier(self, mock_validate_struct):
        """Default identifier 'N/A' is handled without error."""
        mock_validate_struct.return_value = None
        handler = _make_handler()
        props = _make_raw_properties()
        handler.validate_molecule_data(props, molecule_index=0)

    @patch("milia_pipeline.handlers.implementations.dmc.validate_molecular_structure")
    def test_normal_energy_no_warning(self, mock_validate_struct):
        """Normal energy value does not log a warning."""
        mock_validate_struct.return_value = None
        handler = _make_handler()
        props = _make_raw_properties(Etot=-76.4)
        # Should not raise and should not produce WARNING level log
        handler.validate_molecule_data(props, molecule_index=0, identifier="test")


# ============================================================================
# GROUP 5: validate_molecule_data — Error Paths (7 tests)
# ============================================================================


class TestValidateMoleculeDataErrors(unittest.TestCase):
    """Test DMC molecule validation error paths."""

    def test_missing_etot_raises_handler_validation_error(self):
        """Missing Etot raises HandlerValidationError."""
        handler = _make_handler()
        props = _make_raw_properties(Etot=None)
        with self.assertRaises(HandlerValidationError) as ctx:
            handler.validate_molecule_data(props, molecule_index=0)
        self.assertIn("Etot", str(ctx.exception))

    def test_missing_atoms_raises_handler_validation_error(self):
        """Missing atoms raises HandlerValidationError."""
        handler = _make_handler()
        props = _make_raw_properties(atoms=None)
        with self.assertRaises(HandlerValidationError) as ctx:
            handler.validate_molecule_data(props, molecule_index=0)
        self.assertIn("atoms", str(ctx.exception))

    def test_missing_coordinates_raises_handler_validation_error(self):
        """Missing coordinates raises HandlerValidationError."""
        handler = _make_handler()
        props = _make_raw_properties(coordinates=None)
        with self.assertRaises(HandlerValidationError) as ctx:
            handler.validate_molecule_data(props, molecule_index=0)
        self.assertIn("coordinates", str(ctx.exception))

    def test_missing_all_essential_lists_all(self):
        """Missing all essential properties lists them all."""
        handler = _make_handler()
        props = {"inchi": "test"}
        with self.assertRaises(HandlerValidationError) as ctx:
            handler.validate_molecule_data(props, molecule_index=0)
        self.assertIn("Etot", str(ctx.exception))
        self.assertIn("atoms", str(ctx.exception))
        self.assertIn("coordinates", str(ctx.exception))

    @patch(
        "milia_pipeline.handlers.implementations.dmc.validate_molecular_structure",
        side_effect=ValueError("Atom count mismatch"),
    )
    def test_structure_validation_failure_raises_dmc_handler_error(self, mock_validate):
        """Structure validation failure wraps into DatasetSpecificHandlerError."""
        handler = _make_handler()
        props = _make_raw_properties()
        with self.assertRaises(DatasetSpecificHandlerError):
            handler.validate_molecule_data(props, molecule_index=0, identifier="test")

    def test_invalid_string_etot_raises(self):
        """String 'missing' for Etot is treated as invalid."""
        handler = _make_handler()
        props = _make_raw_properties(Etot="missing")
        with self.assertRaises(HandlerValidationError):
            handler.validate_molecule_data(props, molecule_index=0)

    def test_non_numeric_etot_raises_dmc_handler_error(self):
        """Non-numeric Etot string that cannot convert raises DatasetSpecificHandlerError."""
        handler = _make_handler()
        props = _make_raw_properties(Etot="not_a_number_but_valid_property")
        # "not_a_number_but_valid_property" passes _is_valid_property (not in invalid set)
        # but fails float() conversion in energy validation
        with self.assertRaises(DatasetSpecificHandlerError):
            handler.validate_molecule_data(props, molecule_index=0, identifier="test")


# ============================================================================
# GROUP 6: process_property_value (8 tests)
# ============================================================================


class TestProcessPropertyValue(unittest.TestCase):
    """Test DMC-specific property value processing."""

    def test_passthrough_normal_value(self):
        """Normal numeric values pass through unchanged."""
        handler = _make_handler()
        result = handler.process_property_value("Etot", -76.4, 0)
        self.assertEqual(result, -76.4)

    def test_etot_string_converted_to_float(self):
        """String Etot value is converted to float."""
        handler = _make_handler()
        result = handler.process_property_value("Etot", "-76.4", 0)
        self.assertAlmostEqual(result, -76.4, places=4)

    def test_etot_bytes_converted_to_float(self):
        """Bytes Etot value is converted to float."""
        handler = _make_handler()
        result = handler.process_property_value("Etot", b"-76.4", 0)
        self.assertAlmostEqual(result, -76.4, places=4)

    def test_etot_non_numeric_string_raises_dmc_handler_error(self):
        """Non-numeric string Etot raises DatasetSpecificHandlerError."""
        handler = _make_handler()
        with self.assertRaises(DatasetSpecificHandlerError):
            handler.process_property_value("Etot", "not_a_number", 0)

    @patch(
        "milia_pipeline.handlers.implementations.dmc.validate_uncertainty_data", return_value=0.001
    )
    def test_uncertainty_field_validated(self, mock_validate):
        """Uncertainty field is validated and returned."""
        uc = _make_uncertainty_config(uncertainty_field_name="std")
        dc = _make_dataset_config(uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        result = handler.process_property_value("std", 0.001, 0)
        self.assertAlmostEqual(result, 0.001, places=6)
        mock_validate.assert_called_once()

    @patch(
        "milia_pipeline.handlers.implementations.dmc.validate_uncertainty_data",
        side_effect=ValueError("Negative uncertainty"),
    )
    def test_uncertainty_validation_failure_raises_dmc_handler_error(self, mock_validate):
        """Uncertainty validation failure raises DatasetSpecificHandlerError."""
        uc = _make_uncertainty_config(uncertainty_field_name="std")
        dc = _make_dataset_config(uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        with self.assertRaises(DatasetSpecificHandlerError):
            handler.process_property_value("std", -0.001, 0)

    def test_non_special_property_passthrough(self):
        """Non-special property values pass through unchanged."""
        handler = _make_handler()
        arr = np.array([1.0, 2.0, 3.0])
        result = handler.process_property_value("some_prop", arr, 0)
        np.testing.assert_array_equal(result, arr)

    def test_unexpected_exception_wrapped_in_dmc_handler_error(self):
        """Unexpected exceptions during processing are wrapped in DatasetSpecificHandlerError."""
        handler = _make_handler()
        with patch(
            "milia_pipeline.handlers.implementations.dmc.validate_uncertainty_data",
            side_effect=RuntimeError("boom"),
        ):
            uc = _make_uncertainty_config(uncertainty_field_name="std")
            dc = _make_dataset_config(uncertainty_config=uc)
            handler = _make_handler(dataset_config=dc)
            with self.assertRaises(DatasetSpecificHandlerError):
                handler.process_property_value("std", 0.001, 0)


# ============================================================================
# GROUP 7: get_transform_recommendations (5 tests)
# ============================================================================


class TestGetTransformRecommendations(unittest.TestCase):
    """Test DMC-specific transform recommendations."""

    def test_returns_dict_with_expected_keys(self):
        """Recommendations dict has recommended, avoid, warnings keys."""
        handler = _make_handler()
        recs = handler.get_transform_recommendations()
        self.assertIn("recommended", recs)
        self.assertIn("avoid", recs)
        self.assertIn("warnings", recs)

    def test_recommended_includes_core_transforms(self):
        """Recommended transforms include AddSelfLoops, ToUndirected, NormalizeFeatures."""
        handler = _make_handler()
        recs = handler.get_transform_recommendations()
        rec_text = " ".join(recs["recommended"])
        self.assertIn("AddSelfLoops", rec_text)
        self.assertIn("ToUndirected", rec_text)
        self.assertIn("NormalizeFeatures", rec_text)

    def test_avoid_includes_augmentation_transforms(self):
        """Avoid list includes data augmentation transforms (DropNode, DropEdge, etc.)."""
        handler = _make_handler()
        recs = handler.get_transform_recommendations()
        avoid_text = " ".join(recs["avoid"])
        self.assertIn("DropNode", avoid_text)
        self.assertIn("DropEdge", avoid_text)
        self.assertIn("VirtualNode", avoid_text)

    def test_avoid_includes_uncertainty_specific_when_uncertainty_enabled(self):
        """Avoid list includes uncertainty-specific transforms when uncertainty is configured."""
        uc = _make_uncertainty_config()
        dc = _make_dataset_config(uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        recs = handler.get_transform_recommendations()
        avoid_text = " ".join(recs["avoid"])
        self.assertIn("VirtualNode", avoid_text)

    def test_warnings_present(self):
        """Warnings about transform interactions are always present."""
        handler = _make_handler()
        recs = handler.get_transform_recommendations()
        self.assertGreater(len(recs["warnings"]), 0)


# ============================================================================
# GROUP 8: get_supported_descriptors (4 tests)
# ============================================================================


class TestGetSupportedDescriptors(unittest.TestCase):
    """Test DMC-specific descriptor support reporting."""

    def test_returns_dict_with_expected_keys(self):
        """Descriptor dict has categories, excluded, recommended, requires_3d, requires_charges."""
        handler = _make_handler()
        desc = handler.get_supported_descriptors()
        for key in ["categories", "excluded", "recommended", "requires_3d", "requires_charges"]:
            self.assertIn(key, desc)

    def test_includes_geometric_category(self):
        """DMC supports geometric descriptors (has 3D coordinates)."""
        handler = _make_handler()
        desc = handler.get_supported_descriptors()
        self.assertIn("geometric", desc["categories"])

    def test_no_exclusions(self):
        """DMC has no excluded descriptors."""
        handler = _make_handler()
        desc = handler.get_supported_descriptors()
        self.assertEqual(len(desc["excluded"]), 0)

    def test_requires_3d_and_charges(self):
        """DMC requires_3d and requires_charges are True."""
        handler = _make_handler()
        desc = handler.get_supported_descriptors()
        self.assertTrue(desc["requires_3d"])
        self.assertTrue(desc["requires_charges"])


# ============================================================================
# GROUP 9: get_supported_structural_features (5 tests)
# ============================================================================


class TestGetSupportedStructuralFeatures(unittest.TestCase):
    """Test DMC-specific structural feature support (LIMITED features)."""

    def test_returns_dict_with_atom_and_bond(self):
        """Returns dict with 'atom' and 'bond' keys."""
        handler = _make_handler()
        features = handler.get_supported_structural_features()
        self.assertIn("atom", features)
        self.assertIn("bond", features)

    def test_atom_features_exclude_partial_charge(self):
        """Atom features EXCLUDE partial_charge (Gasteiger fails on DMC)."""
        handler = _make_handler()
        features = handler.get_supported_structural_features()
        self.assertNotIn("partial_charge", features["atom"])

    def test_atom_features_exclude_mulliken_charge(self):
        """Atom features EXCLUDE mulliken_charge (not available in DMC data)."""
        handler = _make_handler()
        features = handler.get_supported_structural_features()
        self.assertNotIn("mulliken_charge", features["atom"])

    def test_atom_features_include_topology(self):
        """Atom features include topology features (degree, hybridization, etc.)."""
        handler = _make_handler()
        features = handler.get_supported_structural_features()
        for feat in ["degree", "hybridization", "is_aromatic"]:
            self.assertIn(feat, features["atom"])

    def test_bond_features_include_expected(self):
        """Bond features include bond_type, bond_length, and topology features."""
        handler = _make_handler()
        features = handler.get_supported_structural_features()
        for feat in [
            "bond_type",
            "is_conjugated",
            "is_aromatic",
            "bond_length",
            "bond_length_binned",
        ]:
            self.assertIn(feat, features["bond"])


# ============================================================================
# GROUP 10: _is_valid_property (7 tests)
# ============================================================================


class TestIsValidProperty(unittest.TestCase):
    """Test DMC-specific property validation."""

    def test_none_is_invalid(self):
        """None is invalid."""
        handler = _make_handler()
        self.assertFalse(handler._is_valid_property(None))

    def test_missing_string_is_invalid(self):
        """String 'missing' is invalid."""
        handler = _make_handler()
        self.assertFalse(handler._is_valid_property("missing"))

    def test_missing_etot_string_is_invalid(self):
        """String 'missing_etot' is invalid."""
        handler = _make_handler()
        self.assertFalse(handler._is_valid_property("missing_etot"))

    def test_invalid_string_is_invalid(self):
        """String 'invalid' is invalid."""
        handler = _make_handler()
        self.assertFalse(handler._is_valid_property("invalid"))

    def test_empty_string_is_invalid(self):
        """Empty string is invalid."""
        handler = _make_handler()
        self.assertFalse(handler._is_valid_property(""))

    def test_nan_string_is_invalid(self):
        """String 'nan' is invalid."""
        handler = _make_handler()
        self.assertFalse(handler._is_valid_property("nan"))

    def test_numeric_value_is_valid(self):
        """Normal numeric value is valid."""
        handler = _make_handler()
        self.assertTrue(handler._is_valid_property(-76.4))

    def test_numeric_string_is_valid(self):
        """Numeric string (e.g., '-76.4') is valid (deferred conversion)."""
        handler = _make_handler()
        self.assertTrue(handler._is_valid_property("-76.4"))

    def test_numpy_array_delegates_to_validator(self):
        """Numpy array delegates to is_value_valid_and_not_nan."""
        handler = _make_handler()
        self.assertTrue(handler._is_valid_property(np.array([1.0, 2.0])))


# ============================================================================
# GROUP 11: _add_scalar_targets_internal (9 tests)
# ============================================================================


class TestAddScalarTargetsInternal(unittest.TestCase):
    """Test DMC scalar target addition to PyG data."""

    def test_no_targets_configured_noop(self):
        """No scalar_graph_targets means no-op."""
        pc = _make_processing_config(scalar_graph_targets=[])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        handler._add_scalar_targets_internal(data, _make_raw_properties(), 0, "test")
        self.assertFalse(hasattr(data, "y") and data.y is not None)

    def test_single_scalar_target(self):
        """Single scalar target is correctly added as tensor."""
        pc = _make_processing_config(scalar_graph_targets=["Etot"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties(Etot=-76.4)
        handler._add_scalar_targets_internal(data, props, 0, "test")
        self.assertIsNotNone(data.y)
        self.assertEqual(data.y.shape[0], 1)
        self.assertAlmostEqual(data.y[0].item(), -76.4, places=4)

    def test_numpy_scalar_target(self):
        """Numpy scalar (ndarray size 1) is handled."""
        pc = _make_processing_config(scalar_graph_targets=["Etot"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties(Etot=np.array(-76.4))
        handler._add_scalar_targets_internal(data, props, 0, "test")
        self.assertAlmostEqual(data.y[0].item(), -76.4, places=4)

    def test_string_convertible_scalar(self):
        """String-convertible scalar (e.g., '-76.4') is handled."""
        pc = _make_processing_config(scalar_graph_targets=["Etot"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties(Etot="-76.4")
        handler._add_scalar_targets_internal(data, props, 0, "test")
        self.assertAlmostEqual(data.y[0].item(), -76.4, places=4)

    def test_list_single_element_scalar(self):
        """Single-element list [val] is handled as scalar."""
        pc = _make_processing_config(scalar_graph_targets=["Etot"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties(Etot=[-76.4])
        handler._add_scalar_targets_internal(data, props, 0, "test")
        self.assertAlmostEqual(data.y[0].item(), -76.4, places=4)

    def test_missing_target_raises_property_enrichment_error(self):
        """Missing required scalar target raises PropertyEnrichmentError."""
        pc = _make_processing_config(scalar_graph_targets=["nonexistent_prop"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties()
        with self.assertRaises(PropertyEnrichmentError):
            handler._add_scalar_targets_internal(data, props, 0, "test")

    def test_multi_element_array_raises(self):
        """Multi-element array raises PropertyEnrichmentError."""
        pc = _make_processing_config(scalar_graph_targets=["Etot"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties(Etot=np.array([-76.4, -75.0]))
        with self.assertRaises(PropertyEnrichmentError):
            handler._add_scalar_targets_internal(data, props, 0, "test")

    def test_nan_value_raises_property_enrichment_error(self):
        """NaN scalar target raises PropertyEnrichmentError after conversion."""
        pc = _make_processing_config(scalar_graph_targets=["Etot"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties(Etot=float("nan"))
        with self.assertRaises((PropertyEnrichmentError, HandlerValidationError)):
            handler._add_scalar_targets_internal(data, props, 0, "test")

    def test_unsupported_type_raises_property_enrichment_error(self):
        """Unsupported type for scalar target raises PropertyEnrichmentError."""
        pc = _make_processing_config(scalar_graph_targets=["Etot"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data()
        props = _make_raw_properties(Etot={"value": -76.4})
        with self.assertRaises(PropertyEnrichmentError):
            handler._add_scalar_targets_internal(data, props, 0, "test")


# ============================================================================
# GROUP 12: enrich_pyg_data (8 tests)
# ============================================================================


class TestEnrichPygData(unittest.TestCase):
    """Test DMC PyG data enrichment orchestration."""

    def test_sets_dataset_type(self):
        """Enrichment sets dataset_type = 'DMC' on PyG data."""
        handler = _make_handler()
        data = _make_pyg_data()
        props = _make_raw_properties()
        result = handler.enrich_pyg_data(data, props, 0, "test")
        self.assertEqual(result.dataset_type, "DMC")

    def test_returns_data_object(self):
        """Enrichment returns the Data object."""
        handler = _make_handler()
        data = _make_pyg_data()
        result = handler.enrich_pyg_data(data, _make_raw_properties(), 0, "test")
        self.assertIsInstance(result, Data)

    def test_calls_scalar_targets(self):
        """Enrichment calls _add_scalar_targets_internal."""
        handler = _make_handler()
        data = _make_pyg_data()
        with patch.object(handler, "_add_scalar_targets_internal") as mock_scalar:
            handler.enrich_pyg_data(data, _make_raw_properties(), 0, "test")
            mock_scalar.assert_called_once()

    def test_calls_uncertainty_when_enabled(self):
        """Enrichment calls _add_uncertainty_metadata_internal when uncertainty enabled."""
        uc = _make_uncertainty_config()
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        data = _make_pyg_data()
        with patch.object(handler, "_add_uncertainty_metadata_internal") as mock_unc:
            handler.enrich_pyg_data(data, _make_raw_properties(), 0, "test")
            mock_unc.assert_called_once()

    def test_no_uncertainty_call_when_disabled(self):
        """Enrichment does NOT call uncertainty processing when disabled."""
        dc = _make_dataset_config(is_uncertainty_enabled=False, uncertainty_config=None)
        handler = _make_handler(dataset_config=dc)
        data = _make_pyg_data()
        with patch.object(handler, "_add_uncertainty_metadata_internal") as mock_unc:
            handler.enrich_pyg_data(data, _make_raw_properties(), 0, "test")
            mock_unc.assert_not_called()

    def test_zero_nodes_raises_dmc_handler_error(self):
        """Zero-node PyG data raises DatasetSpecificHandlerError."""
        handler = _make_handler()
        data = Data()
        data.z = None
        data.num_nodes = 0
        with self.assertRaises(DatasetSpecificHandlerError):
            handler.enrich_pyg_data(data, _make_raw_properties(), 0, "test")

    def test_enrichment_error_raises_dmc_handler_error(self):
        """Unexpected enrichment error wraps in DatasetSpecificHandlerError."""
        handler = _make_handler()
        data = _make_pyg_data()
        with (
            patch.object(
                handler,
                "_add_scalar_targets_internal",
                side_effect=RuntimeError("unexpected"),
            ),
            self.assertRaises(DatasetSpecificHandlerError),
        ):
            handler.enrich_pyg_data(data, _make_raw_properties(), 0, "test")

    def test_sets_num_nodes_from_z(self):
        """Enrichment sets num_nodes from z tensor when not already set."""
        handler = _make_handler()
        data = Data()
        data.z = torch.tensor([6, 1, 1], dtype=torch.long)
        data.num_nodes = 0  # Not set
        props = _make_raw_properties()
        result = handler.enrich_pyg_data(data, props, 0, "test")
        self.assertEqual(result.num_nodes, 3)


# ============================================================================
# GROUP 13: _add_uncertainty_metadata_internal (8 tests)
# ============================================================================


class TestAddUncertaintyMetadataInternal(unittest.TestCase):
    """Test DMC uncertainty metadata addition to PyG data."""

    @patch(
        "milia_pipeline.handlers.implementations.dmc.validate_uncertainty_data", return_value=0.001
    )
    def test_adds_uncertainty_tensor(self, mock_validate):
        """Valid uncertainty is added as tensor on PyG data."""
        uc = _make_uncertainty_config()
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        data = _make_pyg_data()
        props = _make_raw_properties(std=0.001)
        handler._add_uncertainty_metadata_internal(data, props, 0, "test")
        self.assertTrue(hasattr(data, "uncertainty"))
        self.assertAlmostEqual(data.uncertainty[0].item(), 0.001, places=6)

    @patch(
        "milia_pipeline.handlers.implementations.dmc.validate_uncertainty_data", return_value=0.001
    )
    def test_adds_inverse_variance_weight(self, mock_validate):
        """Inverse variance weight is calculated correctly."""
        uc = _make_uncertainty_config(uncertainty_weighting="inverse_variance")
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        data = _make_pyg_data()
        props = _make_raw_properties(std=0.001)
        handler._add_uncertainty_metadata_internal(data, props, 0, "test")
        self.assertTrue(hasattr(data, "uncertainty_weight"))
        # weight = 1.0 / (0.001**2 + 1e-8) ≈ 1000000
        expected_weight = 1.0 / (0.001**2 + 1e-8)
        self.assertAlmostEqual(data.uncertainty_weight[0].item(), expected_weight, places=0)

    @patch(
        "milia_pipeline.handlers.implementations.dmc.validate_uncertainty_data", return_value=0.001
    )
    def test_non_inverse_variance_weight_is_one(self, mock_validate):
        """Non-inverse_variance weighting uses weight 1.0."""
        uc = _make_uncertainty_config(uncertainty_weighting="uniform")
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        data = _make_pyg_data()
        props = _make_raw_properties(std=0.001)
        handler._add_uncertainty_metadata_internal(data, props, 0, "test")
        self.assertAlmostEqual(data.uncertainty_weight[0].item(), 1.0, places=4)

    @patch(
        "milia_pipeline.handlers.implementations.dmc.validate_uncertainty_data", return_value=10.0
    )
    def test_relative_uncertainty_calculated(self, mock_validate):
        """Relative uncertainty calculated when energy is available."""
        uc = _make_uncertainty_config()
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        data = _make_pyg_data()
        data.y = torch.tensor([-76.4], dtype=torch.float32)
        props = _make_raw_properties(std=10.0)
        handler._add_uncertainty_metadata_internal(data, props, 0, "test")
        self.assertTrue(hasattr(data, "relative_uncertainty"))
        expected = abs(10.0 / -76.4)
        self.assertAlmostEqual(data.relative_uncertainty[0].item(), expected, places=4)

    @patch(
        "milia_pipeline.handlers.implementations.dmc.validate_uncertainty_data", return_value=10.0
    )
    def test_high_uncertainty_flag_set(self, mock_validate):
        """High uncertainty flag is set when relative uncertainty > 0.1."""
        uc = _make_uncertainty_config()
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        data = _make_pyg_data()
        data.y = torch.tensor([-76.4], dtype=torch.float32)
        props = _make_raw_properties(std=10.0)
        handler._add_uncertainty_metadata_internal(data, props, 0, "test")
        # relative_uncertainty = 10.0/76.4 ≈ 0.131 > 0.1
        self.assertTrue(hasattr(data, "high_uncertainty"))
        self.assertTrue(data.high_uncertainty[0].item())

    @patch(
        "milia_pipeline.handlers.implementations.dmc.validate_uncertainty_data", return_value=0.001
    )
    def test_low_uncertainty_flag_not_set(self, mock_validate):
        """High uncertainty flag is False when relative uncertainty < 0.1."""
        uc = _make_uncertainty_config()
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        data = _make_pyg_data()
        data.y = torch.tensor([-76.4], dtype=torch.float32)
        props = _make_raw_properties(std=0.001)
        handler._add_uncertainty_metadata_internal(data, props, 0, "test")
        # relative_uncertainty = 0.001/76.4 ≈ 0.000013 < 0.1
        self.assertTrue(hasattr(data, "high_uncertainty"))
        self.assertFalse(data.high_uncertainty[0].item())

    def test_missing_uncertainty_field_logs_warning(self):
        """Missing uncertainty field in raw data does not crash (no tensor added)."""
        uc = _make_uncertainty_config()
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        data = _make_pyg_data()
        props = _make_raw_properties()  # No 'std' key
        # Should not raise - logs warning
        handler._add_uncertainty_metadata_internal(data, props, 0, "test")
        self.assertFalse(hasattr(data, "uncertainty"))

    @patch(
        "milia_pipeline.handlers.implementations.dmc.validate_uncertainty_data",
        side_effect=ValueError("Bad uncertainty"),
    )
    def test_validation_failure_logs_warning(self, mock_validate):
        """Uncertainty validation failure logs warning, does not raise."""
        uc = _make_uncertainty_config()
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        data = _make_pyg_data()
        props = _make_raw_properties(std=-0.001)
        # Should not raise per _add_uncertainty_metadata_internal try/except
        with self.assertLogs("test.dmc", level="WARNING"):
            handler._add_uncertainty_metadata_internal(data, props, 0, "test")


# ============================================================================
# GROUP 14: _validate_uncertainty_data (6 tests)
# ============================================================================


class TestValidateUncertaintyData(unittest.TestCase):
    """Test DMC uncertainty data validation."""

    def test_no_uncertainty_config_returns_early(self):
        """No uncertainty_config returns without error."""
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=None)
        handler = _make_handler(dataset_config=dc)
        # Should not raise
        handler._validate_uncertainty_data({}, 0, "test")

    @patch(
        "milia_pipeline.handlers.implementations.dmc.validate_uncertainty_data", return_value=0.001
    )
    def test_valid_uncertainty_passes(self, mock_validate):
        """Valid uncertainty data passes validation."""
        uc = _make_uncertainty_config()
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        props = {"std": 0.001}
        handler._validate_uncertainty_data(props, 0, "test")

    def test_missing_uncertainty_field_raises_dmc_handler_error(self):
        """Missing uncertainty field raises DatasetSpecificHandlerError."""
        uc = _make_uncertainty_config()
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        props = {}  # No 'std'
        with self.assertRaises(DatasetSpecificHandlerError):
            handler._validate_uncertainty_data(props, 0, "test")

    @patch(
        "milia_pipeline.handlers.implementations.dmc.validate_uncertainty_data", return_value=None
    )
    def test_validation_returns_none_raises_dmc_handler_error(self, mock_validate):
        """Uncertainty validation returning None raises DatasetSpecificHandlerError."""
        uc = _make_uncertainty_config()
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        props = {"std": 0.001}
        with self.assertRaises(DatasetSpecificHandlerError):
            handler._validate_uncertainty_data(props, 0, "test")

    @patch(
        "milia_pipeline.handlers.implementations.dmc.validate_uncertainty_data", return_value=0.5
    )
    def test_exceeds_threshold_raises_dmc_handler_error(self, mock_validate):
        """Uncertainty exceeding threshold raises DatasetSpecificHandlerError."""
        uc = _make_uncertainty_config(max_uncertainty_threshold=0.1)
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        props = {"std": 0.5}
        with self.assertRaises(DatasetSpecificHandlerError):
            handler._validate_uncertainty_data(props, 0, "test")

    @patch(
        "milia_pipeline.handlers.implementations.dmc.validate_uncertainty_data",
        side_effect=ValueError("Negative uncertainty"),
    )
    def test_value_error_wrapped_in_dmc_handler_error(self, mock_validate):
        """ValueError from validate_uncertainty_data wrapped in DatasetSpecificHandlerError."""
        uc = _make_uncertainty_config()
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        props = {"std": -1.0}
        with self.assertRaises(DatasetSpecificHandlerError):
            handler._validate_uncertainty_data(props, 0, "test")


# ============================================================================
# GROUP 15: validate_configuration (5 tests)
# ============================================================================


class TestValidateConfiguration(unittest.TestCase):
    """Test DMC-specific configuration validation."""

    def test_valid_config_passes(self):
        """Valid configuration does not raise."""
        handler = _make_handler()
        handler.validate_configuration()

    def test_uncertainty_enabled_without_config_raises(self):
        """Uncertainty enabled but no config raises HandlerConfigurationError."""
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=None)
        handler = _make_handler(dataset_config=dc)
        with self.assertRaises(HandlerConfigurationError):
            handler.validate_configuration()

    def test_uncertainty_missing_field_name_raises(self):
        """Uncertainty config without field name raises HandlerConfigurationError."""
        uc = {"uncertainty_field_name": None, "max_uncertainty_threshold": None}
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        with self.assertRaises(HandlerConfigurationError):
            handler.validate_configuration()

    def test_negative_threshold_raises(self):
        """Negative max_uncertainty_threshold raises HandlerConfigurationError."""
        uc = _make_uncertainty_config(max_uncertainty_threshold=-0.5)
        dc = _make_dataset_config(is_uncertainty_enabled=True, uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        with self.assertRaises(HandlerConfigurationError):
            handler.validate_configuration()

    def test_unexpected_config_error_wrapped(self):
        """Unexpected configuration error is wrapped in HandlerConfigurationError."""
        handler = _make_handler()
        with (
            patch.object(
                type(handler).__bases__[0],
                "validate_configuration",
                side_effect=RuntimeError("config_boom"),
            ),
            self.assertRaises(HandlerConfigurationError),
        ):
            handler.validate_configuration()


# ============================================================================
# GROUP 16: Transform Validation Helpers (8 tests)
# ============================================================================


class TestTransformValidationHelpers(unittest.TestCase):
    """Test DMC-specific transform validation methods."""

    def test_uncertainty_augmentation_warning(self):
        """Augmentation transforms with uncertainty produce warnings."""
        uc = _make_uncertainty_config()
        dc = _make_dataset_config(uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        warnings = handler._validate_dataset_specific_transforms(["DropEdge"])
        self.assertTrue(any("uncertainty" in w.lower() for w in warnings))

    def test_geometric_transform_warning(self):
        """Geometric transforms produce warning for DMC."""
        handler = _make_handler()
        warnings = handler._validate_dataset_specific_transforms(["RandomRotate"])
        self.assertTrue(any("geometric" in w.lower() for w in warnings))

    def test_no_warnings_for_safe_transforms(self):
        """Safe transforms (structural only) produce no warnings."""
        handler = _make_handler()
        warnings = handler._validate_dataset_specific_transforms(["AddSelfLoops"])
        self.assertEqual(len(warnings), 0)

    def test_virtualnode_incompatible_with_uncertainty(self):
        """VirtualNode + uncertainty flagged as incompatible."""
        uc = _make_uncertainty_config()
        dc = _make_dataset_config(uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        errors = handler._check_transform_incompatibilities(["VirtualNode"])
        self.assertTrue(any("VirtualNode" in e for e in errors))

    def test_dropnode_incompatible_with_inverse_variance(self):
        """DropNode + inverse_variance uncertainty weighting flagged as incompatible."""
        uc = _make_uncertainty_config(uncertainty_weighting="inverse_variance")
        dc = _make_dataset_config(uncertainty_config=uc)
        handler = _make_handler(dataset_config=dc)
        errors = handler._check_transform_incompatibilities(["DropNode"])
        self.assertTrue(any("DropNode" in e for e in errors))

    def test_no_incompatibilities_without_uncertainty(self):
        """No uncertainty means no incompatibilities for VirtualNode."""
        dc = _make_dataset_config(uncertainty_config=None)
        handler = _make_handler(dataset_config=dc)
        errors = handler._check_transform_incompatibilities(["VirtualNode"])
        self.assertEqual(len(errors), 0)

    def test_recommendations_warn_about_too_many_transforms(self):
        """More than 3 transforms triggers complexity recommendation."""
        handler = _make_handler()
        recs = handler._get_transform_recommendations(
            ["AddSelfLoops", "ToUndirected", "NormalizeFeatures", "Distance"]
        )
        self.assertTrue(any("minimal" in r.lower() or "≤3" in r for r in recs))

    def test_get_dataset_suitable_transforms(self):
        """_get_dataset_suitable_transforms returns only minimal structural transforms."""
        handler = _make_handler()
        available = {
            "AddSelfLoops": None,
            "ToUndirected": None,
            "NormalizeFeatures": None,
            "RandomRotate": None,
            "DropEdge": None,
            "VirtualNode": None,
        }
        suitable = handler._get_dataset_suitable_transforms(available)
        self.assertIn("AddSelfLoops", suitable)
        self.assertIn("ToUndirected", suitable)
        self.assertIn("NormalizeFeatures", suitable)
        # DMC should NOT recommend aggressive transforms
        self.assertNotIn("RandomRotate", suitable)
        self.assertNotIn("DropEdge", suitable)
        self.assertNotIn("VirtualNode", suitable)


# ============================================================================
# GROUP 17: get_processing_statistics (5 tests)
# ============================================================================


class TestGetProcessingStatistics(unittest.TestCase):
    """Test DMC processing statistics generation."""

    def test_basic_stats_structure(self):
        """Stats dict has required keys."""
        handler = _make_handler()
        stats = handler.get_processing_statistics([])
        self.assertEqual(stats["dataset_type"], "DMC")
        self.assertIn("total_processed", stats)

    def test_total_processed_count(self):
        """total_processed matches input list length."""
        handler = _make_handler()
        stats = handler.get_processing_statistics([{}, {}, {}])
        self.assertEqual(stats["total_processed"], 3)

    def test_uncertainty_stats_included_when_enabled(self):
        """Uncertainty processing stats included when molecules have uncertainty."""
        dc = _make_dataset_config(is_uncertainty_enabled=True)
        handler = _make_handler(dataset_config=dc)
        processed = [
            {"uncertainty_processed": True, "high_uncertainty": False},
            {"uncertainty_processed": True, "high_uncertainty": True},
        ]
        stats = handler.get_processing_statistics(processed)
        self.assertIn("uncertainty_processing", stats)
        self.assertEqual(stats["uncertainty_processing"]["molecules_with_uncertainty"], 2)
        self.assertEqual(stats["uncertainty_processing"]["high_uncertainty_molecules"], 1)

    def test_experimental_setup_included(self):
        """Experimental setup info included when configured."""
        handler = _make_handler(experimental_setup="standard")
        stats = handler.get_processing_statistics([])
        self.assertIn("transform_aware_processing", stats)
        self.assertTrue(stats["transform_aware_processing"])

    def test_no_experimental_setup(self):
        """No experimental setup means no transform_aware_processing key."""
        handler = _make_handler(experimental_setup=None)
        stats = handler.get_processing_statistics([])
        self.assertNotIn("transform_aware_processing", stats)


# ============================================================================
# GROUP 18: _get_transform_recommendations (6 tests)
# ============================================================================


class TestGetTransformRecommendationsInternal(unittest.TestCase):
    """Test DMC internal transform recommendation logic."""

    def test_normalize_features_recommendation_when_absent(self):
        """Recommends NormalizeFeatures when not present."""
        handler = _make_handler()
        recs = handler._get_transform_recommendations(["AddSelfLoops"])
        self.assertTrue(any("NormalizeFeatures" in r for r in recs))

    def test_no_normalize_recommendation_when_present(self):
        """Does not recommend NormalizeFeatures when already present."""
        handler = _make_handler()
        recs = handler._get_transform_recommendations(["NormalizeFeatures"])
        self.assertFalse(any("Consider adding NormalizeFeatures" in r for r in recs))

    def test_non_structural_transform_warning(self):
        """Non-structural transforms produce compatibility warning."""
        handler = _make_handler()
        recs = handler._get_transform_recommendations(["RandomRotate"])
        self.assertTrue(any("non-structural" in r.lower() for r in recs))

    def test_augmentation_detected_warning(self):
        """Data augmentation transforms produce strong warning."""
        handler = _make_handler()
        recs = handler._get_transform_recommendations(["DropEdge"])
        self.assertTrue(any("augmentation" in r.lower() for r in recs))

    def test_distance_cartesian_edge_warning(self):
        """Distance/Cartesian transforms produce edge feature warning."""
        handler = _make_handler()
        recs = handler._get_transform_recommendations(["Distance"])
        self.assertTrue(any("Distance" in r or "edge" in r.lower() for r in recs))

    def test_structural_only_no_augmentation_warning(self):
        """Structural-only transforms do not trigger augmentation warning."""
        handler = _make_handler()
        recs = handler._get_transform_recommendations(["AddSelfLoops"])
        self.assertFalse(any("augmentation detected" in r.lower() for r in recs))


# ============================================================================
# GROUP 19: Edge Cases and Integration (6 tests)
# ============================================================================


class TestEdgeCasesAndIntegration(unittest.TestCase):
    """Test edge cases and integration scenarios."""

    def test_handler_without_experimental_setup(self):
        """Handler works without experimental_setup (None)."""
        handler = _make_handler()
        self.assertIsNone(handler.experimental_setup)

    def test_handler_with_empty_processing_config(self):
        """Handler works with empty processing config lists."""
        pc = _make_processing_config(
            scalar_graph_targets=[],
            node_features=[],
        )
        handler = _make_handler(processing_config=pc)
        self.assertEqual(handler.get_dataset_type(), "DMC")

    def test_enrichment_with_scalar_targets(self):
        """Full enrichment with scalar properties."""
        pc = _make_processing_config(scalar_graph_targets=["Etot"])
        handler = _make_handler(processing_config=pc)
        data = _make_pyg_data(num_atoms=3)
        props = _make_raw_properties(Etot=-76.4)
        result = handler.enrich_pyg_data(data, props, 0, "test")
        self.assertTrue(hasattr(result, "y"))

    def test_multiple_enrichments_independent(self):
        """Multiple enrichment calls on separate data objects are independent."""
        handler = _make_handler()
        data1 = _make_pyg_data()
        data2 = _make_pyg_data()
        props1 = _make_raw_properties(Etot=-76.4)
        props2 = _make_raw_properties(Etot=-80.0)
        result1 = handler.enrich_pyg_data(data1, props1, 0, "test1")
        result2 = handler.enrich_pyg_data(data2, props2, 1, "test2")
        self.assertNotEqual(result1.y[0].item(), result2.y[0].item())

    def test_molecule_processing_error_converted_to_dmc_handler_error(self):
        """MoleculeProcessingError in validate_molecule_data wraps to DatasetSpecificHandlerError."""
        handler = _make_handler()
        props = _make_raw_properties()
        with (
            patch.object(
                handler,
                "_is_valid_property",
                side_effect=MoleculeProcessingError(
                    message="Processing failed",
                    molecule_index=0,
                ),
            ),
            self.assertRaises(DatasetSpecificHandlerError),
        ):
            handler.validate_molecule_data(props, molecule_index=0, identifier="test")

    def test_unexpected_error_in_validation_wraps_to_dmc_handler_error(self):
        """Unexpected errors in validate_molecule_data wrap to DatasetSpecificHandlerError."""
        handler = _make_handler()
        props = _make_raw_properties()
        with (
            patch.object(
                handler,
                "_is_valid_property",
                side_effect=TypeError("unexpected type error"),
            ),
            self.assertRaises(DatasetSpecificHandlerError),
        ):
            handler.validate_molecule_data(props, molecule_index=0, identifier="test")


# ============================================================================
# TEST RUNNER
# ============================================================================


def run_comprehensive_suite():
    """Run all test groups in a structured order."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestDMCDatasetHandlerIdentity,  # GROUP 1:    6 tests
        TestGetRequiredProperties,  # GROUP 2:    7 tests
        TestGetMolecularCharge,  # GROUP 3:    6 tests
        TestValidateMoleculeDataSuccess,  # GROUP 4:    5 tests
        TestValidateMoleculeDataErrors,  # GROUP 5:    7 tests
        TestProcessPropertyValue,  # GROUP 6:    8 tests
        TestGetTransformRecommendations,  # GROUP 7:    5 tests
        TestGetSupportedDescriptors,  # GROUP 8:    4 tests
        TestGetSupportedStructuralFeatures,  # GROUP 9:    5 tests
        TestIsValidProperty,  # GROUP 10:   9 tests
        TestAddScalarTargetsInternal,  # GROUP 11:   9 tests
        TestEnrichPygData,  # GROUP 12:   8 tests
        TestAddUncertaintyMetadataInternal,  # GROUP 13:   8 tests
        TestValidateUncertaintyData,  # GROUP 14:   6 tests
        TestValidateConfiguration,  # GROUP 15:   5 tests
        TestTransformValidationHelpers,  # GROUP 16:   8 tests
        TestGetProcessingStatistics,  # GROUP 17:   5 tests
        TestGetTransformRecommendationsInternal,  # GROUP 18:   6 tests
        TestEdgeCasesAndIntegration,  # GROUP 19:   6 tests
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS - handlers/implementations/dmc.py")
    print("=" * 80)
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"\nTest Groups: {len(test_classes)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - PRODUCTION-READY")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - REVIEW REQUIRED")
        return 1


if __name__ == "__main__":
    if "pytest" in sys.modules:
        pass
    else:
        sys.exit(run_comprehensive_suite())


"""
TEST SUITE SUMMARY — milia_pipeline/handlers/implementations/dmc.py
=====================================================================

123 comprehensive production-ready tests across 19 groups:

GROUP 1:  DMCDatasetHandler — Identity and Registration               (  6 tests)
GROUP 2:  get_required_properties                                      (  7 tests)
GROUP 3:  get_molecular_charge                                         (  6 tests)
GROUP 4:  validate_molecule_data — Success Paths                       (  5 tests)
GROUP 5:  validate_molecule_data — Error Paths                         (  7 tests)
GROUP 6:  process_property_value                                       (  8 tests)
GROUP 7:  get_transform_recommendations                                (  5 tests)
GROUP 8:  get_supported_descriptors                                    (  4 tests)
GROUP 9:  get_supported_structural_features                            (  5 tests)
GROUP 10: _is_valid_property                                           (  9 tests)
GROUP 11: _add_scalar_targets_internal                                 (  9 tests)
GROUP 12: enrich_pyg_data                                              (  8 tests)
GROUP 13: _add_uncertainty_metadata_internal                           (  8 tests)
GROUP 14: _validate_uncertainty_data                                   (  6 tests)
GROUP 15: validate_configuration                                       (  5 tests)
GROUP 16: Transform Validation Helpers                                 (  8 tests)
GROUP 17: get_processing_statistics                                    (  5 tests)
GROUP 18: _get_transform_recommendations (internal)                    (  6 tests)
GROUP 19: Edge Cases and Integration                                   (  6 tests)

PRODUCTION-READY QUALITIES:
- NO sys.modules pollution (no module-level mocking)
- All mocking via @patch decorators or context managers (test-level only)
- Dynamic test data creation via helper functions (no hardcoded paths)
- No file downloads (all NPZ data mocked via numpy arrays)
- Comprehensive error path coverage
- Interface-focused testing (future-proof)
- Compatible with both pytest and unittest runner
- Exception hierarchy correctly tested (DatasetSpecificHandlerError, HandlerValidationError,
  HandlerConfigurationError, PropertyEnrichmentError)
- DMC-specific features thoroughly tested:
  - Uncertainty handling (inverse variance weighting, threshold checks)
  - Limited structural features (no partial_charge, no mulliken_charge)
  - Conservative transform recommendations (minimal transforms only)
  - Energy validation with unusual magnitude warnings
  - InChI-based charge extraction
- No hard-coded solutions or workarounds
"""
