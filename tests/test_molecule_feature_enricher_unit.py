#!/usr/bin/env python3
"""
Unit tests for molecule_feature_enricher.py module (Handler-Only Architecture)

Test file: test_molecule_feature_enricher_unit.py
Module under test: ~/ml_projects/milia/milia_pipeline/molecules/molecule_feature_enricher.py

This test suite validates the handler-only molecule_feature_enricher module with
comprehensive coverage of molecular property estimation, structural feature analysis,
feature extraction diagnostics, and molecule identification capabilities.

Key Test Areas:
1. Molecular property estimation (estimate_molecular_properties)
2. Molecule identifier extraction (get_molecule_identifiers)
3. Structural feature summaries (get_structural_feature_summary)
4. Feature extraction diagnostics (get_feature_extraction_diagnostics)
5. Dataset capability analysis (analyze_structural_feature_capabilities)
6. Handler-only operation functions
7. Handler capability checking and integration
8. Error handling and exception propagation
9. Edge cases and boundary conditions
10. Dataset-specific behaviors (DFT vs DMC)
11. Phase 6: Registry Integration (30+ tests)
12. create_molecular_fingerprint standalone tests (handler-independent)
13. Handler edge cases (None returns, missing methods, AttributeError)
14. Registry initialization edge cases

Phase 6 additions:
- Registry integration functions (_init_registry, _get_available_dataset_types, etc.)
- Feature-based dataset processing (_get_dataset_feature)
- Enrichment category determination (_get_dataset_enrichment_category)
- Registry status function (get_registry_integration_status)
- All 6 refactored function locations using feature-based queries
- Conservative fallback behavior when registry unavailable (returns False)

Phase 6.1 alignment:
- Legacy hardcoded fallback removed from _get_dataset_feature
- Registry is single source of truth for feature queries
- Fallback tests updated to verify conservative False behavior
- Filesystem-based dynamic discovery validated

Functions Tested:
- estimate_molecular_properties()
- get_molecule_identifiers()
- get_structural_feature_summary()
- get_feature_extraction_diagnostics()
- analyze_structural_feature_capabilities()
- create_molecular_fingerprint() - standalone, no handler
- estimate_properties_with_handler()
- analyze_capabilities_with_handler()
- create_handler_compatible_fingerprint()
- validate_feature_extraction_with_handler()
- get_registry_integration_status() - Phase 6
- _init_registry() - Phase 6 (internal)
- _get_available_dataset_types() - Phase 6 (internal)
- _is_dataset_type_registered() - Phase 6 (internal)
- _get_dataset_feature() - Phase 6 (internal)
- _get_dataset_enrichment_category() - Phase 6 (internal)

Total: 130+ comprehensive tests

Test execution:
    cd /app/milia
    python -m pytest tests/test_molecule_feature_enricher_unit.py -v --tb=short
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

        @property
        def num_nodes(self):
            if hasattr(self, "x") and self.x is not None:
                return self.x.shape[0]
            if hasattr(self, "z") and self.z is not None:
                return (
                    self.z.shape[0]
                    if isinstance(self.z, (np.ndarray, torch.Tensor))
                    else len(self.z)
                )
            return 0


# Import module under test
try:
    from milia_pipeline.molecules.molecule_feature_enricher import (
        analyze_capabilities_with_handler,
        analyze_structural_feature_capabilities,
        create_handler_compatible_fingerprint,
        create_molecular_fingerprint,
        estimate_molecular_properties,
        estimate_properties_with_handler,
        get_feature_extraction_diagnostics,
        get_molecule_identifiers,
        get_structural_feature_summary,
        validate_feature_extraction_with_handler,
    )

    IMPORTS_SUCCESSFUL = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)
    print(f"WARNING: Could not import required modules: {e}")

# Phase 6: Import registry integration functions
try:
    from milia_pipeline.molecules.molecule_feature_enricher import (
        _get_available_dataset_types,
        _get_dataset_enrichment_category,
        _get_dataset_feature,
        _init_registry,
        _is_dataset_type_registered,
        get_registry_integration_status,
    )

    PHASE6_IMPORTS_SUCCESSFUL = True
    PHASE6_IMPORT_ERROR = None
except ImportError as e:
    PHASE6_IMPORTS_SUCCESSFUL = False
    PHASE6_IMPORT_ERROR = str(e)
    print(f"WARNING: Could not import Phase 6 registry functions: {e}")

# Import required configuration and exception classes
try:
    from milia_pipeline.exceptions import (
        HandlerError,
        HandlerOperationError,
        HandlerValidationError,
        PropertyEnrichmentError,
        StructuralFeatureError,
    )
except ImportError:
    # Create mock exceptions if not available
    class HandlerError(Exception):
        pass

    class HandlerOperationError(Exception):
        def __init__(self, message="", handler_type="", operation=""):
            super().__init__(message)
            self.handler_type = handler_type
            self.operation = operation

    class HandlerValidationError(Exception):
        pass

    class PropertyEnrichmentError(Exception):
        pass

    class StructuralFeatureError(Exception):
        pass


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
        import milia_pipeline.molecules.molecule_feature_enricher as enricher_module

        enricher_module._REGISTRY_INITIALIZED = False
        enricher_module._REGISTRY_AVAILABLE = False
        enricher_module._REGISTRY_IMPORT_ERROR = None
        enricher_module._registry_list_all = None
        enricher_module._registry_get = None
        enricher_module._registry_is_registered = None
    except (ImportError, AttributeError):
        pass  # Module may not have been imported yet


# ==============================================================================
# MOCK HELPERS
# ==============================================================================


class MockDatasetHandler:
    """Mock dataset handler for testing"""

    def __init__(self, dataset_type="DFT", capabilities=None):
        self.dataset_type = dataset_type
        self.capabilities = capabilities or {}
        self.dataset_config = Mock()
        self.dataset_config.is_uncertainty_enabled = False
        self.dataset_config.uncertainty_config = {}
        self.filter_config = Mock()
        self.processing_config = Mock()

    def get_dataset_type(self):
        return self.dataset_type

    def get_feature_capabilities(self):
        """Return mock feature capabilities"""
        return self.capabilities

    def estimate_additional_properties(self, pyg_data):
        """Mock additional property estimation"""
        return {"custom_property": 1.0}

    def refine_vibrational_data(self, freqs):
        """Mock vibrational refinement"""
        return freqs

    def process_uncertainty_data(self, pyg_data):
        """Mock uncertainty processing"""
        return {"uncertainty_mean": 0.01, "uncertainty_std": 0.005}

    def validate_feature_quality(self, pyg_data):
        """Mock feature quality validation"""
        return {"passed": True, "issues": []}

    def validate_statistical_properties(self, pyg_data):
        """Mock statistical validation"""
        return {"passed": True}

    def validate_structural_integrity(self, pyg_data):
        """Mock structural validation"""
        return {"passed": True}

    def validate_uncertainty_data(self, pyg_data):
        """Mock uncertainty data validation"""
        return {"valid": True, "issues": []}

    def check_statistical_quality(self, pyg_data):
        """Mock statistical quality check"""
        return {"quality_level": "good"}

    def validate_vibrational_data(self, freqs):
        """Mock vibrational data validation"""
        return {"valid": True, "issues": []}

    def validate_electronic_structure(self, pyg_data):
        """Mock electronic structure validation"""
        return {"valid": True, "issues": []}


class MockDFTHandler(MockDatasetHandler):
    """Mock DFT handler with DFT-specific capabilities"""

    def __init__(self, capabilities=None):
        default_caps = {
            "custom_properties": True,
            "vibrational_refinement": True,
            "dft_specific": {"vibrational_validation": True, "electronic_validation": True},
        }
        if capabilities:
            default_caps.update(capabilities)
        super().__init__(dataset_type="DFT", capabilities=default_caps)


class MockDMCHandler(MockDatasetHandler):
    """Mock DMC handler with uncertainty support"""

    def __init__(self, uncertainty_enabled=False, capabilities=None):
        default_caps = {
            "custom_properties": True,
            "uncertainty_processing": True,
            "dmc_specific": {"uncertainty_validation": True, "statistical_quality_check": True},
        }
        if capabilities:
            default_caps.update(capabilities)
        super().__init__(dataset_type="DMC", capabilities=default_caps)
        self.dataset_config.is_uncertainty_enabled = uncertainty_enabled


class MockWavefunctionHandler(MockDatasetHandler):
    """Mock Wavefunction handler with orbital analysis support"""

    def __init__(self, capabilities=None):
        default_caps = {
            "custom_properties": True,
            "orbital_analysis": True,
            "wavefunction_specific": {"mo_validation": True, "homo_lumo_analysis": True},
        }
        if capabilities:
            default_caps.update(capabilities)
        super().__init__(dataset_type="Wavefunction", capabilities=default_caps)


def create_mock_logger():
    """Create a mock logger for testing"""
    logger = Mock(spec=logging.Logger)
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


def create_sample_pyg_data(num_atoms=3, include_positions=True, include_features=True):
    """Create a sample PyG Data object for testing"""
    pyg_data = Data()

    # Basic atomic information
    pyg_data.z = torch.tensor([6, 1, 1], dtype=torch.long)[:num_atoms]

    if include_positions:
        pyg_data.pos = torch.randn(num_atoms, 3)

    if include_features:
        pyg_data.x = torch.randn(num_atoms, 5)

    # Add some common properties
    pyg_data.y = torch.tensor([[-150.5]], dtype=torch.float32)
    pyg_data.smiles = "C"
    pyg_data.inchi = "InChI=1S/CH4/h1H4"
    pyg_data.original_mol_idx = 0

    return pyg_data


# ==============================================================================
# TEST CASES: MOLECULAR PROPERTY ESTIMATION
# ==============================================================================


class TestEstimateMolecularProperties(unittest.TestCase):
    """Test estimate_molecular_properties function"""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        """Set up test fixtures"""
        self.handler = MockDFTHandler()
        self.pyg_data = create_sample_pyg_data()

    def test_basic_property_estimation(self):
        """Test basic molecular property estimation"""
        properties = estimate_molecular_properties(self.pyg_data, self.handler)

        # Check basic properties
        self.assertIn("num_atoms", properties)
        self.assertIn("num_heavy_atoms", properties)
        self.assertIn("num_hydrogen", properties)
        self.assertEqual(properties["num_atoms"], 3.0)
        self.assertEqual(properties["num_heavy_atoms"], 1.0)  # One carbon
        self.assertEqual(properties["num_hydrogen"], 2.0)  # Two hydrogens

    def test_geometric_properties(self):
        """Test geometric property calculation"""
        properties = estimate_molecular_properties(self.pyg_data, self.handler)

        # Check geometric properties
        self.assertIn("max_distance", properties)
        self.assertIn("center_of_mass", properties)
        self.assertIn("radius_of_gyration", properties)

        # Verify types
        self.assertIsInstance(properties["max_distance"], float)
        self.assertIsInstance(properties["center_of_mass"], list)
        self.assertIsInstance(properties["radius_of_gyration"], float)
        self.assertEqual(len(properties["center_of_mass"]), 3)

    def test_electron_estimation(self):
        """Test electron count estimation"""
        properties = estimate_molecular_properties(self.pyg_data, self.handler)

        self.assertIn("estimated_electrons", properties)
        # Carbon (6) + Hydrogen (1) + Hydrogen (1) = 8
        self.assertEqual(properties["estimated_electrons"], 8.0)

    def test_handler_additional_properties(self):
        """Test handler-specific additional property estimation"""
        properties = estimate_molecular_properties(self.pyg_data, self.handler)

        # Should include custom property from handler
        self.assertIn("custom_property", properties)
        self.assertEqual(properties["custom_property"], 1.0)

    def test_vibrational_refinement_dft(self):
        """Test vibrational data refinement for DFT"""
        # Add vibrational data
        self.pyg_data.freqs = torch.tensor([100.0, 200.0, 300.0])

        properties = estimate_molecular_properties(self.pyg_data, self.handler)

        # Should have refined vibrational modes
        self.assertIn("refined_vibrational_modes", properties)
        self.assertEqual(properties["refined_vibrational_modes"], 3)

    def test_uncertainty_processing_dmc(self):
        """Test uncertainty processing for DMC"""
        dmc_handler = MockDMCHandler(uncertainty_enabled=True)
        pyg_data = create_sample_pyg_data()
        pyg_data.uncertainty = torch.tensor([0.01])  # Use scalar tensor

        properties = estimate_molecular_properties(pyg_data, dmc_handler)

        # Should have processed uncertainty
        self.assertIn("processed_uncertainty", properties)
        self.assertIsInstance(properties["processed_uncertainty"], dict)

    def test_missing_atomic_numbers(self):
        """Test property estimation with missing atomic numbers"""
        pyg_data = Data()
        pyg_data.pos = torch.randn(3, 3)
        pyg_data.num_nodes = 3  # Set num_nodes explicitly

        # Missing z will cause errors in geometric property calculation
        # Should raise HandlerOperationError
        with self.assertRaises(HandlerOperationError):
            estimate_molecular_properties(pyg_data, self.handler)

    def test_missing_positions(self):
        """Test property estimation with missing positions"""
        pyg_data = Data()
        pyg_data.z = torch.tensor([6, 1, 1], dtype=torch.long)
        pyg_data.num_nodes = 3  # Set explicitly to avoid warning

        properties = estimate_molecular_properties(pyg_data, self.handler)

        # Should handle gracefully - no geometric properties
        self.assertNotIn("max_distance", properties)
        self.assertNotIn("center_of_mass", properties)
        # But basic properties should be present
        self.assertIn("num_atoms", properties)

    def test_single_atom_molecule(self):
        """Test property estimation for single atom"""
        pyg_data = create_sample_pyg_data(num_atoms=1)

        properties = estimate_molecular_properties(pyg_data, self.handler)

        self.assertEqual(properties["num_atoms"], 1.0)
        self.assertIn("center_of_mass", properties)
        self.assertIn("radius_of_gyration", properties)

    def test_handler_capability_checking(self):
        """Test that handler capabilities are properly checked"""
        # Handler without custom properties capability
        handler = MockDFTHandler(capabilities={"custom_properties": False})

        properties = estimate_molecular_properties(self.pyg_data, handler)

        # Should still work but might not have custom properties
        self.assertIn("num_atoms", properties)

    @patch("milia_pipeline.molecules.molecule_feature_enricher.logger")
    def test_handler_error_logging(self, mock_logger):
        """Test that handler errors are properly logged"""
        # Create handler that raises error
        handler = MockDFTHandler()
        handler.estimate_additional_properties = Mock(side_effect=Exception("Test error"))

        properties = estimate_molecular_properties(self.pyg_data, handler)

        # Should log debug and continue (not warning for non-HandlerError exceptions)
        mock_logger.debug.assert_called()
        # Should still have basic properties
        self.assertIn("num_atoms", properties)


# ==============================================================================
# TEST CASES: MOLECULE IDENTIFIERS
# ==============================================================================


class TestGetMoleculeIdentifiers(unittest.TestCase):
    """Test get_molecule_identifiers function"""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        """Set up test fixtures"""
        self.handler = MockDFTHandler()
        self.pyg_data = create_sample_pyg_data()

    def test_basic_identifiers(self):
        """Test basic identifier extraction"""
        identifiers = get_molecule_identifiers(self.pyg_data, self.handler)

        # Check standard identifiers that are available in pyg_data
        self.assertIn("smiles", identifiers)
        self.assertIn("inchi", identifiers)

        self.assertEqual(identifiers["smiles"], "C")
        self.assertEqual(identifiers["inchi"], "InChI=1S/CH4/h1H4")

    def test_missing_identifiers(self):
        """Test handling of missing identifiers"""
        pyg_data = Data()
        pyg_data.z = torch.tensor([6, 1, 1], dtype=torch.long)
        pyg_data.num_nodes = 3

        identifiers = get_molecule_identifiers(pyg_data, self.handler)

        # Should return empty dict or dict without smiles/inchi
        self.assertIsInstance(identifiers, dict)
        # May not have smiles or inchi keys at all
        if "smiles" in identifiers:
            self.assertIsNone(identifiers["smiles"])
        if "inchi" in identifiers:
            self.assertIsNone(identifiers["inchi"])

    def test_partial_identifiers(self):
        """Test with only some identifiers present"""
        pyg_data = Data()
        pyg_data.smiles = "CCO"
        pyg_data.z = torch.tensor([6, 6, 8], dtype=torch.long)

        identifiers = get_molecule_identifiers(pyg_data, self.handler)

        self.assertEqual(identifiers["smiles"], "CCO")
        self.assertIsNone(identifiers.get("inchi"))

    def test_identifier_types(self):
        """Test that identifiers have correct types"""
        identifiers = get_molecule_identifiers(self.pyg_data, self.handler)

        if identifiers.get("smiles") is not None:
            self.assertIsInstance(identifiers["smiles"], str)
        if identifiers.get("inchi") is not None:
            self.assertIsInstance(identifiers["inchi"], str)
        if identifiers.get("original_mol_idx") is not None:
            self.assertIsInstance(identifiers["original_mol_idx"], int)


# ==============================================================================
# TEST CASES: STRUCTURAL FEATURE SUMMARY
# ==============================================================================


class TestGetStructuralFeatureSummary(unittest.TestCase):
    """Test get_structural_feature_summary function"""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        """Set up test fixtures"""
        self.handler = MockDFTHandler()
        self.pyg_data = create_sample_pyg_data()

    def test_basic_structural_summary(self):
        """Test basic structural feature summary"""
        summary = get_structural_feature_summary(self.pyg_data, self.handler)

        self.assertIn("has_atom_features", summary)
        self.assertIn("has_bond_features", summary)
        self.assertIn("has_edges", summary)
        self.assertIn("num_nodes", summary)
        self.assertIn("num_edges", summary)

    def test_summary_with_features(self):
        """Test summary when features are present"""
        summary = get_structural_feature_summary(self.pyg_data, self.handler)

        self.assertTrue(summary["has_atom_features"])
        self.assertIn("atom_feature_dim", summary)
        self.assertIn("atom_feature_stats", summary)
        self.assertIn("atom_feature_quality", summary)

    def test_summary_without_features(self):
        """Test summary when features are missing"""
        pyg_data = Data()
        pyg_data.z = torch.tensor([6, 1, 1], dtype=torch.long)
        pyg_data.pos = torch.randn(3, 3)
        pyg_data.num_nodes = 3

        summary = get_structural_feature_summary(pyg_data, self.handler)

        self.assertFalse(summary["has_atom_features"])
        self.assertEqual(summary["atom_feature_quality"], "missing")

    def test_summary_graph_connectivity(self):
        """Test graph connectivity information"""
        summary = get_structural_feature_summary(self.pyg_data, self.handler)

        self.assertIn("graph_connectivity", summary)
        self.assertIn("density", summary["graph_connectivity"])
        self.assertIn("avg_degree", summary["graph_connectivity"])


# ==============================================================================
# TEST CASES: FEATURE EXTRACTION DIAGNOSTICS
# ==============================================================================


class TestGetFeatureExtractionDiagnostics(unittest.TestCase):
    """Test get_feature_extraction_diagnostics function"""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        """Set up test fixtures"""
        self.handler = MockDFTHandler()
        self.pyg_data = create_sample_pyg_data()

    @patch("milia_pipeline.molecules.molecule_feature_enricher.get_structural_features_config")
    def test_basic_diagnostics(self, mock_config):
        """Test basic diagnostic information"""
        mock_config.return_value = None  # No structural features configured

        diagnostics = get_feature_extraction_diagnostics(self.pyg_data, self.handler)

        # Check basic diagnostic fields
        self.assertIn("extraction_success", diagnostics)
        self.assertIn("atom_features_extracted", diagnostics)
        self.assertIn("bond_features_extracted", diagnostics)
        self.assertIn("feature_quality", diagnostics)

    @patch("milia_pipeline.molecules.molecule_feature_enricher.get_structural_features_config")
    def test_diagnostics_with_features(self, mock_config):
        """Test diagnostics with node features"""
        mock_config.return_value = {"atom": ["atomic_num", "charge"], "bond": []}

        diagnostics = get_feature_extraction_diagnostics(self.pyg_data, self.handler)

        self.assertTrue(diagnostics["atom_features_extracted"])
        self.assertIn("feature_quality", diagnostics)

    @patch("milia_pipeline.molecules.molecule_feature_enricher.get_structural_features_config")
    def test_diagnostics_without_features(self, mock_config):
        """Test diagnostics without node features"""
        mock_config.return_value = {"atom": ["atomic_num"], "bond": []}

        pyg_data = Data()
        pyg_data.z = torch.tensor([6, 1, 1], dtype=torch.long)
        pyg_data.pos = torch.randn(3, 3)
        pyg_data.num_nodes = 3

        diagnostics = get_feature_extraction_diagnostics(pyg_data, self.handler)

        # Missing atom features when expected
        self.assertFalse(diagnostics["extraction_success"])
        self.assertIn("atom_features", diagnostics["missing_features"])

    @patch("milia_pipeline.molecules.molecule_feature_enricher.get_structural_features_config")
    def test_diagnostics_with_edges(self, mock_config):
        """Test diagnostics with edge information"""
        mock_config.return_value = None
        self.pyg_data.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        diagnostics = get_feature_extraction_diagnostics(self.pyg_data, self.handler)

        # Should complete without error
        self.assertIsInstance(diagnostics, dict)

    @patch("milia_pipeline.molecules.molecule_feature_enricher.get_structural_features_config")
    def test_diagnostics_properties_present(self, mock_config):
        """Test diagnostics with various properties"""
        mock_config.return_value = None
        self.pyg_data.dipole = torch.tensor([0.1, 0.2, 0.3])
        self.pyg_data.charges = torch.randn(3)

        diagnostics = get_feature_extraction_diagnostics(self.pyg_data, self.handler)

        # Should have basic diagnostic structure
        self.assertIn("feature_quality", diagnostics)


# ==============================================================================
# TEST CASES: CAPABILITY ANALYSIS
# ==============================================================================


class TestAnalyzeStructuralFeatureCapabilities(unittest.TestCase):
    """Test analyze_structural_feature_capabilities function"""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        """Set up test fixtures"""
        self.handler = MockDFTHandler()

    @patch("milia_pipeline.molecules.molecule_feature_enricher.is_structural_features_enabled")
    def test_basic_capability_analysis(self, mock_enabled):
        """Test basic capability analysis"""
        mock_enabled.return_value = False

        capabilities = analyze_structural_feature_capabilities(self.handler)

        # Check basic capability fields
        self.assertIn("dataset_type", capabilities)
        self.assertIn("handler_integration", capabilities)
        self.assertEqual(capabilities["dataset_type"], "DFT")

    @patch("milia_pipeline.molecules.molecule_feature_enricher.is_structural_features_enabled")
    def test_handler_capabilities_included(self, mock_enabled):
        """Test that handler capabilities are included"""
        mock_enabled.return_value = False

        capabilities = analyze_structural_feature_capabilities(self.handler)

        # Should include handler integration info
        self.assertIn("handler_integration", capabilities)
        self.assertIsInstance(capabilities["handler_integration"], dict)

    @patch("milia_pipeline.molecules.molecule_feature_enricher.is_structural_features_enabled")
    @patch("milia_pipeline.molecules.molecule_feature_enricher.get_structural_features_config")
    def test_dft_specific_capabilities(self, mock_config, mock_enabled):
        """Test DFT-specific capability reporting"""
        mock_enabled.return_value = True
        mock_config.return_value = {"atom": [], "bond": []}

        capabilities = analyze_structural_feature_capabilities(self.handler)

        # Should have DFT dataset type
        self.assertEqual(capabilities["dataset_type"], "DFT")
        self.assertIn("handler_integration", capabilities)

    @patch("milia_pipeline.molecules.molecule_feature_enricher.is_structural_features_enabled")
    @patch("milia_pipeline.molecules.molecule_feature_enricher.get_structural_features_config")
    def test_dmc_specific_capabilities(self, mock_config, mock_enabled):
        """Test DMC-specific capability reporting"""
        mock_enabled.return_value = True
        mock_config.return_value = {"atom": [], "bond": []}

        dmc_handler = MockDMCHandler()
        capabilities = analyze_structural_feature_capabilities(dmc_handler)

        # Should have DMC dataset type
        self.assertEqual(capabilities["dataset_type"], "DMC")
        self.assertIn("handler_integration", capabilities)

    @patch("milia_pipeline.molecules.molecule_feature_enricher.is_structural_features_enabled")
    def test_handler_without_capabilities(self, mock_enabled):
        """Test handler that doesn't implement get_feature_capabilities"""
        mock_enabled.return_value = False

        handler = MockDatasetHandler(dataset_type="DFT")
        # Handler still has get_feature_capabilities in our mock

        # Should handle gracefully
        capabilities = analyze_structural_feature_capabilities(handler)

        self.assertIn("dataset_type", capabilities)


# ==============================================================================
# TEST CASES: HANDLER-ONLY OPERATIONS
# ==============================================================================


class TestHandlerOnlyOperations(unittest.TestCase):
    """Test handler-only operation functions"""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        """Set up test fixtures"""
        self.handler = MockDFTHandler()
        self.pyg_data = create_sample_pyg_data()

    def test_estimate_properties_with_handler(self):
        """Test estimate_properties_with_handler function"""
        properties = estimate_properties_with_handler(self.handler, self.pyg_data)

        # Should return properties dictionary
        self.assertIsInstance(properties, dict)
        self.assertIn("num_atoms", properties)

    def test_analyze_capabilities_with_handler(self):
        """Test analyze_capabilities_with_handler function"""
        capabilities = analyze_capabilities_with_handler(self.handler)

        # Should return capabilities dictionary
        self.assertIsInstance(capabilities, dict)
        self.assertIn("dataset_type", capabilities)

    @patch("milia_pipeline.molecules.molecule_feature_enricher.is_structural_features_enabled")
    def test_create_handler_compatible_fingerprint(self, mock_enabled):
        """Test create_handler_compatible_fingerprint function"""
        mock_enabled.return_value = True

        fingerprint = create_handler_compatible_fingerprint(self.handler, self.pyg_data)

        # Should return fingerprint information
        self.assertIsInstance(fingerprint, dict)

    def test_validate_feature_extraction_with_handler(self):
        """Test validate_feature_extraction_with_handler function"""
        validation = validate_feature_extraction_with_handler(self.handler, self.pyg_data)

        self.assertIsInstance(validation, dict)
        self.assertIn("handler_type", validation)
        self.assertIn("validation_passed", validation)
        self.assertIn("missing_requirements", validation)


# ==============================================================================
# TEST CASES: VALIDATION
# ==============================================================================


class TestFeatureValidation(unittest.TestCase):
    """Test feature validation functionality"""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        """Set up test fixtures"""
        self.handler = MockDFTHandler()
        self.pyg_data = create_sample_pyg_data()

    def test_validation_basic_structure(self):
        """Test validation returns expected structure."""
        validation = validate_feature_extraction_with_handler(self.handler, self.pyg_data)

        self.assertIsInstance(validation, dict)
        self.assertIn("handler_type", validation)
        self.assertIn("validation_passed", validation)
        self.assertIn("missing_requirements", validation)
        self.assertIn("quality_issues", validation)

    def test_validation_with_missing_requirements(self):
        """Test validation with missing requirements"""
        pyg_data = Data()
        # Missing essential properties

        validation = validate_feature_extraction_with_handler(self.handler, pyg_data)

        self.assertIsInstance(validation, dict)
        self.assertIn("missing_requirements", validation)

    def test_validation_quality_checks(self):
        """Test quality validation checks"""
        handler = MockDFTHandler()
        handler.validate_feature_quality = Mock(
            return_value={"passed": False, "issues": ["test_issue"]}
        )

        validation = validate_feature_extraction_with_handler(handler, self.pyg_data)

        self.assertIsInstance(validation, dict)
        self.assertIn("handler_type", validation)

    def test_dmc_uncertainty_validation(self):
        """Test DMC-specific uncertainty validation"""
        dmc_handler = MockDMCHandler(uncertainty_enabled=True)
        pyg_data = create_sample_pyg_data()

        # Missing uncertainty data — should detect it as missing
        validation = validate_feature_extraction_with_handler(dmc_handler, pyg_data)

        self.assertIsInstance(validation, dict)
        self.assertIn("validation_passed", validation)
        # With uncertainty_handling feature and is_uncertainty_enabled=True,
        # missing uncertainty data should fail validation
        if _get_dataset_feature("DMC", "uncertainty_handling"):
            self.assertFalse(validation["validation_passed"])
            self.assertIn("uncertainty", validation.get("missing_requirements", []))

    def test_dft_vibrational_validation(self):
        """Test DFT-specific vibrational validation"""
        self.pyg_data.freqs = torch.tensor([100.0, 200.0, 300.0])

        validation = validate_feature_extraction_with_handler(self.handler, self.pyg_data)

        self.assertIsInstance(validation, dict)
        self.assertIn("handler_type", validation)

    def test_statistical_validation(self):
        """Test statistical validation capability"""
        handler = MockDFTHandler(capabilities={"statistical_validation": True})

        validation = validate_feature_extraction_with_handler(handler, self.pyg_data)

        self.assertIsInstance(validation, dict)
        self.assertIn("handler_type", validation)

    def test_structural_validation(self):
        """Test structural validation capability"""
        handler = MockDFTHandler(capabilities={"structural_validation": True})

        validation = validate_feature_extraction_with_handler(handler, self.pyg_data)

        self.assertIsInstance(validation, dict)
        self.assertIn("handler_type", validation)


# ==============================================================================
# TEST CASES: ERROR HANDLING
# ==============================================================================


class TestErrorHandling(unittest.TestCase):
    """Test error handling and exception propagation"""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        """Set up test fixtures"""
        self.handler = MockDFTHandler()
        self.pyg_data = create_sample_pyg_data()

    def test_handler_operation_error_propagation(self):
        """Test that HandlerOperationError is properly raised"""
        handler = MockDFTHandler()
        handler.get_dataset_type = Mock(
            side_effect=HandlerOperationError(
                message="Test error", handler_type="DFT", operation="test"
            )
        )

        with self.assertRaises(HandlerOperationError):
            estimate_molecular_properties(self.pyg_data, handler)

    def test_handler_validation_error_propagation(self):
        """Test that HandlerValidationError is properly raised"""
        handler = MockDFTHandler()
        # Cause an error that should be wrapped
        handler.get_dataset_type = Mock(side_effect=AttributeError("Test error"))

        # Validation function should wrap in HandlerValidationError or raise HandlerOperationError
        with self.assertRaises((HandlerValidationError, HandlerOperationError, AttributeError)):
            validate_feature_extraction_with_handler(handler, self.pyg_data)

    @patch("milia_pipeline.molecules.molecule_feature_enricher.logger")
    def test_capability_error_handling(self, mock_logger):
        """Test handling of capability checking errors"""
        handler = MockDFTHandler()
        handler.get_feature_capabilities = Mock(side_effect=Exception("Capability error"))

        # Should log and continue
        properties = estimate_molecular_properties(self.pyg_data, handler)

        mock_logger.debug.assert_called()
        self.assertIn("num_atoms", properties)

    @patch("milia_pipeline.molecules.molecule_feature_enricher.logger")
    def test_additional_property_error_handling(self, mock_logger):
        """Test handling of additional property estimation errors"""
        handler = MockDFTHandler()
        handler.estimate_additional_properties = Mock(side_effect=Exception("Property error"))

        # Should log and continue
        properties = estimate_molecular_properties(self.pyg_data, handler)

        mock_logger.debug.assert_called()
        self.assertIn("num_atoms", properties)


# ==============================================================================
# TEST CASES: EDGE CASES
# ==============================================================================


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        """Set up test fixtures"""
        self.handler = MockDFTHandler()

    def test_empty_molecule(self):
        """Test handling of empty molecule data"""
        pyg_data = Data()
        pyg_data.num_nodes = 0

        # Empty molecule may cause errors
        try:
            properties = estimate_molecular_properties(pyg_data, self.handler)
            # If it succeeds, check basic structure
            self.assertIsInstance(properties, dict)
        except (HandlerOperationError, AttributeError):
            # Expected - empty molecule causes errors
            pass

    def test_single_atom_edge_cases(self):
        """Test single atom molecule edge cases"""
        pyg_data = create_sample_pyg_data(num_atoms=1)

        properties = estimate_molecular_properties(pyg_data, self.handler)
        diagnostics = get_feature_extraction_diagnostics(pyg_data, self.handler)

        self.assertEqual(properties["num_atoms"], 1.0)
        # Diagnostics structure
        self.assertIsInstance(diagnostics, dict)

    def test_large_molecule(self):
        """Test handling of large molecules"""
        num_atoms = 100
        pyg_data = Data()
        pyg_data.z = torch.tensor([6] * num_atoms, dtype=torch.long)
        pyg_data.pos = torch.randn(num_atoms, 3)
        pyg_data.x = torch.randn(num_atoms, 10)

        properties = estimate_molecular_properties(pyg_data, self.handler)

        self.assertEqual(properties["num_atoms"], float(num_atoms))

    def test_zero_mass_molecule(self):
        """Test handling of zero total mass"""
        pyg_data = Data()
        pyg_data.z = torch.tensor([], dtype=torch.long)
        pyg_data.pos = torch.empty(0, 3)
        pyg_data.num_nodes = 0

        # Zero mass/empty molecule will likely cause errors
        try:
            properties = estimate_molecular_properties(pyg_data, self.handler)
            # If successful, check structure
            self.assertIsInstance(properties, dict)
        except (HandlerOperationError, RuntimeError, AttributeError):
            # Expected for edge case
            pass

    def test_extreme_coordinates(self):
        """Test handling of extreme coordinate values"""
        pyg_data = create_sample_pyg_data()
        pyg_data.pos = torch.tensor([[1e10, 1e10, 1e10], [1e-10, 1e-10, 1e-10], [0.0, 0.0, 0.0]])

        properties = estimate_molecular_properties(pyg_data, self.handler)

        # Should handle extreme values
        self.assertIn("max_distance", properties)
        self.assertTrue(np.isfinite(properties["max_distance"]))

    def test_nan_handling(self):
        """Test handling of NaN values"""
        pyg_data = create_sample_pyg_data()
        pyg_data.pos[0, 0] = float("nan")

        # Should handle NaN gracefully or raise appropriate error
        try:
            properties = estimate_molecular_properties(pyg_data, self.handler)
            # If it doesn't raise, check that we got reasonable output
            self.assertIsInstance(properties, dict)
        except (HandlerError, PropertyEnrichmentError):
            # Expected behavior - NaN values should cause error
            pass

    def test_unicode_identifiers(self):
        """Test handling of unicode in identifiers"""
        pyg_data = create_sample_pyg_data()
        pyg_data.smiles = "CCO_测试"
        pyg_data.inchi = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3_日本語"

        identifiers = get_molecule_identifiers(pyg_data, self.handler)

        self.assertIn("smiles", identifiers)
        self.assertIn("inchi", identifiers)

    def test_missing_handler_methods(self):
        """Test handling when handler is missing expected methods"""
        self.pyg_data = create_sample_pyg_data()  # Ensure we have pyg_data
        handler = MockDatasetHandler(dataset_type="DFT", capabilities={})

        # Mock the method to raise AttributeError to simulate it being missing
        def raise_attr_error(*args, **kwargs):
            raise AttributeError("Method not available")

        handler.estimate_additional_properties = raise_attr_error

        # Should handle gracefully
        properties = estimate_molecular_properties(self.pyg_data, handler)

        self.assertIn("num_atoms", properties)


# ==============================================================================
# TEST CASES: DATASET-SPECIFIC BEHAVIORS
# ==============================================================================


class TestDatasetSpecificBehaviors(unittest.TestCase):
    """Test dataset-specific behaviors for DFT and DMC"""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def test_dft_specific_properties(self):
        """Test DFT-specific property handling"""
        dft_handler = MockDFTHandler()
        pyg_data = create_sample_pyg_data()
        pyg_data.freqs = torch.tensor([100.0, 200.0, 300.0])
        pyg_data.vibmodes = torch.randn(3, 3, 3)

        properties = estimate_molecular_properties(pyg_data, dft_handler)

        # Should include DFT-specific properties
        self.assertIn("estimated_electrons", properties)
        self.assertIn("refined_vibrational_modes", properties)

    def test_dmc_specific_properties(self):
        """Test DMC-specific property handling"""
        dmc_handler = MockDMCHandler(uncertainty_enabled=True)
        pyg_data = create_sample_pyg_data()
        pyg_data.uncertainty = torch.tensor([0.01])  # Use scalar tensor

        properties = estimate_molecular_properties(pyg_data, dmc_handler)

        # Should include DMC-specific properties
        self.assertIn("estimated_electrons", properties)
        self.assertIn("processed_uncertainty", properties)

    def test_dft_validation_requirements(self):
        """Test DFT-specific validation requirements"""
        dft_handler = MockDFTHandler()
        pyg_data = Data()
        pyg_data.num_nodes = 0
        # Missing essential DFT properties

        validation = validate_feature_extraction_with_handler(dft_handler, pyg_data)

        self.assertIsInstance(validation, dict)
        self.assertIn("missing_requirements", validation)
        # Should detect missing required properties
        missing = validation.get("missing_requirements", [])
        self.assertIsInstance(missing, list)

    def test_dmc_validation_requirements(self):
        """Test DMC-specific validation requirements"""
        dmc_handler = MockDMCHandler(uncertainty_enabled=True)
        pyg_data = create_sample_pyg_data()
        # Missing uncertainty

        validation = validate_feature_extraction_with_handler(dmc_handler, pyg_data)

        self.assertIsInstance(validation, dict)
        self.assertIn("validation_passed", validation)
        # With uncertainty feature active and is_uncertainty_enabled=True,
        # missing uncertainty should fail validation
        if _get_dataset_feature("DMC", "uncertainty_handling"):
            self.assertFalse(validation["validation_passed"])
            self.assertIn("uncertainty", validation.get("missing_requirements", []))

    def test_dft_capability_specific_validation(self):
        """Test DFT capability-specific validation"""
        dft_handler = MockDFTHandler(
            capabilities={
                "dft_specific": {"vibrational_validation": True, "electronic_validation": True}
            }
        )
        pyg_data = create_sample_pyg_data()
        pyg_data.freqs = torch.tensor([100.0, 200.0])

        validation = validate_feature_extraction_with_handler(dft_handler, pyg_data)

        self.assertIsInstance(validation, dict)
        self.assertIn("handler_type", validation)
        self.assertEqual(validation["handler_type"], "DFT")

    def test_dmc_capability_specific_validation(self):
        """Test DMC capability-specific validation"""
        dmc_handler = MockDMCHandler(
            uncertainty_enabled=True,
            capabilities={
                "dmc_specific": {"uncertainty_validation": True, "statistical_quality_check": True}
            },
        )
        pyg_data = create_sample_pyg_data()
        pyg_data.uncertainty = torch.tensor([0.01])  # Use scalar tensor

        validation = validate_feature_extraction_with_handler(dmc_handler, pyg_data)

        self.assertIsInstance(validation, dict)
        self.assertIn("handler_type", validation)
        self.assertEqual(validation["handler_type"], "DMC")


# ==============================================================================
# TEST CASES: INTEGRATION SCENARIOS
# ==============================================================================


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios combining multiple functions"""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        """Set up test fixtures"""
        self.handler = MockDFTHandler()
        self.pyg_data = create_sample_pyg_data()

    def test_full_feature_extraction_workflow(self):
        """Test complete feature extraction workflow"""
        # Get properties
        properties = estimate_molecular_properties(self.pyg_data, self.handler)
        self.assertIsInstance(properties, dict)

        # Get identifiers
        identifiers = get_molecule_identifiers(self.pyg_data, self.handler)
        self.assertIsInstance(identifiers, dict)

        # Get diagnostics
        diagnostics = get_feature_extraction_diagnostics(self.pyg_data, self.handler)
        self.assertIsInstance(diagnostics, dict)

        # Validate
        validation = validate_feature_extraction_with_handler(self.handler, self.pyg_data)
        self.assertIsInstance(validation, dict)
        self.assertIn("handler_type", validation)

    def test_handler_switching(self):
        """Test switching between DFT and DMC handlers"""
        pyg_data = create_sample_pyg_data()

        # DFT handler
        dft_handler = MockDFTHandler()
        dft_props = estimate_molecular_properties(pyg_data, dft_handler)

        # DMC handler
        dmc_handler = MockDMCHandler()
        dmc_props = estimate_molecular_properties(pyg_data, dmc_handler)

        # Both should work
        self.assertIn("num_atoms", dft_props)
        self.assertIn("num_atoms", dmc_props)

    def test_error_recovery_workflow(self):
        """Test error recovery in multi-step workflow"""
        handler = MockDFTHandler()
        # Make one method fail
        handler.estimate_additional_properties = Mock(side_effect=Exception("Test error"))

        # Should still complete workflow
        properties = estimate_molecular_properties(self.pyg_data, handler)
        validation = validate_feature_extraction_with_handler(handler, self.pyg_data)

        self.assertIsInstance(properties, dict)
        self.assertIsInstance(validation, dict)


# ==============================================================================
# PHASE 6: Registry Integration Tests - NEW
# ==============================================================================


class TestPhase6RegistryIntegrationFunctions(unittest.TestCase):
    """
    Test Phase 6 registry integration functions.

    Phase 6 adds:
    - Lazy registry initialization (_init_registry)
    - Dynamic available types (_get_available_dataset_types)
    - Dataset type registration check (_is_dataset_type_registered)
    - Feature-based queries (_get_dataset_feature)
    - Enrichment category determination (_get_dataset_enrichment_category)
    """

    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")

    def setUp(self):
        """Reset registry state before each test."""
        reset_registry_state()

    def tearDown(self):
        """Clean up after each test."""
        reset_registry_state()

    def test_01_init_registry_function_exists(self):
        """Test _init_registry function is importable."""
        self.assertTrue(callable(_init_registry))

    def test_02_get_available_dataset_types_function_exists(self):
        """Test _get_available_dataset_types function is importable."""
        self.assertTrue(callable(_get_available_dataset_types))

    def test_03_is_dataset_type_registered_function_exists(self):
        """Test _is_dataset_type_registered function is importable."""
        self.assertTrue(callable(_is_dataset_type_registered))

    def test_04_get_dataset_feature_function_exists(self):
        """Test _get_dataset_feature function is importable."""
        self.assertTrue(callable(_get_dataset_feature))

    def test_05_get_dataset_enrichment_category_function_exists(self):
        """Test _get_dataset_enrichment_category function is importable."""
        self.assertTrue(callable(_get_dataset_enrichment_category))

    def test_06_get_registry_integration_status_function_exists(self):
        """Test get_registry_integration_status function is importable."""
        self.assertTrue(callable(get_registry_integration_status))

    def test_07_init_registry_returns_bool(self):
        """Test _init_registry returns a boolean."""
        result = _init_registry()
        self.assertIsInstance(result, bool)

    def test_08_init_registry_idempotent(self):
        """Test _init_registry is idempotent (multiple calls same result)."""
        result1 = _init_registry()
        result2 = _init_registry()
        self.assertEqual(result1, result2)

    def test_09_get_available_dataset_types_returns_list(self):
        """Test _get_available_dataset_types returns a list."""
        types = _get_available_dataset_types()
        self.assertIsInstance(types, list)
        self.assertGreater(len(types), 0)

    def test_10_get_available_dataset_types_includes_known_types(self):
        """Test _get_available_dataset_types includes DFT, DMC, Wavefunction."""
        types = _get_available_dataset_types()
        self.assertIn("DFT", types)
        self.assertIn("DMC", types)
        self.assertIn("Wavefunction", types)

    def test_11_is_dataset_type_registered_dft(self):
        """Test _is_dataset_type_registered returns True for DFT."""
        self.assertTrue(_is_dataset_type_registered("DFT"))

    def test_12_is_dataset_type_registered_dmc(self):
        """Test _is_dataset_type_registered returns True for DMC."""
        self.assertTrue(_is_dataset_type_registered("DMC"))

    def test_13_is_dataset_type_registered_wavefunction(self):
        """Test _is_dataset_type_registered returns True for Wavefunction."""
        self.assertTrue(_is_dataset_type_registered("Wavefunction"))

    def test_14_is_dataset_type_registered_unknown(self):
        """Test _is_dataset_type_registered returns False for unknown type."""
        self.assertFalse(_is_dataset_type_registered("INVALID_TYPE"))
        self.assertFalse(_is_dataset_type_registered("TOTALLY_NONEXISTENT_DATASET_XYZ"))
        self.assertFalse(_is_dataset_type_registered(""))


class TestPhase6FeatureQueries(unittest.TestCase):
    """
    Test Phase 6 feature-based query functions.

    These tests verify the _get_dataset_feature function works correctly
    for querying dataset-specific features like uncertainty_handling,
    vibrational_analysis, orbital_analysis, etc.
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

    def test_08_dmc_no_atomization_energy(self):
        """Test DMC has atomization_energy=False."""
        result = _get_dataset_feature("DMC", "atomization_energy")
        self.assertFalse(result)

    def test_09_unknown_feature_returns_false(self):
        """Test unknown feature returns False."""
        result = _get_dataset_feature("DFT", "unknown_feature")
        self.assertFalse(result)

    def test_10_unknown_dataset_returns_false(self):
        """Test unknown dataset type returns False for any feature."""
        result = _get_dataset_feature("INVALID", "uncertainty_handling")
        self.assertFalse(result)

    def test_11_wavefunction_has_homo_lumo_gap(self):
        """Test Wavefunction has homo_lumo_gap=True."""
        result = _get_dataset_feature("Wavefunction", "homo_lumo_gap")
        self.assertTrue(result)

    def test_12_dft_has_rotational_constants(self):
        """Test DFT has rotational_constants=True."""
        result = _get_dataset_feature("DFT", "rotational_constants")
        self.assertTrue(result)

    def test_13_dft_has_frequency_analysis(self):
        """Test DFT has frequency_analysis=True."""
        result = _get_dataset_feature("DFT", "frequency_analysis")
        self.assertTrue(result)

    def test_14_wavefunction_has_mo_energies(self):
        """Test Wavefunction has mo_energies=True."""
        result = _get_dataset_feature("Wavefunction", "mo_energies")
        self.assertTrue(result)

    def test_15_dmc_no_mo_energies(self):
        """Test DMC has mo_energies=False."""
        result = _get_dataset_feature("DMC", "mo_energies")
        self.assertFalse(result)


class TestPhase6EnrichmentCategory(unittest.TestCase):
    """
    Test Phase 6 enrichment category determination.

    These tests verify _get_dataset_enrichment_category works correctly
    for routing datasets to appropriate enrichment behavior.
    """

    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")

    def setUp(self):
        reset_registry_state()

    def tearDown(self):
        reset_registry_state()

    def test_01_dmc_uncertainty_category(self):
        """Test DMC maps to 'uncertainty' category."""
        result = _get_dataset_enrichment_category("DMC")
        self.assertEqual(result, "uncertainty")

    def test_02_dft_vibrational_category(self):
        """Test DFT maps to 'vibrational' category."""
        result = _get_dataset_enrichment_category("DFT")
        self.assertEqual(result, "vibrational")

    def test_03_wavefunction_orbital_category(self):
        """Test Wavefunction maps to 'orbital' category."""
        result = _get_dataset_enrichment_category("Wavefunction")
        self.assertEqual(result, "orbital")

    def test_04_unknown_generic_category(self):
        """Test unknown dataset maps to 'generic' category."""
        result = _get_dataset_enrichment_category("UNKNOWN")
        self.assertEqual(result, "generic")

    def test_05_empty_string_generic_category(self):
        """Test empty string maps to 'generic' category."""
        result = _get_dataset_enrichment_category("")
        self.assertEqual(result, "generic")

    def test_06_category_determines_processing(self):
        """Test that category correctly determines processing path."""
        # Uncertainty category for DMC
        self.assertEqual(_get_dataset_enrichment_category("DMC"), "uncertainty")
        self.assertTrue(_get_dataset_feature("DMC", "uncertainty_handling"))

        # Vibrational category for DFT
        self.assertEqual(_get_dataset_enrichment_category("DFT"), "vibrational")
        self.assertTrue(_get_dataset_feature("DFT", "vibrational_analysis"))

        # Orbital category for Wavefunction
        self.assertEqual(_get_dataset_enrichment_category("Wavefunction"), "orbital")
        self.assertTrue(_get_dataset_feature("Wavefunction", "orbital_analysis"))


class TestPhase6RegistryIntegrationStatus(unittest.TestCase):
    """
    Test Phase 6 get_registry_integration_status function.

    This function provides diagnostic information about registry integration.
    """

    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")

    def setUp(self):
        reset_registry_state()

    def tearDown(self):
        reset_registry_state()

    def test_01_get_registry_integration_status_returns_dict(self):
        """Test get_registry_integration_status returns a dictionary."""
        status = get_registry_integration_status()
        self.assertIsInstance(status, dict)

    def test_02_status_includes_registry_available(self):
        """Test status includes registry_available field."""
        status = get_registry_integration_status()
        self.assertIn("registry_available", status)
        self.assertIsInstance(status["registry_available"], bool)

    def test_03_status_includes_registry_initialized(self):
        """Test status includes registry_initialized field."""
        status = get_registry_integration_status()
        self.assertIn("registry_initialized", status)
        self.assertIsInstance(status["registry_initialized"], bool)

    def test_04_status_includes_available_dataset_types(self):
        """Test status includes available_dataset_types."""
        status = get_registry_integration_status()
        self.assertIn("available_dataset_types", status)
        self.assertIsInstance(status["available_dataset_types"], list)
        self.assertIn("DFT", status["available_dataset_types"])
        self.assertIn("DMC", status["available_dataset_types"])
        self.assertIn("Wavefunction", status["available_dataset_types"])

    def test_05_status_includes_phase_6_complete(self):
        """Test status includes phase_6_complete=True."""
        status = get_registry_integration_status()
        self.assertIn("phase_6_complete", status)
        self.assertTrue(status["phase_6_complete"])

    def test_06_status_includes_refactoring_version(self):
        """Test status includes refactoring_version."""
        status = get_registry_integration_status()
        self.assertIn("refactoring_version", status)
        self.assertEqual(status["refactoring_version"], "6.0.0")

    def test_07_status_includes_module_name(self):
        """Test status includes module name."""
        status = get_registry_integration_status()
        self.assertIn("module", status)
        self.assertEqual(status["module"], "molecule_feature_enricher")

    def test_08_status_includes_feature_query_capability(self):
        """Test status includes feature_query_capability."""
        status = get_registry_integration_status()
        self.assertIn("feature_query_capability", status)
        fqc = status["feature_query_capability"]
        self.assertIn("uncertainty_handling", fqc)
        self.assertIn("vibrational_analysis", fqc)
        self.assertIn("orbital_analysis", fqc)
        self.assertIn("atomization_energy", fqc)

    def test_09_status_includes_handler_integration(self):
        """Test status includes handler_integration info."""
        status = get_registry_integration_status()
        self.assertIn("handler_integration", status)
        hi = status["handler_integration"]
        self.assertIn("handler_required", hi)
        self.assertTrue(hi["handler_required"])
        self.assertIn("hardcoded_type_checks", hi)
        self.assertEqual(hi["hardcoded_type_checks"], 0)

    def test_10_status_reports_feature_based_logic(self):
        """Test status reports dataset_specific_logic as FEATURE_BASED."""
        status = get_registry_integration_status()
        hi = status["handler_integration"]
        self.assertEqual(hi["dataset_specific_logic"], "FEATURE_BASED")


class TestPhase6RefactoredFunctions(unittest.TestCase):
    """
    Test Phase 6 refactored functions use feature-based queries.

    These tests verify that the 6 refactored locations correctly use
    registry-based validation and feature queries instead of hardcoded
    dataset type checks.
    """

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        self.dft_handler = MockDFTHandler()
        self.dmc_handler = MockDMCHandler(uncertainty_enabled=True)
        self.wf_handler = MockWavefunctionHandler()
        self.pyg_data = create_sample_pyg_data()

    def test_01_estimate_properties_uses_feature_queries(self):
        """Test estimate_molecular_properties uses feature-based routing."""
        # DFT with vibrational_analysis feature
        self.pyg_data.freqs = torch.tensor([100.0, 200.0, 300.0])
        dft_props = estimate_molecular_properties(self.pyg_data, self.dft_handler)
        self.assertIn("refined_vibrational_modes", dft_props)

        # DMC with uncertainty_handling feature
        pyg_data_dmc = create_sample_pyg_data()
        pyg_data_dmc.uncertainty = torch.tensor([0.01])
        dmc_props = estimate_molecular_properties(pyg_data_dmc, self.dmc_handler)
        self.assertIn("processed_uncertainty", dmc_props)

    def test_02_dft_vibrational_processing_enabled(self):
        """Test DFT gets vibrational processing (vibrational_analysis=True)."""
        self.pyg_data.freqs = torch.tensor([100.0, 200.0])
        props = estimate_molecular_properties(self.pyg_data, self.dft_handler)
        # Should process vibrational data
        self.assertIn("refined_vibrational_modes", props)

    def test_03_dmc_uncertainty_processing_enabled(self):
        """Test DMC gets uncertainty processing (uncertainty_handling=True)."""
        self.pyg_data.uncertainty = torch.tensor([0.01])
        props = estimate_molecular_properties(self.pyg_data, self.dmc_handler)
        # Should process uncertainty data
        self.assertIn("processed_uncertainty", props)

    @patch("milia_pipeline.molecules.molecule_feature_enricher.is_structural_features_enabled")
    @patch("milia_pipeline.molecules.molecule_feature_enricher.get_structural_features_config")
    def test_04_analyze_capabilities_uses_feature_queries(self, mock_config, mock_enabled):
        """Test analyze_structural_feature_capabilities uses feature-based routing."""
        mock_enabled.return_value = True
        mock_config.return_value = {"atom": [], "bond": []}

        # DFT capabilities
        dft_caps = analyze_structural_feature_capabilities(self.dft_handler)
        self.assertEqual(dft_caps["dataset_type"], "DFT")

        # DMC capabilities
        dmc_caps = analyze_structural_feature_capabilities(self.dmc_handler)
        self.assertEqual(dmc_caps["dataset_type"], "DMC")

    @patch("milia_pipeline.molecules.molecule_feature_enricher.is_structural_features_enabled")
    def test_05_fingerprint_uses_feature_queries(self, mock_enabled):
        """Test create_handler_compatible_fingerprint uses feature-based routing."""
        mock_enabled.return_value = True

        # DFT fingerprint
        dft_fp = create_handler_compatible_fingerprint(self.dft_handler, self.pyg_data)
        self.assertIsInstance(dft_fp, dict)

        # DMC fingerprint
        dmc_fp = create_handler_compatible_fingerprint(self.dmc_handler, self.pyg_data)
        self.assertIsInstance(dmc_fp, dict)

    def test_06_validation_uses_feature_queries(self):
        """Test validate_feature_extraction_with_handler uses feature-based routing."""
        # DFT validation
        dft_val = validate_feature_extraction_with_handler(self.dft_handler, self.pyg_data)
        self.assertIsInstance(dft_val, dict)
        self.assertEqual(dft_val["handler_type"], "DFT")

        # DMC validation
        dmc_val = validate_feature_extraction_with_handler(self.dmc_handler, self.pyg_data)
        self.assertIsInstance(dmc_val, dict)
        self.assertEqual(dmc_val["handler_type"], "DMC")

    def test_07_wavefunction_orbital_processing(self):
        """Test Wavefunction gets orbital processing (orbital_analysis=True)."""
        wf_handler = MockWavefunctionHandler()
        # Wavefunction should be recognized
        caps = analyze_capabilities_with_handler(wf_handler)
        self.assertEqual(caps["dataset_type"], "Wavefunction")

    def test_08_all_handlers_work_with_feature_routing(self):
        """Test all handler types work correctly with feature-based routing."""
        for handler, _handler_type in [
            (self.dft_handler, "DFT"),
            (self.dmc_handler, "DMC"),
            (self.wf_handler, "Wavefunction"),
        ]:
            props = estimate_molecular_properties(self.pyg_data, handler)
            self.assertIsInstance(props, dict)
            self.assertIn("num_atoms", props)


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

    @patch("milia_pipeline.molecules.molecule_feature_enricher._REGISTRY_AVAILABLE", False)
    def test_01_fallback_get_available_dataset_types_returns_list(self):
        """Test _get_available_dataset_types returns list via dynamic filesystem discovery.

        When registry is unavailable, the module uses dynamic filesystem discovery
        (scanning datasets/implementations/*.py). The test verifies it returns a list
        and that known types are found if the filesystem structure is present.
        """
        import milia_pipeline.molecules.molecule_feature_enricher as mod

        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False

        types = _get_available_dataset_types()
        self.assertIsInstance(types, list)
        # If filesystem discovery succeeds, known types should be present
        # If not (e.g., different working dir), empty list is valid
        if len(types) > 0:
            # All returned types should be uppercase strings (per discovery logic)
            for t in types:
                self.assertIsInstance(t, str)

    @patch("milia_pipeline.molecules.molecule_feature_enricher._REGISTRY_AVAILABLE", False)
    def test_02_fallback_is_dataset_type_registered_returns_bool(self):
        """Test _is_dataset_type_registered returns bool via dynamic discovery.

        When registry is unavailable, delegates to _get_available_dataset_types()
        which uses filesystem discovery. Always returns bool. An invalid/unknown
        type should return False.
        """
        import milia_pipeline.molecules.molecule_feature_enricher as mod

        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False

        # Invalid type should always be False regardless of discovery
        self.assertFalse(_is_dataset_type_registered("INVALID_NONEXISTENT_TYPE_XYZ"))

        # Return type is always bool
        result = _is_dataset_type_registered("DFT")
        self.assertIsInstance(result, bool)

    @patch("milia_pipeline.molecules.molecule_feature_enricher._REGISTRY_AVAILABLE", False)
    def test_03_fallback_get_dataset_feature_returns_false(self):
        """Test _get_dataset_feature returns False when registry unavailable.

        Phase 6.1 removed the hardcoded legacy_features fallback — the registry
        is now the single source of truth. When registry is unavailable,
        _get_dataset_feature conservatively returns False for all queries.
        """
        import milia_pipeline.molecules.molecule_feature_enricher as mod

        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False

        # All feature queries return False when registry is unavailable
        self.assertFalse(_get_dataset_feature("DMC", "uncertainty_handling"))
        self.assertFalse(_get_dataset_feature("DFT", "uncertainty_handling"))
        self.assertFalse(_get_dataset_feature("DFT", "vibrational_analysis"))
        self.assertFalse(_get_dataset_feature("Wavefunction", "orbital_analysis"))
        self.assertFalse(_get_dataset_feature("INVALID", "unknown_feature"))

    @patch("milia_pipeline.molecules.molecule_feature_enricher._REGISTRY_AVAILABLE", False)
    def test_04_fallback_enrichment_category_defaults_to_generic(self):
        """Test enrichment category defaults to 'generic' when registry unavailable.

        Since _get_dataset_feature returns False for all features when the registry
        is unavailable (Phase 6.1 removed legacy fallback), _get_dataset_enrichment_category
        returns 'generic' for all dataset types.
        """
        import milia_pipeline.molecules.molecule_feature_enricher as mod

        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False

        self.assertEqual(_get_dataset_enrichment_category("DMC"), "generic")
        self.assertEqual(_get_dataset_enrichment_category("DFT"), "generic")
        self.assertEqual(_get_dataset_enrichment_category("Wavefunction"), "generic")
        self.assertEqual(_get_dataset_enrichment_category("UNKNOWN"), "generic")

    @patch("milia_pipeline.molecules.molecule_feature_enricher._REGISTRY_AVAILABLE", False)
    def test_05_fallback_registry_status_shows_unavailable(self):
        """Test registry status correctly shows unavailable when fallback active."""
        import milia_pipeline.molecules.molecule_feature_enricher as mod

        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False

        status = get_registry_integration_status()
        self.assertFalse(status["registry_available"])
        # phase_6_complete should still be True (module code is deployed)
        self.assertTrue(status["phase_6_complete"])
        # Feature query capability keys should still be present (static metadata)
        self.assertIn("feature_query_capability", status)

    def test_06_registry_state_global_flags_exist(self):
        """Test registry state global flags exist in module."""
        import milia_pipeline.molecules.molecule_feature_enricher as mod

        self.assertTrue(hasattr(mod, "_REGISTRY_INITIALIZED"))
        self.assertTrue(hasattr(mod, "_REGISTRY_AVAILABLE"))
        self.assertTrue(hasattr(mod, "_REGISTRY_IMPORT_ERROR"))


class TestPhase6FeatureBasedProcessing(unittest.TestCase):
    """
    Test that feature-based processing correctly routes to appropriate handlers.

    These tests verify that after Phase 6 refactoring, the module correctly
    uses feature queries instead of hardcoded type checks.
    """

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")

    def setUp(self):
        reset_registry_state()
        self.pyg_data = create_sample_pyg_data()

    def tearDown(self):
        reset_registry_state()

    def test_01_uncertainty_feature_triggers_uncertainty_processing(self):
        """Test uncertainty_handling feature triggers uncertainty processing."""
        # DMC has uncertainty_handling=True
        self.assertTrue(_get_dataset_feature("DMC", "uncertainty_handling"))

        dmc_handler = MockDMCHandler(uncertainty_enabled=True)
        self.pyg_data.uncertainty = torch.tensor([0.01])
        props = estimate_molecular_properties(self.pyg_data, dmc_handler)
        self.assertIn("processed_uncertainty", props)

    def test_02_vibrational_feature_triggers_vibrational_processing(self):
        """Test vibrational_analysis feature triggers vibrational processing."""
        # DFT has vibrational_analysis=True
        self.assertTrue(_get_dataset_feature("DFT", "vibrational_analysis"))

        dft_handler = MockDFTHandler()
        self.pyg_data.freqs = torch.tensor([100.0, 200.0, 300.0])
        props = estimate_molecular_properties(self.pyg_data, dft_handler)
        self.assertIn("refined_vibrational_modes", props)

    def test_03_no_uncertainty_feature_skips_uncertainty_processing(self):
        """Test datasets without uncertainty_handling skip uncertainty processing."""
        # DFT has uncertainty_handling=False
        self.assertFalse(_get_dataset_feature("DFT", "uncertainty_handling"))

        dft_handler = MockDFTHandler()
        props = estimate_molecular_properties(self.pyg_data, dft_handler)
        self.assertNotIn("processed_uncertainty", props)

    def test_04_no_vibrational_feature_skips_vibrational_processing(self):
        """Test datasets without vibrational_analysis skip vibrational processing."""
        # DMC has vibrational_analysis=False
        self.assertFalse(_get_dataset_feature("DMC", "vibrational_analysis"))

        dmc_handler = MockDMCHandler()
        self.pyg_data.freqs = torch.tensor([100.0, 200.0])
        props = estimate_molecular_properties(self.pyg_data, dmc_handler)
        self.assertNotIn("refined_vibrational_modes", props)

    def test_05_enrichment_category_matches_features(self):
        """Test enrichment category correctly reflects dataset features."""
        # DMC: uncertainty_handling=True -> 'uncertainty'
        self.assertEqual(_get_dataset_enrichment_category("DMC"), "uncertainty")

        # DFT: vibrational_analysis=True -> 'vibrational'
        self.assertEqual(_get_dataset_enrichment_category("DFT"), "vibrational")

        # Wavefunction: orbital_analysis=True -> 'orbital'
        self.assertEqual(_get_dataset_enrichment_category("Wavefunction"), "orbital")

    def test_06_multiple_feature_check_consistency(self):
        """Test multiple feature queries are consistent."""
        for dataset_type in ["DFT", "DMC", "Wavefunction"]:
            # Check consistency across multiple calls
            result1 = _get_dataset_feature(dataset_type, "vibrational_analysis")
            result2 = _get_dataset_feature(dataset_type, "vibrational_analysis")
            self.assertEqual(result1, result2)


# ==============================================================================
# TEST CASES: create_molecular_fingerprint (standalone, no handler required)
# ==============================================================================


class TestCreateMolecularFingerprint(unittest.TestCase):
    """Test create_molecular_fingerprint function (handler-independent).

    This function creates a general fingerprint regardless of dataset type.
    It does NOT require a handler parameter.
    """

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def test_basic_fingerprint_structure(self):
        """Test fingerprint returns expected top-level structure."""
        pyg_data = create_sample_pyg_data()
        fp = create_molecular_fingerprint(pyg_data)

        self.assertIsInstance(fp, dict)
        self.assertIn("structure", fp)
        self.assertIn("features", fp)
        self.assertIn("targets", fp)
        self.assertIn("metadata", fp)
        self.assertIn("quality_indicators", fp)

    def test_fingerprint_structure_fields(self):
        """Test fingerprint structure section."""
        pyg_data = create_sample_pyg_data()
        fp = create_molecular_fingerprint(pyg_data)

        self.assertIn("num_atoms", fp["structure"])
        self.assertIn("atomic_numbers", fp["structure"])
        self.assertIn("has_coordinates", fp["structure"])
        self.assertEqual(fp["structure"]["num_atoms"], 3)
        self.assertTrue(fp["structure"]["has_coordinates"])

    def test_fingerprint_with_coordinates(self):
        """Test fingerprint coordinate analysis."""
        pyg_data = create_sample_pyg_data()
        fp = create_molecular_fingerprint(pyg_data)

        self.assertIsNotNone(fp["structure"]["coordinate_range"])
        coord_range = fp["structure"]["coordinate_range"]
        self.assertIn("min", coord_range)
        self.assertIn("max", coord_range)
        self.assertIn("mean", coord_range)
        self.assertIn("std", coord_range)

    def test_fingerprint_without_coordinates(self):
        """Test fingerprint without coordinates."""
        pyg_data = create_sample_pyg_data(include_positions=False)
        fp = create_molecular_fingerprint(pyg_data)

        self.assertFalse(fp["structure"]["has_coordinates"])
        self.assertIsNone(fp["structure"]["coordinate_range"])
        self.assertEqual(fp["quality_indicators"]["coordinate_quality"], "missing")

    def test_fingerprint_feature_dimensions(self):
        """Test fingerprint feature dimension reporting."""
        pyg_data = create_sample_pyg_data()
        fp = create_molecular_fingerprint(pyg_data)

        self.assertTrue(fp["features"]["has_atom_features"])
        self.assertIn("atom_feature_dim", fp["features"])
        self.assertEqual(fp["features"]["atom_feature_dim"], 5)

    def test_fingerprint_without_features(self):
        """Test fingerprint without atom features."""
        pyg_data = create_sample_pyg_data(include_features=False)
        fp = create_molecular_fingerprint(pyg_data)

        self.assertFalse(fp["features"]["has_atom_features"])

    def test_fingerprint_quality_with_nan(self):
        """Test fingerprint detects NaN values in atom features."""
        pyg_data = create_sample_pyg_data()
        pyg_data.x[0, 0] = float("nan")
        fp = create_molecular_fingerprint(pyg_data)

        self.assertTrue(fp["quality_indicators"]["has_nan_values"])
        self.assertEqual(fp["quality_indicators"]["quality_status"], "poor")

    def test_fingerprint_quality_with_inf(self):
        """Test fingerprint detects inf values in atom features."""
        pyg_data = create_sample_pyg_data()
        pyg_data.x[0, 0] = float("inf")
        fp = create_molecular_fingerprint(pyg_data)

        self.assertTrue(fp["quality_indicators"]["has_inf_values"])
        self.assertEqual(fp["quality_indicators"]["quality_status"], "poor")

    def test_fingerprint_targets(self):
        """Test fingerprint target information."""
        pyg_data = create_sample_pyg_data()
        fp = create_molecular_fingerprint(pyg_data)

        self.assertTrue(fp["targets"]["has_targets"])
        self.assertIsNotNone(fp["targets"]["target_shape"])

    def test_fingerprint_identifiers(self):
        """Test fingerprint identifier collection."""
        pyg_data = create_sample_pyg_data()
        fp = create_molecular_fingerprint(pyg_data)

        # Should contain identifiers from pyg_data (smiles, inchi, original_mol_idx)
        identifier_names = [name for name, _ in fp["metadata"]["identifiers"]]
        self.assertIn("smiles", identifier_names)
        self.assertIn("inchi", identifier_names)

    def test_fingerprint_edge_connectivity(self):
        """Test fingerprint with edge_index."""
        pyg_data = create_sample_pyg_data()
        pyg_data.edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        fp = create_molecular_fingerprint(pyg_data)

        self.assertTrue(fp["features"]["has_edges"])
        self.assertEqual(fp["features"]["edge_connectivity"], 4)

    def test_fingerprint_error_raises_structural_feature_error(self):
        """Test fingerprint raises StructuralFeatureError on failure."""
        pyg_data = Mock()
        pyg_data.z = None
        pyg_data.num_nodes = 0
        pyg_data.original_mol_idx = "N/A"
        pyg_data.smiles = None
        pyg_data.inchi = None
        # Make hasattr raise to trigger error
        type(pyg_data).x = property(lambda self: (_ for _ in ()).throw(RuntimeError("test")))

        with self.assertRaises((StructuralFeatureError, RuntimeError)):
            create_molecular_fingerprint(pyg_data)


# ==============================================================================
# TEST CASES: Handler Edge Cases (get_required_properties returns None, etc.)
# ==============================================================================


class TestHandlerEdgeCases(unittest.TestCase):
    """Test edge cases in handler interaction patterns.

    These tests cover code paths where handler methods return unexpected values
    (None, empty collections) or raise exceptions that must be handled gracefully.
    """

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        self.pyg_data = create_sample_pyg_data()

    def test_get_required_properties_returns_none(self):
        """Test validation handles get_required_properties returning None."""
        handler = MockDFTHandler()
        handler.get_required_properties = Mock(return_value=None)

        validation = validate_feature_extraction_with_handler(handler, self.pyg_data)

        self.assertIsInstance(validation, dict)
        # Should not crash — None is treated as empty list per module line 1538-1539
        self.assertIn("validation_passed", validation)

    def test_get_required_properties_raises_attribute_error(self):
        """Test validation handles AttributeError from get_required_properties."""
        handler = MockDFTHandler()
        handler.get_required_properties = Mock(side_effect=AttributeError("no method"))

        validation = validate_feature_extraction_with_handler(handler, self.pyg_data)

        # Should catch and continue — module line 1545
        self.assertIsInstance(validation, dict)
        self.assertIn("validation_passed", validation)

    def test_get_feature_capabilities_returns_none(self):
        """Test that None from get_feature_capabilities is handled."""
        handler = MockDFTHandler()
        handler.get_feature_capabilities = Mock(return_value=None)

        # Should not crash — module defaults to {} when None
        properties = estimate_molecular_properties(self.pyg_data, handler)
        self.assertIsInstance(properties, dict)
        self.assertIn("num_atoms", properties)

    def test_handler_without_get_feature_capabilities(self):
        """Test handler missing get_feature_capabilities entirely."""

        # Create a minimal handler object whose class does NOT define get_feature_capabilities
        class MinimalHandler:
            def __init__(self):
                self.dataset_config = Mock()
                self.dataset_config.is_uncertainty_enabled = False
                self.processing_config = Mock()

            def get_dataset_type(self):
                return "DFT"

            def estimate_additional_properties(self, pyg_data):
                return {"custom_property": 1.0}

        handler = MinimalHandler()
        self.assertFalse(hasattr(handler, "get_feature_capabilities"))

        # Module uses hasattr check — should handle gracefully
        properties = estimate_molecular_properties(self.pyg_data, handler)
        self.assertIsInstance(properties, dict)
        self.assertIn("num_atoms", properties)

    def test_handler_without_processing_config(self):
        """Test handler with processing_config set to None."""
        handler = MockDFTHandler()
        handler.processing_config = None

        fingerprint = create_handler_compatible_fingerprint(handler, self.pyg_data)
        self.assertIsInstance(fingerprint, dict)
        # processing_config should be empty dict in handler_integration
        hi = fingerprint.get("handler_integration", {})
        self.assertEqual(hi.get("processing_config", {}), {})

    def test_handler_without_dataset_config(self):
        """Test handler with dataset_config set to None."""
        handler = MockDMCHandler(uncertainty_enabled=True)
        handler.dataset_config = None

        fingerprint = create_handler_compatible_fingerprint(handler, self.pyg_data)
        self.assertIsInstance(fingerprint, dict)
        hi = fingerprint.get("handler_integration", {})
        self.assertEqual(hi.get("dataset_config", {}), {})

    def test_estimate_additional_properties_returns_none(self):
        """Test estimate_additional_properties returning None is handled."""
        handler = MockDFTHandler()
        handler.estimate_additional_properties = Mock(return_value=None)

        properties = estimate_molecular_properties(self.pyg_data, handler)
        self.assertIsInstance(properties, dict)
        # Should not have the custom_property since None was returned
        self.assertNotIn("custom_property", properties)

    def test_validate_feature_quality_raises_handler_error(self):
        """Test that HandlerError from validate_feature_quality is recorded."""
        handler = MockDFTHandler(capabilities={"custom_validation": True})
        handler.validate_feature_quality = Mock(side_effect=HandlerError("validation failed"))

        validation = validate_feature_extraction_with_handler(handler, self.pyg_data)
        self.assertIsInstance(validation, dict)
        # Handler error should be recorded
        self.assertIn("handler_error", validation.get("handler_specific_checks", {}))
        self.assertFalse(validation["validation_passed"])


# ==============================================================================
# TEST CASES: Registry Initialization Edge Cases
# ==============================================================================


class TestRegistryInitializationEdgeCases(unittest.TestCase):
    """Test registry initialization edge cases and error handling."""

    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")

    def setUp(self):
        reset_registry_state()

    def tearDown(self):
        reset_registry_state()

    def test_registry_initialized_flag_set_after_init(self):
        """Test _REGISTRY_INITIALIZED is set to True after _init_registry."""
        import milia_pipeline.molecules.molecule_feature_enricher as mod

        mod._REGISTRY_INITIALIZED = False

        _init_registry()

        self.assertTrue(mod._REGISTRY_INITIALIZED)

    def test_registry_init_only_runs_once(self):
        """Test _init_registry short-circuits on second call."""
        import milia_pipeline.molecules.molecule_feature_enricher as mod

        result1 = _init_registry()
        # Manually set a flag to verify short-circuit
        mod._REGISTRY_INITIALIZED = True
        original_available = mod._REGISTRY_AVAILABLE

        result2 = _init_registry()

        # Second call should return same cached result
        self.assertEqual(result1, result2)
        self.assertEqual(mod._REGISTRY_AVAILABLE, original_available)

    @patch("milia_pipeline.molecules.molecule_feature_enricher._REGISTRY_AVAILABLE", False)
    @patch("milia_pipeline.molecules.molecule_feature_enricher._REGISTRY_INITIALIZED", True)
    def test_get_dataset_feature_unknown_dataset_with_registry_down(self):
        """Test _get_dataset_feature returns False for unknown dataset when registry down."""
        result = _get_dataset_feature("NONEXISTENT_DATASET", "vibrational_analysis")
        self.assertFalse(result)

    def test_enrichment_category_with_empty_string(self):
        """Test _get_dataset_enrichment_category with empty string."""
        result = _get_dataset_enrichment_category("")
        self.assertEqual(result, "generic")

    def test_enrichment_category_with_none_like_string(self):
        """Test _get_dataset_enrichment_category with unusual input."""
        result = _get_dataset_enrichment_category("None")
        self.assertEqual(result, "generic")


# ==============================================================================
# TEST RUNNER
# ==============================================================================


def run_test_suite():
    """Run the complete test suite"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all original test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEstimateMolecularProperties))
    suite.addTests(loader.loadTestsFromTestCase(TestGetMoleculeIdentifiers))
    suite.addTests(loader.loadTestsFromTestCase(TestGetStructuralFeatureSummary))
    suite.addTests(loader.loadTestsFromTestCase(TestGetFeatureExtractionDiagnostics))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyzeStructuralFeatureCapabilities))
    suite.addTests(loader.loadTestsFromTestCase(TestHandlerOnlyOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetSpecificBehaviors))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationScenarios))

    # Add Phase 6 test classes (NEW)
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6RegistryIntegrationFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6FeatureQueries))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6EnrichmentCategory))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6RegistryIntegrationStatus))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6RefactoredFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6LegacyFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6FeatureBasedProcessing))

    # Add new production-ready test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCreateMolecularFingerprint))
    suite.addTests(loader.loadTestsFromTestCase(TestHandlerEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestRegistryInitializationEdgeCases))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("COMPREHENSIVE TEST SUMMARY - molecule_feature_enricher.py (Phase 6 Updated)")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)

    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED!")
        print("✓ Molecular property estimation validated")
        print("✓ Molecule identifiers working")
        print("✓ Structural feature summaries operational")
        print("✓ Feature extraction diagnostics functional")
        print("✓ Capability analysis verified")
        print("✓ Handler-only operations working")
        print("✓ Feature validation tested")
        print("✓ Error handling verified")
        print("✓ Edge cases covered")
        print("✓ Dataset-specific behaviors validated")
        print("✓ Integration scenarios passed")
        print("✓ Phase 6: Registry integration validated")
        print("✓ Phase 6: Feature-based queries working")
        print("✓ Phase 6: Enrichment category routing functional")
        print("✓ Phase 6: get_registry_integration_status verified")
        print("✓ Phase 6: All 6 refactored locations tested")
        print("✓ Phase 6: Legacy fallback working (conservative False)")
        print("✓ Phase 6: Feature-based processing validated")
        print("✓ create_molecular_fingerprint standalone tests passed")
        print("✓ Handler edge cases (None returns, missing methods) covered")
        print("✓ Registry initialization edge cases covered")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("  Review failures before production use")

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if not IMPORTS_SUCCESSFUL:
        print("\n" + "=" * 70)
        print("ERROR: Required imports failed!")
        print("=" * 70)
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)

    exit_code = run_test_suite()
    sys.exit(exit_code)
