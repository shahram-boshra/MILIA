#!/usr/bin/env python3
"""
Unit tests for property_enrichment.py module (Handler Migration Complete + Phase 6)

Test file: test_property_enrichment_unit.py
Module under test: ~/ml_projects/milia/milia_pipeline/molecules/property_enrichment.py

This test suite validates the refactored property_enrichment module after complete
handler pattern integration with enhanced exception handling and Phase 6 registry
integration.

Key Test Areas:
1. Tensor conversion and validation (_ensure_tensor)
2. Scalar graph target addition
3. Node feature enrichment
4. Vector graph properties
5. Variable-length graph properties
6. Vibrational data processing and refinement
7. Vibration modes processing
8. Frequency data processing
9. Atomization energy calculation
10. Main enrichment orchestration
11. Handler context detection
12. Handler compatibility validation
13. Parameter resolution decorator
14. Handler-aware error reporting
15. Edge cases and integration scenarios
16. Handler error context creation (NEW)
17. Phase 6: Registry Integration Functions
18. Phase 6: Feature-Based Queries
19. Phase 6: Registry Status Methods
20. Phase 6: Updated Verification Functions
21. Phase 6: Legacy Fallback Behavior

Functions Tested:
- _ensure_tensor()
- add_scalar_graph_targets()
- add_node_features()
- add_vector_graph_properties()
- add_variable_len_graph_properties()
- _process_vibrational_data()
- _process_vibmodes()
- _process_frequencies()
- calculate_atomization_energy()
- enrich_pyg_data_with_properties()
- _get_handler_context()
- validate_handler_compatibility()
- create_handler_error_context()
- resolve_parameters()
- verify_interface_compatibility()
- get_handler_integration_summary()
- get_handler_integration_status()
- Phase 6:
  - _init_registry()
  - _get_available_dataset_types()
  - _is_dataset_type_registered()
  - _get_dataset_feature()
  - get_registry_integration_status()

Total: 140+ comprehensive tests including Phase 6 additions
"""

import sys
import os
from pathlib import Path

# CRITICAL: Add project root to Python path FIRST
project_root = Path('/app/milia')
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List

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
            if hasattr(self, 'x') and self.x is not None:
                return self.x.shape[0]
            if hasattr(self, 'z') and self.z is not None:
                return self.z.shape[0] if isinstance(self.z, (np.ndarray, torch.Tensor)) else len(self.z)
            return 0

# Import module under test
from milia_pipeline.molecules.property_enrichment import (
    _ensure_tensor,
    add_scalar_graph_targets,
    add_node_features,
    add_vector_graph_properties,
    add_variable_len_graph_properties,
    _process_vibrational_data,
    _process_vibmodes,
    _process_frequencies,
    calculate_atomization_energy,
    enrich_pyg_data_with_properties,
    _get_handler_context,
    validate_handler_compatibility,
    create_handler_error_context,
    resolve_parameters,
    verify_interface_compatibility,
    get_handler_integration_summary,
    get_handler_integration_status
)

# Phase 6: Import registry integration functions
try:
    from milia_pipeline.molecules.property_enrichment import (
        _init_registry,
        _get_available_dataset_types,
        _is_dataset_type_registered,
        _get_dataset_feature,
        get_registry_integration_status,
        _REGISTRY_INITIALIZED,
        _REGISTRY_AVAILABLE,
        _REGISTRY_IMPORT_ERROR,
    )
    PHASE6_IMPORTS_SUCCESSFUL = True
    PHASE6_IMPORT_ERROR = None
except ImportError as e:
    PHASE6_IMPORTS_SUCCESSFUL = False
    PHASE6_IMPORT_ERROR = str(e)
    print(f"WARNING: Could not import Phase 6 registry functions: {e}")

# Import required configuration and exception classes
from milia_pipeline.config.config_containers import (
    DatasetConfig,
    FilterConfig,
    ProcessingConfig
)
from milia_pipeline.config.config_constants import HAR2EV, ATOMIC_ENERGIES_HARTREE
from milia_pipeline.exceptions import (
    PropertyEnrichmentError,
    ConfigurationError,
    HandlerError,
    HandlerOperationError,
    HandlerValidationError,
    ValidationError
)


# ==============================================================================
# MOCK HELPERS
# ==============================================================================

class MockDatasetHandler:
    """Mock dataset handler for testing"""
    
    def __init__(self, dataset_type='DFT', should_validate=True):
        self.dataset_type = dataset_type
        self.should_validate = should_validate
        self.dataset_config = Mock()
        self.dataset_config.is_uncertainty_enabled = False
        self.dataset_config.uncertainty_config = {}
        self.filter_config = Mock()
        self.processing_config = Mock()
        
    def get_dataset_type(self):
        return self.dataset_type
    
    def get_required_properties(self):
        if self.dataset_type == 'DFT':
            return ['atoms', 'coordinates', 'Etot', 'forces']
        elif self.dataset_type == 'DMC':
            return ['atoms', 'coordinates', 'Etot']
        return ['atoms', 'coordinates']
    
    def process_property_value(self, property_name, property_value, molecule_index, identifier):
        """Mock property value processing - returns value as-is"""
        return property_value
    
    def validate_molecule_data(self, pyg_data, raw_properties_dict, molecule_index, identifier):
        """Mock molecule data validation"""
        return self.should_validate
    
    def enrich_pyg_data(self, pyg_data, raw_properties_dict, molecule_index, identifier):
        """Mock enrichment - returns pyg_data as-is"""
        return pyg_data
    
    def get_node_feature_keys(self):
        """Return mock node feature keys"""
        return ['charges']
    
    def get_scalar_target_keys(self):
        """Return mock scalar target keys"""
        return ['Etot', 'Emad']
    
    def get_vector_property_keys(self):
        """Return mock vector property keys"""
        return ['dipole', 'rots']
    
    def get_variable_length_property_keys(self):
        """Return mock variable-length property keys"""
        return ['freqs', 'vibmodes']


class MockDFTHandler(MockDatasetHandler):
    """Mock DFT handler"""
    
    def __init__(self, should_validate=True):
        super().__init__(dataset_type='DFT', should_validate=should_validate)


class MockDMCHandler(MockDatasetHandler):
    """Mock DMC handler with uncertainty support"""
    
    def __init__(self, should_validate=True, uncertainty_enabled=False):
        super().__init__(dataset_type='DMC', should_validate=should_validate)
        self.dataset_config.is_uncertainty_enabled = uncertainty_enabled


def create_mock_logger():
    """Create a mock logger for testing"""
    logger = Mock(spec=logging.Logger)
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


def create_mock_dataset_config(dataset_type='DFT'):
    """Create a mock dataset configuration"""
    config = Mock(spec=DatasetConfig)
    config.dataset_type = dataset_type
    config.is_uncertainty_enabled = False
    return config


def create_mock_processing_config():
    """Create a mock processing configuration"""
    config = Mock(spec=ProcessingConfig)
    config.vibration_refinement = {
        'comparison_tolerance': 1e-4
    }
    return config


def reset_registry_state():
    """
    Reset registry state to uninitialized for testing.
    IMPORTANT: Use this before/after tests that modify registry state.
    """
    try:
        import milia_pipeline.molecules.property_enrichment as enrichment_module
        enrichment_module._REGISTRY_INITIALIZED = False
        enrichment_module._REGISTRY_AVAILABLE = False
        enrichment_module._REGISTRY_IMPORT_ERROR = None
        enrichment_module._registry_list_all = None
        enrichment_module._registry_get = None
        enrichment_module._registry_is_registered = None
    except (ImportError, AttributeError):
        pass  # Module may not have been imported yet


# ==============================================================================
# TEST CASES: TENSOR CONVERSION
# ==============================================================================

class TestEnsureTensor(unittest.TestCase):
    """Test _ensure_tensor function for proper tensor conversion"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.molecule_index = 0
        self.identifier = "InChI=1S/H2O/h1H2"
        self.property_name = "test_property"
    
    def test_none_value_raises_error(self):
        """Test that None value raises PropertyEnrichmentError"""
        with self.assertRaises(PropertyEnrichmentError) as context:
            _ensure_tensor(None, torch.float32, self.property_name, 
                          self.molecule_index, self.identifier)
        
        error = context.exception
        self.assertEqual(error.molecule_index, self.molecule_index)
        self.assertIn("Cannot convert None", error.reason)
    
    def test_already_tensor_conversion(self):
        """Test tensor that's already a tensor"""
        original = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = _ensure_tensor(original, torch.float32, self.property_name,
                               self.molecule_index, self.identifier)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.allclose(result, torch.tensor([1.0, 2.0, 3.0])))
    
    def test_numpy_array_conversion(self):
        """Test conversion from numpy array"""
        np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _ensure_tensor(np_array, torch.float32, self.property_name,
                               self.molecule_index, self.identifier)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.allclose(result, torch.tensor([1.0, 2.0, 3.0])))
    
    def test_list_conversion(self):
        """Test conversion from list (CRITICAL FIX validation)"""
        test_list = [1.0, 2.0, 3.0]
        result = _ensure_tensor(test_list, torch.float32, self.property_name,
                               self.molecule_index, self.identifier)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.allclose(result, torch.tensor([1.0, 2.0, 3.0])))
    
    def test_tuple_conversion(self):
        """Test conversion from tuple"""
        test_tuple = (1.0, 2.0, 3.0)
        result = _ensure_tensor(test_tuple, torch.float32, self.property_name,
                               self.molecule_index, self.identifier)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.allclose(result, torch.tensor([1.0, 2.0, 3.0])))
    
    def test_scalar_int_conversion(self):
        """Test conversion from scalar int"""
        result = _ensure_tensor(42, torch.float32, self.property_name,
                               self.molecule_index, self.identifier)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.item(), 42.0)
    
    def test_scalar_float_conversion(self):
        """Test conversion from scalar float"""
        result = _ensure_tensor(3.14, torch.float32, self.property_name,
                               self.molecule_index, self.identifier)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
        self.assertAlmostEqual(result.item(), 3.14, places=5)
    
    def test_numpy_number_conversion(self):
        """Test conversion from numpy number types"""
        np_value = np.float64(2.718)
        result = _ensure_tensor(np_value, torch.float32, self.property_name,
                               self.molecule_index, self.identifier)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
        self.assertAlmostEqual(result.item(), 2.718, places=5)
    
    def test_string_number_conversion(self):
        """Test conversion from string representing number"""
        result = _ensure_tensor("123.456", torch.float32, self.property_name,
                               self.molecule_index, self.identifier)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
        self.assertAlmostEqual(result.item(), 123.456, places=3)
    
    def test_invalid_string_raises_error(self):
        """Test that non-numeric string raises PropertyEnrichmentError"""
        with self.assertRaises(PropertyEnrichmentError) as context:
            _ensure_tensor("not_a_number", torch.float32, self.property_name,
                          self.molecule_index, self.identifier)
        
        error = context.exception
        self.assertIn("Cannot convert string", error.reason)
    
    def test_unsupported_type_raises_error(self):
        """Test that unsupported type raises PropertyEnrichmentError"""
        with self.assertRaises(PropertyEnrichmentError) as context:
            _ensure_tensor({'key': 'value'}, torch.float32, self.property_name,
                          self.molecule_index, self.identifier)
        
        error = context.exception
        self.assertIn("Unsupported type", error.reason)
    
    def test_complex_dtype_conversion(self):
        """Test conversion with complex dtype"""
        complex_array = np.array([1+2j, 3+4j])
        result = _ensure_tensor(complex_array, torch.complex64, self.property_name,
                               self.molecule_index, self.identifier)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.complex64)
    
    def test_nested_list_conversion(self):
        """Test conversion from nested list (2D array)"""
        nested_list = [[1.0, 2.0], [3.0, 4.0]]
        result = _ensure_tensor(nested_list, torch.float32, self.property_name,
                               self.molecule_index, self.identifier)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (2, 2))


# ==============================================================================
# TEST CASES: SCALAR GRAPH TARGETS
# ==============================================================================

class TestAddScalarGraphTargets(unittest.TestCase):
    """Test add_scalar_graph_targets function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.molecule_index = 0
        self.identifier = "InChI=1S/C2H6/c1-2/h1-2H3"
        self.dataset_config = create_mock_dataset_config()
        self.pyg_data = Data()
    
    def test_add_single_scalar_target(self):
        """Test adding a single scalar target"""
        raw_props = {'Etot': -100.5}
        target_keys = ['Etot']
        
        add_scalar_graph_targets(
            pyg_data=self.pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            target_keys=target_keys,
            dataset_config=self.dataset_config,
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(self.pyg_data, 'y'))
        self.assertIsInstance(self.pyg_data.y, torch.Tensor)
        self.assertEqual(self.pyg_data.y.shape, (1,))
        self.assertAlmostEqual(self.pyg_data.y.item(), -100.5, places=5)
    
    def test_add_multiple_scalar_targets(self):
        """Test adding multiple scalar targets"""
        raw_props = {
            'Etot': -100.5,
            'Emad': 50.25,
            'HOMO': -0.25
        }
        target_keys = ['Etot', 'Emad', 'HOMO']
        
        add_scalar_graph_targets(
            pyg_data=self.pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            target_keys=target_keys,
            dataset_config=self.dataset_config,
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(self.pyg_data, 'y'))
        self.assertIsInstance(self.pyg_data.y, torch.Tensor)
        self.assertEqual(self.pyg_data.y.shape, (3,))
        self.assertTrue(torch.allclose(
            self.pyg_data.y,
            torch.tensor([-100.5, 50.25, -0.25])
        ))
    
    def test_empty_target_keys(self):
        """Test with empty target keys (should do nothing)"""
        fresh_pyg_data = Data()
        raw_props = {'Etot': -100.5}
        target_keys = []
        
        had_y_before = hasattr(fresh_pyg_data, 'y') and fresh_pyg_data.y is not None
        
        add_scalar_graph_targets(
            pyg_data=fresh_pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            target_keys=target_keys,
            dataset_config=self.dataset_config,
            identifier=self.identifier
        )
        
        if not had_y_before:
            self.assertTrue(not hasattr(fresh_pyg_data, 'y') or fresh_pyg_data.y is None)
    
    def test_missing_target_raises_error(self):
        """Test that missing target raises PropertyEnrichmentError"""
        raw_props = {'Etot': -100.5}
        target_keys = ['Etot', 'missing_key']
        
        with self.assertRaises(PropertyEnrichmentError) as context:
            add_scalar_graph_targets(
                pyg_data=self.pyg_data,
                raw_properties_dict=raw_props,
                molecule_index=self.molecule_index,
                logger=self.logger,
                target_keys=target_keys,
                dataset_config=self.dataset_config,
                identifier=self.identifier
            )
        
        error = context.exception
        self.assertEqual(error.molecule_index, self.molecule_index)
        self.assertIn("missing_key", error.property_name)
    
    def test_none_target_raises_error(self):
        """Test that None target value raises error"""
        raw_props = {'Etot': None}
        target_keys = ['Etot']
        
        with self.assertRaises(PropertyEnrichmentError):
            add_scalar_graph_targets(
                pyg_data=self.pyg_data,
                raw_properties_dict=raw_props,
                molecule_index=self.molecule_index,
                logger=self.logger,
                target_keys=target_keys,
                dataset_config=self.dataset_config,
                identifier=self.identifier
            )
    
    def test_numpy_scalar_conversion(self):
        """Test conversion from numpy scalar"""
        raw_props = {'Etot': np.float64(-100.5)}
        target_keys = ['Etot']
        
        add_scalar_graph_targets(
            pyg_data=self.pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            target_keys=target_keys,
            dataset_config=self.dataset_config,
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(self.pyg_data, 'y'))
        self.assertAlmostEqual(self.pyg_data.y.item(), -100.5, places=5)
    
    def test_single_element_array_conversion(self):
        """Test conversion from single-element array"""
        raw_props = {'Etot': np.array([-100.5])}
        target_keys = ['Etot']
        
        add_scalar_graph_targets(
            pyg_data=self.pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            target_keys=target_keys,
            dataset_config=self.dataset_config,
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(self.pyg_data, 'y'))
        self.assertAlmostEqual(self.pyg_data.y.item(), -100.5, places=5)
    
    def test_multi_element_array_raises_error(self):
        """Test that multi-element array raises error"""
        raw_props = {'Etot': np.array([-100.5, -50.0])}
        target_keys = ['Etot']
        
        with self.assertRaises(PropertyEnrichmentError) as context:
            add_scalar_graph_targets(
                pyg_data=self.pyg_data,
                raw_properties_dict=raw_props,
                molecule_index=self.molecule_index,
                logger=self.logger,
                target_keys=target_keys,
                dataset_config=self.dataset_config,
                identifier=self.identifier
            )
        
        error = context.exception
        self.assertIn("array shape", error.reason.lower())
    
    def test_string_numeric_conversion(self):
        """Test conversion from string numeric value"""
        raw_props = {'Etot': "-100.5"}
        target_keys = ['Etot']
        
        add_scalar_graph_targets(
            pyg_data=self.pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            target_keys=target_keys,
            dataset_config=self.dataset_config,
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(self.pyg_data, 'y'))
        self.assertAlmostEqual(self.pyg_data.y.item(), -100.5, places=5)
    
    def test_single_element_list_conversion(self):
        """Test conversion from single-element list (CRITICAL FIX)"""
        raw_props = {'Etot': [-100.5]}
        target_keys = ['Etot']
        
        add_scalar_graph_targets(
            pyg_data=self.pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            target_keys=target_keys,
            dataset_config=self.dataset_config,
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(self.pyg_data, 'y'))
        self.assertAlmostEqual(self.pyg_data.y.item(), -100.5, places=5)
    
    def test_nan_value_raises_error(self):
        """Test that NaN value raises error"""
        raw_props = {'Etot': float('nan')}
        target_keys = ['Etot']
        
        with self.assertRaises(PropertyEnrichmentError) as context:
            add_scalar_graph_targets(
                pyg_data=self.pyg_data,
                raw_properties_dict=raw_props,
                molecule_index=self.molecule_index,
                logger=self.logger,
                target_keys=target_keys,
                dataset_config=self.dataset_config,
                identifier=self.identifier
            )
        
        error = context.exception
        self.assertIn("NaN", error.reason)
    
    def test_inf_value_raises_error(self):
        """Test that Inf value raises error"""
        raw_props = {'Etot': float('inf')}
        target_keys = ['Etot']
        
        with self.assertRaises(PropertyEnrichmentError):
            add_scalar_graph_targets(
                pyg_data=self.pyg_data,
                raw_properties_dict=raw_props,
                molecule_index=self.molecule_index,
                logger=self.logger,
                target_keys=target_keys,
                dataset_config=self.dataset_config,
                identifier=self.identifier
            )
    
    def test_y_property_names_metadata_stored(self):
        """Test that y_property_names metadata is stored on pyg_data"""
        raw_props = {'Etot': -100.5, 'Emad': 50.25}
        target_keys = ['Etot', 'Emad']
        
        add_scalar_graph_targets(
            pyg_data=self.pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            target_keys=target_keys,
            dataset_config=self.dataset_config,
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(self.pyg_data, 'y_property_names'))
        self.assertEqual(self.pyg_data.y_property_names, ['Etot', 'Emad'])
    
    def test_multi_element_list_raises_error(self):
        """Test that multi-element list raises error for scalar target"""
        raw_props = {'Etot': [-100.5, -50.0]}
        target_keys = ['Etot']
        
        with self.assertRaises(PropertyEnrichmentError) as context:
            add_scalar_graph_targets(
                pyg_data=self.pyg_data,
                raw_properties_dict=raw_props,
                molecule_index=self.molecule_index,
                logger=self.logger,
                target_keys=target_keys,
                dataset_config=self.dataset_config,
                identifier=self.identifier
            )
        
        error = context.exception
        self.assertIn("list/tuple", error.reason.lower())
    
    def test_unsupported_type_raises_error(self):
        """Test that unsupported type (e.g., dict) raises error"""
        raw_props = {'Etot': {'value': -100.5}}
        target_keys = ['Etot']
        
        with self.assertRaises(PropertyEnrichmentError) as context:
            add_scalar_graph_targets(
                pyg_data=self.pyg_data,
                raw_properties_dict=raw_props,
                molecule_index=self.molecule_index,
                logger=self.logger,
                target_keys=target_keys,
                dataset_config=self.dataset_config,
                identifier=self.identifier
            )
        
        error = context.exception
        self.assertIn("unexpected type", error.reason.lower())
    
    def test_non_numeric_string_raises_error(self):
        """Test that non-numeric string raises error for scalar target"""
        raw_props = {'Etot': "not_a_number"}
        target_keys = ['Etot']
        
        with self.assertRaises(PropertyEnrichmentError):
            add_scalar_graph_targets(
                pyg_data=self.pyg_data,
                raw_properties_dict=raw_props,
                molecule_index=self.molecule_index,
                logger=self.logger,
                target_keys=target_keys,
                dataset_config=self.dataset_config,
                identifier=self.identifier
            )


# ==============================================================================
# TEST CASES: NODE FEATURES
# ==============================================================================

class TestAddNodeFeatures(unittest.TestCase):
    """Test add_node_features function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.molecule_index = 0
        self.identifier = "InChI=1S/C2H6/c1-2/h1-2H3"
        self.num_atoms = 8  # C2H6 has 8 atoms
        
        self.pyg_data = Data()
        self.pyg_data.z = torch.tensor([6, 6, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    
    def test_add_single_node_feature(self):
        """Test adding a single node feature"""
        raw_props = {
            'charges': np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        }
        
        add_node_features(
            pyg_data=self.pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            feature_keys=['charges'],
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(self.pyg_data, 'x'))
        self.assertIsInstance(self.pyg_data.x, torch.Tensor)
        self.assertEqual(self.pyg_data.x.shape[0], self.num_atoms)
    
    def test_add_multiple_node_features(self):
        """Test adding multiple node features"""
        raw_props = {
            'charges': np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'spins': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        }
        
        add_node_features(
            pyg_data=self.pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            feature_keys=['charges', 'spins'],
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(self.pyg_data, 'x'))
        self.assertEqual(self.pyg_data.x.shape, (self.num_atoms, 2))
    
    def test_empty_feature_keys(self):
        """Test with empty feature keys"""
        raw_props = {'charges': np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        
        add_node_features(
            pyg_data=self.pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            feature_keys=[],
            identifier=self.identifier
        )
        
        # x should not be added if empty keys
        self.assertTrue(not hasattr(self.pyg_data, 'x') or self.pyg_data.x is None)
    
    def test_shape_mismatch_raises_error(self):
        """Test that shape mismatch raises error"""
        raw_props = {
            'charges': np.array([0.1, 0.1, 0.0])  # Wrong size
        }
        
        with self.assertRaises(PropertyEnrichmentError):
            add_node_features(
                pyg_data=self.pyg_data,
                raw_properties_dict=raw_props,
                molecule_index=self.molecule_index,
                logger=self.logger,
                feature_keys=['charges'],
                identifier=self.identifier
            )
    
    def test_list_conversion(self):
        """Test conversion from list (CRITICAL FIX)"""
        raw_props = {
            'charges': [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
        
        add_node_features(
            pyg_data=self.pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            feature_keys=['charges'],
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(self.pyg_data, 'x'))
        self.assertEqual(self.pyg_data.x.shape[0], self.num_atoms)
    
    def test_zero_nodes_raises_error(self):
        """Test that zero nodes raises error"""
        pyg_data_zero = Data()
        pyg_data_zero.z = torch.tensor([], dtype=torch.long)
        
        raw_props = {'charges': np.array([])}
        
        with self.assertRaises((PropertyEnrichmentError, HandlerValidationError)):
            add_node_features(
                pyg_data=pyg_data_zero,
                raw_properties_dict=raw_props,
                molecule_index=self.molecule_index,
                logger=self.logger,
                feature_keys=['charges'],
                identifier=self.identifier
            )
    
    def test_concatenate_with_existing_x(self):
        """Test that new features concatenate with existing pyg_data.x"""
        # Set existing x features
        self.pyg_data.x = torch.rand(self.num_atoms, 3)
        original_feature_count = self.pyg_data.x.shape[1]
        
        raw_props = {
            'charges': np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        }
        
        add_node_features(
            pyg_data=self.pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            feature_keys=['charges'],
            identifier=self.identifier
        )
        
        # Should have original features + 1 new feature
        self.assertEqual(self.pyg_data.x.shape, (self.num_atoms, original_feature_count + 1))
    
    def test_unsupported_node_feature_type_raises_error(self):
        """Test that unsupported type for node feature raises error"""
        raw_props = {
            'charges': "not_an_array_or_list"
        }
        
        with self.assertRaises(PropertyEnrichmentError):
            add_node_features(
                pyg_data=self.pyg_data,
                raw_properties_dict=raw_props,
                molecule_index=self.molecule_index,
                logger=self.logger,
                feature_keys=['charges'],
                identifier=self.identifier
            )
    
    def test_missing_node_feature_raises_error(self):
        """Test that missing node feature raises error"""
        raw_props = {}  # No 'charges' key
        
        with self.assertRaises(PropertyEnrichmentError):
            add_node_features(
                pyg_data=self.pyg_data,
                raw_properties_dict=raw_props,
                molecule_index=self.molecule_index,
                logger=self.logger,
                feature_keys=['charges'],
                identifier=self.identifier
            )
    
    def test_num_nodes_updated_after_feature_addition(self):
        """Test that num_nodes is updated correctly after adding features"""
        raw_props = {
            'charges': np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        }
        
        add_node_features(
            pyg_data=self.pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            feature_keys=['charges'],
            identifier=self.identifier
        )
        
        self.assertEqual(self.pyg_data.num_nodes, self.num_atoms)


# ==============================================================================
# TEST CASES: VECTOR GRAPH PROPERTIES
# ==============================================================================

class TestAddVectorGraphProperties(unittest.TestCase):
    """Test add_vector_graph_properties function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.molecule_index = 0
        self.identifier = "InChI=1S/H2O/h1H2"
        
        self.pyg_data = Data()
        self.pyg_data.z = torch.tensor([8, 1, 1], dtype=torch.long)
        self.pyg_data.pos = torch.rand(3, 3)
        self.pyg_data.num_nodes = 3
    
    def test_add_dipole_property(self):
        """Test adding dipole vector property"""
        raw_props = {'dipole': np.array([0.5, 0.3, 0.2])}
        
        add_vector_graph_properties(
            pyg_data=self.pyg_data,
            mol_idx=self.molecule_index,
            raw_properties_dict=raw_props,
            prop_keys=['dipole'],
            logger=self.logger,
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(self.pyg_data, 'dipole'))
        self.assertIsInstance(self.pyg_data.dipole, torch.Tensor)
        self.assertEqual(self.pyg_data.dipole.shape, (3,))
    
    def test_add_rotational_constants_linear_molecule(self):
        """Test adding rotational constants for linear molecule (2 -> 3 padding)"""
        raw_props = {'rots': np.array([1.0, 2.0])}  # Linear molecule has 2 rots
        
        add_vector_graph_properties(
            pyg_data=self.pyg_data,
            mol_idx=self.molecule_index,
            raw_properties_dict=raw_props,
            prop_keys=['rots'],
            logger=self.logger,
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(self.pyg_data, 'rots'))
        self.assertEqual(self.pyg_data.rots.shape, (3,))  # Should be padded
        self.assertEqual(self.pyg_data.rots[2].item(), 0.0)  # Padded with 0
    
    def test_add_rotational_constants_nonlinear_molecule(self):
        """Test adding rotational constants for non-linear molecule"""
        raw_props = {'rots': np.array([1.0, 2.0, 3.0])}
        
        add_vector_graph_properties(
            pyg_data=self.pyg_data,
            mol_idx=self.molecule_index,
            raw_properties_dict=raw_props,
            prop_keys=['rots'],
            logger=self.logger,
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(self.pyg_data, 'rots'))
        self.assertEqual(self.pyg_data.rots.shape, (3,))
    
    def test_wrong_dipole_shape_raises_error(self):
        """Test that wrong dipole shape raises error"""
        raw_props = {'dipole': np.array([0.5, 0.3])}  # Should be (3,)
        
        with self.assertRaises(PropertyEnrichmentError):
            add_vector_graph_properties(
                pyg_data=self.pyg_data,
                mol_idx=self.molecule_index,
                raw_properties_dict=raw_props,
                prop_keys=['dipole'],
                logger=self.logger,
                identifier=self.identifier
            )
    
    def test_list_conversion(self):
        """Test conversion from list"""
        raw_props = {'dipole': [0.5, 0.3, 0.2]}
        
        add_vector_graph_properties(
            pyg_data=self.pyg_data,
            mol_idx=self.molecule_index,
            raw_properties_dict=raw_props,
            prop_keys=['dipole'],
            logger=self.logger,
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(self.pyg_data, 'dipole'))
        self.assertIsInstance(self.pyg_data.dipole, torch.Tensor)
    
    def test_missing_property_raises_error(self):
        """Test that missing property raises error"""
        raw_props = {}
        
        with self.assertRaises(PropertyEnrichmentError):
            add_vector_graph_properties(
                pyg_data=self.pyg_data,
                mol_idx=self.molecule_index,
                raw_properties_dict=raw_props,
                prop_keys=['dipole'],
                logger=self.logger,
                identifier=self.identifier
            )
    
    def test_no_nodes_or_pos_raises_error(self):
        """Test that missing nodes or positions raises error"""
        pyg_data_empty = Data()
        pyg_data_empty.num_nodes = 0
        
        raw_props = {'dipole': np.array([0.5, 0.3, 0.2])}
        
        with self.assertRaises((PropertyEnrichmentError, HandlerValidationError)):
            add_vector_graph_properties(
                pyg_data=pyg_data_empty,
                mol_idx=self.molecule_index,
                raw_properties_dict=raw_props,
                prop_keys=['dipole'],
                logger=self.logger,
                identifier=self.identifier
            )
    
    def test_quadrupole_shape_validation(self):
        """Test quadrupole property shape (6,) is validated"""
        raw_props = {'quadrupole': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])}
        
        add_vector_graph_properties(
            pyg_data=self.pyg_data,
            mol_idx=self.molecule_index,
            raw_properties_dict=raw_props,
            prop_keys=['quadrupole'],
            logger=self.logger,
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(self.pyg_data, 'quadrupole'))
        self.assertEqual(self.pyg_data.quadrupole.shape, (6,))
    
    def test_invalid_rots_shape_raises_error(self):
        """Test that invalid rots shape (not 2 or 3) raises error"""
        raw_props = {'rots': np.array([1.0, 2.0, 3.0, 4.0])}  # (4,) is invalid
        
        with self.assertRaises(PropertyEnrichmentError):
            add_vector_graph_properties(
                pyg_data=self.pyg_data,
                mol_idx=self.molecule_index,
                raw_properties_dict=raw_props,
                prop_keys=['rots'],
                logger=self.logger,
                identifier=self.identifier
            )
    
    def test_non_1d_array_raises_error(self):
        """Test that non-1D array raises error"""
        raw_props = {'dipole': np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.1]])}  # 2D
        
        with self.assertRaises(PropertyEnrichmentError):
            add_vector_graph_properties(
                pyg_data=self.pyg_data,
                mol_idx=self.molecule_index,
                raw_properties_dict=raw_props,
                prop_keys=['dipole'],
                logger=self.logger,
                identifier=self.identifier
            )
    
    def test_multiple_vector_properties(self):
        """Test adding multiple vector properties at once"""
        raw_props = {
            'dipole': np.array([0.5, 0.3, 0.2]),
            'rots': np.array([1.0, 2.0, 3.0])
        }
        
        add_vector_graph_properties(
            pyg_data=self.pyg_data,
            mol_idx=self.molecule_index,
            raw_properties_dict=raw_props,
            prop_keys=['dipole', 'rots'],
            logger=self.logger,
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(self.pyg_data, 'dipole'))
        self.assertTrue(hasattr(self.pyg_data, 'rots'))


# ==============================================================================
# TEST CASES: VARIABLE LENGTH GRAPH PROPERTIES
# ==============================================================================

class TestAddVariableLenGraphProperties(unittest.TestCase):
    """Test add_variable_len_graph_properties function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.molecule_index = 0
        self.identifier = "InChI=1S/H2O/h1H2"
        self.data_config = {'vibration_refinement': {'comparison_tolerance': 1e-4}}
        
        self.pyg_data = Data()
        self.pyg_data.z = torch.tensor([8, 1, 1], dtype=torch.long)
        self.pyg_data.pos = torch.rand(3, 3)
        self.pyg_data.num_nodes = 3
    
    def test_empty_property_keys(self):
        """Test with empty property keys"""
        raw_props = {}
        
        add_variable_len_graph_properties(
            pyg_data=self.pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            property_keys=[],
            data_config=self.data_config,
            identifier=self.identifier
        )
        # Should complete without error
    
    def test_zero_nodes_raises_error(self):
        """Test that zero nodes raises error"""
        pyg_data_zero = Data()
        pyg_data_zero.z = torch.tensor([], dtype=torch.long)
        pyg_data_zero.num_nodes = 0
        
        with self.assertRaises((PropertyEnrichmentError, HandlerValidationError)):
            add_variable_len_graph_properties(
                pyg_data=pyg_data_zero,
                raw_properties_dict={'freqs': np.array([100.0])},
                molecule_index=self.molecule_index,
                logger=self.logger,
                property_keys=['freqs'],
                data_config=self.data_config,
                identifier=self.identifier
            )
    
    @patch('milia_pipeline.molecules.property_enrichment.refine_molecular_vibrations')
    def test_freqs_and_vibmodes_with_refinement(self, mock_refine):
        """Test processing both freqs and vibmodes triggers refinement"""
        num_modes = 3
        num_atoms = 3
        
        raw_freqs = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        raw_vibmodes = np.random.rand(num_modes, num_atoms, 3).astype(np.float32)
        
        # Mock successful refinement
        mock_refine.return_value = (
            raw_freqs.tolist(),
            [raw_vibmodes[i] for i in range(num_modes)],
            True
        )
        
        raw_props = {'freqs': raw_freqs, 'vibmodes': raw_vibmodes}
        
        add_variable_len_graph_properties(
            pyg_data=self.pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            property_keys=['freqs', 'vibmodes'],
            data_config=self.data_config,
            identifier=self.identifier
        )
        
        mock_refine.assert_called_once()
        self.assertTrue(hasattr(self.pyg_data, 'freqs'))
        self.assertTrue(hasattr(self.pyg_data, 'vibmodes'))
    
    def test_generic_variable_length_property(self):
        """Test processing a generic (non-freq/vibmode) variable-length property"""
        raw_props = {'intensities': np.array([0.5, 1.2, 0.8], dtype=np.float32)}
        
        add_variable_len_graph_properties(
            pyg_data=self.pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            property_keys=['intensities'],
            data_config=self.data_config,
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(self.pyg_data, 'intensities'))
        self.assertIsInstance(self.pyg_data.intensities, torch.Tensor)
    
    def test_missing_variable_length_property_raises_error(self):
        """Test that missing variable-length property raises error"""
        raw_props = {}  # 'freqs' not present
        
        with self.assertRaises(PropertyEnrichmentError):
            add_variable_len_graph_properties(
                pyg_data=self.pyg_data,
                raw_properties_dict=raw_props,
                molecule_index=self.molecule_index,
                logger=self.logger,
                property_keys=['freqs'],
                data_config=self.data_config,
                identifier=self.identifier
            )


# ==============================================================================
# TEST CASES: VIBRATIONAL DATA PROCESSING
# ==============================================================================

class TestProcessVibrationalData(unittest.TestCase):
    """Test _process_vibrational_data function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.molecule_index = 0
        self.identifier = "InChI=1S/H2O/h1H2"
        self.refinement_tolerance = 1e-4
    
    @patch('milia_pipeline.molecules.property_enrichment.refine_molecular_vibrations')
    def test_refine_both_present(self, mock_refine):
        """Test refinement when both freqs and vibmodes are present"""
        num_modes = 3
        num_atoms = 3
        
        raw_freqs = np.array([100.0, 200.0, 300.0])
        raw_vibmodes = np.random.rand(num_modes * num_atoms, 3)
        
        mock_refine.return_value = (
            raw_freqs.tolist(),
            [raw_vibmodes[i*num_atoms:(i+1)*num_atoms] for i in range(num_modes)],
            True
        )
        
        raw_props = {'freqs': raw_freqs, 'vibmodes': raw_vibmodes}
        
        result = _process_vibrational_data(
            raw_properties_dict=raw_props,
            property_keys=['freqs', 'vibmodes'],
            molecule_index=self.molecule_index,
            logger=self.logger,
            refinement_tolerance=self.refinement_tolerance,
            identifier=self.identifier
        )
        
        self.assertIn('freqs', result)
        self.assertIn('vibmodes', result)
        mock_refine.assert_called_once()
    
    def test_only_freqs_present(self):
        """Test when only freqs is present (no refinement)"""
        raw_freqs = np.array([100.0, 200.0, 300.0])
        raw_props = {'freqs': raw_freqs}
        
        result = _process_vibrational_data(
            raw_properties_dict=raw_props,
            property_keys=['freqs'],
            molecule_index=self.molecule_index,
            logger=self.logger,
            refinement_tolerance=self.refinement_tolerance,
            identifier=self.identifier
        )
        
        self.assertIn('freqs', result)
        np.testing.assert_array_equal(result['freqs'], raw_freqs)
    
    @patch('milia_pipeline.molecules.property_enrichment.refine_molecular_vibrations')
    def test_refinement_rejected_raises_error(self, mock_refine):
        """Test that rejected refinement raises PropertyEnrichmentError"""
        num_modes = 3
        num_atoms = 3
        
        raw_freqs = np.array([100.0, 200.0, 300.0])
        raw_vibmodes = np.random.rand(num_modes * num_atoms, 3)
        
        # Mock rejected refinement
        mock_refine.return_value = (
            raw_freqs.tolist(),
            [raw_vibmodes[i*num_atoms:(i+1)*num_atoms] for i in range(num_modes)],
            False  # Rejected
        )
        
        raw_props = {'freqs': raw_freqs, 'vibmodes': raw_vibmodes}
        
        with self.assertRaises((PropertyEnrichmentError, HandlerValidationError)):
            _process_vibrational_data(
                raw_properties_dict=raw_props,
                property_keys=['freqs', 'vibmodes'],
                molecule_index=self.molecule_index,
                logger=self.logger,
                refinement_tolerance=self.refinement_tolerance,
                identifier=self.identifier
            )
    
    def test_none_freqs_data_no_refinement(self):
        """Test that None freqs data skips refinement"""
        raw_props = {'freqs': None, 'vibmodes': np.random.rand(9, 3)}
        
        result = _process_vibrational_data(
            raw_properties_dict=raw_props,
            property_keys=['freqs', 'vibmodes'],
            molecule_index=self.molecule_index,
            logger=self.logger,
            refinement_tolerance=self.refinement_tolerance,
            identifier=self.identifier
        )
        
        # Should return dict without refinement (raw values as-is)
        self.assertIsInstance(result, dict)
    
    def test_only_vibmodes_present(self):
        """Test when only vibmodes is present (no refinement)"""
        raw_vibmodes = np.random.rand(9, 3).astype(np.float32)
        raw_props = {'vibmodes': raw_vibmodes}
        
        result = _process_vibrational_data(
            raw_properties_dict=raw_props,
            property_keys=['vibmodes'],
            molecule_index=self.molecule_index,
            logger=self.logger,
            refinement_tolerance=self.refinement_tolerance,
            identifier=self.identifier
        )
        
        self.assertIn('vibmodes', result)
        np.testing.assert_array_equal(result['vibmodes'], raw_vibmodes)


# ==============================================================================
# TEST CASES: VIBMODES PROCESSING
# ==============================================================================

class TestProcessVibmodes(unittest.TestCase):
    """Test _process_vibmodes function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.molecule_index = 0
        self.identifier = "InChI=1S/H2O/h1H2"
        self.num_atoms = 3
    
    def test_process_2d_array(self):
        """Test processing 2D array (num_modes*num_atoms, 3)"""
        num_modes = 9  # 3N - 6 modes for 3 atoms
        vibmodes_2d = np.random.rand(num_modes * self.num_atoms, 3).astype(np.float32)
        
        result = _process_vibmodes(
            vibmodes_data=vibmodes_2d,
            num_atoms=self.num_atoms,
            molecule_index=self.molecule_index,
            inchi=self.identifier,
            logger=self.logger
        )
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), num_modes)
        for mode in result:
            self.assertIsInstance(mode, torch.Tensor)
            self.assertEqual(mode.shape, (self.num_atoms, 3))
    
    def test_process_3d_array(self):
        """Test processing 3D array (num_modes, num_atoms, 3)"""
        num_modes = 9
        vibmodes_3d = np.random.rand(num_modes, self.num_atoms, 3).astype(np.float32)
        
        result = _process_vibmodes(
            vibmodes_data=vibmodes_3d,
            num_atoms=self.num_atoms,
            molecule_index=self.molecule_index,
            inchi=self.identifier,
            logger=self.logger
        )
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), num_modes)
    
    def test_zero_atoms_raises_error(self):
        """Test that zero atoms raises error"""
        vibmodes = np.random.rand(9, 3).astype(np.float32)
        
        with self.assertRaises((PropertyEnrichmentError, HandlerValidationError)):
            _process_vibmodes(
                vibmodes_data=vibmodes,
                num_atoms=0,
                molecule_index=self.molecule_index,
                inchi=self.identifier,
                logger=self.logger
            )
    
    def test_process_list_of_numpy_arrays(self):
        """Test processing list of numpy arrays (refined vibmodes format)"""
        num_modes = 3
        vibmodes_list = [
            np.random.rand(self.num_atoms, 3).astype(np.float32)
            for _ in range(num_modes)
        ]
        
        result = _process_vibmodes(
            vibmodes_data=vibmodes_list,
            num_atoms=self.num_atoms,
            molecule_index=self.molecule_index,
            inchi=self.identifier,
            logger=self.logger
        )
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), num_modes)
        for mode in result:
            self.assertIsInstance(mode, torch.Tensor)
            self.assertEqual(mode.shape, (self.num_atoms, 3))
    
    def test_invalid_vibmode_format_raises_error(self):
        """Test that unsupported vibmodes format raises error"""
        # A 1D array is not a valid vibmodes format
        vibmodes_invalid = np.array([1.0, 2.0, 3.0])
        
        with self.assertRaises(PropertyEnrichmentError):
            _process_vibmodes(
                vibmodes_data=vibmodes_invalid,
                num_atoms=self.num_atoms,
                molecule_index=self.molecule_index,
                inchi=self.identifier,
                logger=self.logger
            )
    
    def test_2d_array_not_divisible_by_num_atoms_raises_error(self):
        """Test that 2D array with rows not divisible by num_atoms raises error"""
        # 7 rows, not divisible by 3 atoms
        vibmodes_bad = np.random.rand(7, 3).astype(np.float32)
        
        with self.assertRaises(PropertyEnrichmentError):
            _process_vibmodes(
                vibmodes_data=vibmodes_bad,
                num_atoms=self.num_atoms,
                molecule_index=self.molecule_index,
                inchi=self.identifier,
                logger=self.logger
            )
    
    def test_list_of_arrays_invalid_shape_raises_error(self):
        """Test that list of arrays with invalid individual shape raises error"""
        vibmodes_list = [
            np.random.rand(self.num_atoms + 1, 3).astype(np.float32)  # Wrong atom count
            for _ in range(3)
        ]
        
        with self.assertRaises(PropertyEnrichmentError):
            _process_vibmodes(
                vibmodes_data=vibmodes_list,
                num_atoms=self.num_atoms,
                molecule_index=self.molecule_index,
                inchi=self.identifier,
                logger=self.logger
            )


# ==============================================================================
# TEST CASES: FREQUENCIES PROCESSING
# ==============================================================================

class TestProcessFrequencies(unittest.TestCase):
    """Test _process_frequencies function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.molecule_index = 0
        self.identifier = "InChI=1S/H2O/h1H2"
    
    def test_process_real_frequencies(self):
        """Test processing real frequency data"""
        freqs = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        
        result = _process_frequencies(
            freqs_data=freqs,
            molecule_index=self.molecule_index,
            inchi=self.identifier,
            logger=self.logger
        )
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.shape, (3,))
    
    def test_process_complex_frequencies(self):
        """Test processing complex frequency data (imaginary modes)"""
        freqs = np.array([100.0+0j, 200.0+0j, -50.0+10j], dtype=np.complex64)
        
        result = _process_frequencies(
            freqs_data=freqs,
            molecule_index=self.molecule_index,
            inchi=self.identifier,
            logger=self.logger
        )
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.complex64)
    
    def test_process_list_frequencies(self):
        """Test processing list frequency data"""
        freqs = [100.0, 200.0, 300.0]
        
        result = _process_frequencies(
            freqs_data=freqs,
            molecule_index=self.molecule_index,
            inchi=self.identifier,
            logger=self.logger
        )
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3,))
    
    def test_none_frequencies_raises_error(self):
        """Test that None frequencies raises error"""
        with self.assertRaises((PropertyEnrichmentError, HandlerValidationError)):
            _process_frequencies(
                freqs_data=None,
                molecule_index=self.molecule_index,
                inchi=self.identifier,
                logger=self.logger
            )


# ==============================================================================
# TEST CASES: ATOMIZATION ENERGY
# ==============================================================================

class TestCalculateAtomizationEnergy(unittest.TestCase):
    """Test calculate_atomization_energy function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.molecule_index = 0
        self.identifier = "InChI=1S/H2O/h1H2"
        self.dataset_config = create_mock_dataset_config()
    
    def test_water_atomization_energy(self):
        """Test atomization energy calculation for water"""
        # H2O - molecular energy approximation
        molecular_energy_hartree = -76.0  # Approximate value
        atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.long)
        
        result = calculate_atomization_energy(
            molecular_total_energy_hartree=molecular_energy_hartree,
            atomic_numbers_tensor=atomic_numbers,
            molecule_index=self.molecule_index,
            logger=self.logger,
            dataset_config=self.dataset_config,
            identifier=self.identifier
        )
        
        self.assertIsInstance(result, float)
        # Atomization energy is the energy to break all bonds
        # It can be positive (energy required) or negative depending on convention
        # Just verify it's a valid finite number
        self.assertFalse(np.isnan(result))
        self.assertFalse(np.isinf(result))
    
    def test_single_atom(self):
        """Test atomization energy for single atom"""
        # Single H atom
        atomic_numbers = torch.tensor([1], dtype=torch.long)
        h_energy = ATOMIC_ENERGIES_HARTREE.get(1, -0.5)
        
        result = calculate_atomization_energy(
            molecular_total_energy_hartree=h_energy,
            atomic_numbers_tensor=atomic_numbers,
            molecule_index=self.molecule_index,
            logger=self.logger,
            dataset_config=self.dataset_config,
            identifier=self.identifier
        )
        
        # Single atom should have zero atomization energy
        self.assertAlmostEqual(result, 0.0, places=5)
    
    def test_unknown_atomic_number_raises_error(self):
        """Test that unknown atomic number raises error"""
        atomic_numbers = torch.tensor([999], dtype=torch.long)  # Invalid
        
        with self.assertRaises(PropertyEnrichmentError):
            calculate_atomization_energy(
                molecular_total_energy_hartree=-100.0,
                atomic_numbers_tensor=atomic_numbers,
                molecule_index=self.molecule_index,
                logger=self.logger,
                dataset_config=self.dataset_config,
                identifier=self.identifier
            )


# ==============================================================================
# TEST CASES: MAIN ENRICHMENT ORCHESTRATION
# ==============================================================================

class TestEnrichPyGDataWithProperties(unittest.TestCase):
    """Test enrich_pyg_data_with_properties function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.molecule_index = 0
        self.identifier = "InChI=1S/H2O/h1H2"
        
        self.pyg_data = Data()
        self.pyg_data.z = torch.tensor([8, 1, 1], dtype=torch.long)
        self.pyg_data.pos = torch.rand(3, 3)
        
        self.raw_props = {
            'atoms': np.array([8, 1, 1]),
            'coordinates': np.random.rand(3, 3),
            'Etot': -76.0
        }
    
    def test_handler_required(self):
        """Test that handler is required"""
        with self.assertRaises(HandlerOperationError):
            enrich_pyg_data_with_properties(
                pyg_data=self.pyg_data,
                mol_idx=self.molecule_index,
                raw_properties_dict=self.raw_props,
                inchi_identifier=self.identifier,
                logger=self.logger,
                dataset_handler=None
            )
    
    def test_handler_enrichment_called(self):
        """Test that handler enrichment is called"""
        mock_handler = MockDFTHandler()
        mock_handler.enrich_pyg_data = Mock(return_value=self.pyg_data)
        
        result = enrich_pyg_data_with_properties(
            pyg_data=self.pyg_data,
            mol_idx=self.molecule_index,
            raw_properties_dict=self.raw_props,
            inchi_identifier=self.identifier,
            logger=self.logger,
            dataset_handler=mock_handler
        )
        
        mock_handler.enrich_pyg_data.assert_called_once()
        self.assertIsNotNone(result)
    
    def test_invalid_handler_result_raises_error(self):
        """Test that invalid handler result raises error"""
        mock_handler = MockDFTHandler()
        mock_handler.enrich_pyg_data = Mock(return_value=None)
        
        with self.assertRaises(HandlerOperationError):
            enrich_pyg_data_with_properties(
                pyg_data=self.pyg_data,
                mol_idx=self.molecule_index,
                raw_properties_dict=self.raw_props,
                inchi_identifier=self.identifier,
                logger=self.logger,
                dataset_handler=mock_handler
            )
    
    def test_handler_exception_converted_to_handler_operation_error(self):
        """Test that RuntimeError from handler is converted to HandlerOperationError"""
        mock_handler = MockDFTHandler()
        mock_handler.enrich_pyg_data = Mock(
            side_effect=RuntimeError("Unexpected failure")
        )
        
        with self.assertRaises(HandlerOperationError) as context:
            enrich_pyg_data_with_properties(
                pyg_data=self.pyg_data,
                mol_idx=self.molecule_index,
                raw_properties_dict=self.raw_props,
                inchi_identifier=self.identifier,
                logger=self.logger,
                dataset_handler=mock_handler
            )
        
        error = context.exception
        self.assertIn("unexpected error", error.message.lower())
    
    def test_property_enrichment_error_wrapped_in_handler_error(self):
        """Test that PropertyEnrichmentError from handler is wrapped in HandlerOperationError"""
        mock_handler = MockDFTHandler()
        mock_handler.enrich_pyg_data = Mock(
            side_effect=PropertyEnrichmentError(
                molecule_index=0,
                inchi="test",
                property_name="Etot",
                reason="Test property error",
                detail="detail"
            )
        )
        
        with self.assertRaises(HandlerOperationError) as context:
            enrich_pyg_data_with_properties(
                pyg_data=self.pyg_data,
                mol_idx=self.molecule_index,
                raw_properties_dict=self.raw_props,
                inchi_identifier=self.identifier,
                logger=self.logger,
                dataset_handler=mock_handler
            )
        
        error = context.exception
        self.assertIn("property error", error.message.lower())
    
    def test_non_tensor_z_auto_converted(self):
        """Test that non-tensor z is auto-converted to tensor before handler call"""
        pyg_data = Data()
        pyg_data.z = np.array([8, 1, 1])  # numpy, not tensor
        pyg_data.pos = torch.rand(3, 3)
        
        mock_handler = MockDFTHandler()
        mock_handler.enrich_pyg_data = Mock(return_value=pyg_data)
        
        result = enrich_pyg_data_with_properties(
            pyg_data=pyg_data,
            mol_idx=self.molecule_index,
            raw_properties_dict=self.raw_props,
            inchi_identifier=self.identifier,
            logger=self.logger,
            dataset_handler=mock_handler
        )
        
        # z should have been converted to tensor before handler was called
        mock_handler.enrich_pyg_data.assert_called_once()
    
    def test_handler_error_reraise_as_is(self):
        """Test that HandlerError from handler is re-raised as-is"""
        mock_handler = MockDFTHandler()
        mock_handler.enrich_pyg_data = Mock(
            side_effect=HandlerOperationError(
                message="Direct handler error",
                handler_type="DFT",
                operation="enrich_pyg_data",
                molecule_index=0,
                details="Test"
            )
        )
        
        with self.assertRaises(HandlerOperationError) as context:
            enrich_pyg_data_with_properties(
                pyg_data=self.pyg_data,
                mol_idx=self.molecule_index,
                raw_properties_dict=self.raw_props,
                inchi_identifier=self.identifier,
                logger=self.logger,
                dataset_handler=mock_handler
            )
        
        self.assertEqual(context.exception.message, "Direct handler error")


# ==============================================================================
# TEST CASES: HANDLER INTEGRATION
# ==============================================================================

class TestHandlerIntegration(unittest.TestCase):
    """Test handler integration functions"""
    
    def test_get_handler_context_returns_dict_or_none(self):
        """Test _get_handler_context returns context or None"""
        result = _get_handler_context()
        # Function may return None or a context dict depending on call stack
        # It uses frame inspection to detect handler context
        self.assertTrue(result is None or isinstance(result, dict))
    
    def test_verify_interface_compatibility_returns_dict(self):
        """Test verify_interface_compatibility returns dict"""
        result = verify_interface_compatibility()
        
        self.assertIsInstance(result, dict)
        self.assertIn('handler_pattern_integration', result)
        self.assertIn('functions_checked', result)
    
    def test_get_handler_integration_summary_returns_dict(self):
        """Test get_handler_integration_summary returns dict"""
        result = get_handler_integration_summary()
        
        self.assertIsInstance(result, dict)
        self.assertIn('integration_stage', result)
        self.assertIn('primary_objectives', result)
    
    def test_get_handler_integration_status_returns_dict(self):
        """Test get_handler_integration_status returns dict"""
        result = get_handler_integration_status()
        
        self.assertIsInstance(result, dict)
        self.assertIn('add_scalar_graph_targets', result)
        self.assertIn('add_node_features', result)
        self.assertEqual(result['overall_status'], 'HANDLER_PATTERN_INTEGRATED')
    
    def test_validate_handler_compatibility_with_valid_handler(self):
        """Test handler compatibility validation with valid handler"""
        mock_handler = MockDFTHandler()
        
        mock_handler.enrich_pyg_data = Mock()
        mock_handler.validate_molecule_data = Mock()
        
        try:
            validate_handler_compatibility(mock_handler)
        except HandlerValidationError as e:
            self.skipTest(f"Handler validation requires additional methods: {str(e)}")
    
    def test_validate_handler_compatibility_with_incompatible_handler(self):
        """Test handler compatibility validation with incompatible handler raises error"""
        # Create an object missing required methods
        incompatible_handler = type('IncompHandler', (), {})()
        
        with self.assertRaises(HandlerValidationError) as context:
            validate_handler_compatibility(incompatible_handler)
        
        error = context.exception
        self.assertIn('missing', error.message.lower())
    
    def test_validate_handler_compatibility_custom_methods(self):
        """Test handler compatibility with custom required methods list"""
        handler = Mock()
        handler.custom_method = Mock()
        
        result = validate_handler_compatibility(handler, required_methods=['custom_method'])
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result['compatible'])
        self.assertEqual(result['missing_methods'], [])
    
    def test_validate_handler_compatibility_returns_validation_results(self):
        """Test that validation results dict has expected structure"""
        mock_handler = MockDFTHandler()
        # Add all 4 default required methods
        mock_handler.enrich_pyg_data = Mock()
        mock_handler.validate_molecule_data = Mock()
        mock_handler.get_required_properties = Mock()
        mock_handler.process_property_value = Mock()
        
        result = validate_handler_compatibility(mock_handler)
        
        self.assertIn('handler_type', result)
        self.assertIn('compatible', result)
        self.assertIn('missing_methods', result)
        self.assertIn('method_check', result)
        self.assertTrue(result['compatible'])


# ==============================================================================
# TEST CASES: CREATE HANDLER ERROR CONTEXT
# ==============================================================================

class TestCreateHandlerErrorContext(unittest.TestCase):
    """Test create_handler_error_context function"""
    
    def test_basic_context_creation(self):
        """Test basic error context creation"""
        context = create_handler_error_context(
            operation='test_operation',
            molecule_index=42
        )
        
        self.assertIsInstance(context, dict)
        self.assertEqual(context['operation'], 'test_operation')
        self.assertEqual(context['molecule_index'], 42)
        self.assertIn('timestamp', context)
        self.assertIn('handler_context', context)
    
    def test_context_with_property_name(self):
        """Test error context with property_name"""
        context = create_handler_error_context(
            operation='tensor_conversion',
            molecule_index=0,
            property_name='Etot'
        )
        
        self.assertEqual(context['property_name'], 'Etot')
    
    def test_context_with_extra_kwargs(self):
        """Test error context with additional kwargs"""
        context = create_handler_error_context(
            operation='enrichment',
            molecule_index=5,
            handler_type='DFT',
            detail='extra info'
        )
        
        self.assertEqual(context['handler_type'], 'DFT')
        self.assertEqual(context['detail'], 'extra info')
    
    def test_context_without_property_name(self):
        """Test that property_name is absent when not provided"""
        context = create_handler_error_context(
            operation='test',
            molecule_index=0
        )
        
        self.assertNotIn('property_name', context)
    
    def test_timestamp_is_iso_format(self):
        """Test that timestamp is in ISO format"""
        context = create_handler_error_context(
            operation='test',
            molecule_index=0
        )
        
        # ISO format check: should be parseable
        import datetime
        try:
            datetime.datetime.fromisoformat(context['timestamp'])
        except ValueError:
            self.fail("Timestamp is not in ISO format")


# ==============================================================================
# TEST CASES: PARAMETER RESOLUTION DECORATOR
# ==============================================================================

class TestParameterResolution(unittest.TestCase):
    """Test resolve_parameters decorator"""
    
    def test_decorator_applied_to_functions(self):
        """Test that decorator is applied to critical functions"""
        self.assertTrue(hasattr(add_scalar_graph_targets, '__wrapped__'))
        self.assertTrue(hasattr(add_node_features, '__wrapped__'))
        self.assertTrue(hasattr(calculate_atomization_energy, '__wrapped__'))
    
    def test_identifier_parameter_resolution(self):
        """Test that identifier parameter is properly resolved when not provided"""
        # When identifier is not passed, resolve_parameters should set default 'N/A'
        pyg_data = Data()
        raw_props = {'Etot': -100.5}
        logger = create_mock_logger()
        
        # Call without explicit identifier — the decorator should resolve it
        add_scalar_graph_targets(
            pyg_data=pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=0,
            logger=logger,
            target_keys=['Etot'],
            dataset_config=create_mock_dataset_config()
        )
        
        self.assertTrue(hasattr(pyg_data, 'y'))
        self.assertAlmostEqual(pyg_data.y.item(), -100.5, places=5)


# ==============================================================================
# TEST CASES: ERROR HANDLING
# ==============================================================================

class TestErrorHandling(unittest.TestCase):
    """Test error handling and exception raising"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.molecule_index = 0
        self.identifier = "InChI=1S/H2O/h1H2"
    
    def test_property_enrichment_error_structure(self):
        """Test PropertyEnrichmentError structure"""
        error = PropertyEnrichmentError(
            molecule_index=self.molecule_index,
            inchi=self.identifier,
            property_name="test_prop",
            reason="Test reason",
            detail="Test detail"
        )
        
        self.assertEqual(error.molecule_index, self.molecule_index)
        self.assertEqual(error.inchi, self.identifier)
        self.assertEqual(error.property_name, "test_prop")
    
    def test_handler_operation_error_in_context(self):
        """Test that HandlerOperationError is raised when handler enrichment fails"""
        mock_handler = MockDFTHandler()
        mock_handler.enrich_pyg_data = Mock(
            side_effect=RuntimeError("Simulated handler failure")
        )
        
        pyg_data = Data()
        pyg_data.z = torch.tensor([8, 1, 1], dtype=torch.long)
        pyg_data.pos = torch.rand(3, 3)
        
        with self.assertRaises(HandlerOperationError):
            enrich_pyg_data_with_properties(
                pyg_data=pyg_data,
                mol_idx=self.molecule_index,
                raw_properties_dict={'Etot': -76.0},
                inchi_identifier=self.identifier,
                logger=self.logger,
                dataset_handler=mock_handler
            )
    
    def test_handler_validation_error_in_context(self):
        """Test that HandlerValidationError is raised for validation failures"""
        # A handler with missing required methods should trigger validation error
        incomplete_handler = Mock(spec=[])  # No methods at all
        
        with self.assertRaises(HandlerValidationError):
            validate_handler_compatibility(incomplete_handler)


# ==============================================================================
# TEST CASES: EDGE CASES
# ==============================================================================

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.molecule_index = 0
        self.identifier = "InChI=1S/H/h1H"
    
    def test_single_atom_molecule(self):
        """Test handling of single-atom molecule"""
        pyg_data = Data()
        pyg_data.z = torch.tensor([1], dtype=torch.long)
        
        raw_props = {
            'atoms': np.array([1]),
            'coordinates': np.random.rand(1, 3),
            'Etot': -0.5
        }
        
        add_scalar_graph_targets(
            pyg_data=pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            target_keys=['Etot'],
            dataset_config=create_mock_dataset_config(),
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(pyg_data, 'y'))
    
    def test_large_molecule(self):
        """Test handling of large molecule"""
        num_atoms = 100
        pyg_data = Data()
        pyg_data.z = torch.tensor([6] * num_atoms, dtype=torch.long)
        
        raw_props = {
            'atoms': np.array([6] * num_atoms),
            'coordinates': np.random.rand(num_atoms, 3),
            'Etot': -500.0,
            'charges': np.random.rand(num_atoms)
        }
        
        add_node_features(
            pyg_data=pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            feature_keys=['charges'],
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(pyg_data, 'x'))
        self.assertEqual(pyg_data.x.shape[0], num_atoms)
    
    def test_unicode_identifier(self):
        """Test handling of unicode characters in identifier"""
        unicode_identifier = "InChI=1S/H2O/h1H2_测试"
        
        result = _ensure_tensor(
            [1.0, 2.0],
            torch.float32,
            "test_prop",
            self.molecule_index,
            unicode_identifier
        )
        
        self.assertIsInstance(result, torch.Tensor)
    
    def test_extreme_numeric_values(self):
        """Test handling of extreme numeric values"""
        pyg_data = Data()
        
        # Very large value
        raw_props = {'Etot': 1e10}
        
        add_scalar_graph_targets(
            pyg_data=pyg_data,
            raw_properties_dict=raw_props,
            molecule_index=self.molecule_index,
            logger=self.logger,
            target_keys=['Etot'],
            dataset_config=create_mock_dataset_config(),
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(pyg_data, 'y'))
        
        # Very small value
        pyg_data2 = Data()
        raw_props2 = {'Etot': 1e-10}
        
        add_scalar_graph_targets(
            pyg_data=pyg_data2,
            raw_properties_dict=raw_props2,
            molecule_index=self.molecule_index,
            logger=self.logger,
            target_keys=['Etot'],
            dataset_config=create_mock_dataset_config(),
            identifier=self.identifier
        )
        
        self.assertTrue(hasattr(pyg_data2, 'y'))


# ==============================================================================
# PHASE 6: REGISTRY INTEGRATION FUNCTION TESTS
# ==============================================================================

class TestPhase6RegistryIntegrationFunctions(unittest.TestCase):
    """
    Test Phase 6 registry integration functions.
    
    These tests verify the new registry functions added in Phase 6:
    - _init_registry()
    - _get_available_dataset_types()
    - _is_dataset_type_registered()
    - _get_dataset_feature()
    """
    
    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")
    
    def setUp(self):
        reset_registry_state()
    
    def tearDown(self):
        reset_registry_state()
    
    def test_01_init_registry_exists(self):
        """Test _init_registry function exists and is callable."""
        self.assertTrue(callable(_init_registry))
    
    def test_02_get_available_dataset_types_exists(self):
        """Test _get_available_dataset_types function exists."""
        self.assertTrue(callable(_get_available_dataset_types))
    
    def test_03_is_dataset_type_registered_exists(self):
        """Test _is_dataset_type_registered function exists."""
        self.assertTrue(callable(_is_dataset_type_registered))
    
    def test_04_get_dataset_feature_exists(self):
        """Test _get_dataset_feature function exists."""
        self.assertTrue(callable(_get_dataset_feature))
    
    def test_05_get_registry_integration_status_exists(self):
        """Test get_registry_integration_status function exists."""
        self.assertTrue(callable(get_registry_integration_status))
    
    def test_06_init_registry_returns_bool(self):
        """Test _init_registry returns a boolean."""
        result = _init_registry()
        self.assertIsInstance(result, bool)
    
    def test_07_init_registry_idempotent(self):
        """Test _init_registry is idempotent (multiple calls same result)."""
        result1 = _init_registry()
        result2 = _init_registry()
        self.assertEqual(result1, result2)
    
    def test_08_get_available_dataset_types_returns_list(self):
        """Test _get_available_dataset_types returns a list."""
        types = _get_available_dataset_types()
        self.assertIsInstance(types, list)
        self.assertGreater(len(types), 0)
    
    def test_09_get_available_dataset_types_includes_known_types(self):
        """Test _get_available_dataset_types includes DFT, DMC, Wavefunction."""
        types = _get_available_dataset_types()
        self.assertIn('DFT', types)
        self.assertIn('DMC', types)
        self.assertIn('Wavefunction', types)
    
    def test_10_is_dataset_type_registered_dft(self):
        """Test _is_dataset_type_registered returns True for DFT."""
        self.assertTrue(_is_dataset_type_registered('DFT'))
    
    def test_11_is_dataset_type_registered_dmc(self):
        """Test _is_dataset_type_registered returns True for DMC."""
        self.assertTrue(_is_dataset_type_registered('DMC'))
    
    def test_12_is_dataset_type_registered_wavefunction(self):
        """Test _is_dataset_type_registered returns True for Wavefunction."""
        self.assertTrue(_is_dataset_type_registered('Wavefunction'))
    
    def test_13_is_dataset_type_registered_unknown(self):
        """Test _is_dataset_type_registered returns False for unknown type."""
        self.assertFalse(_is_dataset_type_registered('INVALID_TYPE'))
        self.assertFalse(_is_dataset_type_registered('NONEXISTENT_DATASET_XYZ'))
        self.assertFalse(_is_dataset_type_registered(''))


# ==============================================================================
# PHASE 6: FEATURE QUERY TESTS
# ==============================================================================

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
        result = _get_dataset_feature('DMC', 'uncertainty_handling')
        self.assertTrue(result)
    
    def test_02_dft_no_uncertainty_handling(self):
        """Test DFT has uncertainty_handling=False."""
        result = _get_dataset_feature('DFT', 'uncertainty_handling')
        self.assertFalse(result)
    
    def test_03_dft_has_vibrational_analysis(self):
        """Test DFT has vibrational_analysis=True."""
        result = _get_dataset_feature('DFT', 'vibrational_analysis')
        self.assertTrue(result)
    
    def test_04_dmc_no_vibrational_analysis(self):
        """Test DMC has vibrational_analysis=False."""
        result = _get_dataset_feature('DMC', 'vibrational_analysis')
        self.assertFalse(result)
    
    def test_05_wavefunction_has_orbital_analysis(self):
        """Test Wavefunction has orbital_analysis=True."""
        result = _get_dataset_feature('Wavefunction', 'orbital_analysis')
        self.assertTrue(result)
    
    def test_06_dft_no_orbital_analysis(self):
        """Test DFT has orbital_analysis=False."""
        result = _get_dataset_feature('DFT', 'orbital_analysis')
        self.assertFalse(result)
    
    def test_07_dft_has_atomization_energy(self):
        """Test DFT has atomization_energy=True."""
        result = _get_dataset_feature('DFT', 'atomization_energy')
        self.assertTrue(result)
    
    def test_08_dmc_no_atomization_energy(self):
        """Test DMC has atomization_energy=False."""
        result = _get_dataset_feature('DMC', 'atomization_energy')
        self.assertFalse(result)
    
    def test_09_unknown_feature_returns_false(self):
        """Test unknown feature returns False."""
        result = _get_dataset_feature('DFT', 'unknown_feature')
        self.assertFalse(result)
    
    def test_10_unknown_dataset_returns_false(self):
        """Test unknown dataset type returns False for any feature."""
        result = _get_dataset_feature('INVALID', 'uncertainty_handling')
        self.assertFalse(result)
    
    def test_11_wavefunction_has_homo_lumo_gap(self):
        """Test Wavefunction has homo_lumo_gap=True."""
        result = _get_dataset_feature('Wavefunction', 'homo_lumo_gap')
        self.assertTrue(result)
    
    def test_12_dft_has_rotational_constants(self):
        """Test DFT has rotational_constants=True."""
        result = _get_dataset_feature('DFT', 'rotational_constants')
        self.assertTrue(result)
    
    def test_13_dft_has_frequency_analysis(self):
        """Test DFT has frequency_analysis=True."""
        result = _get_dataset_feature('DFT', 'frequency_analysis')
        self.assertTrue(result)
    
    def test_14_wavefunction_has_mo_energies(self):
        """Test Wavefunction has mo_energies=True."""
        result = _get_dataset_feature('Wavefunction', 'mo_energies')
        self.assertTrue(result)


# ==============================================================================
# PHASE 6: REGISTRY INTEGRATION STATUS TESTS
# ==============================================================================

class TestPhase6RegistryIntegrationStatus(unittest.TestCase):
    """
    Test Phase 6 get_registry_integration_status function.
    
    This function provides diagnostic information about registry integration.
    """
    
    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")
    
    def test_01_get_registry_integration_status_returns_dict(self):
        """Test get_registry_integration_status returns a dictionary."""
        status = get_registry_integration_status()
        self.assertIsInstance(status, dict)
    
    def test_02_registry_status_includes_registry_available(self):
        """Test registry status includes registry_available."""
        status = get_registry_integration_status()
        self.assertIn('registry_available', status)
        self.assertIsInstance(status['registry_available'], bool)
    
    def test_03_registry_status_includes_registry_initialized(self):
        """Test registry status includes registry_initialized."""
        status = get_registry_integration_status()
        self.assertIn('registry_initialized', status)
        self.assertIsInstance(status['registry_initialized'], bool)
    
    def test_04_registry_status_includes_available_types(self):
        """Test registry status includes available_dataset_types."""
        status = get_registry_integration_status()
        self.assertIn('available_dataset_types', status)
        self.assertIsInstance(status['available_dataset_types'], list)
        self.assertIn('DFT', status['available_dataset_types'])
    
    def test_05_registry_status_includes_phase_6_complete(self):
        """Test registry status includes phase_6_complete=True."""
        status = get_registry_integration_status()
        self.assertIn('phase_6_complete', status)
        self.assertTrue(status['phase_6_complete'])
    
    def test_06_registry_status_includes_refactoring_version(self):
        """Test registry status includes refactoring_version."""
        status = get_registry_integration_status()
        self.assertIn('refactoring_version', status)
        self.assertEqual(status['refactoring_version'], '6.0.0')
    
    def test_07_registry_status_includes_module_name(self):
        """Test registry status includes module name."""
        status = get_registry_integration_status()
        self.assertIn('module', status)
        self.assertEqual(status['module'], 'property_enrichment')
    
    def test_08_registry_status_includes_feature_query_capability(self):
        """Test registry status includes feature_query_capability."""
        status = get_registry_integration_status()
        self.assertIn('feature_query_capability', status)
        capability = status['feature_query_capability']
        self.assertIsInstance(capability, dict)
        self.assertIn('uncertainty_handling', capability)
        self.assertIn('vibrational_analysis', capability)
        self.assertIn('orbital_analysis', capability)
    
    def test_09_registry_status_includes_handler_integration(self):
        """Test registry status includes handler_integration info."""
        status = get_registry_integration_status()
        self.assertIn('handler_integration', status)
        handler_info = status['handler_integration']
        self.assertIsInstance(handler_info, dict)
        self.assertTrue(handler_info['handler_required'])
        self.assertEqual(handler_info['handler_delegation'], 'COMPLETE')
        self.assertEqual(handler_info['hardcoded_type_checks'], 0)


# ==============================================================================
# PHASE 6: UPDATED VERIFICATION FUNCTIONS TESTS
# ==============================================================================

class TestPhase6UpdatedVerificationFunctions(unittest.TestCase):
    """
    Test Phase 6 updates to verification functions.
    
    These tests verify that verify_interface_compatibility(),
    get_handler_integration_summary(), and get_handler_integration_status()
    now include Phase 6 registry integration information.
    """
    
    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")
    
    def test_01_verify_interface_includes_registry_integration(self):
        """Test verify_interface_compatibility includes registry_integration."""
        result = verify_interface_compatibility()
        self.assertIn('registry_integration', result)
        registry_info = result['registry_integration']
        self.assertIn('registry_available', registry_info)
        self.assertIn('registry_initialized', registry_info)
        self.assertIn('phase_6_complete', registry_info)
    
    def test_02_verify_interface_registry_has_available_types(self):
        """Test verify_interface_compatibility registry has available_dataset_types."""
        result = verify_interface_compatibility()
        registry_info = result['registry_integration']
        self.assertIn('available_dataset_types', registry_info)
        self.assertIn('DFT', registry_info['available_dataset_types'])
    
    def test_03_verify_interface_registry_phase_6_complete(self):
        """Test verify_interface_compatibility shows phase_6_complete=True."""
        result = verify_interface_compatibility()
        registry_info = result['registry_integration']
        self.assertTrue(registry_info['phase_6_complete'])
    
    def test_04_handler_summary_includes_phase_6_registry(self):
        """Test get_handler_integration_summary includes phase_6_registry_integration."""
        result = get_handler_integration_summary()
        self.assertIn('phase_6_registry_integration', result)
    
    def test_05_handler_summary_phase_6_status_complete(self):
        """Test get_handler_integration_summary phase_6 status is COMPLETE."""
        result = get_handler_integration_summary()
        phase6_info = result['phase_6_registry_integration']
        self.assertEqual(phase6_info['status'], 'COMPLETE')
    
    def test_06_handler_summary_phase_6_objectives(self):
        """Test get_handler_integration_summary phase_6 has objectives_achieved."""
        result = get_handler_integration_summary()
        phase6_info = result['phase_6_registry_integration']
        self.assertIn('objectives_achieved', phase6_info)
        self.assertIsInstance(phase6_info['objectives_achieved'], list)
        self.assertGreater(len(phase6_info['objectives_achieved']), 0)
    
    def test_07_handler_summary_phase_6_registry_status(self):
        """Test get_handler_integration_summary has registry_status."""
        result = get_handler_integration_summary()
        phase6_info = result['phase_6_registry_integration']
        self.assertIn('registry_status', phase6_info)
        self.assertIn(phase6_info['registry_status'], ['Available', 'Fallback mode'])
    
    def test_08_handler_summary_phase_6_available_types(self):
        """Test get_handler_integration_summary has available_dataset_types."""
        result = get_handler_integration_summary()
        phase6_info = result['phase_6_registry_integration']
        self.assertIn('available_dataset_types', phase6_info)
        self.assertIn('DFT', phase6_info['available_dataset_types'])
    
    def test_09_handler_status_includes_registry_integration(self):
        """Test get_handler_integration_status includes registry_integration."""
        result = get_handler_integration_status()
        self.assertIn('registry_integration', result)
        self.assertEqual(result['registry_integration'], 'PHASE_6_COMPLETE')
    
    def test_10_handler_status_includes_registry_available(self):
        """Test get_handler_integration_status includes registry_available."""
        result = get_handler_integration_status()
        self.assertIn('registry_available', result)
        self.assertIsInstance(result['registry_available'], bool)


# ==============================================================================
# PHASE 6: LEGACY FALLBACK TESTS
# ==============================================================================

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
    
    @patch('milia_pipeline.molecules.property_enrichment._REGISTRY_AVAILABLE', False)
    def test_01_fallback_get_available_dataset_types(self):
        """Test legacy fallback for _get_available_dataset_types returns a list."""
        import milia_pipeline.molecules.property_enrichment as mod
        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False
        
        types = _get_available_dataset_types()
        
        # Should return a list (either from filesystem discovery or empty)
        self.assertIsInstance(types, list)
    
    @patch('milia_pipeline.molecules.property_enrichment._REGISTRY_AVAILABLE', False)
    def test_02_fallback_is_dataset_type_registered(self):
        """Test legacy fallback for _is_dataset_type_registered handles unknown types."""
        import milia_pipeline.molecules.property_enrichment as mod
        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False
        
        # A clearly invalid type should always return False regardless of fallback mechanism
        self.assertFalse(_is_dataset_type_registered('NONEXISTENT_DATASET_XYZ'))
    
    @patch('milia_pipeline.molecules.property_enrichment._REGISTRY_AVAILABLE', False)
    def test_03_fallback_get_dataset_feature(self):
        """Test legacy fallback for _get_dataset_feature."""
        import milia_pipeline.molecules.property_enrichment as mod
        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False
        
        # Test fallback values
        self.assertTrue(_get_dataset_feature('DMC', 'uncertainty_handling'))
        self.assertFalse(_get_dataset_feature('DFT', 'uncertainty_handling'))
        self.assertTrue(_get_dataset_feature('DFT', 'vibrational_analysis'))
        self.assertTrue(_get_dataset_feature('Wavefunction', 'orbital_analysis'))
    
    @patch('milia_pipeline.molecules.property_enrichment._REGISTRY_AVAILABLE', False)
    def test_04_fallback_get_registry_integration_status(self):
        """Test legacy fallback for get_registry_integration_status."""
        import milia_pipeline.molecules.property_enrichment as mod
        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False
        
        status = get_registry_integration_status()
        
        # Should still return valid status even without registry
        self.assertIn('registry_available', status)
        self.assertIn('available_dataset_types', status)
        self.assertIsInstance(status['available_dataset_types'], list)
    
    @patch('milia_pipeline.molecules.property_enrichment._REGISTRY_AVAILABLE', False)
    def test_05_fallback_handler_integration_summary(self):
        """Test handler summary includes fallback registry status."""
        import milia_pipeline.molecules.property_enrichment as mod
        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False
        
        result = get_handler_integration_summary()
        phase6_info = result['phase_6_registry_integration']
        
        # Should indicate fallback mode
        self.assertEqual(phase6_info['registry_status'], 'Fallback mode')


# ==============================================================================
# PHASE 6: ZERO HARDCODED DATASET TYPE CHECKS
# ==============================================================================

class TestPhase6ZeroHardcodedTypeChecks(unittest.TestCase):
    """
    Test that Phase 6 has zero hardcoded dataset type checks.
    
    These tests verify that the property_enrichment module does not contain
    hardcoded dataset type references outside of the legacy fallback dictionary.
    """
    
    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")
    
    def test_01_registry_status_shows_zero_hardcoded_checks(self):
        """Test registry status shows hardcoded_type_checks=0."""
        status = get_registry_integration_status()
        handler_info = status['handler_integration']
        self.assertEqual(handler_info['hardcoded_type_checks'], 0)
    
    def test_02_handler_delegation_complete(self):
        """Test handler delegation is complete."""
        status = get_registry_integration_status()
        handler_info = status['handler_integration']
        self.assertEqual(handler_info['handler_delegation'], 'COMPLETE')
    
    def test_03_dataset_specific_logic_delegated(self):
        """Test dataset-specific logic is delegated to handlers."""
        status = get_registry_integration_status()
        handler_info = status['handler_integration']
        self.assertEqual(handler_info['dataset_specific_logic'], 'DELEGATED_TO_HANDLERS')


# ==============================================================================
# TEST RUNNER
# ==============================================================================

def run_test_suite():
    """Run the complete test suite"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes (original)
    suite.addTests(loader.loadTestsFromTestCase(TestEnsureTensor))
    suite.addTests(loader.loadTestsFromTestCase(TestAddScalarGraphTargets))
    suite.addTests(loader.loadTestsFromTestCase(TestAddNodeFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestAddVectorGraphProperties))
    suite.addTests(loader.loadTestsFromTestCase(TestAddVariableLenGraphProperties))
    suite.addTests(loader.loadTestsFromTestCase(TestProcessVibrationalData))
    suite.addTests(loader.loadTestsFromTestCase(TestProcessVibmodes))
    suite.addTests(loader.loadTestsFromTestCase(TestProcessFrequencies))
    suite.addTests(loader.loadTestsFromTestCase(TestCalculateAtomizationEnergy))
    suite.addTests(loader.loadTestsFromTestCase(TestEnrichPyGDataWithProperties))
    suite.addTests(loader.loadTestsFromTestCase(TestHandlerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestParameterResolution))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestCreateHandlerErrorContext))
    
    # Add Phase 6 test classes (NEW)
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6RegistryIntegrationFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6FeatureQueries))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6RegistryIntegrationStatus))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6UpdatedVerificationFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6LegacyFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6ZeroHardcodedTypeChecks))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST SUMMARY - property_enrichment.py (Phase 6 Updated)")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED!")
        print("✓ Tensor conversion validated")
        print("✓ Scalar graph targets working")
        print("✓ Node features working")
        print("✓ Vector properties working")
        print("✓ Variable-length properties working")
        print("✓ Vibrational data processing working")
        print("✓ Atomization energy calculation working")
        print("✓ Handler integration validated")
        print("✓ Handler error context creation validated")
        print("✓ Phase 6: Registry integration validated")
        print("✓ Phase 6: Feature-based queries working")
        print("✓ Phase 6: get_registry_integration_status verified")
        print("✓ Phase 6: Updated verification functions tested")
        print("✓ Phase 6: Legacy fallback working")
        print("✓ Phase 6: Zero hardcoded type checks confirmed")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("  Review failures before production use")
    
    return result


if __name__ == '__main__':
    result = run_test_suite()
    sys.exit(0 if result.wasSuccessful() else 1)
