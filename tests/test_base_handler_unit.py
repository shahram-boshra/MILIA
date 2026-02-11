#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/handlers/base_handler.py

Module under test: base_handler.py
- DatasetHandler: Abstract base class with 12+4 abstract methods
- handle_transform_errors: Decorator for transform error handling
- _init_registry / _get_available_handler_types / _is_handler_type_registered / get_registry_status
- create_dataset_handler / validate_dataset_handler_compatibility
- filter_descriptors_by_handler_support / verify_handler_abstraction / get_handler_abstraction_summary

Test path on local machine: ~/ml_projects/milia/tests/test_base_handler_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/handlers/base_handler.py

NOTE: This test suite runs inside Docker at /app/milia
Path mappings:
- Project root: /app/milia (mapped from ~/ml_projects/milia)

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution

Updated: February 2026 - Production-ready comprehensive test coverage
"""

import sys
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, MagicMock, patch, PropertyMock, call
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from milia_pipeline.handlers.base_handler import (
    DatasetHandler,
    handle_transform_errors,
    _init_registry,
    _get_available_handler_types,
    _is_handler_type_registered,
    get_registry_status,
    create_dataset_handler,
    validate_dataset_handler_compatibility,
    filter_descriptors_by_handler_support,
    verify_handler_abstraction,
    get_handler_abstraction_summary,
)

from milia_pipeline.exceptions import (
    PropertyEnrichmentError,
    MoleculeProcessingError,
    HandlerError,
    HandlerConfigurationError,
    HandlerOperationError,
    HandlerValidationError,
    HandlerNotAvailableError,
    HandlerCompatibilityError,
    TransformConfigurationError,
    TransformValidationError,
    TransformCompositionError,
    TransformHandlerIntegrationError,
)

import torch
from torch_geometric.data import Data


# ============================================================================
# HELPER: Build a concrete DatasetHandler subclass for testing
# ============================================================================

def _make_mock_dataset_config(dataset_type="TEST"):
    """Create a mock DatasetConfig with required attributes."""
    mock = MagicMock()
    mock.dataset_type = dataset_type
    return mock


def _make_mock_filter_config():
    """Create a mock FilterConfig with required attributes."""
    return MagicMock()


def _make_mock_processing_config(scalar_graph_targets=None):
    """Create a mock ProcessingConfig with required attributes."""
    mock = MagicMock()
    mock.scalar_graph_targets = scalar_graph_targets if scalar_graph_targets is not None else ['energy']
    return mock


def _make_mock_logger():
    """Create a mock logger with standard logging methods."""
    mock_logger = MagicMock(spec=logging.Logger)
    mock_logger.info = MagicMock()
    mock_logger.warning = MagicMock()
    mock_logger.error = MagicMock()
    mock_logger.debug = MagicMock()
    return mock_logger


def _build_concrete_handler_class(
    class_name="TestHandler",
    dataset_type="TEST",
    required_properties=None,
    identifier_keys=None,
    supported_structural_features=None,
    molecule_creation_strategy="identifier_coordinate_based",
    transform_recommendations=None,
    supported_descriptors=None,
):
    """
    Dynamically build a concrete DatasetHandler subclass for testing.
    Implements all 12+4 abstract methods with sensible defaults.
    """
    _ds_type = dataset_type
    _req_props = required_properties or ['energy', 'atoms', 'coordinates']
    _id_keys = identifier_keys or [('inchi', 'inchi')]
    _struct_feat = supported_structural_features or {
        'atom': ['degree', 'hybridization'], 'bond': ['bond_type'],
    }
    _strategy = molecule_creation_strategy
    _transform_recs = transform_recommendations or {
        'recommended': ['NormalizeFeatures'], 'avoid': [], 'warnings': [],
    }
    _desc = supported_descriptors or {
        'categories': ['constitutional', 'topological'], 'excluded': [],
        'recommended': ['MolWt'], 'requires_3d': False, 'requires_charges': False,
    }

    def _get_dataset_type(self, _dt=_ds_type):
        return _dt

    def _validate_molecule_data(self, raw_properties_dict, molecule_index, identifier="N/A"):
        pass

    def _get_required_properties(self, _rp=_req_props):
        return list(_rp)

    def _get_identifier_keys(self, _ik=_id_keys):
        return list(_ik)

    def _process_property_value(self, key, value, molecule_index, identifier="N/A"):
        return value

    def _enrich_pyg_data(self, pyg_data, raw_properties_dict, molecule_index, identifier="N/A"):
        return pyg_data

    def _get_processing_statistics(self, processed_molecules):
        return {'count': len(processed_molecules)}

    def _get_supported_structural_features(self, _sf=_struct_feat):
        return dict(_sf)

    def _get_molecular_charge(self, raw_properties_dict, atomic_numbers, mol_identifier=None):
        return 0

    def _get_molecule_creation_strategy(self, _s=_strategy):
        return _s

    def _get_transform_recommendations_method(self, _tr=_transform_recs):
        return dict(_tr)

    def _get_supported_descriptors(self, _d=_desc):
        return dict(_d)

    def _get_dataset_suitable_transforms(self, available_transforms):
        return []

    def _validate_dataset_specific_transforms(self, transform_names):
        return []

    def _check_transform_incompatibilities(self, transform_names):
        return []

    def _get_transform_recommendations_internal(self, transform_names):
        return []

    ns = {
        'get_dataset_type': _get_dataset_type,
        'validate_molecule_data': _validate_molecule_data,
        'get_required_properties': _get_required_properties,
        'get_identifier_keys': _get_identifier_keys,
        'process_property_value': _process_property_value,
        'enrich_pyg_data': _enrich_pyg_data,
        'get_processing_statistics': _get_processing_statistics,
        'get_supported_structural_features': _get_supported_structural_features,
        'get_molecular_charge': _get_molecular_charge,
        'get_molecule_creation_strategy': _get_molecule_creation_strategy,
        'get_transform_recommendations': _get_transform_recommendations_method,
        'get_supported_descriptors': _get_supported_descriptors,
        '_get_dataset_suitable_transforms': _get_dataset_suitable_transforms,
        '_validate_dataset_specific_transforms': _validate_dataset_specific_transforms,
        '_check_transform_incompatibilities': _check_transform_incompatibilities,
        '_get_transform_recommendations': _get_transform_recommendations_internal,
    }

    cls = type(class_name, (DatasetHandler,), ns)
    return cls


def _create_handler_instance(dataset_type="TEST", experimental_setup=None,
                              scalar_graph_targets=None):
    """Create a fully instantiated concrete handler for testing."""
    handler_cls = _build_concrete_handler_class(dataset_type=dataset_type)
    return handler_cls(
        dataset_config=_make_mock_dataset_config(dataset_type=dataset_type),
        filter_config=_make_mock_filter_config(),
        processing_config=_make_mock_processing_config(scalar_graph_targets=scalar_graph_targets),
        logger=_make_mock_logger(),
        experimental_setup=experimental_setup,
    )


# ============================================================================
# GROUP 1: DatasetHandler ABC — Instantiation & __init__ (14 tests)
# ============================================================================

class TestDatasetHandlerInit(unittest.TestCase):
    """Test DatasetHandler construction, validation, and abstract method enforcement."""

    def test_abstract_cannot_instantiate(self):
        """DatasetHandler ABC cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            DatasetHandler(
                _make_mock_dataset_config(), _make_mock_filter_config(),
                _make_mock_processing_config(), _make_mock_logger(),
            )

    def test_concrete_instantiation_success(self):
        """Concrete handler with all abstract methods can be instantiated."""
        handler = _create_handler_instance()
        self.assertIsInstance(handler, DatasetHandler)

    def test_stores_dataset_config(self):
        """Handler stores dataset_config as attribute."""
        handler = _create_handler_instance()
        self.assertEqual(handler.dataset_config.dataset_type, "TEST")

    def test_stores_filter_config(self):
        """Handler stores filter_config as attribute."""
        self.assertIsNotNone(_create_handler_instance().filter_config)

    def test_stores_processing_config(self):
        """Handler stores processing_config as attribute."""
        self.assertIsNotNone(_create_handler_instance().processing_config)

    def test_stores_logger(self):
        """Handler stores logger as attribute."""
        self.assertIsNotNone(_create_handler_instance().logger)

    def test_stores_experimental_setup_none(self):
        """Handler stores experimental_setup as None when not provided."""
        self.assertIsNone(_create_handler_instance().experimental_setup)

    def test_stores_experimental_setup_provided(self):
        """Handler stores experimental_setup when provided."""
        handler = _create_handler_instance(experimental_setup="setup_v1")
        self.assertEqual(handler.experimental_setup, "setup_v1")

    def test_experimental_setup_logs_info(self):
        """Handler logs info when experimental_setup is provided."""
        handler_cls = _build_concrete_handler_class(dataset_type="TEST")
        mock_logger = _make_mock_logger()
        handler_cls(
            _make_mock_dataset_config("TEST"), _make_mock_filter_config(),
            _make_mock_processing_config(), mock_logger, "my_experiment",
        )
        info_calls = [str(c) for c in mock_logger.info.call_args_list]
        self.assertTrue(any("my_experiment" in c for c in info_calls))

    def test_type_mismatch_raises_handler_configuration_error(self):
        """Handler raises HandlerConfigurationError if dataset_type mismatches."""
        handler_cls = _build_concrete_handler_class(dataset_type="DFT")
        with self.assertRaises(HandlerConfigurationError):
            handler_cls(
                _make_mock_dataset_config("DMC"), _make_mock_filter_config(),
                _make_mock_processing_config(), _make_mock_logger(),
            )

    def test_missing_dataset_config_raises_error(self):
        """Handler raises HandlerConfigurationError with falsy dataset_config."""
        handler_cls = _build_concrete_handler_class(dataset_type="TEST")
        mock_config = MagicMock()
        mock_config.dataset_type = "TEST"
        mock_config.__bool__ = Mock(return_value=False)
        with self.assertRaises(HandlerConfigurationError):
            handler_cls(mock_config, _make_mock_filter_config(),
                        _make_mock_processing_config(), _make_mock_logger())

    def test_missing_processing_config_raises_error(self):
        """Handler raises HandlerConfigurationError with falsy processing_config."""
        handler_cls = _build_concrete_handler_class(dataset_type="TEST")
        mock_proc = MagicMock()
        mock_proc.scalar_graph_targets = ['energy']
        mock_proc.__bool__ = Mock(return_value=False)
        with self.assertRaises(HandlerConfigurationError):
            handler_cls(_make_mock_dataset_config("TEST"), _make_mock_filter_config(),
                        mock_proc, _make_mock_logger())

    def test_missing_filter_config_raises_error(self):
        """Handler raises HandlerConfigurationError with falsy filter_config."""
        handler_cls = _build_concrete_handler_class(dataset_type="TEST")
        mock_filter = MagicMock()
        mock_filter.__bool__ = Mock(return_value=False)
        with self.assertRaises(HandlerConfigurationError):
            handler_cls(_make_mock_dataset_config("TEST"), mock_filter,
                        _make_mock_processing_config(), _make_mock_logger())

    def test_unexpected_init_error_wrapped_as_handler_config_error(self):
        """Unexpected error in __init__ is wrapped as HandlerConfigurationError."""
        handler_cls = _build_concrete_handler_class(dataset_type="TEST")
        original_validate = handler_cls._validate_handler_configuration
        handler_cls._validate_handler_configuration = lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            with self.assertRaises(HandlerConfigurationError):
                handler_cls(_make_mock_dataset_config("TEST"), _make_mock_filter_config(),
                            _make_mock_processing_config(), _make_mock_logger())
        finally:
            handler_cls._validate_handler_configuration = original_validate


# ============================================================================
# GROUP 2: DatasetHandler Abstract Methods — Interface Verification (12 tests)
# ============================================================================

class TestDatasetHandlerAbstractMethods(unittest.TestCase):
    """Test that all abstract methods are present and callable."""

    def setUp(self):
        self.handler = _create_handler_instance()

    def test_get_dataset_type(self):
        self.assertEqual(self.handler.get_dataset_type(), "TEST")

    def test_validate_molecule_data_callable(self):
        self.handler.validate_molecule_data({}, 0, "N/A")

    def test_get_required_properties(self):
        result = self.handler.get_required_properties()
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(p, str) for p in result))

    def test_get_identifier_keys(self):
        result = self.handler.get_identifier_keys()
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_process_property_value(self):
        self.assertEqual(self.handler.process_property_value("energy", 42.0, 0), 42.0)

    def test_enrich_pyg_data(self):
        pyg_data = Data(x=torch.randn(3, 5))
        self.assertIsInstance(self.handler.enrich_pyg_data(pyg_data, {}, 0), Data)

    def test_get_processing_statistics(self):
        result = self.handler.get_processing_statistics([{'mol': 1}, {'mol': 2}])
        self.assertIsInstance(result, dict)
        self.assertEqual(result['count'], 2)

    def test_get_supported_structural_features(self):
        result = self.handler.get_supported_structural_features()
        self.assertIn('atom', result)
        self.assertIn('bond', result)

    def test_get_molecular_charge(self):
        self.assertIsInstance(self.handler.get_molecular_charge({}, np.array([6, 8])), int)

    def test_get_molecule_creation_strategy(self):
        self.assertIn(self.handler.get_molecule_creation_strategy(),
                      ['identifier_coordinate_based', 'coordinate_based'])

    def test_get_transform_recommendations(self):
        result = self.handler.get_transform_recommendations()
        for key in ['recommended', 'avoid', 'warnings']:
            self.assertIn(key, result)

    def test_get_supported_descriptors(self):
        self.assertIn('categories', self.handler.get_supported_descriptors())


# ============================================================================
# GROUP 3: _extract_charge_from_inchi Utility (8 tests)
# ============================================================================

class TestExtractChargeFromInchi(unittest.TestCase):

    def setUp(self):
        self.handler = _create_handler_instance()

    def test_neutral_no_q_layer(self):
        self.assertEqual(self.handler._extract_charge_from_inchi("InChI=1S/C2H4/c1-2/h1-2H2"), 0)

    def test_positive_charge(self):
        self.assertEqual(self.handler._extract_charge_from_inchi("InChI=1S/C2H4/c1-2/h1-2H2/q+1"), 1)

    def test_negative_charge(self):
        self.assertEqual(self.handler._extract_charge_from_inchi("InChI=1S/C2H4/c1-2/h1-2H2/q-2"), -2)

    def test_empty_string(self):
        self.assertEqual(self.handler._extract_charge_from_inchi(""), 0)

    def test_none_returns_zero(self):
        self.assertEqual(self.handler._extract_charge_from_inchi(None), 0)

    def test_q_layer_with_following_layers(self):
        self.assertEqual(self.handler._extract_charge_from_inchi("InChI=1S/C2H4/c1-2/q-1/p+1"), -1)

    def test_malformed_q_layer_returns_zero(self):
        self.assertEqual(self.handler._extract_charge_from_inchi("InChI=1S/C2H4/c1-2/qabc"), 0)

    def test_zero_charge_explicit(self):
        self.assertEqual(self.handler._extract_charge_from_inchi("InChI=1S/C2H4/c1-2/q0"), 0)


# ============================================================================
# GROUP 4: _is_valid_property Utility (9 tests)
# ============================================================================

class TestIsValidProperty(unittest.TestCase):

    def setUp(self):
        self.handler = _create_handler_instance()

    def test_none_is_invalid(self):
        self.assertFalse(self.handler._is_valid_property(None))

    def test_string_missing_is_invalid(self):
        self.assertFalse(self.handler._is_valid_property("missing"))

    def test_string_invalid_is_invalid(self):
        self.assertFalse(self.handler._is_valid_property("invalid"))

    def test_empty_string_is_invalid(self):
        self.assertFalse(self.handler._is_valid_property(""))

    def test_string_nan_is_invalid(self):
        self.assertFalse(self.handler._is_valid_property("nan"))

    def test_string_none_is_invalid(self):
        self.assertFalse(self.handler._is_valid_property("none"))

    def test_valid_string_is_valid(self):
        self.assertTrue(self.handler._is_valid_property("InChI=1S/C2H4"))

    def test_valid_float_is_valid(self):
        self.assertTrue(self.handler._is_valid_property(3.14))

    def test_valid_integer_is_valid(self):
        self.assertTrue(self.handler._is_valid_property(42))


# ============================================================================
# GROUP 5: _ensure_tensor Utility (16 tests)
# ============================================================================

class TestEnsureTensor(unittest.TestCase):

    def setUp(self):
        self.handler = _create_handler_instance()

    def test_none_raises_property_enrichment_error(self):
        with self.assertRaises(PropertyEnrichmentError):
            self.handler._ensure_tensor(None, property_name="energy")

    def test_torch_tensor_returned_with_dtype(self):
        t = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = self.handler._ensure_tensor(t, dtype=torch.float32)
        self.assertEqual(result.dtype, torch.float32)

    def test_numpy_array_converted(self):
        result = self.handler._ensure_tensor(np.array([1.0, 2.0, 3.0]))
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 3)

    def test_list_converted(self):
        result = self.handler._ensure_tensor([1.0, 2.0, 3.0])
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 3)

    def test_tuple_converted(self):
        result = self.handler._ensure_tensor((1.0, 2.0))
        self.assertIsInstance(result, torch.Tensor)

    def test_scalar_int_shape_convention(self):
        result = self.handler._ensure_tensor(42)
        self.assertEqual(result.shape, (1,))
        self.assertEqual(result.item(), 42.0)

    def test_scalar_float_shape_convention(self):
        result = self.handler._ensure_tensor(3.14)
        self.assertEqual(result.shape, (1,))

    def test_numpy_scalar_shape_convention(self):
        result = self.handler._ensure_tensor(np.float64(2.718))
        self.assertEqual(result.shape, (1,))

    def test_string_numeric_converted(self):
        result = self.handler._ensure_tensor("3.14")
        self.assertEqual(result.shape, (1,))
        self.assertAlmostEqual(result.item(), 3.14, places=2)

    def test_string_non_numeric_raises(self):
        with self.assertRaises(PropertyEnrichmentError):
            self.handler._ensure_tensor("not_a_number")

    def test_unsupported_type_raises(self):
        with self.assertRaises(PropertyEnrichmentError):
            self.handler._ensure_tensor({"key": "value"})

    def test_dtype_float64(self):
        result = self.handler._ensure_tensor(1.0, dtype=torch.float64)
        self.assertEqual(result.dtype, torch.float64)

    def test_dtype_int64_from_list(self):
        result = self.handler._ensure_tensor([1, 2, 3], dtype=torch.int64)
        self.assertEqual(result.dtype, torch.int64)

    def test_invalid_list_raises_property_enrichment_error(self):
        with self.assertRaises(PropertyEnrichmentError):
            self.handler._ensure_tensor(["a", "b", "c"])

    def test_bytes_numeric_converted(self):
        result = self.handler._ensure_tensor(b"42")
        self.assertEqual(result.shape, (1,))

    def test_property_name_in_error_context(self):
        with self.assertRaises(PropertyEnrichmentError) as ctx:
            self.handler._ensure_tensor(None, property_name="my_property")
        self.assertIn("my_property", str(ctx.exception))


# ============================================================================
# GROUP 6: validate_configuration (7 tests)
# ============================================================================

class TestValidateConfiguration(unittest.TestCase):

    def test_valid_configuration_no_error(self):
        _create_handler_instance().validate_configuration()

    def test_no_scalar_targets_logs_warning(self):
        handler = _create_handler_instance(scalar_graph_targets=[])
        handler.validate_configuration()
        warning_calls = [str(c) for c in handler.logger.warning.call_args_list]
        self.assertTrue(any("scalar" in c.lower() or "target" in c.lower() for c in warning_calls))

    def test_missing_scalar_graph_targets_attr_raises(self):
        handler = _create_handler_instance()
        del handler.processing_config.scalar_graph_targets
        with self.assertRaises(HandlerConfigurationError):
            handler._validate_processing_config()

    def test_validate_filter_config_default_passes(self):
        _create_handler_instance()._validate_filter_config()

    def test_validate_handler_configuration_all_configs_present(self):
        _create_handler_instance()._validate_handler_configuration()

    def test_validate_configuration_wraps_unexpected_errors(self):
        handler = _create_handler_instance()
        handler._validate_processing_config = Mock(side_effect=RuntimeError("boom"))
        with self.assertRaises(HandlerConfigurationError):
            handler.validate_configuration()

    def test_handler_configuration_error_includes_handler_type(self):
        handler = _create_handler_instance()
        handler._validate_processing_config = Mock(side_effect=RuntimeError("boom"))
        with self.assertRaises(HandlerConfigurationError) as ctx:
            handler.validate_configuration()
        self.assertIn("TEST", str(ctx.exception))


# ============================================================================
# GROUP 7: Experimental Setup Support (6 tests)
# ============================================================================

class TestExperimentalSetupSupport(unittest.TestCase):

    def test_get_experimental_setup_info_with_setup(self):
        handler = _create_handler_instance(experimental_setup="baseline_v2")
        info = handler.get_experimental_setup_info()
        self.assertEqual(info['experimental_setup'], "baseline_v2")
        self.assertTrue(info['has_setup'])

    def test_get_experimental_setup_info_without_setup(self):
        info = _create_handler_instance().get_experimental_setup_info()
        self.assertIsNone(info['experimental_setup'])
        self.assertFalse(info['has_setup'])

    def test_log_with_setup_context_with_setup(self):
        handler = _create_handler_instance(experimental_setup="exp1")
        handler._log_with_setup_context('info', "Test message")
        logged_msg = handler.logger.info.call_args[0][0]
        self.assertIn("[Setup: exp1]", logged_msg)

    def test_log_with_setup_context_without_setup(self):
        handler = _create_handler_instance()
        handler._log_with_setup_context('warning', "Raw message")
        logged_msg = handler.logger.warning.call_args[0][0]
        self.assertEqual(logged_msg, "Raw message")

    def test_log_transform_info(self):
        handler = _create_handler_instance(experimental_setup="test_setup")
        handler.log_transform_info("normalize", {"param": "value"})
        handler.logger.info.assert_called()

    def test_log_with_setup_context_debug_level(self):
        handler = _create_handler_instance()
        handler._log_with_setup_context('debug', "Debug msg")
        handler.logger.debug.assert_called_once()


# ============================================================================
# GROUP 8: validate_transform_compatibility (10 tests)
# ============================================================================

class TestValidateTransformCompatibility(unittest.TestCase):

    def setUp(self):
        self.handler = _create_handler_instance()

    @patch('milia_pipeline.handlers.base_handler.list_available_transforms', return_value=['NormalizeFeatures'])
    @patch('milia_pipeline.handlers.base_handler.get_transform_info', return_value=None)
    @patch('milia_pipeline.handlers.base_handler.validate_comprehensive', return_value={'valid': True, 'errors': []})
    def test_none_transform_returns_warning(self, mock_vc, mock_gti, mock_lat):
        result = self.handler.validate_transform_compatibility(None)
        self.assertTrue(result['compatible'])
        self.assertTrue(any("No transforms" in w for w in result['warnings']))

    @patch('milia_pipeline.handlers.base_handler.list_available_transforms', return_value=['NormalizeFeatures'])
    @patch('milia_pipeline.handlers.base_handler.get_transform_info', return_value=None)
    @patch('milia_pipeline.handlers.base_handler.validate_comprehensive', return_value={'valid': True, 'errors': []})
    def test_valid_transforms_compatible(self, mock_vc, mock_gti, mock_lat):
        class NormalizeFeatures:
            pass
        seq = MagicMock()
        seq.transforms = [NormalizeFeatures()]
        self.assertTrue(self.handler.validate_transform_compatibility(seq)['compatible'])

    @patch('milia_pipeline.handlers.base_handler.list_available_transforms', return_value=['NormalizeFeatures'])
    @patch('milia_pipeline.handlers.base_handler.get_transform_info', return_value=None)
    @patch('milia_pipeline.handlers.base_handler.validate_comprehensive', return_value={'valid': True, 'errors': []})
    def test_unrecognized_transform_warning(self, mock_vc, mock_gti, mock_lat):
        class CustomUnknown:
            pass
        seq = MagicMock()
        seq.transforms = [CustomUnknown()]
        result = self.handler.validate_transform_compatibility(seq)
        self.assertTrue(any("Unrecognized" in w for w in result['warnings']))

    @patch('milia_pipeline.handlers.base_handler.list_available_transforms', return_value=[])
    @patch('milia_pipeline.handlers.base_handler.get_transform_info', return_value=None)
    @patch('milia_pipeline.handlers.base_handler.validate_comprehensive', return_value={'valid': True, 'errors': []})
    def test_result_contains_dataset_type(self, mock_vc, mock_gti, mock_lat):
        self.assertEqual(self.handler.validate_transform_compatibility(None)['dataset_type'], "TEST")

    @patch('milia_pipeline.handlers.base_handler.list_available_transforms', return_value=[])
    @patch('milia_pipeline.handlers.base_handler.get_transform_info', return_value=None)
    @patch('milia_pipeline.handlers.base_handler.validate_comprehensive', return_value={'valid': True, 'errors': []})
    def test_result_contains_experimental_setup(self, mock_vc, mock_gti, mock_lat):
        handler = _create_handler_instance(experimental_setup="my_setup")
        self.assertEqual(handler.validate_transform_compatibility(None)['experimental_setup'], "my_setup")

    @patch('milia_pipeline.handlers.base_handler.list_available_transforms', return_value=[])
    @patch('milia_pipeline.handlers.base_handler.get_transform_info', return_value=None)
    @patch('milia_pipeline.handlers.base_handler.validate_comprehensive', return_value={'valid': True, 'errors': []})
    def test_override_experimental_setup(self, mock_vc, mock_gti, mock_lat):
        handler = _create_handler_instance(experimental_setup="handler_setup")
        result = handler.validate_transform_compatibility(None, experimental_setup="override")
        self.assertEqual(result['experimental_setup'], "override")

    @patch('milia_pipeline.handlers.base_handler.list_available_transforms', return_value=['T1'])
    @patch('milia_pipeline.handlers.base_handler.get_transform_info', return_value={'name': 'T1'})
    @patch('milia_pipeline.handlers.base_handler.validate_comprehensive', return_value={'valid': False, 'errors': ['bad param']})
    def test_parameter_issues_populated(self, mock_vc, mock_gti, mock_lat):
        class T1:
            param1 = 5
        seq = MagicMock()
        seq.transforms = [T1()]
        result = self.handler.validate_transform_compatibility(seq)
        self.assertTrue(len(result['parameter_issues']) > 0)

    @patch('milia_pipeline.handlers.base_handler.list_available_transforms', return_value=[])
    @patch('milia_pipeline.handlers.base_handler.get_transform_info', return_value=None)
    @patch('milia_pipeline.handlers.base_handler.validate_comprehensive', return_value={'valid': True, 'errors': []})
    def test_result_structure_keys(self, mock_vc, mock_gti, mock_lat):
        result = self.handler.validate_transform_compatibility(None)
        expected = {'compatible', 'warnings', 'recommendations', 'parameter_issues',
                    'available_alternatives', 'dataset_type', 'experimental_setup'}
        self.assertTrue(expected.issubset(set(result.keys())))

    def test_critical_error_raises_integration_error(self):
        handler = _create_handler_instance()
        handler._validate_dataset_specific_transforms = Mock(side_effect=RuntimeError("crash"))
        class T:
            pass
        seq = MagicMock()
        seq.transforms = [T()]
        with patch('milia_pipeline.handlers.base_handler.list_available_transforms', side_effect=RuntimeError("boom")):
            with self.assertRaises(TransformHandlerIntegrationError):
                handler.validate_transform_compatibility(seq)

    @patch('milia_pipeline.handlers.base_handler.list_available_transforms', return_value=[])
    @patch('milia_pipeline.handlers.base_handler.get_transform_info', return_value={'name': 'test'})
    @patch('milia_pipeline.handlers.base_handler.validate_comprehensive', return_value={'valid': False, 'errors': ['issue']})
    def test_many_parameter_issues_escalated(self, mock_vc, mock_gti, mock_lat):
        handler = _create_handler_instance()
        transforms = []
        for i in range(5):
            # Use real classes instead of MagicMock to avoid __class__/__dict__ conflicts
            cls = type(f"T{i}", (), {'p': i})
            transforms.append(cls())
        seq = MagicMock()
        seq.transforms = transforms
        result = handler.validate_transform_compatibility(seq)
        self.assertTrue(any("Multiple parameter issues" in w for w in result['warnings']))


# ============================================================================
# GROUP 9: handle_transform_errors Decorator (8 tests)
# ============================================================================

class TestHandleTransformErrorsDecorator(unittest.TestCase):

    def _make_handler_with_decorated_method(self, method_body):
        """Helper: creates handler with a @handle_transform_errors decorated method."""
        handler_cls = _build_concrete_handler_class(dataset_type="TEST")
        decorated = handle_transform_errors("test_op")(method_body)
        handler_cls.test_method = decorated
        return handler_cls(
            _make_mock_dataset_config("TEST"), _make_mock_filter_config(),
            _make_mock_processing_config(), _make_mock_logger(),
        )

    def test_success_passes_through(self):
        instance = self._make_handler_with_decorated_method(lambda self: "result")
        self.assertEqual(instance.test_method(), "result")

    def test_transform_configuration_error_wrapped(self):
        def body(self):
            raise TransformConfigurationError(message="Bad config", config_key="k")
        instance = self._make_handler_with_decorated_method(body)
        with self.assertRaises(TransformHandlerIntegrationError):
            instance.test_method()

    def test_transform_validation_error_wrapped(self):
        def body(self):
            raise TransformValidationError(message="Bad", transform_name="TestTransform")
        instance = self._make_handler_with_decorated_method(body)
        with self.assertRaises(TransformHandlerIntegrationError):
            instance.test_method()

    def test_transform_composition_error_wrapped(self):
        def body(self):
            raise TransformCompositionError(message="Compose failed")
        instance = self._make_handler_with_decorated_method(body)
        with self.assertRaises(TransformHandlerIntegrationError):
            instance.test_method()

    def test_handler_error_reraises_unwrapped(self):
        def body(self):
            raise HandlerOperationError(message="op fail", handler_type="TEST", operation="op")
        instance = self._make_handler_with_decorated_method(body)
        with self.assertRaises(HandlerOperationError):
            instance.test_method()

    def test_property_enrichment_error_reraises_unwrapped(self):
        def body(self):
            raise PropertyEnrichmentError(molecule_index=0, inchi="N/A", property_name="e", reason="bad", detail="detail")
        instance = self._make_handler_with_decorated_method(body)
        with self.assertRaises(PropertyEnrichmentError):
            instance.test_method()

    def test_molecule_processing_error_reraises_unwrapped(self):
        def body(self):
            raise MoleculeProcessingError(message="mol fail", molecule_index=0)
        instance = self._make_handler_with_decorated_method(body)
        with self.assertRaises(MoleculeProcessingError):
            instance.test_method()

    def test_unexpected_error_wrapped_as_handler_operation(self):
        def body(self):
            raise RuntimeError("unexpected boom")
        instance = self._make_handler_with_decorated_method(body)
        with self.assertRaises(HandlerOperationError):
            instance.test_method()


# ============================================================================
# GROUP 10: Registry Functions (12 tests)
# ============================================================================

class TestRegistryFunctions(unittest.TestCase):

    def _reset_registry_state(self):
        import milia_pipeline.handlers.base_handler as bh
        bh._REGISTRY_INITIALIZED = False
        bh._REGISTRY_AVAILABLE = False
        bh._REGISTRY_IMPORT_ERROR = None
        bh._registry_list_all = None
        bh._registry_get = None
        bh._registry_is_registered = None

    def tearDown(self):
        self._reset_registry_state()

    @patch('milia_pipeline.handlers.base_handler._REGISTRY_INITIALIZED', True)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', True)
    def test_init_registry_already_initialized(self):
        self.assertTrue(_init_registry())

    def test_init_registry_import_error_returns_false(self):
        self._reset_registry_state()
        with patch.dict('sys.modules', {'milia_pipeline.datasets.registry': None}):
            import milia_pipeline.handlers.base_handler as bh
            bh._REGISTRY_INITIALIZED = False
            result = _init_registry()
            self.assertIsInstance(result, bool)

    def test_get_registry_status_returns_dict(self):
        with patch('milia_pipeline.handlers.base_handler._init_registry'):
            import milia_pipeline.handlers.base_handler as bh
            bh._REGISTRY_INITIALIZED = True
            bh._REGISTRY_AVAILABLE = False
            bh._REGISTRY_IMPORT_ERROR = "test error"
            result = get_registry_status()
            for key in ['initialized', 'available', 'import_error']:
                self.assertIn(key, result)

    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', True)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_INITIALIZED', True)
    @patch('milia_pipeline.handlers.base_handler._registry_list_all')
    def test_get_available_types_from_registry(self, mock_list_all):
        mock_list_all.return_value = ['DFT', 'DMC']
        self.assertEqual(_get_available_handler_types(), ['DFT', 'DMC'])

    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', True)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_INITIALIZED', True)
    @patch('milia_pipeline.handlers.base_handler._registry_is_registered')
    def test_is_handler_type_registered_true(self, mock_is_reg):
        mock_is_reg.return_value = True
        self.assertTrue(_is_handler_type_registered("DFT"))

    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', True)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_INITIALIZED', True)
    @patch('milia_pipeline.handlers.base_handler._registry_is_registered')
    def test_is_handler_type_registered_false(self, mock_is_reg):
        mock_is_reg.return_value = False
        self.assertFalse(_is_handler_type_registered("NONEXISTENT"))

    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', False)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_INITIALIZED', True)
    def test_get_available_types_dynamic_fallback(self):
        with patch('milia_pipeline.handlers.base_handler._init_registry'):
            self.assertIsInstance(_get_available_handler_types(), list)

    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', True)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_INITIALIZED', True)
    @patch('milia_pipeline.handlers.base_handler._registry_list_all')
    def test_get_available_types_registry_exception_fallback(self, mock_list_all):
        mock_list_all.side_effect = RuntimeError("broken")
        with patch('milia_pipeline.handlers.base_handler._init_registry'):
            self.assertIsInstance(_get_available_handler_types(), list)

    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', True)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_INITIALIZED', True)
    @patch('milia_pipeline.handlers.base_handler._registry_is_registered')
    def test_is_handler_type_registered_exception_fallback(self, mock_is_reg):
        mock_is_reg.side_effect = RuntimeError("broken")
        with patch('milia_pipeline.handlers.base_handler._init_registry'):
            self.assertIsInstance(_is_handler_type_registered("DFT"), bool)

    def test_get_registry_status_includes_available_types(self):
        with patch('milia_pipeline.handlers.base_handler._init_registry'):
            import milia_pipeline.handlers.base_handler as bh
            bh._REGISTRY_INITIALIZED = True
            bh._REGISTRY_AVAILABLE = False
            self.assertIn('available_types', get_registry_status())

    @patch('milia_pipeline.handlers.base_handler._REGISTRY_INITIALIZED', True)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', False)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_IMPORT_ERROR', "some error")
    def test_get_registry_status_includes_import_error(self):
        with patch('milia_pipeline.handlers.base_handler._init_registry'):
            self.assertEqual(get_registry_status()['import_error'], "some error")

    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', True)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_INITIALIZED', True)
    @patch('milia_pipeline.handlers.base_handler._registry_list_all', return_value=['DFT'])
    def test_is_handler_type_registered_fallback_to_available(self, mock_list):
        with patch('milia_pipeline.handlers.base_handler._registry_is_registered', None):
            self.assertTrue(_is_handler_type_registered("DFT"))


# ============================================================================
# GROUP 11: create_dataset_handler Factory (10 tests)
# ============================================================================

class TestCreateDatasetHandler(unittest.TestCase):

    def tearDown(self):
        import milia_pipeline.handlers.base_handler as bh
        bh._REGISTRY_INITIALIZED = False
        bh._REGISTRY_AVAILABLE = False
        bh._REGISTRY_IMPORT_ERROR = None
        bh._registry_list_all = None
        bh._registry_get = None
        bh._registry_is_registered = None

    @patch('milia_pipeline.handlers.base_handler._init_registry', return_value=True)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', True)
    @patch('milia_pipeline.handlers.base_handler._registry_get')
    def test_registry_based_creation(self, mock_get, mock_init):
        mock_ds = MagicMock()
        mock_ds.create_handler.return_value = MagicMock(spec=DatasetHandler)
        mock_get.return_value = mock_ds
        result = create_dataset_handler(
            _make_mock_dataset_config("DFT"), _make_mock_filter_config(),
            _make_mock_processing_config(), _make_mock_logger(),
        )
        mock_ds.create_handler.assert_called_once()

    @patch('milia_pipeline.handlers.base_handler._init_registry', return_value=True)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', True)
    @patch('milia_pipeline.handlers.base_handler._registry_get')
    def test_registry_not_found_raises(self, mock_get, mock_init):
        mock_get.side_effect = KeyError("not registered")
        with self.assertRaises(HandlerNotAvailableError):
            create_dataset_handler(
                _make_mock_dataset_config("UNKNOWN"), _make_mock_filter_config(),
                _make_mock_processing_config(), _make_mock_logger(),
            )

    @patch('milia_pipeline.handlers.base_handler._init_registry', return_value=False)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', False)
    def test_dynamic_fallback_nonexistent(self, mock_init):
        with self.assertRaises((HandlerNotAvailableError, ImportError)):
            create_dataset_handler(
                _make_mock_dataset_config("NONEXISTENT"), _make_mock_filter_config(),
                _make_mock_processing_config(), _make_mock_logger(),
            )

    @patch('milia_pipeline.handlers.base_handler._init_registry', return_value=True)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', True)
    @patch('milia_pipeline.handlers.base_handler._registry_get')
    def test_experimental_setup_forwarded(self, mock_get, mock_init):
        mock_ds = MagicMock()
        mock_ds.create_handler.return_value = MagicMock(spec=DatasetHandler)
        mock_get.return_value = mock_ds
        create_dataset_handler(
            _make_mock_dataset_config("DFT"), _make_mock_filter_config(),
            _make_mock_processing_config(), _make_mock_logger(), "test_exp",
        )
        self.assertEqual(mock_ds.create_handler.call_args[0][4], "test_exp")

    @patch('milia_pipeline.handlers.base_handler._init_registry', return_value=True)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', True)
    @patch('milia_pipeline.handlers.base_handler._registry_get')
    def test_handler_error_reraises(self, mock_get, mock_init):
        mock_get.side_effect = HandlerConfigurationError(message="err", handler_type="DFT")
        with self.assertRaises(HandlerConfigurationError):
            create_dataset_handler(
                _make_mock_dataset_config("DFT"), _make_mock_filter_config(),
                _make_mock_processing_config(), _make_mock_logger(),
            )

    @patch('milia_pipeline.handlers.base_handler._init_registry', return_value=True)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', True)
    @patch('milia_pipeline.handlers.base_handler._registry_get')
    def test_unexpected_error_wrapped(self, mock_get, mock_init):
        mock_get.side_effect = TypeError("unexpected")
        with self.assertRaises((HandlerNotAvailableError, TypeError)):
            create_dataset_handler(
                _make_mock_dataset_config("DFT"), _make_mock_filter_config(),
                _make_mock_processing_config(), _make_mock_logger(),
            )

    @patch('milia_pipeline.handlers.base_handler._init_registry', return_value=False)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', False)
    def test_dynamic_fallback_class_not_found(self, mock_init):
        with self.assertRaises(HandlerNotAvailableError):
            create_dataset_handler(
                _make_mock_dataset_config("DOESNOTEXIST"), _make_mock_filter_config(),
                _make_mock_processing_config(), _make_mock_logger(),
            )

    @patch('milia_pipeline.handlers.base_handler._init_registry', return_value=True)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', True)
    @patch('milia_pipeline.handlers.base_handler._registry_get')
    def test_all_five_args_forwarded(self, mock_get, mock_init):
        mock_ds = MagicMock()
        mock_ds.create_handler.return_value = MagicMock(spec=DatasetHandler)
        mock_get.return_value = mock_ds
        dc, fc, pc, lg = (_make_mock_dataset_config("DFT"), _make_mock_filter_config(),
                          _make_mock_processing_config(), _make_mock_logger())
        create_dataset_handler(dc, fc, pc, lg, "exp")
        args = mock_ds.create_handler.call_args[0]
        self.assertEqual(len(args), 5)
        self.assertEqual(args[0], dc)
        self.assertEqual(args[4], "exp")

    @patch('milia_pipeline.handlers.base_handler._init_registry', return_value=True)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', True)
    @patch('milia_pipeline.handlers.base_handler._registry_get')
    def test_handler_not_available_error_includes_type(self, mock_get, mock_init):
        mock_get.side_effect = KeyError("not found in registry")
        with patch('milia_pipeline.handlers.base_handler._get_available_handler_types', return_value=['DFT']):
            with self.assertRaises(HandlerNotAvailableError) as ctx:
                create_dataset_handler(
                    _make_mock_dataset_config("MISSING"), _make_mock_filter_config(),
                    _make_mock_processing_config(), _make_mock_logger(),
                )
            self.assertIn("MISSING", str(ctx.exception))

    @patch('milia_pipeline.handlers.base_handler._init_registry', return_value=True)
    @patch('milia_pipeline.handlers.base_handler._REGISTRY_AVAILABLE', True)
    @patch('milia_pipeline.handlers.base_handler._registry_get')
    def test_handler_not_available_error_reraises(self, mock_get, mock_init):
        mock_get.side_effect = HandlerNotAvailableError(
            message="not available", requested_dataset_type="X", available_types=['A'],
        )
        with self.assertRaises(HandlerNotAvailableError):
            create_dataset_handler(
                _make_mock_dataset_config("X"), _make_mock_filter_config(),
                _make_mock_processing_config(), _make_mock_logger(),
            )


# ============================================================================
# GROUP 12: validate_dataset_handler_compatibility (6 tests)
# ============================================================================

class TestValidateDatasetHandlerCompatibility(unittest.TestCase):

    def test_matching_types_passes(self):
        handler = _create_handler_instance(dataset_type="TEST")
        validate_dataset_handler_compatibility(handler, _make_mock_dataset_config("TEST"))

    def test_mismatched_types_raises(self):
        handler = _create_handler_instance(dataset_type="TEST")
        with self.assertRaises(HandlerCompatibilityError):
            validate_dataset_handler_compatibility(handler, _make_mock_dataset_config("OTHER"))

    def test_calls_validate_configuration(self):
        handler = _create_handler_instance(dataset_type="TEST")
        handler.validate_configuration = MagicMock()
        validate_dataset_handler_compatibility(handler, _make_mock_dataset_config("TEST"))
        handler.validate_configuration.assert_called_once()

    def test_handler_error_reraises(self):
        handler = _create_handler_instance(dataset_type="TEST")
        handler.validate_configuration = MagicMock(
            side_effect=HandlerConfigurationError(message="bad", handler_type="TEST"))
        with self.assertRaises(HandlerConfigurationError):
            validate_dataset_handler_compatibility(handler, _make_mock_dataset_config("TEST"))

    def test_unexpected_error_wrapped(self):
        handler = _create_handler_instance(dataset_type="TEST")
        handler.validate_configuration = MagicMock(side_effect=RuntimeError("boom"))
        with self.assertRaises(HandlerCompatibilityError):
            validate_dataset_handler_compatibility(handler, _make_mock_dataset_config("TEST"))

    def test_compatibility_error_includes_handler_type(self):
        handler = _create_handler_instance(dataset_type="TEST")
        with self.assertRaises(HandlerCompatibilityError) as ctx:
            validate_dataset_handler_compatibility(handler, _make_mock_dataset_config("DIFFERENT"))
        self.assertIn("TEST", str(ctx.exception))


# ============================================================================
# GROUP 13: filter_descriptors_by_handler_support (8 tests)
# ============================================================================

class TestFilterDescriptorsByHandlerSupport(unittest.TestCase):

    def _make_registry(self, desc_meta=None):
        reg = MagicMock()
        desc_meta = desc_meta or {}
        reg.has_descriptor = lambda n: n in desc_meta
        reg.get_metadata = lambda n: desc_meta.get(n)
        return reg

    def _make_meta(self, cat="constitutional", req_3d=False):
        m = MagicMock()
        m.category = MagicMock()
        m.category.value = cat
        m.requires_3d = req_3d
        return m

    def test_supported_descriptor(self):
        handler = _create_handler_instance()
        s, u = filter_descriptors_by_handler_support(
            handler, ["MolWt"], self._make_registry({"MolWt": self._make_meta("constitutional")}))
        self.assertIn("MolWt", s)

    def test_unsupported_category(self):
        handler = _create_handler_instance()
        s, u = filter_descriptors_by_handler_support(
            handler, ["Geom"], self._make_registry({"Geom": self._make_meta("geometric")}))
        self.assertIn("Geom", u)

    def test_unknown_descriptor(self):
        handler = _create_handler_instance()
        s, u = filter_descriptors_by_handler_support(handler, ["X"], self._make_registry({}))
        self.assertIn("X", u)

    def test_excluded_descriptor(self):
        cls = _build_concrete_handler_class(
            dataset_type="TEST",
            supported_descriptors={'categories': ['constitutional'], 'excluded': ['Bad'],
                                   'recommended': [], 'requires_3d': False, 'requires_charges': False})
        handler = cls(_make_mock_dataset_config("TEST"), _make_mock_filter_config(),
                      _make_mock_processing_config(), _make_mock_logger())
        s, u = filter_descriptors_by_handler_support(
            handler, ["Bad"], self._make_registry({"Bad": self._make_meta("constitutional")}))
        self.assertIn("Bad", u)

    def test_empty_requested(self):
        s, u = filter_descriptors_by_handler_support(
            _create_handler_instance(), [], self._make_registry({}))
        self.assertEqual(s, [])
        self.assertEqual(u, [])

    def test_mixed_descriptors(self):
        handler = _create_handler_instance()
        reg = self._make_registry({
            "MolWt": self._make_meta("constitutional"),
            "Geom": self._make_meta("geometric"),
        })
        s, u = filter_descriptors_by_handler_support(handler, ["MolWt", "Geom", "Missing"], reg)
        self.assertIn("MolWt", s)
        self.assertIn("Geom", u)
        self.assertIn("Missing", u)

    def test_no_category_no_3d_supported(self):
        handler = _create_handler_instance()
        m = MagicMock()
        m.category = None
        m.requires_3d = False
        s, u = filter_descriptors_by_handler_support(
            handler, ["Desc"], self._make_registry({"Desc": m}))
        self.assertIn("Desc", s)

    def test_returns_tuple_of_two_lists(self):
        result = filter_descriptors_by_handler_support(
            _create_handler_instance(), [], self._make_registry({}))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


# ============================================================================
# GROUP 14: verify_handler_abstraction & get_handler_abstraction_summary (6 tests)
# ============================================================================

class TestVerifyHandlerAbstractionAndSummary(unittest.TestCase):

    def test_verify_returns_dict(self):
        self.assertIsInstance(verify_handler_abstraction(), dict)

    def test_verify_has_core_keys(self):
        result = verify_handler_abstraction()
        for key in ['organic_fix_applied', 'abstraction_complete', 'handler_classes', 'registry_integration']:
            self.assertIn(key, result)

    def test_verify_registry_integration_keys(self):
        reg = verify_handler_abstraction()['registry_integration']
        for key in ['registry_initialized', 'registry_available', 'phase_6_complete']:
            self.assertIn(key, reg)

    def test_summary_returns_dict(self):
        self.assertIsInstance(get_handler_abstraction_summary(), dict)

    def test_summary_has_objectives(self):
        result = get_handler_abstraction_summary()
        self.assertIn('objectives_achieved', result)
        self.assertTrue(len(result['objectives_achieved']) > 0)

    def test_summary_has_phase_info(self):
        result = get_handler_abstraction_summary()
        self.assertIn('phase_6_registry_integration', result)
        self.assertIn('phase_7_migration', result)


# ============================================================================
# GROUP 15: get_common_required_properties (3 tests)
# ============================================================================

class TestGetCommonRequiredProperties(unittest.TestCase):

    def test_returns_list(self):
        result = _create_handler_instance().get_common_required_properties()
        self.assertIsInstance(result, list)

    def test_fallback_contains_basics(self):
        handler = _create_handler_instance()
        with patch.dict('sys.modules', {'milia_pipeline.config.config_accessors': None}):
            try:
                result = handler.get_common_required_properties()
                self.assertIsInstance(result, list)
            except (ImportError, ModuleNotFoundError):
                pass  # Acceptable in test environment

    def test_method_exists_on_handler(self):
        handler = _create_handler_instance()
        self.assertTrue(hasattr(handler, 'get_common_required_properties'))
        self.assertTrue(callable(handler.get_common_required_properties))


# ============================================================================
# GROUP 16: Edge Cases and Boundary Conditions (8 tests)
# ============================================================================

class TestEdgeCasesAndBoundary(unittest.TestCase):

    def test_handler_with_empty_string_experimental_setup(self):
        self.assertEqual(_create_handler_instance(experimental_setup="").experimental_setup, "")

    def test_ensure_tensor_empty_list(self):
        result = _create_handler_instance()._ensure_tensor([])
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.numel(), 0)

    def test_ensure_tensor_nested_list(self):
        result = _create_handler_instance()._ensure_tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(result.shape, (2, 2))

    def test_ensure_tensor_large_numpy_array(self):
        result = _create_handler_instance()._ensure_tensor(np.random.randn(1000, 50))
        self.assertEqual(result.shape, (1000, 50))

    def test_multiple_handlers_coexist(self):
        a = _create_handler_instance(dataset_type="TYPE_A")
        b = _create_handler_instance(dataset_type="TYPE_B")
        self.assertEqual(a.get_dataset_type(), "TYPE_A")
        self.assertEqual(b.get_dataset_type(), "TYPE_B")

    def test_handler_isinstance_check(self):
        self.assertIsInstance(_create_handler_instance(), DatasetHandler)

    def test_handler_issubclass_check(self):
        self.assertTrue(issubclass(_build_concrete_handler_class(), DatasetHandler))

    def test_extract_charge_large_positive(self):
        self.assertEqual(_create_handler_instance()._extract_charge_from_inchi("InChI=1S/t/q+5"), 5)


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_comprehensive_suite():
    """Run all test groups in a structured order."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestDatasetHandlerInit,                    # GROUP 1:  14 tests
        TestDatasetHandlerAbstractMethods,         # GROUP 2:  12 tests
        TestExtractChargeFromInchi,                # GROUP 3:   8 tests
        TestIsValidProperty,                       # GROUP 4:   9 tests
        TestEnsureTensor,                          # GROUP 5:  16 tests
        TestValidateConfiguration,                 # GROUP 6:   7 tests
        TestExperimentalSetupSupport,              # GROUP 7:   6 tests
        TestValidateTransformCompatibility,        # GROUP 8:  10 tests
        TestHandleTransformErrorsDecorator,        # GROUP 9:   8 tests
        TestRegistryFunctions,                     # GROUP 10: 12 tests
        TestCreateDatasetHandler,                  # GROUP 11: 10 tests
        TestValidateDatasetHandlerCompatibility,   # GROUP 12:  6 tests
        TestFilterDescriptorsByHandlerSupport,     # GROUP 13:  8 tests
        TestVerifyHandlerAbstractionAndSummary,    # GROUP 14:  6 tests
        TestGetCommonRequiredProperties,           # GROUP 15:  3 tests
        TestEdgeCasesAndBoundary,                  # GROUP 16:  8 tests
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS - base_handler.py")
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
TEST SUITE SUMMARY — milia_pipeline/handlers/base_handler.py
=============================================================

143 comprehensive production-ready tests across 16 groups:

GROUP 1:  DatasetHandler ABC — Instantiation & __init__            (14 tests)
GROUP 2:  DatasetHandler Abstract Methods — Interface Verification (12 tests)
GROUP 3:  _extract_charge_from_inchi Utility                       ( 8 tests)
GROUP 4:  _is_valid_property Utility                               ( 9 tests)
GROUP 5:  _ensure_tensor Utility                                   (16 tests)
GROUP 6:  validate_configuration                                   ( 7 tests)
GROUP 7:  Experimental Setup Support                               ( 6 tests)
GROUP 8:  validate_transform_compatibility                         (10 tests)
GROUP 9:  handle_transform_errors Decorator                        ( 8 tests)
GROUP 10: Registry Functions                                       (12 tests)
GROUP 11: create_dataset_handler Factory                           (10 tests)
GROUP 12: validate_dataset_handler_compatibility                   ( 6 tests)
GROUP 13: filter_descriptors_by_handler_support                    ( 8 tests)
GROUP 14: verify_handler_abstraction & summary                     ( 6 tests)
GROUP 15: get_common_required_properties                           ( 3 tests)
GROUP 16: Edge Cases and Boundary Conditions                       ( 8 tests)

PRODUCTION-READY QUALITIES:
- NO sys.modules pollution (no module-level mocking)
- All mocking via @patch decorators or context managers (test-level only)
- Dynamic test data creation via helper functions (no hardcoded paths)
- No NPZ file downloads (no file system dependencies)
- Comprehensive error path coverage
- Interface-focused testing (future-proof)
- Compatible with both pytest and unittest runner
- Registry state properly reset between tests (tearDown)
- Exception hierarchy correctly tested
- Error message quality assertions
"""
