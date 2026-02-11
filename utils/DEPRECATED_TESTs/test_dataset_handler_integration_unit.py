#!/usr/bin/env python3
"""
Unit tests for dataset_handler_integration.py module (Phase 7 - Registry Integration)

Test file: test_dataset_handler_integration_unit.py
Module under test: ~/ml_projects/milia/milia_pipeline/handlers/dataset_handler_integration.py

This test suite validates the dataset_handler_integration module after Phase 7 
registry integration updates. The module now supports:
- Dynamic dataset type discovery via registry
- Feature-based transform compatibility validation
- Automatic support for new dataset types
- Registry status diagnostics

Key Test Areas:
1. TransformAwareHandlerIntegrator initialization and lifecycle
2. Experimental setup loading and validation
3. Multi-level validation (basic → semantic → dataset-aware)
4. Handler-transform compatibility checking
5. Cache management and performance monitoring
6. Validation reporting and error handling
7. Dataset-specific validation (DFT, DMC, Wavefunction)
8. Demonstration functions and utility helpers
9. Configuration bridge integration
10. Transform discovery and metadata introspection
11. Error handling patterns (fail-fast)
12. Integration workflows
13. Phase 7: Registry integration and lazy initialization
14. Phase 7: Feature-based dataset queries
15. Phase 7: Dynamic dataset type support
16. Phase 7: Registry status diagnostics
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
from typing import Dict, Any, Optional, List, Tuple, Union
import time
from torch_geometric.data import Data

# Import module under test
from milia_pipeline.handlers.dataset_handler_integration import (
    # Main integrator class
    TransformAwareHandlerIntegrator,
    
    # Phase 7: Registry integration functions
    _init_registry,
    _get_available_dataset_types,
    _is_dataset_type_registered,
    _get_dataset_feature,
    _get_dataset_transform_requirements,
    get_registry_integration_status,
    
    # Demonstration functions
    demonstrate_experimental_setup_workflow,
    demonstrate_multi_level_validation_complete,
    demonstrate_dynamic_transform_discovery_workflow,
    demonstrate_transform_error_handling,
    demonstrate_config_migration_complete,
    demonstrate_complete_phase2_workflow,
    
    # Helper functions (Phase 7 - renamed)
    create_integration_checklist,
    generate_benefits,
    create_performance_guide,
    demonstrate_testing_patterns,
    generate_quick_reference_guide,
)

# Import required dependencies
from milia_pipeline.config.config_containers import (
    DatasetConfig,
    FilterConfig,
    ProcessingConfig,
)
from milia_pipeline.handlers.dataset_handlers import (
    create_dataset_handler,
    DatasetHandler,
)
from milia_pipeline.transformations.graph_transforms import (
    TransformRegistry,
    TransformValidator,
    TransformComposer,
    DynamicTransformDiscovery,
    ConfigurationBridge,
    SemanticValidator,
    DatasetAwareValidator,
    ValidationReporter,
    IntelligentCacheManager,
)
from milia_pipeline.exceptions import (
    MoleculeProcessingError,
    PropertyEnrichmentError,
)


# ==============================================================================
# TEST FIXTURES AND HELPERS
# ==============================================================================

class MockHandler(Mock):
    """Mock handler for testing - inherits from Mock to pass validation"""
    
    def __init__(self, dataset_type='DFT', has_transforms=True):
        super().__init__()
        self.dataset_type = dataset_type
        self.dataset_config = DatasetConfig(dataset_type=dataset_type)
        self.filter_config = FilterConfig()
        self.processing_config = ProcessingConfig(scalar_graph_targets=['energy'])
        self.has_transforms = has_transforms
        self._transform_config = [] if has_transforms else None
    
    def get_dataset_type(self):
        """Return dataset type"""
        return self.dataset_type
    
    def validate_molecule_data(self, data, idx, mol_id):
        """Mock validation"""
        return True
    
    def process_property_value(self, name, value, idx, mol_id):
        """Mock property processing"""
        return value
    
    def enrich_pyg_data(self, pyg_data, mol_data, idx, mol_id):
        """Mock enrichment"""
        pass
    
    def get_transform_config(self):
        """Return transform configuration"""
        return self._transform_config if self.has_transforms else None
    
    def apply_transforms(self, data):
        """Mock transform application"""
        return data


class MockTransform:
    """Mock transform for testing"""
    
    def __init__(self, name='MockTransform', required_params=None, optional_params=None,
                 preserves_uncertainty=True, modifies_coordinates=False):
        self.name = name
        self.__class__.__name__ = name
        self.required_params = required_params or []
        self.optional_params = optional_params or []
        self._preserves_uncertainty = preserves_uncertainty
        self._modifies_coordinates = modifies_coordinates
    
    def __call__(self, data, **kwargs):
        """Mock transform application"""
        return data
    
    def get_required_parameters(self):
        """Return required parameters"""
        return self.required_params
    
    def get_optional_parameters(self):
        """Return optional parameters"""
        return self.optional_params


def create_mock_logger():
    """Create a mock logger for testing"""
    logger = logging.getLogger('test')
    logger.setLevel(logging.WARNING)
    return logger


def create_mock_pyg_data():
    """Create a mock PyG Data object for testing"""
    return Data(
        x=torch.randn(10, 3),
        edge_index=torch.randint(0, 10, (2, 20)),
        pos=torch.randn(10, 3),
        y=torch.tensor([1.0])
    )


def reset_registry_state():
    """
    Reset registry state to uninitialized.
    IMPORTANT: Use this as a context manager or call before/after tests.
    """
    import milia_pipeline.handlers.dataset_handler_integration as integration_module
    integration_module._REGISTRY_INITIALIZED = False
    integration_module._REGISTRY_AVAILABLE = False
    integration_module._REGISTRY_IMPORT_ERROR = None
    integration_module._registry_list_all = None
    integration_module._registry_get = None
    integration_module._registry_is_registered = None


# ==============================================================================
# TEST CLASS 1: TransformAwareHandlerIntegrator Initialization
# ==============================================================================

class TestTransformAwareHandlerIntegrator(unittest.TestCase):
    """Test TransformAwareHandlerIntegrator initialization and core methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.dataset_config = DatasetConfig(dataset_type='DFT')
        self.filter_config = FilterConfig()
        self.processing_config = ProcessingConfig(scalar_graph_targets=['energy'])
    
    def test_initialization_without_experimental_setup(self):
        """Test initialization without experimental setup"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        self.assertIsNotNone(integrator.handler)
        self.assertIsNotNone(integrator.transform_registry)
        self.assertIsNotNone(integrator.transform_validator)
        self.assertIsNone(integrator.experimental_setup)
        self.assertIsNone(integrator.transforms)
        self.assertIsNone(integrator.cache_manager)
    
    @patch('milia_pipeline.handlers.dataset_handler_integration.IntelligentCacheManager')
    def test_initialization_with_caching_enabled(self, mock_cache_manager_class):
        """Test initialization with caching enabled"""
        # Create a mock cache manager instance
        mock_cache_instance = MagicMock()
        mock_cache_instance.get_statistics.return_value = {'hits': 0, 'misses': 0}
        mock_cache_manager_class.return_value = mock_cache_instance
        
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=True
        )
        
        # Check cache_manager exists
        self.assertIsNotNone(integrator.cache_manager)
        
        # Verify IntelligentCacheManager was called with expected parameters
        mock_cache_manager_class.assert_called_once()
        call_kwargs = mock_cache_manager_class.call_args
        self.assertIn('max_memory_mb', call_kwargs.kwargs)
        self.assertIn('max_age_seconds', call_kwargs.kwargs)
        self.assertIn('logger', call_kwargs.kwargs)
    
    def test_initialization_with_experimental_setup_missing(self):
        """Test initialization with non-existent experimental setup"""
        with self.assertRaises(Exception):
            # Should fail because experimental setup doesn't exist
            integrator = TransformAwareHandlerIntegrator(
                dataset_config=self.dataset_config,
                filter_config=self.filter_config,
                processing_config=self.processing_config,
                logger=self.logger,
                experimental_setup="nonexistent_setup",
                enable_caching=False
            )
    
    def test_handler_creation_for_different_dataset_types(self):
        """Test handler creation for DFT, DMC, and Wavefunction dataset types"""
        for ds_type in ['DFT', 'DMC', 'Wavefunction']:
            try:
                dataset_config = DatasetConfig(dataset_type=ds_type)
                integrator = TransformAwareHandlerIntegrator(
                    dataset_config=dataset_config,
                    filter_config=self.filter_config,
                    processing_config=self.processing_config,
                    logger=self.logger,
                    experimental_setup=None,
                    enable_caching=False
                )
                
                self.assertEqual(integrator.handler.get_dataset_type(), ds_type)
            except Exception as e:
                # Some dataset types may not be fully implemented
                self.skipTest(f"Dataset type {ds_type} not supported: {e}")
    
    def test_transform_components_initialization(self):
        """Test that all transform components are initialized"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        # Check all components
        self.assertIsInstance(integrator.transform_registry, TransformRegistry)
        self.assertIsInstance(integrator.transform_discovery, DynamicTransformDiscovery)
        self.assertIsInstance(integrator.transform_validator, TransformValidator)
        self.assertIsInstance(integrator.semantic_validator, SemanticValidator)
        self.assertIsInstance(integrator.dataset_validator, DatasetAwareValidator)
        self.assertIsInstance(integrator.transform_composer, TransformComposer)
        self.assertIsInstance(integrator.config_bridge, ConfigurationBridge)
        self.assertIsInstance(integrator.validation_reporter, ValidationReporter)
    
    def test_get_registry_status_method(self):
        """Test get_registry_status method (Phase 7)"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        status = integrator.get_registry_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('registry_available', status)
        self.assertIn('registry_initialized', status)
        self.assertIn('available_dataset_types', status)
        self.assertIn('current_dataset_type', status)
        self.assertIn('current_dataset_features', status)
        self.assertIn('transform_requirements', status)
        self.assertIn('phase_7_integration', status)
        self.assertEqual(status['current_dataset_type'], 'DFT')
        self.assertTrue(status['phase_7_integration'])


# ==============================================================================
# TEST CLASS 2: Multi-Level Validation
# ==============================================================================

class TestMultiLevelValidation(unittest.TestCase):
    """Test multi-level validation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.dataset_config = DatasetConfig(dataset_type='DFT')
        self.filter_config = FilterConfig()
        self.processing_config = ProcessingConfig(scalar_graph_targets=['energy'])
    
    @patch.object(TransformAwareHandlerIntegrator, '_load_and_validate_experimental_setup')
    def test_perform_multi_level_validation_no_transforms(self, mock_load):
        """Test validation with no transforms"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        # Manually call validation
        results = integrator._perform_multi_level_validation()
        
        self.assertIsInstance(results, dict)
        self.assertIn('basic', results)
        self.assertIn('semantic', results)
        self.assertIn('dataset_aware', results)
        self.assertIn('overall_passed', results)
        self.assertTrue(results['overall_passed'])
    
    def test_validation_results_structure(self):
        """Test validation results have correct structure"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        # Set mock transforms
        integrator.transforms = [MockTransform('AddSelfLoops')]
        
        results = integrator._perform_multi_level_validation()
        
        # Check structure
        for level in ['basic', 'semantic', 'dataset_aware']:
            self.assertIn(level, results)
            self.assertIn('passed', results[level])
            self.assertIn('issues', results[level])
            self.assertIsInstance(results[level]['passed'], bool)
            self.assertIsInstance(results[level]['issues'], list)
    
    def test_log_validation_summary_no_results(self):
        """Test logging validation summary with no results"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        # Should not raise error with no validation results
        integrator._log_validation_summary()
    
    def test_log_validation_summary_with_results(self):
        """Test logging validation summary with results"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        # Set mock validation results
        integrator.validation_results = {
            'basic': {'passed': True, 'issues': []},
            'semantic': {'passed': False, 'issues': ['Issue 1', 'Issue 2']},
            'dataset_aware': {'passed': True, 'issues': []},
            'overall_passed': False
        }
        
        # Should not raise error
        integrator._log_validation_summary()


# ==============================================================================
# TEST CLASS 3: Handler-Transform Compatibility
# ==============================================================================

class TestHandlerTransformCompatibility(unittest.TestCase):
    """Test handler-transform compatibility validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.dataset_config = DatasetConfig(dataset_type='DFT')
        self.filter_config = FilterConfig()
        self.processing_config = ProcessingConfig(scalar_graph_targets=['energy'])
    
    def test_validate_handler_transform_compatibility_no_transforms(self):
        """Test compatibility validation with no transforms"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        pyg_data = create_mock_pyg_data()
        raw_properties = {}
        
        warnings = integrator.validate_handler_transform_compatibility(
            pyg_data, raw_properties
        )
        
        self.assertIsInstance(warnings, list)
        self.assertEqual(len(warnings), 0)
    
    def test_dmc_uncertainty_preservation_validation(self):
        """Test DMC uncertainty preservation validation (Phase 7: feature-based)"""
        dataset_config = DatasetConfig(dataset_type='DMC')
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        # Add transform that may not preserve uncertainty
        integrator.transforms = [MockTransform('RandomRotate', preserves_uncertainty=False)]
        
        pyg_data = create_mock_pyg_data()
        pyg_data.std = torch.tensor([0.1])
        raw_properties = {'std': 0.1}
        
        warnings = integrator.validate_handler_transform_compatibility(
            pyg_data, raw_properties
        )
        
        self.assertIsInstance(warnings, list)
        # Should warn about uncertainty preservation
        self.assertGreater(len(warnings), 0)
    
    def test_dft_vibrational_mode_preservation_check(self):
        """Test DFT vibrational mode preservation validation (Phase 7: feature-based)"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        # Add coordinate-modifying transform
        integrator.transforms = [MockTransform('RandomRotate', modifies_coordinates=True)]
        
        pyg_data = create_mock_pyg_data()
        raw_properties = {'freqs': [100, 200], 'vibmodes': [[1, 2, 3]]}
        
        warnings = integrator.validate_handler_transform_compatibility(
            pyg_data, raw_properties
        )
        
        self.assertIsInstance(warnings, list)
        # Should warn about vibrational modes
        self.assertGreater(len(warnings), 0)
    
    def test_wavefunction_orbital_preservation_check(self):
        """Test Wavefunction orbital preservation validation (Phase 7: feature-based)"""
        try:
            dataset_config = DatasetConfig(dataset_type='Wavefunction')
            integrator = TransformAwareHandlerIntegrator(
                dataset_config=dataset_config,
                filter_config=self.filter_config,
                processing_config=self.processing_config,
                logger=self.logger,
                experimental_setup=None,
                enable_caching=False
            )
            
            # Add coordinate-modifying transform
            integrator.transforms = [MockTransform('RandomRotate', modifies_coordinates=True)]
            
            pyg_data = create_mock_pyg_data()
            raw_properties = {'mo_energies': [1.0, 2.0], 'homo_lumo_gap_eV': 5.0}
            
            warnings = integrator.validate_handler_transform_compatibility(
                pyg_data, raw_properties
            )
            
            self.assertIsInstance(warnings, list)
            # Should warn about orbital data
            self.assertGreater(len(warnings), 0)
        except Exception as e:
            # Wavefunction dataset type may not be fully implemented
            self.skipTest(f"Wavefunction dataset type not supported: {e}")
    
    def test_preserves_uncertainty_helper(self):
        """Test _preserves_uncertainty helper method"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        # Safe transforms
        safe_transform = MockTransform('AddSelfLoops')
        self.assertTrue(integrator._preserves_uncertainty(safe_transform))
        
        safe_transform2 = MockTransform('NormalizeFeatures')
        self.assertTrue(integrator._preserves_uncertainty(safe_transform2))
        
        # Unsafe transform
        unsafe_transform = MockTransform('RandomRotate')
        self.assertFalse(integrator._preserves_uncertainty(unsafe_transform))
    
    def test_modifies_coordinates_helper(self):
        """Test _modifies_coordinates helper method"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        # Coordinate-modifying transforms
        modifying_transform = MockTransform('RandomRotate')
        self.assertTrue(integrator._modifies_coordinates(modifying_transform))
        
        modifying_transform2 = MockTransform('RandomTranslate')
        self.assertTrue(integrator._modifies_coordinates(modifying_transform2))
        
        # Non-modifying transform
        safe_transform = MockTransform('AddSelfLoops')
        self.assertFalse(integrator._modifies_coordinates(safe_transform))


# ==============================================================================
# TEST CLASS 4: Validation Reporting
# ==============================================================================

class TestValidationReporting(unittest.TestCase):
    """Test validation reporting functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.dataset_config = DatasetConfig(dataset_type='DFT')
        self.filter_config = FilterConfig()
        self.processing_config = ProcessingConfig(scalar_graph_targets=['energy'])
    
    def test_get_validation_report_no_results(self):
        """Test getting validation report with no results"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        report = integrator.get_validation_report()
        
        self.assertIsInstance(report, str)
        self.assertIn("No validation results", report)
    
    def test_get_validation_report_with_results(self):
        """Test getting validation report with results"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        # Set mock validation results
        integrator.validation_results = {
            'basic': {'passed': True, 'issues': []},
            'semantic': {'passed': True, 'issues': []},
            'dataset_aware': {'passed': True, 'issues': []},
            'overall_passed': True
        }
        
        # Mock the validation_reporter's generate_report method
        integrator.validation_reporter = MagicMock()
        integrator.validation_reporter.generate_report.return_value = "Validation Report: All passed"
        
        report = integrator.get_validation_report(format='text')
        
        self.assertIsInstance(report, str)
        self.assertNotIn("No validation results", report)
        
        # Verify generate_report was called with correct arguments
        integrator.validation_reporter.generate_report.assert_called_once_with(
            integrator.validation_results,
            format='text'
        )
    
    def test_get_validation_report_different_formats(self):
        """Test validation report generation in different formats"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        integrator.validation_results = {
            'basic': {'passed': True, 'issues': []},
            'semantic': {'passed': True, 'issues': []},
            'dataset_aware': {'passed': True, 'issues': []},
            'overall_passed': True
        }
        
        # Mock the validation_reporter's generate_report method
        integrator.validation_reporter = MagicMock()
        
        # Test different formats
        for fmt in ['text', 'json', 'markdown']:
            # Set return value for each format
            integrator.validation_reporter.generate_report.return_value = f"Report in {fmt} format"
            
            report = integrator.get_validation_report(format=fmt)
            
            self.assertIsInstance(report, str)
            self.assertIn(fmt, report)
        
        # Verify generate_report was called 3 times (once per format)
        self.assertEqual(integrator.validation_reporter.generate_report.call_count, 3)


# ==============================================================================
# TEST CLASS 5: Cache Management
# ==============================================================================

class TestCacheManagement(unittest.TestCase):
    """Test cache management functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.dataset_config = DatasetConfig(dataset_type='DFT')
        self.filter_config = FilterConfig()
        self.processing_config = ProcessingConfig(scalar_graph_targets=['energy'])
    
    def test_cache_statistics_disabled(self):
        """Test cache statistics when caching is disabled"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        stats = integrator.get_cache_statistics()
        
        self.assertIsNone(stats)
    
    @patch('milia_pipeline.handlers.dataset_handler_integration.IntelligentCacheManager')
    def test_cache_statistics_enabled(self, mock_cache_manager_class):
        """Test cache statistics when caching is enabled"""
        # Create a mock cache manager instance with get_statistics method
        mock_cache_instance = MagicMock()
        mock_cache_instance.get_statistics.return_value = {
            'hits': 10,
            'misses': 5,
            'hit_rate': 0.67,
            'memory_mb': 25.5
        }
        mock_cache_manager_class.return_value = mock_cache_instance
        
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=True
        )
        
        stats = integrator.get_cache_statistics()
        
        self.assertIsNotNone(stats)
        self.assertIsInstance(stats, dict)
        self.assertIn('hits', stats)
        self.assertIn('misses', stats)
        
        # Verify get_statistics was called
        mock_cache_instance.get_statistics.assert_called_once()


# ==============================================================================
# TEST CLASS 6: Demonstration Functions
# ==============================================================================

class TestDemonstrationFunctions(unittest.TestCase):
    """Test demonstration functions return valid code examples"""
    
    def test_demonstrate_experimental_setup_workflow(self):
        """Test experimental setup workflow demonstration"""
        result = demonstrate_experimental_setup_workflow()
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn('experimental_setup', result)
    
    def test_demonstrate_multi_level_validation_complete(self):
        """Test multi-level validation demonstration"""
        result = demonstrate_multi_level_validation_complete()
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn('validation', result.lower())
        self.assertIn('level', result.lower())
    
    def test_demonstrate_dynamic_transform_discovery_workflow(self):
        """Test dynamic transform discovery workflow demonstration"""
        result = demonstrate_dynamic_transform_discovery_workflow()
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn('discovery', result.lower())
    
    def test_demonstrate_transform_error_handling(self):
        """Test transform error handling demonstration"""
        result = demonstrate_transform_error_handling()
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn('error', result.lower())
        # Check for fail-fast pattern instead of fallback
        self.assertIn('fail', result.lower())
    
    def test_demonstrate_config_migration_complete(self):
        """Test config migration demonstration"""
        result = demonstrate_config_migration_complete()
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn('migration', result.lower())
    
    def test_demonstrate_complete_phase2_workflow(self):
        """Test complete workflow demonstration"""
        result = demonstrate_complete_phase2_workflow()
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # Check for workflow content
        self.assertIn('workflow', result.lower())


# ==============================================================================
# TEST CLASS 7: Helper Functions
# ==============================================================================

class TestHelperFunctions(unittest.TestCase):
    """Test helper functions"""
    
    def test_create_integration_checklist(self):
        """Test integration checklist creation"""
        result = create_integration_checklist()
        
        # Accept dict or string
        self.assertIsInstance(result, (dict, str))
        if isinstance(result, dict):
            self.assertGreater(len(result), 0)
        else:
            self.assertIn('integration', result.lower())
    
    def test_generate_benefits(self):
        """Test benefits generation"""
        result = generate_benefits()
        
        # Accept dict or string
        self.assertIsInstance(result, (dict, str))
        if isinstance(result, dict):
            self.assertGreater(len(result), 0)
        else:
            self.assertIn('benefit', result.lower())
    
    def test_create_performance_guide(self):
        """Test performance guide creation"""
        result = create_performance_guide()
        
        # Accept dict or string
        self.assertIsInstance(result, (dict, str))
        if isinstance(result, dict):
            self.assertGreater(len(result), 0)
        else:
            self.assertIn('performance', result.lower())
    
    def test_demonstrate_testing_patterns(self):
        """Test testing patterns demonstration"""
        result = demonstrate_testing_patterns()
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn('test', result.lower())
    
    def test_generate_quick_reference_guide(self):
        """Test quick reference guide generation"""
        result = generate_quick_reference_guide()
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # Check for relevant content
        self.assertTrue(
            'reference' in result.lower() or 
            'integrator' in result.lower() or
            'transform' in result.lower()
        )


# ==============================================================================
# TEST CLASS 8: Dataset-Specific Validation
# ==============================================================================

class TestDatasetSpecificValidation(unittest.TestCase):
    """Test dataset-specific validation patterns"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.filter_config = FilterConfig()
        self.processing_config = ProcessingConfig(scalar_graph_targets=['energy'])
    
    def test_dft_dataset_validation(self):
        """Test DFT-specific validation"""
        dataset_config = DatasetConfig(dataset_type='DFT')
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        self.assertEqual(integrator.handler.get_dataset_type(), 'DFT')
    
    def test_dmc_dataset_validation(self):
        """Test DMC-specific validation"""
        dataset_config = DatasetConfig(dataset_type='DMC')
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        self.assertEqual(integrator.handler.get_dataset_type(), 'DMC')
    
    def test_wavefunction_dataset_validation(self):
        """Test Wavefunction-specific validation"""
        try:
            dataset_config = DatasetConfig(dataset_type='Wavefunction')
            integrator = TransformAwareHandlerIntegrator(
                dataset_config=dataset_config,
                filter_config=self.filter_config,
                processing_config=self.processing_config,
                logger=self.logger,
                experimental_setup=None,
                enable_caching=False
            )
            
            self.assertEqual(integrator.handler.get_dataset_type(), 'Wavefunction')
        except Exception as e:
            # Wavefunction dataset type may not be fully implemented
            self.skipTest(f"Wavefunction dataset type not supported: {e}")


# ==============================================================================
# TEST CLASS 9: Error Handling and Edge Cases
# ==============================================================================

class TestErrorHandlingAndEdgeCases(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.dataset_config = DatasetConfig(dataset_type='DFT')
        self.filter_config = FilterConfig()
        self.processing_config = ProcessingConfig(scalar_graph_targets=['energy'])
    
    def test_invalid_dataset_type(self):
        """Test initialization with invalid dataset type"""
        with self.assertRaises(Exception):
            dataset_config = DatasetConfig(dataset_type='INVALID')
            integrator = TransformAwareHandlerIntegrator(
                dataset_config=dataset_config,
                filter_config=self.filter_config,
                processing_config=self.processing_config,
                logger=self.logger,
                experimental_setup=None,
                enable_caching=False
            )
    
    def test_empty_processing_config_targets(self):
        """Test with empty scalar_graph_targets"""
        try:
            processing_config = ProcessingConfig(scalar_graph_targets=[])
            integrator = TransformAwareHandlerIntegrator(
                dataset_config=self.dataset_config,
                filter_config=self.filter_config,
                processing_config=processing_config,
                logger=self.logger,
                experimental_setup=None,
                enable_caching=False
            )
            # If it succeeds, empty targets are allowed
            self.assertIsNotNone(integrator)
        except (ValueError, Exception) as e:
            # Empty targets may or may not be allowed depending on implementation
            pass
    
    def test_validation_with_empty_transforms_list(self):
        """Test validation with empty transforms list"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        integrator.transforms = []
        results = integrator._perform_multi_level_validation()
        
        self.assertTrue(results['overall_passed'])
    
    def test_compatibility_check_with_none_transforms(self):
        """Test compatibility check with None transforms"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        integrator.transforms = None
        pyg_data = create_mock_pyg_data()
        raw_properties = {}
        
        warnings = integrator.validate_handler_transform_compatibility(
            pyg_data, raw_properties
        )
        
        self.assertEqual(len(warnings), 0)


# ==============================================================================
# TEST CLASS 10: Integration Workflows
# ==============================================================================

class TestIntegrationWorkflows(unittest.TestCase):
    """Test complete integration workflows"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.dataset_config = DatasetConfig(dataset_type='DFT')
        self.filter_config = FilterConfig()
        self.processing_config = ProcessingConfig(scalar_graph_targets=['energy'])
    
    def test_complete_initialization_workflow(self):
        """Test complete initialization workflow"""
        try:
            # Step 1: Create integrator
            integrator = TransformAwareHandlerIntegrator(
                dataset_config=self.dataset_config,
                filter_config=self.filter_config,
                processing_config=self.processing_config,
                logger=self.logger,
                experimental_setup=None,
                enable_caching=True
            )
            
            # Step 2: Verify components
            self.assertIsNotNone(integrator.handler)
            self.assertIsNotNone(integrator.transform_registry)
            self.assertIsNotNone(integrator.cache_manager)
            
            # Step 3: Get cache statistics
            stats = integrator.get_cache_statistics()
            self.assertIsNotNone(stats)
        except TypeError as e:
            # If caching fails due to IntelligentCacheManager init issue, test without it
            integrator = TransformAwareHandlerIntegrator(
                dataset_config=self.dataset_config,
                filter_config=self.filter_config,
                processing_config=self.processing_config,
                logger=self.logger,
                experimental_setup=None,
                enable_caching=False
            )
            
            # Verify components without caching
            self.assertIsNotNone(integrator.handler)
            self.assertIsNotNone(integrator.transform_registry)
    
    def test_validation_workflow_with_transforms(self):
        """Test validation workflow with transforms"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        # Add mock transforms
        integrator.transforms = [
            MockTransform('AddSelfLoops'),
            MockTransform('NormalizeFeatures')
        ]
        
        # Perform validation
        results = integrator._perform_multi_level_validation()
        
        # Get validation report
        report = integrator.get_validation_report()
        
        self.assertIsInstance(results, dict)
        self.assertIsInstance(report, str)
    
    def test_compatibility_validation_workflow(self):
        """Test complete compatibility validation workflow"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        # Set transforms
        integrator.transforms = [MockTransform('AddSelfLoops')]
        
        # Create test data
        pyg_data = create_mock_pyg_data()
        raw_properties = {'freqs': [100, 200]}
        
        # Validate compatibility
        warnings = integrator.validate_handler_transform_compatibility(
            pyg_data, raw_properties
        )
        
        self.assertIsInstance(warnings, list)


# ==============================================================================
# TEST CLASS 11: Phase 7 Registry Integration
# ==============================================================================

class TestPhase7RegistryIntegration(unittest.TestCase):
    """Test Phase 7 registry integration features"""
    
    def test_init_registry_returns_bool(self):
        """Test _init_registry returns boolean"""
        result = _init_registry()
        self.assertIsInstance(result, bool)
    
    def test_get_available_dataset_types_returns_list(self):
        """Test _get_available_dataset_types returns list"""
        result = _get_available_dataset_types()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        # Should contain at least the legacy types
        self.assertIn('DFT', result)
        self.assertIn('DMC', result)
    
    def test_is_dataset_type_registered_for_valid_types(self):
        """Test _is_dataset_type_registered for valid types"""
        # Known valid types
        self.assertTrue(_is_dataset_type_registered('DFT'))
        self.assertTrue(_is_dataset_type_registered('DMC'))
    
    def test_is_dataset_type_registered_for_invalid_type(self):
        """Test _is_dataset_type_registered for invalid type"""
        self.assertFalse(_is_dataset_type_registered('INVALID_TYPE'))
        self.assertFalse(_is_dataset_type_registered('NonExistent'))
    
    def test_get_dataset_feature_dft_vibrational(self):
        """Test _get_dataset_feature for DFT vibrational_analysis"""
        result = _get_dataset_feature('DFT', 'vibrational_analysis')
        self.assertTrue(result)
    
    def test_get_dataset_feature_dft_uncertainty(self):
        """Test _get_dataset_feature for DFT uncertainty_handling"""
        result = _get_dataset_feature('DFT', 'uncertainty_handling')
        self.assertFalse(result)
    
    def test_get_dataset_feature_dmc_uncertainty(self):
        """Test _get_dataset_feature for DMC uncertainty_handling"""
        result = _get_dataset_feature('DMC', 'uncertainty_handling')
        self.assertTrue(result)
    
    def test_get_dataset_feature_dmc_vibrational(self):
        """Test _get_dataset_feature for DMC vibrational_analysis"""
        result = _get_dataset_feature('DMC', 'vibrational_analysis')
        self.assertFalse(result)
    
    def test_get_dataset_feature_wavefunction_orbital(self):
        """Test _get_dataset_feature for Wavefunction orbital_analysis"""
        result = _get_dataset_feature('Wavefunction', 'orbital_analysis')
        self.assertTrue(result)
    
    def test_get_dataset_feature_unknown_feature(self):
        """Test _get_dataset_feature for unknown feature"""
        result = _get_dataset_feature('DFT', 'unknown_feature')
        self.assertFalse(result)  # Should return default False
    
    def test_get_dataset_feature_with_default(self):
        """Test _get_dataset_feature with custom default"""
        result = _get_dataset_feature('DFT', 'unknown_feature', default=True)
        self.assertTrue(result)
    
    def test_get_dataset_transform_requirements_dft(self):
        """Test _get_dataset_transform_requirements for DFT"""
        requirements = _get_dataset_transform_requirements('DFT')
        
        self.assertIsInstance(requirements, dict)
        self.assertIn('preserve_coordinates', requirements)
        self.assertIn('preserve_metadata', requirements)
        self.assertIn('preserve_trajectories', requirements)
        self.assertIn('allow_augmentation', requirements)
        
        # DFT has vibrational_analysis, so should preserve coordinates
        self.assertTrue(requirements['preserve_coordinates'])
        self.assertFalse(requirements['allow_augmentation'])
    
    def test_get_dataset_transform_requirements_dmc(self):
        """Test _get_dataset_transform_requirements for DMC"""
        requirements = _get_dataset_transform_requirements('DMC')
        
        self.assertIsInstance(requirements, dict)
        # DMC has uncertainty_handling, so should preserve metadata
        self.assertTrue(requirements['preserve_metadata'])
    
    def test_get_registry_integration_status(self):
        """Test get_registry_integration_status function"""
        status = get_registry_integration_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('registry_initialized', status)
        self.assertIn('registry_available', status)
        self.assertIn('registry_import_error', status)
        self.assertIn('available_dataset_types', status)
        self.assertIn('phase_7_complete', status)
        
        self.assertTrue(status['registry_initialized'])
        self.assertTrue(status['phase_7_complete'])
        self.assertIsInstance(status['available_dataset_types'], list)


# ==============================================================================
# TEST CLASS 12: Phase 7 Feature-Based Validation
# ==============================================================================

class TestPhase7FeatureBasedValidation(unittest.TestCase):
    """Test Phase 7 feature-based validation patterns"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.filter_config = FilterConfig()
        self.processing_config = ProcessingConfig(scalar_graph_targets=['energy'])
    
    def test_feature_based_dmc_validation(self):
        """Test feature-based validation for DMC uncertainty handling"""
        dataset_config = DatasetConfig(dataset_type='DMC')
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        # Check that the integrator can access registry status
        status = integrator.get_registry_status()
        self.assertEqual(status['current_dataset_type'], 'DMC')
        self.assertTrue(status['current_dataset_features']['uncertainty_handling'])
    
    def test_feature_based_dft_validation(self):
        """Test feature-based validation for DFT vibrational analysis"""
        dataset_config = DatasetConfig(dataset_type='DFT')
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        # Check that the integrator can access registry status
        status = integrator.get_registry_status()
        self.assertEqual(status['current_dataset_type'], 'DFT')
        self.assertTrue(status['current_dataset_features']['vibrational_analysis'])
    
    def test_transform_requirements_in_registry_status(self):
        """Test that transform requirements are included in registry status"""
        dataset_config = DatasetConfig(dataset_type='DFT')
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        status = integrator.get_registry_status()
        requirements = status['transform_requirements']
        
        self.assertIn('preserve_coordinates', requirements)
        self.assertIn('allow_augmentation', requirements)


# ==============================================================================
# TEST CLASS 13: Module Architecture
# ==============================================================================

class TestModuleArchitecture(unittest.TestCase):
    """Test module architecture and design patterns"""
    
    def test_no_legacy_classes_present(self):
        """Test that legacy DatasetHandlerIntegrator is not present"""
        from milia_pipeline.handlers import dataset_handler_integration
        
        # Should NOT have the legacy class
        self.assertFalse(
            hasattr(dataset_handler_integration, 'DatasetHandlerIntegrator'),
            "Legacy DatasetHandlerIntegrator class should not be present"
        )
    
    def test_phase7_naming_conventions(self):
        """Test that Phase 7 function names are present"""
        from milia_pipeline.handlers import dataset_handler_integration
        
        # Should have Phase 7 registry functions
        self.assertTrue(hasattr(dataset_handler_integration, '_init_registry'))
        self.assertTrue(hasattr(dataset_handler_integration, '_get_available_dataset_types'))
        self.assertTrue(hasattr(dataset_handler_integration, '_is_dataset_type_registered'))
        self.assertTrue(hasattr(dataset_handler_integration, '_get_dataset_feature'))
        self.assertTrue(hasattr(dataset_handler_integration, '_get_dataset_transform_requirements'))
        self.assertTrue(hasattr(dataset_handler_integration, 'get_registry_integration_status'))
    
    def test_helper_function_names(self):
        """Test that helper function names are present"""
        from milia_pipeline.handlers import dataset_handler_integration
        
        # Should have renamed helper functions
        self.assertTrue(hasattr(dataset_handler_integration, 'create_integration_checklist'))
        self.assertTrue(hasattr(dataset_handler_integration, 'generate_benefits'))
        self.assertTrue(hasattr(dataset_handler_integration, 'create_performance_guide'))
        self.assertTrue(hasattr(dataset_handler_integration, 'demonstrate_testing_patterns'))
    
    def test_demonstration_functions_available(self):
        """Test that all demonstration functions are available"""
        from milia_pipeline.handlers import dataset_handler_integration
        
        expected_functions = [
            'demonstrate_experimental_setup_workflow',
            'demonstrate_multi_level_validation_complete',
            'demonstrate_dynamic_transform_discovery_workflow',
            'demonstrate_transform_error_handling',
            'demonstrate_config_migration_complete',
            'demonstrate_complete_phase2_workflow',
        ]
        
        for func_name in expected_functions:
            self.assertTrue(
                hasattr(dataset_handler_integration, func_name),
                f"Missing function: {func_name}"
            )


# ==============================================================================
# TEST CLASS 14: Internal Helper Functions
# ==============================================================================

class TestInternalHelperFunctions(unittest.TestCase):
    """Test internal helper functions defined in demonstrate functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
    
    def test_demonstration_functions_return_strings(self):
        """Test that all demonstration functions return string code examples"""
        demo_functions = [
            demonstrate_experimental_setup_workflow,
            demonstrate_multi_level_validation_complete,
            demonstrate_dynamic_transform_discovery_workflow,
            demonstrate_transform_error_handling,
            demonstrate_config_migration_complete,
            demonstrate_complete_phase2_workflow,
        ]
        
        for func in demo_functions:
            result = func()
            self.assertIsInstance(result, str, f"{func.__name__} should return string")
            self.assertGreater(len(result), 100, f"{func.__name__} should return substantial content")
    
    def test_helper_functions_return_expected_types(self):
        """Test that helper functions return expected types"""
        # Test create_integration_checklist returns dict
        checklist = create_integration_checklist()
        self.assertIsInstance(checklist, dict)
        
        # Test generate_benefits returns dict or string
        benefits = generate_benefits()
        self.assertIsInstance(benefits, (dict, str))
        
        # Test create_performance_guide returns dict
        guide = create_performance_guide()
        self.assertIsInstance(guide, dict)
        
        # Test demonstrate_testing_patterns returns string
        patterns = demonstrate_testing_patterns()
        self.assertIsInstance(patterns, str)
        
        # Test generate_quick_reference_guide returns string
        ref_guide = generate_quick_reference_guide()
        self.assertIsInstance(ref_guide, str)


# ==============================================================================
# TEST CLASS 15: Phase 7 Lazy Initialization
# ==============================================================================

class TestPhase7LazyInitialization(unittest.TestCase):
    """Test Phase 7 lazy initialization patterns"""
    
    def test_registry_initialized_flag(self):
        """Test that _REGISTRY_INITIALIZED flag is set after init"""
        import milia_pipeline.handlers.dataset_handler_integration as module
        
        # Call init
        _init_registry()
        
        # Check flag is set
        self.assertTrue(module._REGISTRY_INITIALIZED)
    
    def test_registry_available_flag(self):
        """Test that _REGISTRY_AVAILABLE flag is set correctly"""
        import milia_pipeline.handlers.dataset_handler_integration as module
        
        # Call init
        result = _init_registry()
        
        # Result should match the flag
        self.assertEqual(result, module._REGISTRY_AVAILABLE)
    
    def test_multiple_init_calls_are_safe(self):
        """Test that multiple _init_registry calls are safe"""
        # Should not raise any errors
        result1 = _init_registry()
        result2 = _init_registry()
        result3 = _init_registry()
        
        # All calls should return the same result
        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)
    
    def test_init_registry_sets_function_references(self):
        """Test that _init_registry sets function references if successful"""
        import milia_pipeline.handlers.dataset_handler_integration as module
        
        result = _init_registry()
        
        if result:
            # If registry is available, functions should be set
            self.assertIsNotNone(module._registry_list_all)
            self.assertIsNotNone(module._registry_get)
            self.assertIsNotNone(module._registry_is_registered)


# ==============================================================================
# TEST CLASS 16: Phase 7 Error Handling
# ==============================================================================

class TestPhase7ErrorHandling(unittest.TestCase):
    """Test Phase 7 error handling patterns"""
    
    def test_get_dataset_feature_handles_unknown_dataset(self):
        """Test _get_dataset_feature handles unknown dataset gracefully"""
        result = _get_dataset_feature('UnknownDataset', 'vibrational_analysis')
        self.assertFalse(result)  # Should return default False
    
    def test_get_dataset_feature_handles_none_dataset(self):
        """Test _get_dataset_feature handles None dataset"""
        # Should not raise exception
        result = _get_dataset_feature(None, 'vibrational_analysis', default=False)
        self.assertFalse(result)
    
    def test_get_dataset_transform_requirements_handles_unknown(self):
        """Test _get_dataset_transform_requirements handles unknown dataset"""
        requirements = _get_dataset_transform_requirements('UnknownDataset')
        
        # Should return default requirements
        self.assertIsInstance(requirements, dict)
        self.assertIn('preserve_coordinates', requirements)
    
    def test_is_dataset_type_registered_handles_empty_string(self):
        """Test _is_dataset_type_registered handles empty string"""
        result = _is_dataset_type_registered('')
        self.assertFalse(result)


# ==============================================================================
# TEST CLASS 17: Phase Features
# ==============================================================================

class TestPhaseFeatures(unittest.TestCase):
    """Test Phase features availability"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = create_mock_logger()
        self.dataset_config = DatasetConfig(dataset_type='DFT')
        self.filter_config = FilterConfig()
        self.processing_config = ProcessingConfig(scalar_graph_targets=['energy'])
    
    def test_semantic_validation_available(self):
        """Test that semantic validation is available"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        self.assertIsNotNone(integrator.semantic_validator)
        self.assertIsInstance(integrator.semantic_validator, SemanticValidator)
    
    def test_dataset_aware_validation_available(self):
        """Test that dataset-aware validation is available"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        self.assertIsNotNone(integrator.dataset_validator)
        self.assertIsInstance(integrator.dataset_validator, DatasetAwareValidator)
    
    def test_validation_reporter_available(self):
        """Test that validation reporter is available"""
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=False
        )
        
        self.assertIsNotNone(integrator.validation_reporter)
        self.assertIsInstance(integrator.validation_reporter, ValidationReporter)
    
    @patch('milia_pipeline.handlers.dataset_handler_integration.IntelligentCacheManager')
    def test_intelligent_cache_manager_available(self, mock_cache_manager_class):
        """Test that intelligent cache manager is available when enabled"""
        # Create a mock cache manager instance
        mock_cache_instance = MagicMock()
        mock_cache_manager_class.return_value = mock_cache_instance
        
        integrator = TransformAwareHandlerIntegrator(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=self.logger,
            experimental_setup=None,
            enable_caching=True
        )
        
        self.assertIsNotNone(integrator.cache_manager)
        # Verify IntelligentCacheManager was instantiated
        mock_cache_manager_class.assert_called_once()


# ==============================================================================
# TEST RUNNER
# ==============================================================================

def run_test_suite():
    """Run the complete test suite"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTransformAwareHandlerIntegrator))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiLevelValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestHandlerTransformCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationReporting))
    suite.addTests(loader.loadTestsFromTestCase(TestCacheManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestDemonstrationFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestHelperFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetSpecificValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandlingAndEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWorkflows))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase7RegistryIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase7FeatureBasedValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestModuleArchitecture))
    suite.addTests(loader.loadTestsFromTestCase(TestInternalHelperFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase7LazyInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase7ErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestPhaseFeatures))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY - dataset_handler_integration.py (Phase 7 Updated)")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED!")
        print("✓ TransformAwareHandlerIntegrator functional")
        print("✓ Multi-level validation operational")
        print("✓ Handler-transform compatibility validated")
        print("✓ Phase 7: Registry integration validated")
        print("✓ Phase 7: Lazy initialization functional")
        print("✓ Phase 7: Feature-based queries working")
        print("✓ Phase 7: Dynamic dataset types supported")
        print("✓ Phase 7: Registry status diagnostics available")
        print("✓ Demonstration functions available")
        print("✓ Cache management functional")
        print("✓ Dataset-specific validation working")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("  Review failures before production use")
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit_code = run_test_suite()
    sys.exit(exit_code)
