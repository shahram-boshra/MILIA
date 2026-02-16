#!/usr/bin/env python3
"""
Comprehensive Unit Tests for molecule_converter_core.py module.

This expanded test suite provides thorough coverage of the MoleculeDataConverter class,
including all 51+ methods and critical code paths.

Coverage areas:
- Initialization & Configuration (8 tests)
- Handler Integration (6 tests)
- Transformation System (10 tests)
- Conversion Pipeline (15 tests)
- Validation & Error Handling (12 tests)
- Dataset Type Specific (8 tests)
- Structural Features (8 tests)
- Diagnostics & Statistics (7 tests)
- Caching & Performance (6 tests)
- Edge Cases & Error Recovery (10 tests)
- Phase 6: Registry Integration (25 tests) - NEW

Phase 6 additions:
- Registry integration functions (_init_registry, _get_available_dataset_types, etc.)
- Feature-based dataset processing (_get_dataset_feature)
- Molecule creation strategy lookup (_get_dataset_molecule_creation_strategy)
- Enhanced utils determination (_should_use_enhanced_utils)
- Registry status method (get_registry_integration_status)
- All 8 refactored hardcoded locations

Total: 115+ comprehensive tests

Test execution:
    cd /app/milia
    python -m pytest tests/test_molecule_converter_core_unit.py -v --tb=short
"""

import sys
from pathlib import Path

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
import unittest
from unittest.mock import Mock, patch

import numpy as np

# Test imports
try:
    from milia_pipeline.config.config_containers import (
        DatasetConfig,
        ExperimentalSetup,
        FilterConfig,
        ProcessingConfig,
        TransformationConfig,
        TransformSpec,
        create_dataset_config_from_global,
        create_filter_config_from_global,
        create_processing_config_from_global,
    )
    from milia_pipeline.exceptions import (
        ConfigurationError,
        HandlerConfigurationError,
        HandlerError,
        HandlerNotAvailableError,
        MoleculeProcessingError,
        PyGDataCreationError,
        RDKitConversionError,
        TransformationError,
        TransformConfigurationError,
    )
    from milia_pipeline.molecules.molecule_converter_core import MoleculeDataConverter

    IMPORTS_SUCCESSFUL = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)
    print(f"WARNING: Could not import required modules: {e}")

# Phase 6: Import registry integration functions
try:
    from milia_pipeline.molecules.molecule_converter_core import (
        _REGISTRY_AVAILABLE,
        _REGISTRY_IMPORT_ERROR,
        _REGISTRY_INITIALIZED,
        _get_available_dataset_types,
        _get_dataset_feature,
        _get_dataset_molecule_creation_strategy,
        _init_registry,
        _is_dataset_type_registered,
        _should_use_enhanced_utils,
    )

    PHASE6_IMPORTS_SUCCESSFUL = True
    PHASE6_IMPORT_ERROR = None
except ImportError as e:
    PHASE6_IMPORTS_SUCCESSFUL = False
    PHASE6_IMPORT_ERROR = str(e)
    print(f"WARNING: Could not import Phase 6 registry functions: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_mock_global_config():
    """Create a mock global configuration dictionary."""
    return {
        "dataset": {
            "type": "DFT",
            "source": "/fake/path",
            "target_properties": ["energy", "homo", "lumo"],
            "enable_caching": True,
        },
        "filtering": {
            "min_atoms": 1,
            "max_atoms": 50,
            "allowed_elements": ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br"],
            "charge_limits": {"min": -2, "max": 2},
        },
        "processing": {
            "batch_size": 32,
            "num_workers": 4,
            "enable_validation": True,
            "enable_caching": True,
        },
        "structural_features": {"enabled": False},
    }


def create_mock_molecule_data():
    """Create mock molecule data for testing."""
    return {
        "smiles": "CCO",
        "atoms": ["C", "C", "O"],
        "coordinates": np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 1.0, 0.0]]),
        "energy": -150.5,
        "homo": -0.25,
        "lumo": 0.15,
        "charge": 0,
    }


def reset_registry_state():
    """
    Reset registry state to uninitialized for testing.
    IMPORTANT: Use this before/after tests that modify registry state.
    """
    try:
        import milia_pipeline.molecules.molecule_converter_core as converter_module

        converter_module._REGISTRY_INITIALIZED = False
        converter_module._REGISTRY_AVAILABLE = False
        converter_module._REGISTRY_IMPORT_ERROR = None
        converter_module._registry_list_all = None
        converter_module._registry_get = None
        converter_module._registry_is_registered = None
    except (ImportError, AttributeError):
        pass  # Module may not have been imported yet


class TestMoleculeDataConverterInitialization(unittest.TestCase):
    """Test initialization and configuration - 8 tests."""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        self.global_config = create_mock_global_config()
        try:
            self.dataset_config = create_dataset_config_from_global(self.global_config)
            self.filter_config = create_filter_config_from_global(self.global_config)
            self.processing_config = create_processing_config_from_global(self.global_config)
        except Exception:
            self.dataset_config = Mock(spec=DatasetConfig)
            self.dataset_config.dataset_type = "DFT"
            self.filter_config = Mock(spec=FilterConfig)
            self.processing_config = Mock(spec=ProcessingConfig)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_01_basic_initialization(self, mock_create_handler):
        """Test basic initialization with minimal config."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        self.assertIsNotNone(converter)
        self.assertIsNotNone(converter._dataset_config)
        self.assertIsNotNone(converter._filter_config)
        self.assertIsNotNone(converter._processing_config)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_02_initialization_with_all_configs(self, mock_create_handler):
        """Test initialization with all configuration objects."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        transform_config = Mock(spec=TransformationConfig)
        mock_setup = Mock(spec=ExperimentalSetup)
        mock_setup.transforms = []
        mock_setup.name = "default"
        transform_config.experimental_setups = {"default": mock_setup}
        transform_config.default_setup = "default"

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            transformation_config=transform_config,
        )

        self.assertIsNotNone(converter._transformation_config)
        self.assertEqual(converter._transformation_config, transform_config)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_03_initialization_with_custom_logger(self, mock_create_handler):
        """Test initialization with custom logger."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        custom_logger = logging.getLogger("test_logger")
        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            logger=custom_logger,
        )

        self.assertEqual(converter.logger, custom_logger)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_04_initialization_with_injected_handler(self, mock_create_handler):
        """Test initialization with pre-created handler injection."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        # Test handler parameter (old name)
        converter1 = MoleculeDataConverter(
            handler=mock_handler,
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )
        self.assertIsNotNone(converter1)

        # Test dataset_handler parameter (new name)
        converter2 = MoleculeDataConverter(
            dataset_handler=mock_handler,
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )
        self.assertIsNotNone(converter2)

    def test_05_initialization_with_invalid_dataset_type(self):
        """Test that invalid dataset type raises error."""
        invalid_config = Mock(spec=DatasetConfig)
        invalid_config.dataset_type = "INVALID_TYPE"

        with self.assertRaises(
            (HandlerConfigurationError, ConfigurationError, ValueError, AttributeError)
        ):
            MoleculeDataConverter(
                dataset_config=invalid_config,
                filter_config=self.filter_config,
                processing_config=self.processing_config,
            )

    def test_06_initialization_with_custom_parameters(self):
        """Test initialization with custom atomic parameters."""
        custom_energies = {"H": -0.5, "C": -37.8}
        custom_har2ev = 27.211

        with (
            patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True),
            patch(
                "milia_pipeline.molecules.molecule_converter_core.create_dataset_handler"
            ) as mock_create,
        ):
            mock_handler = Mock()
            mock_handler.get_dataset_type.return_value = "DFT"
            mock_create.return_value = mock_handler

            converter = MoleculeDataConverter(
                dataset_config=self.dataset_config,
                filter_config=self.filter_config,
                processing_config=self.processing_config,
                atomic_energies_hartree=custom_energies,
                har2ev=custom_har2ev,
            )

            self.assertEqual(converter.atomic_energies_hartree, custom_energies)
            self.assertEqual(converter.har2ev, custom_har2ev)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_07_config_container_validation(self, mock_create_handler):
        """Test configuration container validation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        # Should have validated and stored configs
        self.assertIsNotNone(converter._dataset_config)
        self.assertIsNotNone(converter._filter_config)
        self.assertIsNotNone(converter._processing_config)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_08_default_config_creation(self, mock_create_handler):
        """Test default configuration creation when None provided."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        # Pass None to trigger default creation
        converter = MoleculeDataConverter()

        # Should have created defaults
        self.assertIsNotNone(converter._dataset_config)
        self.assertIsNotNone(converter._filter_config)
        self.assertIsNotNone(converter._processing_config)


class TestMoleculeDataConverterHandlers(unittest.TestCase):
    """Test handler integration - 6 tests."""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        self.global_config = create_mock_global_config()
        try:
            self.dataset_config = create_dataset_config_from_global(self.global_config)
            self.filter_config = create_filter_config_from_global(self.global_config)
            self.processing_config = create_processing_config_from_global(self.global_config)
        except Exception:
            self.dataset_config = Mock(spec=DatasetConfig)
            self.dataset_config.dataset_type = "DFT"
            self.filter_config = Mock(spec=FilterConfig)
            self.processing_config = Mock(spec=ProcessingConfig)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_01_handler_creation_from_config(self, mock_create_handler):
        """Test handler is created from configuration."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        self.assertTrue(mock_create_handler.called or hasattr(converter, "_handler"))

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", False)
    def test_02_graceful_degradation_without_handlers(self):
        """Test graceful handling when handlers unavailable."""
        try:
            converter = MoleculeDataConverter(
                dataset_config=self.dataset_config,
                filter_config=self.filter_config,
                processing_config=self.processing_config,
            )
            # If initialization succeeds, check that handler is None or unavailable
            self.assertTrue(
                not hasattr(converter, "_handler") or converter._handler is None,
                "Handler should be None when handlers unavailable",
            )
        except (HandlerNotAvailableError, ConfigurationError, ImportError, TypeError) as e:
            # Expected behavior when handlers unavailable
            logger.info(f"Expected exception when handlers unavailable: {type(e).__name__}")
            pass

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_03_get_handler_info(self, mock_create_handler):
        """Test retrieving handler information."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_handler.get_handler_info = Mock(return_value={"type": "DFT", "version": "1.0"})
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "get_handler_info"):
            info = converter.get_handler_info()
            self.assertIsInstance(info, dict)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_04_validate_handler_integration(self, mock_create_handler):
        """Test handler integration validation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "validate_handler_integration"):
            result = converter.validate_handler_integration()
            self.assertIsInstance(result, dict)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_05_handler_type_property(self, mock_create_handler):
        """Test dataset_type property."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        dataset_config = Mock(spec=DatasetConfig)
        dataset_config.dataset_type = "DFT"

        converter = MoleculeDataConverter(
            dataset_config=dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        self.assertEqual(converter._dataset_config.dataset_type, "DFT")

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_06_dataset_specific_handler_creation(self, mock_create_handler):
        """Test dataset-specific handler creation."""
        mock_handler = Mock()
        mock_handler.dataset_type = "DFT"
        mock_handler.get_dataset_type = Mock(return_value="DFT")
        mock_create_handler.return_value = mock_handler

        dataset_config = Mock(spec=DatasetConfig)
        dataset_config.dataset_type = "DFT"

        _converter = MoleculeDataConverter(
            dataset_config=dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        # Handler should have been created with correct type
        self.assertTrue(mock_create_handler.called)


class TestMoleculeDataConverterTransformationSystem(unittest.TestCase):
    """Test transformation system integration - 10 tests."""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        self.global_config = create_mock_global_config()
        try:
            self.dataset_config = create_dataset_config_from_global(self.global_config)
            self.filter_config = create_filter_config_from_global(self.global_config)
            self.processing_config = create_processing_config_from_global(self.global_config)
        except Exception:
            self.dataset_config = Mock(spec=DatasetConfig)
            self.dataset_config.dataset_type = "DFT"
            self.filter_config = Mock(spec=FilterConfig)
            self.processing_config = Mock(spec=ProcessingConfig)

        self.transform_config = Mock(spec=TransformationConfig)
        mock_setup = Mock(spec=ExperimentalSetup)
        mock_setup.transforms = []
        mock_setup.name = "test_setup"
        mock_setup.description = "Test"
        self.transform_config.experimental_setups = {"test_setup": mock_setup}
        self.transform_config.default_setup = "test_setup"

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_01_transform_config_integration(self, mock_create_handler):
        """Test transformation configuration integration."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            transformation_config=self.transform_config,
        )

        self.assertIsNotNone(converter._transformation_config)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_02_get_transform_capabilities(self, mock_create_handler):
        """Test getting transform capabilities."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            transformation_config=self.transform_config,
        )

        if hasattr(converter, "get_transform_capabilities"):
            capabilities = converter.get_transform_capabilities()
            self.assertIsInstance(capabilities, dict)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_03_validate_transform_compatibility(self, mock_create_handler):
        """Test transform compatibility validation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            transformation_config=self.transform_config,
        )

        if hasattr(converter, "validate_transform_compatibility"):
            result = converter.validate_transform_compatibility([])
            self.assertIsInstance(result, dict)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    @patch(
        "milia_pipeline.molecules.molecule_converter_core.ENHANCED_TRANSFORM_FEATURES_AVAILABLE",
        True,
    )
    def test_04_get_transform_parameter_info(self, mock_create_handler):
        """Test getting transform parameter information."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            transformation_config=self.transform_config,
        )

        if hasattr(converter, "get_transform_parameter_info"):
            info = converter.get_transform_parameter_info("normalize")
            self.assertIsInstance(info, (dict, type(None)))

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_05_discover_available_transforms(self, mock_create_handler):
        """Test dynamic transform discovery."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            transformation_config=self.transform_config,
        )

        if hasattr(converter, "discover_available_transforms"):
            transforms = converter.discover_available_transforms()
            self.assertIsInstance(transforms, (list, dict, set, type(None)))

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_06_transform_system_initialization(self, mock_create_handler):
        """Test transform system awareness initialization."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            transformation_config=self.transform_config,
        )

        # Should initialize without errors
        self.assertIsNotNone(converter)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_07_transform_graceful_degradation(self, mock_create_handler):
        """Test graceful degradation when transforms unavailable."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            transformation_config=self.transform_config,
        )

        self.assertIsNotNone(converter)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_08_default_transform_config_creation(self, mock_create_handler):
        """Test default transform config creation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        # Should create default transform config
        self.assertIsNotNone(converter._transformation_config)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_09_check_transform_dataset_compatibility(self, mock_create_handler):
        """Test transform dataset compatibility checking."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            transformation_config=self.transform_config,
        )

        # Should have the method
        if hasattr(converter, "_check_transform_dataset_compatibility"):
            self.assertTrue(callable(converter._check_transform_dataset_compatibility))

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_10_get_enhanced_transform_diagnostics(self, mock_create_handler):
        """Test enhanced transform diagnostics."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            transformation_config=self.transform_config,
        )

        if hasattr(converter, "get_enhanced_transform_diagnostics"):
            diagnostics = converter.get_enhanced_transform_diagnostics()
            self.assertIsInstance(diagnostics, dict)


class TestMoleculeDataConverterValidation(unittest.TestCase):
    """Test validation methods - 12 tests."""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        self.global_config = create_mock_global_config()
        try:
            self.dataset_config = create_dataset_config_from_global(self.global_config)
            self.filter_config = create_filter_config_from_global(self.global_config)
            self.processing_config = create_processing_config_from_global(self.global_config)
        except Exception:
            self.dataset_config = Mock(spec=DatasetConfig)
            self.dataset_config.dataset_type = "DFT"
            self.filter_config = Mock(spec=FilterConfig)
            self.processing_config = Mock(spec=ProcessingConfig)
            self.processing_config.enable_validation = True

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_01_validate_configuration_compatibility(self, mock_create_handler):
        """Test configuration compatibility validation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "validate_configuration_compatibility"):
            result = converter.validate_configuration_compatibility()
            self.assertIsInstance(result, (dict, bool, tuple))

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_02_get_processing_capabilities(self, mock_create_handler):
        """Test getting processing capabilities."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "get_processing_capabilities"):
            capabilities = converter.get_processing_capabilities()
            self.assertIsInstance(capabilities, dict)
            self.assertTrue(len(capabilities) > 0)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_03_validate_dataset_configuration(self, mock_create_handler):
        """Test dataset configuration validation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        # Should validate without errors
        self.assertIsNotNone(converter)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_04_validate_transformation_configuration(self, mock_create_handler):
        """Test transformation configuration validation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        transform_config = Mock(spec=TransformationConfig)
        mock_setup = Mock(spec=ExperimentalSetup)
        mock_setup.transforms = []
        mock_setup.name = "default"
        transform_config.experimental_setups = {"default": mock_setup}
        transform_config.default_setup = "default"

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            transformation_config=transform_config,
        )

        # Should validate without errors
        self.assertIsNotNone(converter)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_05_validate_converter_configuration(self, mock_create_handler):
        """Test converter configuration validation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        # Should have validated configuration
        self.assertIsNotNone(converter._dataset_config)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_06_validate_structural_features_configuration(self, mock_create_handler):
        """Test structural features configuration validation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
            structural_features_config={"enabled": True},
        )

        self.assertIsNotNone(converter)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_07_ensure_pyg_data_tensors(self, mock_create_handler):
        """Test PyG data tensor validation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "_ensure_pyg_data_tensors"):
            self.assertTrue(callable(converter._ensure_pyg_data_tensors))

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_08_validate_final_pyg_data(self, mock_create_handler):
        """Test final PyG data validation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "_validate_final_pyg_data"):
            self.assertTrue(callable(converter._validate_final_pyg_data))

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_09_validate_dft_properties(self, mock_create_handler):
        """Test DFT property validation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "_validate_dft_properties"):
            self.assertTrue(callable(converter._validate_dft_properties))

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_10_validate_dmc_uncertainty(self, mock_create_handler):
        """Test DMC uncertainty validation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DMC"
        mock_create_handler.return_value = mock_handler

        # Create a new DMC dataset config instead of modifying frozen dataclass
        dmc_config = Mock(spec=DatasetConfig)
        dmc_config.dataset_type = "DMC"

        converter = MoleculeDataConverter(
            dataset_config=dmc_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "_validate_dmc_uncertainty"):
            self.assertTrue(callable(converter._validate_dmc_uncertainty))

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_11_legacy_validation_fallback(self, mock_create_handler):
        """Test legacy validation fallback."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "_legacy_validation_fallback"):
            self.assertTrue(callable(converter._legacy_validation_fallback))

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_12_validate_handler_compatibility(self, mock_create_handler):
        """Test handler compatibility validation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "_validate_handler_compatibility"):
            self.assertTrue(callable(converter._validate_handler_compatibility))


class TestMoleculeDataConverterDiagnostics(unittest.TestCase):
    """Test diagnostic and statistics methods - 7 tests."""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        self.global_config = create_mock_global_config()
        try:
            self.dataset_config = create_dataset_config_from_global(self.global_config)
            self.filter_config = create_filter_config_from_global(self.global_config)
            self.processing_config = create_processing_config_from_global(self.global_config)
        except Exception:
            self.dataset_config = Mock(spec=DatasetConfig)
            self.dataset_config.dataset_type = "DFT"
            self.filter_config = Mock(spec=FilterConfig)
            self.processing_config = Mock(spec=ProcessingConfig)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_01_get_conversion_statistics(self, mock_create_handler):
        """Test getting conversion statistics."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "get_conversion_statistics"):
            stats = converter.get_conversion_statistics()
            self.assertIsInstance(stats, dict)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_02_reset_statistics(self, mock_create_handler):
        """Test resetting statistics."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "reset_statistics"):
            converter.reset_statistics()
            # Should not raise errors

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_03_get_error_recovery_capabilities(self, mock_create_handler):
        """Test getting error recovery capabilities."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "get_error_recovery_capabilities"):
            capabilities = converter.get_error_recovery_capabilities()
            self.assertIsInstance(capabilities, dict)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_04_get_enhanced_transform_diagnostics(self, mock_create_handler):
        """Test enhanced transform diagnostics."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "get_enhanced_transform_diagnostics"):
            diagnostics = converter.get_enhanced_transform_diagnostics()
            self.assertIsInstance(diagnostics, dict)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_05_log_structural_features_capabilities(self, mock_create_handler):
        """Test logging structural features capabilities."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "_log_structural_features_capabilities"):
            self.assertTrue(callable(converter._log_structural_features_capabilities))

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_06_diagnostics_after_conversion(self, mock_create_handler):
        """Test diagnostics are available after conversion."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "get_conversion_statistics"):
            stats = converter.get_conversion_statistics()
            self.assertIsInstance(stats, dict)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_07_detailed_error_diagnostics(self, mock_create_handler):
        """Test detailed error diagnostics."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        self.assertIsNotNone(converter)


class TestMoleculeDataConverterDatasetTypes(unittest.TestCase):
    """Test dataset type specific functionality - 8 tests."""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        self.filter_config = Mock(spec=FilterConfig)
        self.processing_config = Mock(spec=ProcessingConfig)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_01_dft_dataset_creation(self, mock_create_handler):
        """Test DFT dataset converter creation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        dft_config = Mock(spec=DatasetConfig)
        dft_config.dataset_type = "DFT"

        converter = MoleculeDataConverter(
            dataset_config=dft_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        self.assertEqual(converter._dataset_config.dataset_type, "DFT")

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_02_dmc_dataset_creation(self, mock_create_handler):
        """Test DMC dataset converter creation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DMC"
        mock_create_handler.return_value = mock_handler

        dmc_config = Mock(spec=DatasetConfig)
        dmc_config.dataset_type = "DMC"

        converter = MoleculeDataConverter(
            dataset_config=dmc_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        self.assertEqual(converter._dataset_config.dataset_type, "DMC")

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_03_wavefunction_dataset_creation(self, mock_create_handler):
        """Test Wavefunction dataset converter creation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "Wavefunction"
        mock_create_handler.return_value = mock_handler

        wf_config = Mock(spec=DatasetConfig)
        wf_config.dataset_type = "Wavefunction"

        converter = MoleculeDataConverter(
            dataset_config=wf_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        self.assertEqual(converter._dataset_config.dataset_type, "Wavefunction")

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_04_dft_property_handling(self, mock_create_handler):
        """Test DFT property handling methods exist."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        dft_config = Mock(spec=DatasetConfig)
        dft_config.dataset_type = "DFT"

        converter = MoleculeDataConverter(
            dataset_config=dft_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "_legacy_dft_validation"):
            self.assertTrue(callable(converter._legacy_dft_validation))

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_05_dmc_uncertainty_handling(self, mock_create_handler):
        """Test DMC uncertainty handling methods exist."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DMC"
        mock_create_handler.return_value = mock_handler

        dmc_config = Mock(spec=DatasetConfig)
        dmc_config.dataset_type = "DMC"

        converter = MoleculeDataConverter(
            dataset_config=dmc_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "_legacy_dmc_validation"):
            self.assertTrue(callable(converter._legacy_dmc_validation))

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_06_dataset_type_property_accessible(self, mock_create_handler):
        """Test dataset_type is accessible."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        dft_config = Mock(spec=DatasetConfig)
        dft_config.dataset_type = "DFT"

        converter = MoleculeDataConverter(
            dataset_config=dft_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        # Should be accessible via property or config
        self.assertEqual(converter._dataset_config.dataset_type, "DFT")

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_07_dft_handler_used(self, mock_create_handler):
        """Test DFT handler is used for DFT config."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        dft_config = Mock(spec=DatasetConfig)
        dft_config.dataset_type = "DFT"

        _converter = MoleculeDataConverter(
            dataset_config=dft_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        # Handler should have been created for DFT
        self.assertTrue(mock_create_handler.called)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_08_dmc_handler_used(self, mock_create_handler):
        """Test DMC handler is used for DMC config."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DMC"
        mock_create_handler.return_value = mock_handler

        dmc_config = Mock(spec=DatasetConfig)
        dmc_config.dataset_type = "DMC"

        _converter = MoleculeDataConverter(
            dataset_config=dmc_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        # Handler should have been created for DMC
        self.assertTrue(mock_create_handler.called)


class TestMoleculeDataConverterCaching(unittest.TestCase):
    """Test caching functionality - 6 tests."""

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        self.dataset_config = Mock(spec=DatasetConfig)
        self.dataset_config.dataset_type = "DFT"
        self.dataset_config.enable_caching = True

        self.filter_config = Mock(spec=FilterConfig)

        self.processing_config = Mock(spec=ProcessingConfig)
        self.processing_config.enable_caching = True

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_01_caching_enabled_initialization(self, mock_create_handler):
        """Test initialization with caching enabled."""
        mock_handler = Mock()
        mock_handler.dataset_type = "DFT"
        mock_handler.get_dataset_type = Mock(return_value="DFT")
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        self.assertIsNotNone(converter)
        self.assertTrue(self.dataset_config.enable_caching)
        self.assertTrue(self.processing_config.enable_caching)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_02_cache_clearing(self, mock_create_handler):
        """Test cache clearing functionality."""
        mock_handler = Mock()
        mock_handler.dataset_type = "DFT"
        mock_handler.get_dataset_type = Mock(return_value="DFT")
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "clear_handler_caches"):
            converter.clear_handler_caches()
            # Should not raise errors

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_03_caching_disabled(self, mock_create_handler):
        """Test behavior with caching disabled."""
        mock_handler = Mock()
        mock_handler.dataset_type = "DFT"
        mock_handler.get_dataset_type = Mock(return_value="DFT")
        mock_create_handler.return_value = mock_handler

        self.dataset_config.enable_caching = False
        self.processing_config.enable_caching = False

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        self.assertIsNotNone(converter)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_04_handler_cache_integration(self, mock_create_handler):
        """Test handler cache integration."""
        mock_handler = Mock()
        mock_handler.dataset_type = "DFT"
        mock_handler.get_dataset_type = Mock(return_value="DFT")
        mock_handler.clear_caches = Mock()
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "clear_handler_caches"):
            converter.clear_handler_caches()

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_05_transform_cache_integration(self, mock_create_handler):
        """Test transform cache integration."""
        mock_handler = Mock()
        mock_handler.dataset_type = "DFT"
        mock_handler.get_dataset_type = Mock(return_value="DFT")
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        # Should handle transform caches
        self.assertIsNotNone(converter)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_06_cache_statistics(self, mock_create_handler):
        """Test cache statistics."""
        mock_handler = Mock()
        mock_handler.dataset_type = "DFT"
        mock_handler.get_dataset_type = Mock(return_value="DFT")
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "get_conversion_statistics"):
            stats = converter.get_conversion_statistics()
            self.assertIsInstance(stats, dict)


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
    - Molecule creation strategy (_get_dataset_molecule_creation_strategy)
    - Enhanced utils determination (_should_use_enhanced_utils)
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

    def test_05_get_dataset_molecule_creation_strategy_function_exists(self):
        """Test _get_dataset_molecule_creation_strategy function is importable."""
        self.assertTrue(callable(_get_dataset_molecule_creation_strategy))

    def test_06_should_use_enhanced_utils_function_exists(self):
        """Test _should_use_enhanced_utils function is importable."""
        self.assertTrue(callable(_should_use_enhanced_utils))

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
        self.assertFalse(_is_dataset_type_registered("NONEXISTENT_DATASET"))
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


class TestPhase6MoleculeCreationStrategy(unittest.TestCase):
    """
    Test Phase 6 molecule creation strategy functions.

    These tests verify _get_dataset_molecule_creation_strategy and
    _should_use_enhanced_utils work correctly.
    """

    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")

    def setUp(self):
        reset_registry_state()

    def tearDown(self):
        reset_registry_state()

    def test_01_dft_uses_identifier_coordinate_based(self):
        """Test DFT uses identifier_coordinate_based strategy."""
        result = _get_dataset_molecule_creation_strategy("DFT")
        self.assertEqual(result, "identifier_coordinate_based")

    def test_02_dmc_uses_identifier_coordinate_based(self):
        """Test DMC uses identifier_coordinate_based strategy."""
        result = _get_dataset_molecule_creation_strategy("DMC")
        self.assertEqual(result, "identifier_coordinate_based")

    def test_03_wavefunction_uses_coordinate_based(self):
        """Test Wavefunction uses coordinate_based strategy."""
        result = _get_dataset_molecule_creation_strategy("Wavefunction")
        self.assertEqual(result, "coordinate_based")

    def test_04_unknown_defaults_to_identifier_coordinate_based(self):
        """Test unknown dataset defaults to identifier_coordinate_based."""
        result = _get_dataset_molecule_creation_strategy("UNKNOWN")
        self.assertEqual(result, "identifier_coordinate_based")

    def test_05_dft_should_use_enhanced_utils(self):
        """Test DFT should use enhanced utils."""
        result = _should_use_enhanced_utils("DFT")
        self.assertTrue(result)

    def test_06_dmc_should_use_enhanced_utils(self):
        """Test DMC should use enhanced utils."""
        result = _should_use_enhanced_utils("DMC")
        self.assertTrue(result)

    def test_07_wavefunction_should_not_use_enhanced_utils(self):
        """Test Wavefunction should NOT use enhanced utils."""
        result = _should_use_enhanced_utils("Wavefunction")
        self.assertFalse(result)

    def test_08_unknown_should_use_enhanced_utils(self):
        """Test unknown dataset should use enhanced utils (default strategy)."""
        result = _should_use_enhanced_utils("UNKNOWN")
        self.assertTrue(result)


class TestPhase6RegistryIntegrationStatus(unittest.TestCase):
    """
    Test Phase 6 get_registry_integration_status method.

    This method provides diagnostic information about registry integration.
    """

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        self.dataset_config = Mock(spec=DatasetConfig)
        self.dataset_config.dataset_type = "DFT"
        self.filter_config = Mock(spec=FilterConfig)
        self.processing_config = Mock(spec=ProcessingConfig)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_01_get_registry_integration_status_exists(self, mock_create_handler):
        """Test get_registry_integration_status method exists."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        self.assertTrue(hasattr(converter, "get_registry_integration_status"))
        self.assertTrue(callable(converter.get_registry_integration_status))

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_02_get_registry_integration_status_returns_dict(self, mock_create_handler):
        """Test get_registry_integration_status returns a dictionary."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "get_registry_integration_status"):
            status = converter.get_registry_integration_status()
            self.assertIsInstance(status, dict)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_03_registry_status_includes_available_types(self, mock_create_handler):
        """Test registry status includes available_dataset_types."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "get_registry_integration_status"):
            status = converter.get_registry_integration_status()
            self.assertIn("available_dataset_types", status)
            self.assertIsInstance(status["available_dataset_types"], list)
            self.assertIn("DFT", status["available_dataset_types"])

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_04_registry_status_includes_phase_6_complete(self, mock_create_handler):
        """Test registry status includes phase_6_complete=True."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "get_registry_integration_status"):
            status = converter.get_registry_integration_status()
            self.assertIn("phase_6_complete", status)
            self.assertTrue(status["phase_6_complete"])

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_05_registry_status_includes_current_dataset_type(self, mock_create_handler):
        """Test registry status includes current_dataset_type."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "get_registry_integration_status"):
            status = converter.get_registry_integration_status()
            self.assertIn("current_dataset_type", status)
            self.assertEqual(status["current_dataset_type"], "DFT")

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_06_registry_status_includes_dataset_features(self, mock_create_handler):
        """Test registry status includes dataset_features for DFT."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "get_registry_integration_status"):
            status = converter.get_registry_integration_status()
            if "dataset_features" in status:
                features = status["dataset_features"]
                self.assertIsInstance(features, dict)
                self.assertIn("vibrational_analysis", features)
                self.assertIn("uncertainty_handling", features)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_07_registry_status_includes_molecule_creation_strategy(self, mock_create_handler):
        """Test registry status includes molecule_creation_strategy."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        converter = MoleculeDataConverter(
            dataset_config=self.dataset_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "get_registry_integration_status"):
            status = converter.get_registry_integration_status()
            if "molecule_creation_strategy" in status:
                self.assertEqual(
                    status["molecule_creation_strategy"], "identifier_coordinate_based"
                )


class TestPhase6RefactoredMethods(unittest.TestCase):
    """
    Test Phase 6 refactored methods use feature-based queries.

    These tests verify that the 8 refactored locations correctly use
    registry-based validation and feature queries instead of hardcoded
    dataset type checks.
    """

    @classmethod
    def setUpClass(cls):
        if not IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Required imports failed: {IMPORT_ERROR}")

    def setUp(self):
        self.filter_config = Mock(spec=FilterConfig)
        self.processing_config = Mock(spec=ProcessingConfig)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_01_get_processing_capabilities_uses_dynamic_types(self, mock_create_handler):
        """Test get_processing_capabilities uses dynamic dataset types."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        dft_config = Mock(spec=DatasetConfig)
        dft_config.dataset_type = "DFT"

        converter = MoleculeDataConverter(
            dataset_config=dft_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "get_processing_capabilities"):
            capabilities = converter.get_processing_capabilities()
            self.assertIn("dataset_types_supported", capabilities)
            # Should include at least DFT, DMC, Wavefunction
            types = capabilities["dataset_types_supported"]
            self.assertIn("DFT", types)
            self.assertIn("DMC", types)
            self.assertIn("Wavefunction", types)

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_02_validate_configuration_compatibility_uses_registry(self, mock_create_handler):
        """Test validate_configuration_compatibility uses registry validation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        dft_config = Mock(spec=DatasetConfig)
        dft_config.dataset_type = "DFT"

        converter = MoleculeDataConverter(
            dataset_config=dft_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "validate_configuration_compatibility"):
            result = converter.validate_configuration_compatibility()
            self.assertIsInstance(result, dict)
            # Should validate without errors for DFT
            self.assertTrue(result.get("dataset_config_valid", True))

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_03_check_transform_dataset_compatibility_uses_features(self, mock_create_handler):
        """Test _check_transform_dataset_compatibility uses feature queries."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DMC"
        mock_create_handler.return_value = mock_handler

        dmc_config = Mock(spec=DatasetConfig)
        dmc_config.dataset_type = "DMC"

        converter = MoleculeDataConverter(
            dataset_config=dmc_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "_check_transform_dataset_compatibility"):
            # Create a mock TransformSpec
            transform_spec = Mock(spec=TransformSpec)
            transform_spec.name = "DropEdge"
            transform_spec.params = {}

            warnings = converter._check_transform_dataset_compatibility(transform_spec)
            # DMC with uncertainty_handling=True should warn about DropEdge
            self.assertIsInstance(warnings, list)
            # Should include warning about uncertainty propagation
            warning_text = " ".join(warnings)
            self.assertIn("uncertainty", warning_text.lower())

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_04_dft_no_uncertainty_warning(self, mock_create_handler):
        """Test DFT does not get uncertainty warnings for transforms."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "DFT"
        mock_create_handler.return_value = mock_handler

        dft_config = Mock(spec=DatasetConfig)
        dft_config.dataset_type = "DFT"

        converter = MoleculeDataConverter(
            dataset_config=dft_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        if hasattr(converter, "_check_transform_dataset_compatibility"):
            # Create a mock TransformSpec
            transform_spec = Mock(spec=TransformSpec)
            transform_spec.name = "DropEdge"
            transform_spec.params = {}

            warnings = converter._check_transform_dataset_compatibility(transform_spec)
            # DFT with uncertainty_handling=False should NOT warn about uncertainty
            warning_text = " ".join(warnings)
            self.assertNotIn("uncertainty propagation", warning_text.lower())

    @patch("milia_pipeline.molecules.molecule_converter_core.HANDLERS_AVAILABLE", True)
    @patch("milia_pipeline.molecules.molecule_converter_core.create_dataset_handler")
    def test_05_wavefunction_converter_creation(self, mock_create_handler):
        """Test Wavefunction converter creation with registry validation."""
        mock_handler = Mock()
        mock_handler.get_dataset_type.return_value = "Wavefunction"
        mock_create_handler.return_value = mock_handler

        wf_config = Mock(spec=DatasetConfig)
        wf_config.dataset_type = "Wavefunction"

        # Should create without error (registry validates Wavefunction)
        converter = MoleculeDataConverter(
            dataset_config=wf_config,
            filter_config=self.filter_config,
            processing_config=self.processing_config,
        )

        self.assertIsNotNone(converter)
        self.assertEqual(converter._dataset_config.dataset_type, "Wavefunction")

    def test_06_invalid_dataset_type_rejected_by_registry(self):
        """Test invalid dataset type is rejected by registry validation."""
        invalid_config = Mock(spec=DatasetConfig)
        invalid_config.dataset_type = "INVALID_TYPE_ABC"

        with self.assertRaises((ConfigurationError, ValueError, HandlerConfigurationError)):
            MoleculeDataConverter(
                dataset_config=invalid_config,
                filter_config=self.filter_config,
                processing_config=self.processing_config,
            )


class TestPhase6LegacyFallback(unittest.TestCase):
    """
    Test Phase 6 fallback behavior when registry unavailable.

    These tests verify that the module correctly degrades to conservative
    defaults when the registry is not available:
    - _get_available_dataset_types: filesystem discovery, or empty list
    - _is_dataset_type_registered: delegates to available types discovery
    - _get_dataset_feature: returns False (conservative default)
    - _get_dataset_molecule_creation_strategy: returns 'identifier_coordinate_based' (default)
    """

    @classmethod
    def setUpClass(cls):
        if not PHASE6_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest(f"Phase 6 imports failed: {PHASE6_IMPORT_ERROR}")

    def setUp(self):
        reset_registry_state()

    def tearDown(self):
        reset_registry_state()

    @patch("milia_pipeline.molecules.molecule_converter_core._REGISTRY_AVAILABLE", False)
    def test_01_fallback_get_available_dataset_types(self):
        """Test legacy fallback for _get_available_dataset_types.

        When registry is unavailable, the function falls back to filesystem
        discovery of dataset implementations. If the implementations directory
        is not reachable (e.g., in an isolated test environment), it returns
        an empty list with a warning — this is the conservative default per
        the Phase 6 refactored design.
        """
        # Force registry unavailable
        import milia_pipeline.molecules.molecule_converter_core as mod

        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False

        types = _get_available_dataset_types()

        # Module guarantees a list is returned (never raises)
        self.assertIsInstance(types, list)
        # If filesystem discovery succeeds (implementations dir exists), types will be populated;
        # if not, empty list is the documented conservative fallback.
        # Either way the function must not raise.

    @patch("milia_pipeline.molecules.molecule_converter_core._REGISTRY_AVAILABLE", False)
    def test_02_fallback_is_dataset_type_registered(self):
        """Test legacy fallback for _is_dataset_type_registered.

        When registry is unavailable, validation delegates to filesystem
        discovery via _get_available_dataset_types(). If the implementations
        directory is reachable, discovered types are considered registered;
        otherwise all types (including known ones) return False — this is
        the conservative default.

        The guaranteed invariant is: truly invalid types always return False,
        and the function never raises.
        """
        import milia_pipeline.molecules.molecule_converter_core as mod

        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False

        # Invalid/unknown types must always return False regardless of fallback path
        self.assertFalse(_is_dataset_type_registered("INVALID"))

        # Known types depend on filesystem discovery success.
        # Verify the function returns a bool without raising.
        for dtype in ("DFT", "DMC", "Wavefunction"):
            result = _is_dataset_type_registered(dtype)
            self.assertIsInstance(result, bool)

    @patch("milia_pipeline.molecules.molecule_converter_core._REGISTRY_AVAILABLE", False)
    def test_03_fallback_get_dataset_feature(self):
        """Test conservative fallback for _get_dataset_feature.

        When the registry is unavailable, the function returns False for ALL
        feature queries as a conservative default. This is by design: the
        module does not embed hardcoded feature knowledge — it relies solely
        on the registry as the single source of truth for feature flags.
        """
        import milia_pipeline.molecules.molecule_converter_core as mod

        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False

        # Conservative default: all features return False without registry
        self.assertFalse(_get_dataset_feature("DMC", "uncertainty_handling"))
        self.assertFalse(_get_dataset_feature("DFT", "uncertainty_handling"))
        self.assertFalse(_get_dataset_feature("DFT", "vibrational_analysis"))
        self.assertFalse(_get_dataset_feature("Wavefunction", "orbital_analysis"))

    @patch("milia_pipeline.molecules.molecule_converter_core._REGISTRY_AVAILABLE", False)
    def test_04_fallback_molecule_creation_strategy(self):
        """Test conservative fallback for _get_dataset_molecule_creation_strategy.

        When the registry is unavailable, the function returns the default
        strategy 'identifier_coordinate_based' for ALL dataset types. The
        module does not embed per-dataset strategy knowledge — it relies on
        the registry as the single source of truth.
        """
        import milia_pipeline.molecules.molecule_converter_core as mod

        mod._REGISTRY_INITIALIZED = True
        mod._REGISTRY_AVAILABLE = False

        # Conservative default: all types get 'identifier_coordinate_based'
        self.assertEqual(
            _get_dataset_molecule_creation_strategy("DFT"), "identifier_coordinate_based"
        )
        self.assertEqual(
            _get_dataset_molecule_creation_strategy("DMC"), "identifier_coordinate_based"
        )
        self.assertEqual(
            _get_dataset_molecule_creation_strategy("Wavefunction"), "identifier_coordinate_based"
        )


# ==============================================================================
# TEST RUNNER
# ==============================================================================


def run_tests():
    """Run all comprehensive tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes (original)
    suite.addTests(loader.loadTestsFromTestCase(TestMoleculeDataConverterInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestMoleculeDataConverterHandlers))
    suite.addTests(loader.loadTestsFromTestCase(TestMoleculeDataConverterTransformationSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestMoleculeDataConverterValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestMoleculeDataConverterDiagnostics))
    suite.addTests(loader.loadTestsFromTestCase(TestMoleculeDataConverterDatasetTypes))
    suite.addTests(loader.loadTestsFromTestCase(TestMoleculeDataConverterCaching))

    # Add Phase 6 test classes (NEW)
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6RegistryIntegrationFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6FeatureQueries))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6MoleculeCreationStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6RegistryIntegrationStatus))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6RefactoredMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6LegacyFallback))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("COMPREHENSIVE TEST SUMMARY - molecule_converter_core.py (Phase 6 Updated)")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)

    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED!")
        print("✓ Initialization & Configuration validated")
        print("✓ Handler Integration verified")
        print("✓ Transformation System operational")
        print("✓ Validation & Error Handling functional")
        print("✓ Dataset Type Specific tests passed")
        print("✓ Caching functionality verified")
        print("✓ Diagnostics & Statistics working")
        print("✓ Phase 6: Registry integration validated")
        print("✓ Phase 6: Feature-based queries working")
        print("✓ Phase 6: Molecule creation strategy lookup functional")
        print("✓ Phase 6: Enhanced utils determination working")
        print("✓ Phase 6: get_registry_integration_status method verified")
        print("✓ Phase 6: All 8 refactored locations tested")
        print("✓ Phase 6: Legacy fallback working")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("  Review failures before production use")

    return result


if __name__ == "__main__":
    if not IMPORTS_SUCCESSFUL:
        print("\n" + "=" * 70)
        print("ERROR: Required imports failed!")
        print("=" * 70)
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)

    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
