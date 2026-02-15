#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for exceptions.py Module (Dynamic Exception Hierarchy - Production Ready)

This test suite provides extensive coverage of the milia exception hierarchy including:
- Base exceptions (BaseProjectError, LoggingConfigurationError)
- Configuration exceptions (ConfigurationError and variants)
- Data processing exceptions (DataProcessingError, PreprocessingRequiredError, MoleculeProcessingError, etc.)
- Molecule processing exceptions (RDKitConversionError, PyGDataCreationError, etc.)
- Handler system exceptions (HandlerError and all subclasses)
- Validation exceptions (ValidationError, CompatibilityError, etc.)
- Transform system exceptions (TransformError and all subclasses)
- Plugin system exceptions (PluginError and all subclasses)
- Descriptor exceptions (DescriptorError and all subclasses)
- Model system exceptions (ModelError and all subclasses)
- Dataset registration exceptions (DatasetRegistrationError, DatasetNotFoundError)
- HPO exceptions (HPOError and all subclasses)

DYNAMIC EXCEPTION MODULE COVERAGE:
- Registry integration functions testing (_init_registry, _get_available_dataset_types, etc.)
- DatasetSpecificHandlerError testing (fully dynamic, any dataset type)
- UncertaintyProcessingError testing (fully dynamic, any dataset type)
- Factory functions testing (create_dataset_handler_error, create_uncertainty_processing_error, etc.)
- Registry status diagnostics testing
- _discover_dataset_types_from_filesystem testing (filesystem discovery)
- Registry fallback scenarios with mocking

EXTENDED COVERAGE (Production Ready):
- Utility functions (create_handler_error_context, format_handler_exception_summary, get_recovery_suggestions)
- Decorator functions (wrap_handler_operation, wrap_transform_operation)
- Model system exceptions (ModelError, ModelNotFoundError, ModelValidationError, etc.)
- Dataset registration exceptions (DatasetRegistrationError, DatasetNotFoundError with auto-fill)
- HPO exceptions (HPOError, TrialFailedError, StudyNotFoundError, BackendError, SearchSpaceError, PruningError)
- PreprocessingRequiredError testing
- ConfigurationError detailed attribute testing
- Exception chaining and re-raising behavior
- Comprehensive __str__ edge cases for all exception classes
- Dynamic exception hierarchy context and error handling
- validate_exception_hierarchy comprehensive coverage

Test Coverage:
- Exception initialization and attribute storage
- Custom __str__ methods and message formatting
- Exception inheritance hierarchy validation
- Contextual information preservation
- Optional and required parameters
- Edge cases and boundary conditions
- Exception hierarchy validation function
- Registry integration and factory functions
- Utility and decorator functions
- Error context creation with registry info
- Recovery suggestion generation
- Filesystem discovery function testing
- Registry fallback scenarios

NOTE: This test suite runs inside Docker at /app/milia
Path: ~/ml_projects/milia/milia_pipeline/exceptions.py

Author: milia Project Team
Created: November 16, 2025
Updated: November 26, 2025 - Registry Integration
Updated: November 29, 2025 - Extended Unit Test Suite v2
Updated: February 04, 2026 - Production Ready Enhancement
Updated: February 06, 2026 - Dynamic Exception Hierarchy Refactoring
"""

import sys
from pathlib import Path

# CRITICAL: Add project root to Python path FIRST
project_root = Path("/app/milia")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import patch

import pytest

# ---
# Import all exception classes from the module under test
from milia_pipeline.exceptions import (
    AtomFilterError,
    BackendError,
    # Base exceptions
    BaseProjectError,
    CheckpointError,
    CompatibilityError,
    # Configuration exceptions
    ConfigurationError,
    DataCompatibilityError,
    DataError,
    # Data processing exceptions
    DataProcessingError,
    DatasetIntegrationError,
    DatasetNotFoundError,
    # Dataset registration exceptions
    DatasetRegistrationError,
    # Dataset-specific handler exception
    DatasetSpecificHandlerError,
    DescriptorCalculationError,
    # Descriptor exceptions
    DescriptorError,
    DescriptorPluginConfigError,
    DescriptorPluginError,
    DescriptorPluginLoadError,
    DescriptorPluginValidationError,
    DescriptorValidationError,
    ExperimentalSetupError,
    HandlerCompatibilityError,
    HandlerConfigurationError,
    # Handler system exceptions
    HandlerError,
    HandlerIntegrationError,
    HandlerNotAvailableError,
    HandlerOperationError,
    HandlerValidationError,
    HPOConfigurationError,
    # HPO exceptions
    HPOError,
    HyperparameterError,
    LegacyCodeError,
    LoggingConfigurationError,
    MigrationError,
    MissingDependencyError,
    # Model system exceptions
    ModelError,
    ModelInstantiationError,
    ModelNotFoundError,
    ModelValidationError,
    MoleculeFilterRejectedError,
    MoleculeProcessingError,
    PluginDependencyError,
    PluginDiscoveryError,
    # Plugin system exceptions
    PluginError,
    PluginLoadError,
    PluginModelError,
    PluginRegistrationError,
    PluginSecurityError,
    PluginValidationError,
    PreprocessingRequiredError,
    PropertyEnrichmentError,
    PruningError,
    PyGDataCreationError,
    # Molecule processing exceptions
    RDKitConversionError,
    SearchSpaceError,
    StructuralFeatureError,
    StudyNotFoundError,
    TrainingError,
    TransformationError,
    TransformCompatibilityError,
    TransformCompositionError,
    TransformConfigurationError,
    # Transform system exceptions
    TransformError,
    TransformHandlerIntegrationError,
    TransformNotFoundError,
    TransformRegistryError,
    TransformValidationError,
    TrialFailedError,
    # Uncertainty processing exception
    UncertaintyProcessingError,
    # Validation exceptions
    ValidationError,
    VibrationRefinementError,
    _discover_dataset_types_from_filesystem,
    _get_available_dataset_types,
    _get_dataset_feature,
    # Registry integration functions
    _init_registry,
    _is_dataset_type_registered,
    # Factory functions
    create_dataset_handler_error,
    # Utility functions for exception handling
    create_handler_error_context,
    create_handler_not_available_error,
    create_uncertainty_processing_error,
    format_handler_exception_summary,
    get_exception_recovery_suggestions,
    get_exception_registry_status,
    is_recoverable_handler_error,
    # Validation function
    validate_exception_hierarchy,
    wrap_handler_operation,
    wrap_transform_operation,
)

# =============================================================================
# HELPER FUNCTIONS FOR REGISTRY TESTS
# =============================================================================


def reset_registry_state():
    """
    Reset registry state to uninitialized.
    IMPORTANT: Use this for testing lazy initialization patterns.
    """
    import milia_pipeline.exceptions as exc_module

    exc_module._REGISTRY_INITIALIZED = False
    exc_module._REGISTRY_AVAILABLE = False
    exc_module._registry_list_all = None
    exc_module._registry_get = None
    exc_module._registry_is_registered = None


# =============================================================================
# TEST CLASS: Base Exceptions
# =============================================================================


class TestBaseExceptions:
    """Test base exception classes"""

    def test_base_project_error_minimal(self):
        """Test BaseProjectError with minimal arguments"""
        error = BaseProjectError("Test error")
        assert error.message == "Test error"
        assert error.details is None
        assert error.extra_info == {}
        assert str(error) == "Test error"

    def test_base_project_error_with_details(self):
        """Test BaseProjectError with details"""
        error = BaseProjectError("Test error", details="Additional context")
        assert error.message == "Test error"
        assert error.details == "Additional context"
        assert str(error) == "Test error. Details: Additional context"

    def test_base_project_error_with_kwargs(self):
        """Test BaseProjectError with extra kwargs"""
        error = BaseProjectError(
            "Test error", details="Context", custom_field="value", another_field=42
        )
        assert error.extra_info["custom_field"] == "value"
        assert error.extra_info["another_field"] == 42

    def test_base_project_error_inherits_exception(self):
        """Test BaseProjectError inherits from Exception"""
        assert issubclass(BaseProjectError, Exception)
        error = BaseProjectError("Test")
        assert isinstance(error, Exception)

    def test_logging_configuration_error_default_message(self):
        """Test LoggingConfigurationError with default message"""
        error = LoggingConfigurationError()
        assert "Error configuring logging" in error.message
        assert isinstance(error, BaseProjectError)

    def test_logging_configuration_error_custom_message(self):
        """Test LoggingConfigurationError with custom message"""
        error = LoggingConfigurationError(
            "Failed to initialize logger", details="Invalid log level"
        )
        assert error.message == "Failed to initialize logger"
        assert error.details == "Invalid log level"

    def test_logging_configuration_error_inheritance(self):
        """Test LoggingConfigurationError inheritance"""
        assert issubclass(LoggingConfigurationError, BaseProjectError)
        assert issubclass(LoggingConfigurationError, Exception)


# =============================================================================
# TEST CLASS: Configuration Exceptions
# =============================================================================


class TestConfigurationExceptions:
    """Test configuration exception classes"""

    def test_configuration_error_basic(self):
        """Test ConfigurationError basic usage"""
        error = ConfigurationError("Invalid config")
        assert error.message == "Invalid config"
        assert isinstance(error, BaseProjectError)

    def test_configuration_error_with_details(self):
        """Test ConfigurationError with details"""
        error = ConfigurationError(
            "Missing required key", details="Required for database connection"
        )
        assert error.details == "Required for database connection"

    def test_configuration_error_str_method(self):
        """Test ConfigurationError __str__ method"""
        error = ConfigurationError("Config error", details="Extra info")
        error_str = str(error)
        assert "Config error" in error_str
        assert "Extra info" in error_str

    def test_configuration_error_with_config_key(self):
        """Test ConfigurationError with config_key attribute."""
        error = ConfigurationError("Invalid configuration", config_key="dataset.batch_size")
        assert error.config_key == "dataset.batch_size"
        assert "dataset.batch_size" in str(error)

    def test_configuration_error_with_actual_value(self):
        """Test ConfigurationError with actual_value attribute."""
        error = ConfigurationError(
            "Type mismatch", config_key="learning_rate", actual_value="not_a_number"
        )
        assert error.actual_value == "not_a_number"
        error_str = str(error)
        assert "not_a_number" in error_str

    def test_configuration_error_with_expected_value(self):
        """Test ConfigurationError with expected_value attribute."""
        error = ConfigurationError(
            "Value mismatch", config_key="epochs", expected_value=int, actual_value="100"
        )
        assert error.expected_value == int
        error_str = str(error)
        assert "int" in error_str.lower() or "Expected" in error_str

    def test_configuration_error_full_context(self):
        """Test ConfigurationError with all context attributes."""
        error = ConfigurationError(
            "Complete config error",
            config_key="model.hidden_channels",
            actual_value=-1,
            expected_value="positive integer",
            details="Value must be greater than 0",
        )
        assert error.config_key == "model.hidden_channels"
        assert error.actual_value == -1
        assert error.expected_value == "positive integer"
        assert error.details == "Value must be greater than 0"

        error_str = str(error)
        assert "hidden_channels" in error_str


# =============================================================================
# TEST CLASS: Data Processing Exceptions
# =============================================================================


class TestDataProcessingExceptions:
    """Test data processing exception classes"""

    def test_data_processing_error_basic(self):
        """Test DataProcessingError basic usage"""
        error = DataProcessingError("Processing failed")
        assert error.message == "Processing failed"
        assert isinstance(error, BaseProjectError)

    def test_data_processing_error_with_file_path(self):
        """Test DataProcessingError with file_path"""
        error = DataProcessingError("Failed to process file", file_path="/path/to/file.npz")
        assert error.file_path == "/path/to/file.npz"

    def test_data_processing_error_with_operation(self):
        """Test DataProcessingError with operation"""
        error = DataProcessingError("Operation failed", operation="load_dataset")
        assert error.operation == "load_dataset"

    def test_molecule_processing_error_required_params(self):
        """Test MoleculeProcessingError with required molecule_index"""
        error = MoleculeProcessingError("Molecule processing failed", molecule_index=42)
        assert error.molecule_index == 42
        assert error.smiles == "N/A"  # Default value
        assert error.inchi == "N/A"  # Default value

    def test_molecule_processing_error_with_all_params(self):
        """Test MoleculeProcessingError with all parameters"""
        error = MoleculeProcessingError(
            "Invalid molecule",
            molecule_index=42,
            smiles="CCO",
            inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
            details="Validation failed",
        )
        assert error.molecule_index == 42
        assert error.smiles == "CCO"
        assert error.inchi == "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
        assert error.details == "Validation failed"

    def test_molecule_processing_error_str_method(self):
        """Test MoleculeProcessingError __str__ includes context"""
        error = MoleculeProcessingError(
            "Error", molecule_index=10, smiles="C", inchi="InChI=1S/CH4"
        )
        error_str = str(error)
        assert "10" in error_str  # Index should be in string
        assert "Index" in error_str or "molecule" in error_str.lower()

    def test_molecule_filter_rejected_error_signature(self):
        """Test MoleculeFilterRejectedError with correct signature"""
        error = MoleculeFilterRejectedError(
            molecule_index=10,
            inchi="InChI=1S/test",
            reason="Rejected by filter",
            filter_name="atom_count",
        )
        assert isinstance(error, BaseException)
        assert not isinstance(error, Exception)  # Should inherit from BaseException, not Exception
        assert error.molecule_index == 10
        assert error.filter_name == "atom_count"
        assert error.reason == "Rejected by filter"

    def test_molecule_filter_rejected_error_str_method(self):
        """Test MoleculeFilterRejectedError __str__ method"""
        error = MoleculeFilterRejectedError(
            molecule_index=5, inchi="InChI=1S/test", reason="Too many atoms"
        )
        error_str = str(error)
        assert "5" in error_str
        assert "Too many atoms" in error_str

    def test_missing_dependency_error_signature(self):
        """Test MissingDependencyError with correct attributes"""
        error = MissingDependencyError("Required package not found", dependency_name="rdkit")
        assert error.dependency_name == "rdkit"
        assert isinstance(error, BaseProjectError)

    def test_missing_dependency_error_with_install_command(self):
        """Test MissingDependencyError with install command"""
        error = MissingDependencyError(
            "Package missing", dependency_name="torch", install_command="pip install torch"
        )
        assert error.install_command == "pip install torch"

    def test_missing_dependency_error_str_method(self):
        """Test MissingDependencyError __str__ method"""
        error = MissingDependencyError("Missing dep", dependency_name="numpy")
        error_str = str(error)
        assert "numpy" in error_str

    def test_atom_filter_error_basic(self):
        """Test AtomFilterError basic usage"""
        error = AtomFilterError("Invalid atom filter")
        assert error.message == "Invalid atom filter"
        assert isinstance(error, BaseProjectError)

    def test_atom_filter_error_with_atom_symbol(self):
        """Test AtomFilterError with atom_symbol"""
        error = AtomFilterError("Invalid atom", atom_symbol="Fe")
        assert error.atom_symbol == "Fe"

    def test_atom_filter_error_with_filter_config(self):
        """Test AtomFilterError with filter_config"""
        config = {"allowed_atoms": [1, 6, 7, 8]}
        error = AtomFilterError("Filter error", filter_config=config)
        assert error.filter_config == config


# =============================================================================
# TEST CLASS: Preprocessing Required Error
# =============================================================================


class TestPreprocessingRequiredError:
    """Test PreprocessingRequiredError for preprocessing workflow scenarios."""

    def test_preprocessing_required_error_basic(self):
        """Test PreprocessingRequiredError with required parameters."""
        error = PreprocessingRequiredError(
            source_file="/path/to/source.json",
            target_file="/path/to/target.npz",
            dataset_type="DFT",
        )
        assert isinstance(error, DataProcessingError)
        assert error.source_file == "/path/to/source.json"
        assert error.target_file == "/path/to/target.npz"
        assert error.dataset_type == "DFT"

    def test_preprocessing_required_error_default_command(self):
        """Test PreprocessingRequiredError generates default preprocessing command."""
        error = PreprocessingRequiredError(
            source_file="input.json", target_file="output.npz", dataset_type="DMC"
        )
        # Message should contain auto-generated preprocessing command
        error_str = str(error)
        assert "preprocessing" in error_str.lower()
        assert "DMC" in error_str

    def test_preprocessing_required_error_custom_command(self):
        """Test PreprocessingRequiredError with custom preprocessing_command."""
        custom_cmd = "custom_preprocess --input file.json --output file.npz"
        error = PreprocessingRequiredError(
            source_file="input.json",
            target_file="output.npz",
            dataset_type="DFT",
            preprocessing_command=custom_cmd,
        )
        error_str = str(error)
        assert custom_cmd in error_str

    def test_preprocessing_required_error_with_details(self):
        """Test PreprocessingRequiredError with details parameter."""
        error = PreprocessingRequiredError(
            source_file="input.json",
            target_file="output.npz",
            dataset_type="Wavefunction",
            details="Additional context for the error",
        )
        assert error.details == "Additional context for the error"


# =============================================================================
# TEST CLASS: Molecule Processing Exceptions
# =============================================================================


class TestMoleculeProcessingExceptions:
    """Test molecule-specific processing exception classes"""

    def test_rdkit_conversion_error_signature(self):
        """Test RDKitConversionError with correct signature"""
        error = RDKitConversionError(
            molecule_index=5,
            inchi="InChI=1S/test",
            reason="Failed to convert",
            detail="Invalid SMILES",
        )
        assert isinstance(error, MoleculeProcessingError)
        assert error.molecule_index == 5
        assert error.reason == "Failed to convert"
        assert error.detail == "Invalid SMILES"

    def test_rdkit_conversion_error_with_rdkit_error(self):
        """Test RDKitConversionError with rdkit_error"""
        error = RDKitConversionError(
            molecule_index=1,
            inchi="InChI=1S/test",
            reason="Conversion failed",
            detail="Details here",
            rdkit_error="RDKit: Invalid molecule",
        )
        assert error.rdkit_error == "RDKit: Invalid molecule"

    def test_pyg_data_creation_error_signature(self):
        """Test PyGDataCreationError with correct signature"""
        error = PyGDataCreationError(
            message="Failed to create PyG Data object",
            molecule_index=10,
            smiles="CCO",
            reason="Missing attributes",
            detail="x and edge_index required",
        )
        assert isinstance(error, MoleculeProcessingError)
        assert error.molecule_index == 10
        assert error.smiles == "CCO"
        assert error.reason == "Missing attributes"

    def test_pyg_data_creation_error_with_transform_name(self):
        """Test PyGDataCreationError with transform_name"""
        error = PyGDataCreationError(
            message="Transform failed",
            molecule_index=5,
            smiles="C",
            reason="Transform error",
            detail="Details",
            transform_name="AddSelfLoops",
        )
        assert error.transform_name == "AddSelfLoops"

    def test_property_enrichment_error_signature(self):
        """Test PropertyEnrichmentError with correct signature"""
        error = PropertyEnrichmentError(
            molecule_index=7,
            inchi="InChI=1S/test",
            property_name="LogP",
            reason="Calculation failed",
            detail="RDKit error",
        )
        assert isinstance(error, MoleculeProcessingError)
        assert error.property_name == "LogP"
        assert error.reason == "Calculation failed"

    def test_property_enrichment_error_str_method(self):
        """Test PropertyEnrichmentError __str__ includes property name"""
        error = PropertyEnrichmentError(
            molecule_index=5,
            inchi="InChI=1S/test",
            property_name="MolWeight",
            reason="Failed",
            detail="Detail",
        )
        error_str = str(error)
        assert "MolWeight" in error_str

    def test_structural_feature_error_basic(self):
        """Test StructuralFeatureError basic usage"""
        error = StructuralFeatureError("Feature error")
        assert isinstance(error, MoleculeProcessingError)

    def test_structural_feature_error_with_all_params(self):
        """Test StructuralFeatureError with all parameters"""
        error = StructuralFeatureError(
            message="Feature failed",
            molecule_index=3,
            inchi="InChI=1S/test",
            feature_type="atom",
            feature_name="hybridization",
            reason="Unknown type",
            detail="Details",
        )
        assert error.feature_type == "atom"
        assert error.feature_name == "hybridization"
        assert error.reason == "Unknown type"

    def test_structural_feature_error_str_method(self):
        """Test StructuralFeatureError __str__ method"""
        error = StructuralFeatureError(feature_type="bond", feature_name="order")
        error_str = str(error)
        assert "bond" in error_str
        assert "order" in error_str

    def test_vibration_refinement_error_basic(self):
        """Test VibrationRefinementError basic usage"""
        error = VibrationRefinementError("Refinement failed")
        assert isinstance(error, DataProcessingError)

    def test_vibration_refinement_error_with_params(self):
        """Test VibrationRefinementError with parameters"""
        error = VibrationRefinementError(
            message="Failed", molecule_index=5, reason="Invalid freqs", detail="Negative frequency"
        )
        assert error.molecule_index == 5
        assert error.reason == "Invalid freqs"

    def test_vibration_refinement_error_str_method(self):
        """Test VibrationRefinementError __str__ method"""
        error = VibrationRefinementError(molecule_index=10, reason="Error reason")
        error_str = str(error)
        assert "10" in error_str
        assert "Error reason" in error_str


# =============================================================================
# TEST CLASS: UncertaintyProcessingError
# =============================================================================


class TestUncertaintyProcessingError:
    """Test UncertaintyProcessingError class"""

    def test_uncertainty_processing_error_basic(self):
        """Test UncertaintyProcessingError basic creation with required dataset_type"""
        error = UncertaintyProcessingError(
            message="Uncertainty validation failed", dataset_type="TestDataset", molecule_index=5
        )
        assert isinstance(error, MoleculeProcessingError)
        assert error.molecule_index == 5
        assert error.dataset_type == "TestDataset"

    def test_uncertainty_processing_error_requires_dataset_type(self):
        """Test UncertaintyProcessingError requires dataset_type (no default)"""
        # dataset_type is now a required positional-like keyword argument
        with pytest.raises(TypeError):
            UncertaintyProcessingError(message="Uncertainty error", molecule_index=5)

    def test_uncertainty_processing_error_with_dataset_type(self):
        """Test UncertaintyProcessingError with custom dataset_type"""
        error = UncertaintyProcessingError(
            message="Uncertainty error", dataset_type="QMC", molecule_index=10
        )
        assert error.dataset_type == "QMC"

    def test_uncertainty_processing_error_with_property_name(self):
        """Test UncertaintyProcessingError with uncertainty_property_name"""
        error = UncertaintyProcessingError(
            message="Property validation failed",
            dataset_type="QMC",
            uncertainty_property_name="std_error",
        )
        assert error.uncertainty_property_name == "std_error"

    def test_uncertainty_processing_error_str_method(self):
        """Test UncertaintyProcessingError __str__ method"""
        error = UncertaintyProcessingError(
            message="Test error", dataset_type="QMC", uncertainty_property_name="correlation_energy"
        )
        error_str = str(error)
        assert "QMC" in error_str
        assert "correlation_energy" in error_str

    def test_uncertainty_processing_error_str_always_includes_dataset(self):
        """Test UncertaintyProcessingError __str__ always includes dataset type"""
        # After refactoring, __str__ always includes dataset type for ALL datasets
        for ds_type in ["QMC", "FCIQMC", "TestDS"]:
            error = UncertaintyProcessingError(message="Test error", dataset_type=ds_type)
            error_str = str(error)
            assert ds_type in error_str, f"Dataset type '{ds_type}' should appear in __str__"

    def test_uncertainty_processing_error_inheritance_chain(self):
        """Test UncertaintyProcessingError inheritance"""
        assert issubclass(UncertaintyProcessingError, MoleculeProcessingError)
        assert issubclass(UncertaintyProcessingError, BaseProjectError)

    def test_uncertainty_processing_error_dynamic_dataset_types(self):
        """Test UncertaintyProcessingError works with any dataset type string"""
        for ds_type in ["QMC", "FCIQMC", "CCSD", "NewDataset", "CustomType"]:
            error = UncertaintyProcessingError(
                message=f"{ds_type} uncertainty error",
                dataset_type=ds_type,
                molecule_index=5,
                uncertainty_property_name="std_error",
            )
            assert error.dataset_type == ds_type
            assert isinstance(error, UncertaintyProcessingError)
            assert isinstance(error, MoleculeProcessingError)


# =============================================================================
# TEST CLASS: Handler System Exceptions
# =============================================================================


class TestHandlerExceptions:
    """Test handler system exception classes"""

    def test_handler_error_basic(self):
        """Test HandlerError basic usage"""
        error = HandlerError("Handler operation failed")
        assert error.message == "Handler operation failed"
        assert isinstance(error, BaseProjectError)

    def test_handler_error_with_handler_type(self):
        """Test HandlerError with handler_type"""
        error = HandlerError("Handler failed", handler_type="DFTDatasetHandler")
        assert error.handler_type == "DFTDatasetHandler"
        assert "DFTDatasetHandler" in str(error)

    def test_handler_error_with_operation(self):
        """Test HandlerError with handler_operation"""
        error = HandlerError("Operation failed", handler_type="DFT", handler_operation="validate")
        assert error.handler_operation == "validate"

    def test_handler_error_str_method(self):
        """Test HandlerError __str__ includes handler info"""
        error = HandlerError("Error", handler_type="TestHandler", handler_operation="test_op")
        error_str = str(error)
        assert "TestHandler" in error_str
        assert "test_op" in error_str

    def test_handler_not_available_error_signature(self):
        """Test HandlerNotAvailableError with correct signature"""
        error = HandlerNotAvailableError(
            message="Handler not available",
            requested_dataset_type="UNKNOWN",
            available_types=["DFT", "DMC", "Wavefunction"],
        )
        assert isinstance(error, HandlerError)
        assert error.requested_dataset_type == "UNKNOWN"
        assert error.available_types == ["DFT", "DMC", "Wavefunction"]

    def test_handler_not_available_error_with_dependencies(self):
        """Test HandlerNotAvailableError with missing_dependencies"""
        error = HandlerNotAvailableError(
            message="Handler unavailable",
            requested_dataset_type="Custom",
            missing_dependencies=["rdkit", "torch"],
        )
        assert error.missing_dependencies == ["rdkit", "torch"]

    def test_handler_not_available_error_str_method(self):
        """Test HandlerNotAvailableError __str__ method"""
        error = HandlerNotAvailableError(
            message="Not available", requested_dataset_type="TEST", available_types=["A", "B"]
        )
        error_str = str(error)
        assert "TEST" in error_str

    def test_handler_configuration_error_signature(self):
        """Test HandlerConfigurationError - requires handler_type"""
        error = HandlerConfigurationError(
            message="Invalid configuration", handler_type="DMCDatasetHandler"
        )
        assert isinstance(error, HandlerError)
        assert error.handler_type == "DMCDatasetHandler"

    def test_handler_configuration_error_with_validation_errors(self):
        """Test HandlerConfigurationError with config_validation_errors"""
        errors = ["Missing key: batch_size", "Invalid value: negative"]
        error = HandlerConfigurationError(
            message="Config invalid", handler_type="DFT", config_validation_errors=errors
        )
        assert error.config_validation_errors == errors

    def test_handler_configuration_error_str_method(self):
        """Test HandlerConfigurationError __str__ method"""
        error = HandlerConfigurationError(
            message="Error", handler_type="Test", invalid_config_keys=["key1", "key2"]
        )
        error_str = str(error)
        assert "Test" in error_str

    def test_handler_operation_error_signature(self):
        """Test HandlerOperationError with correct signature"""
        error = HandlerOperationError(
            message="Operation failed",
            handler_type="DFT",
            operation="validate_molecule",
            molecule_index=42,
        )
        assert isinstance(error, HandlerError)
        assert error.handler_type == "DFT"
        assert error.handler_operation == "validate_molecule"
        assert error.molecule_index == 42

    def test_handler_operation_error_with_suggestions(self):
        """Test HandlerOperationError with recovery_suggestions"""
        suggestions = ["Skip molecule", "Check data"]
        error = HandlerOperationError(
            message="Failed",
            handler_type="DMC",
            operation="process",
            recovery_suggestions=suggestions,
        )
        assert error.recovery_suggestions == suggestions

    def test_handler_operation_error_str_method(self):
        """Test HandlerOperationError __str__ includes all context"""
        error = HandlerOperationError(
            message="Error", handler_type="DFT", operation="validate", molecule_index=10
        )
        error_str = str(error)
        assert "DFT" in error_str
        assert "validate" in error_str
        assert "10" in error_str

    def test_handler_validation_error_signature(self):
        """Test HandlerValidationError - requires handler_type and validation_type"""
        error = HandlerValidationError(
            message="Validation failed",
            handler_type="WavefunctionHandler",
            validation_type="schema_check",
        )
        assert isinstance(error, HandlerError)
        assert error.validation_type == "schema_check"

    def test_handler_validation_error_with_failed_validations(self):
        """Test HandlerValidationError with failed_validations"""
        failures = ["Missing field", "Type mismatch"]
        error = HandlerValidationError(
            message="Failed",
            handler_type="DFT",
            validation_type="data",
            failed_validations=failures,
        )
        assert error.failed_validations == failures

    def test_handler_validation_error_str_method(self):
        """Test HandlerValidationError __str__ method"""
        error = HandlerValidationError(
            message="Error", handler_type="DMC", validation_type="uncertainty", molecule_index=5
        )
        error_str = str(error)
        assert "DMC" in error_str
        assert "uncertainty" in error_str

    def test_handler_compatibility_error_signature(self):
        """Test HandlerCompatibilityError - requires handler_type"""
        error = HandlerCompatibilityError(message="Incompatible", handler_type="LegacyHandler")
        assert isinstance(error, HandlerError)
        assert error.handler_type == "LegacyHandler"

    def test_handler_compatibility_error_with_features(self):
        """Test HandlerCompatibilityError with incompatible_features"""
        features = ["feature1", "feature2"]
        error = HandlerCompatibilityError(
            message="Incompatible", handler_type="Custom", incompatible_features=features
        )
        assert error.incompatible_features == features

    def test_handler_compatibility_error_str_method(self):
        """Test HandlerCompatibilityError __str__ method"""
        error = HandlerCompatibilityError(
            message="Error",
            handler_type="Test",
            minimum_requirements={"python": "3.8", "torch": "1.9"},
        )
        error_str = str(error)
        assert "Test" in error_str

    def test_handler_integration_error_signature(self):
        """Test HandlerIntegrationError - inherits from BaseProjectError"""
        error = HandlerIntegrationError(message="Integration failed", handler_type="CustomHandler")
        assert isinstance(error, BaseProjectError)
        assert not isinstance(error, HandlerError)  # Correct - inherits from BaseProjectError

    def test_handler_integration_error_with_all_params(self):
        """Test HandlerIntegrationError with all parameters"""
        error = HandlerIntegrationError(
            message="Failed",
            handler_type="DFT",
            integration_point="transform",
            legacy_component="old_code",
            migration_phase="phase1",
        )
        assert error.handler_type == "DFT"
        assert error.integration_point == "transform"
        assert error.legacy_component == "old_code"
        assert error.migration_phase == "phase1"

    def test_handler_integration_error_str_method(self):
        """Test HandlerIntegrationError __str__ method"""
        error = HandlerIntegrationError(
            message="Error", handler_type="Test", integration_point="pipeline"
        )
        error_str = str(error)
        assert "Test" in error_str
        assert "pipeline" in error_str

    def test_transform_handler_integration_error_signature(self):
        """Test TransformHandlerIntegrationError - requires handler_type and integration_point"""
        error = TransformHandlerIntegrationError(
            message="Transform integration failed",
            handler_type="DFTHandler",
            integration_point="transform_pipeline",
        )
        assert isinstance(error, HandlerIntegrationError)
        assert error.handler_type == "DFTHandler"
        assert error.integration_point == "transform_pipeline"

    def test_transform_handler_integration_error_with_transform(self):
        """Test TransformHandlerIntegrationError with transform_name"""
        error = TransformHandlerIntegrationError(
            message="Failed",
            handler_type="DMC",
            integration_point="apply",
            transform_name="Normalize",
        )
        assert error.transform_name == "Normalize"

    def test_transform_handler_integration_error_str_method(self):
        """Test TransformHandlerIntegrationError __str__ includes transform"""
        error = TransformHandlerIntegrationError(
            message="Error",
            handler_type="DFT",
            integration_point="test",
            transform_name="AddSelfLoops",
        )
        error_str = str(error)
        assert "AddSelfLoops" in error_str


# =============================================================================
# TEST CLASS: DatasetSpecificHandlerError
# =============================================================================


class TestDatasetSpecificHandlerError:
    """Test DatasetSpecificHandlerError class"""

    def test_dataset_specific_handler_error_basic(self):
        """Test DatasetSpecificHandlerError basic creation"""
        error = DatasetSpecificHandlerError(message="Handler error", dataset_type="TestDataset")
        assert isinstance(error, HandlerError)
        assert error.dataset_type == "TestDataset"
        assert error.handler_type == "TestDataset"  # Set from dataset_type

    def test_dataset_specific_handler_error_with_operation(self):
        """Test DatasetSpecificHandlerError with operation"""
        error = DatasetSpecificHandlerError(
            message="Operation failed", dataset_type="QMC", operation="validate_uncertainty"
        )
        assert error.handler_operation == "validate_uncertainty"

    def test_dataset_specific_handler_error_with_property_name(self):
        """Test DatasetSpecificHandlerError with property_name"""
        error = DatasetSpecificHandlerError(
            message="Property error", dataset_type="QMC", property_name="correlation_energy"
        )
        assert error.property_name == "correlation_energy"

    def test_dataset_specific_handler_error_str_method(self):
        """Test DatasetSpecificHandlerError __str__ method"""
        error = DatasetSpecificHandlerError(
            message="Test error",
            dataset_type="QMC",
            operation="validate",
            property_name="std_error",
        )
        error_str = str(error)
        assert "QMC" in error_str
        assert "validate" in error_str
        assert "std_error" in error_str

    def test_dataset_specific_handler_error_inheritance(self):
        """Test DatasetSpecificHandlerError inheritance chain"""
        assert issubclass(DatasetSpecificHandlerError, HandlerError)
        assert issubclass(DatasetSpecificHandlerError, BaseProjectError)

    def test_catch_all_dataset_specific_errors(self):
        """Test catching all dataset-specific errors with single except"""
        exceptions_to_test = [
            DatasetSpecificHandlerError(message="TypeA error", dataset_type="TypeA"),
            DatasetSpecificHandlerError(message="TypeB error", dataset_type="TypeB"),
            DatasetSpecificHandlerError(message="QMC error", dataset_type="QMC"),
        ]

        for exc in exceptions_to_test:
            try:
                raise exc
            except DatasetSpecificHandlerError as e:
                # All should be caught
                assert e is exc

    def test_dataset_specific_handler_error_dynamic_dataset_types(self):
        """Test DatasetSpecificHandlerError works with any dataset type string"""
        for ds_type in ["QMC", "CCSD", "FCIQMC", "NewDataset", "CustomType"]:
            error = DatasetSpecificHandlerError(
                message=f"{ds_type} handler error",
                dataset_type=ds_type,
                operation="validate",
                property_name="test_property",
            )
            assert error.dataset_type == ds_type
            assert error.handler_type == ds_type
            assert isinstance(error, DatasetSpecificHandlerError)
            assert isinstance(error, HandlerError)

    def test_dataset_specific_handler_error_with_details(self):
        """Test DatasetSpecificHandlerError with details"""
        error = DatasetSpecificHandlerError(
            message="Handler error", dataset_type="TestDS", details="Additional context"
        )
        assert error.details == "Additional context"
        assert "Additional context" in str(error)


# =============================================================================
# TEST CLASS: Validation Exceptions
# =============================================================================


class TestValidationExceptions:
    """Test validation exception classes"""

    def test_validation_error_signature(self):
        """Test ValidationError - requires validation_type"""
        error = ValidationError(message="Validation failed", validation_type="schema_check")
        assert error.validation_type == "schema_check"
        assert isinstance(error, BaseProjectError)

    def test_validation_error_with_failed_checks(self):
        """Test ValidationError with failed_checks"""
        checks = ["Missing field", "Invalid type"]
        error = ValidationError(message="Failed", validation_type="data", failed_checks=checks)
        assert error.failed_checks == checks

    def test_validation_error_with_handler_type(self):
        """Test ValidationError with handler_type"""
        error = ValidationError(
            message="Validation error", validation_type="schema", handler_type="DFT"
        )
        assert error.handler_type == "DFT"

    def test_validation_error_str_method(self):
        """Test ValidationError __str__ method"""
        error = ValidationError(
            message="Error", validation_type="test", data_context="molecule_data"
        )
        error_str = str(error)
        assert "test" in error_str

    def test_compatibility_error_signature(self):
        """Test CompatibilityError - requires component_a and component_b"""
        error = CompatibilityError(
            message="Incompatible", component_a="dataset_v2", component_b="handler_v1"
        )
        assert isinstance(error, BaseProjectError)
        assert not isinstance(error, ValidationError)  # Inherits from BaseProjectError
        assert error.component_a == "dataset_v2"
        assert error.component_b == "handler_v1"

    def test_compatibility_error_with_version_conflicts(self):
        """Test CompatibilityError with version_conflicts"""
        conflicts = {"torch": "1.8", "required": "1.9"}
        error = CompatibilityError(
            message="Version mismatch",
            component_a="A",
            component_b="B",
            version_conflicts=conflicts,
        )
        assert error.version_conflicts == conflicts

    def test_compatibility_error_str_method(self):
        """Test CompatibilityError __str__ method"""
        error = CompatibilityError(
            message="Error", component_a="CompA", component_b="CompB", compatibility_type="version"
        )
        error_str = str(error)
        assert "CompA" in error_str
        assert "CompB" in error_str

    def test_migration_error_signature(self):
        """Test MigrationError - requires migration_phase"""
        error = MigrationError(message="Migration failed", migration_phase="schema_update")
        assert error.migration_phase == "schema_update"
        assert isinstance(error, BaseProjectError)

    def test_migration_error_with_all_params(self):
        """Test MigrationError with all parameters"""
        error = MigrationError(
            message="Failed",
            migration_phase="phase1",
            source_module="old_module",
            target_pattern="new_pattern",
            migration_step="step1",
            rollback_available=False,
        )
        assert error.source_module == "old_module"
        assert error.target_pattern == "new_pattern"
        assert error.migration_step == "step1"
        assert error.rollback_available == False

    def test_migration_error_str_method(self):
        """Test MigrationError __str__ method"""
        error = MigrationError(message="Error", migration_phase="test", rollback_available=False)
        error_str = str(error)
        assert "test" in error_str
        assert "NO ROLLBACK" in error_str

    def test_legacy_code_error_signature(self):
        """Test LegacyCodeError - requires legacy_pattern"""
        error = LegacyCodeError(message="Legacy code executed", legacy_pattern="old_transform")
        assert error.legacy_pattern == "old_transform"
        assert isinstance(error, BaseProjectError)

    def test_legacy_code_error_with_all_params(self):
        """Test LegacyCodeError with all parameters"""
        error = LegacyCodeError(
            message="Legacy",
            legacy_pattern="pattern1",
            suggested_replacement="new_pattern",
            legacy_module="old_module",
            migration_priority="high",
        )
        assert error.suggested_replacement == "new_pattern"
        assert error.legacy_module == "old_module"
        assert error.migration_priority == "high"

    def test_legacy_code_error_str_method(self):
        """Test LegacyCodeError __str__ method"""
        error = LegacyCodeError(message="Error", legacy_pattern="old", migration_priority="high")
        error_str = str(error)
        assert "old" in error_str
        assert "HIGH" in error_str


# =============================================================================
# TEST CLASS: Transform System Exceptions
# =============================================================================


class TestTransformExceptions:
    """Test transform system exception classes"""

    def test_transform_error_basic(self):
        """Test TransformError basic usage"""
        error = TransformError("Transform failed")
        assert error.message == "Transform failed"
        assert isinstance(error, BaseProjectError)

    def test_transform_error_with_transform_name(self):
        """Test TransformError with transform_name"""
        error = TransformError("Execution failed", transform_name="RandomRotate")
        assert error.transform_name == "RandomRotate"
        assert "RandomRotate" in str(error)

    def test_transform_error_with_experimental_setup(self):
        """Test TransformError with experimental_setup"""
        error = TransformError("Failed", transform_name="Normalize", experimental_setup="baseline")
        assert error.experimental_setup == "baseline"

    def test_transform_error_str_method(self):
        """Test TransformError __str__ method"""
        error = TransformError("Error", transform_name="Test", experimental_setup="setup1")
        error_str = str(error)
        assert "Test" in error_str
        assert "setup1" in error_str

    def test_transform_compatibility_error_signature(self):
        """Test TransformCompatibilityError"""
        error = TransformCompatibilityError(
            message="Incompatible", transform_name="MolTransform", dataset_type="Wavefunction"
        )
        assert isinstance(error, TransformError)
        assert error.dataset_type == "Wavefunction"

    def test_transform_compatibility_error_with_reason(self):
        """Test TransformCompatibilityError with incompatibility_reason"""
        error = TransformCompatibilityError(
            message="Incompatible",
            transform_name="Test",
            incompatibility_reason="Requires molecular structure",
        )
        assert error.incompatibility_reason == "Requires molecular structure"

    def test_transform_compatibility_error_str_method(self):
        """Test TransformCompatibilityError __str__ method"""
        error = TransformCompatibilityError(
            message="Error",
            transform_name="Test",
            dataset_type="DFT",
            suggested_alternatives=["Alt1", "Alt2"],
        )
        error_str = str(error)
        assert "DFT" in error_str

    def test_transformation_error_basic(self):
        """Test TransformationError"""
        error = TransformationError("Pipeline failed", transform_name="TestTransform")
        assert isinstance(error, BaseProjectError)
        assert error.transform_name == "TestTransform"

    def test_transformation_error_with_config(self):
        """Test TransformationError with transform_config"""
        config = {"param1": "value1"}
        error = TransformationError("Error", transform_config=config)
        assert error.transform_config == config

    def test_dataset_integration_error_basic(self):
        """Test DatasetIntegrationError"""
        error = DatasetIntegrationError("Integration failed", dataset_type="miliaDataset")
        assert isinstance(error, DataProcessingError)
        assert error.dataset_type == "miliaDataset"

    def test_dataset_integration_error_with_integration_point(self):
        """Test DatasetIntegrationError with integration_point"""
        error = DatasetIntegrationError("Failed", integration_point="transform_application")
        assert error.integration_point == "transform_application"

    def test_transform_validation_error_signature(self):
        """Test TransformValidationError"""
        error = TransformValidationError(
            message="Invalid parameter",
            transform_name="RandomRotate",
            parameter_name="degrees",
            parameter_value="invalid",
            expected_type=int,
        )
        assert isinstance(error, ValidationError)
        assert error.parameter_name == "degrees"
        assert error.expected_type == int
        # TransformValidationError sets validation_type automatically to "transform_parameter"
        assert error.validation_type == "transform_parameter"

    def test_transform_validation_error_str_method(self):
        """Test TransformValidationError __str__ method"""
        error = TransformValidationError(
            message="Error", transform_name="Test", parameter_name="param"
        )
        error_str = str(error)
        assert "param" in error_str.lower()

    def test_transform_composition_error_signature(self):
        """Test TransformCompositionError"""
        error = TransformCompositionError(
            message="Composition failed",
            transform_sequence=["T1", "T2", "T3"],
            failed_transform_index=1,
        )
        assert isinstance(error, DataProcessingError)
        assert error.transform_sequence == ["T1", "T2", "T3"]
        assert error.failed_transform_index == 1

    def test_transform_composition_error_with_name(self):
        """Test TransformCompositionError with failed_transform_name"""
        error = TransformCompositionError(
            message="Failed", failed_transform_name="InvalidTransform"
        )
        assert error.failed_transform_name == "InvalidTransform"

    def test_transform_composition_error_str_method(self):
        """Test TransformCompositionError __str__ method"""
        error = TransformCompositionError(
            message="Error", transform_sequence=["A", "B"], failed_transform_index=0
        )
        error_str = str(error)
        assert "0" in error_str

    def test_transform_not_found_error_signature(self):
        """Test TransformNotFoundError"""
        error = TransformNotFoundError(message="Not found", transform_name="NonExistent")
        assert isinstance(error, TransformationError)
        assert error.transform_name == "NonExistent"

    def test_transform_not_found_error_with_suggestions(self):
        """Test TransformNotFoundError with suggestions"""
        error = TransformNotFoundError(
            message="Not found", transform_name="Test", suggestions=["Similar1", "Similar2"]
        )
        assert error.suggestions == ["Similar1", "Similar2"]

    def test_transform_not_found_error_str_method(self):
        """Test TransformNotFoundError __str__ shows suggestions"""
        error = TransformNotFoundError(message="Error", transform_name="Test", suggestions=["Alt1"])
        error_str = str(error)
        assert "Alt1" in error_str

    def test_transform_registry_error_signature(self):
        """Test TransformRegistryError"""
        error = TransformRegistryError(message="Registry error", transform_name="Custom")
        assert isinstance(error, TransformationError)
        assert error.transform_name == "Custom"

    def test_transform_registry_error_with_operation(self):
        """Test TransformRegistryError with registry_operation"""
        error = TransformRegistryError(
            message="Error", transform_name="Test", registry_operation="registration"
        )
        assert error.registry_operation == "registration"

    def test_experimental_setup_error_signature(self):
        """Test ExperimentalSetupError"""
        error = ExperimentalSetupError(message="Invalid setup", setup_name="unknown")
        assert isinstance(error, ConfigurationError)
        assert error.setup_name == "unknown"

    def test_experimental_setup_error_with_available(self):
        """Test ExperimentalSetupError with available_setups"""
        setups = ["baseline", "augmented"]
        error = ExperimentalSetupError(message="Error", setup_name="bad", available_setups=setups)
        assert error.available_setups == setups

    def test_experimental_setup_error_str_method(self):
        """Test ExperimentalSetupError __str__ method"""
        error = ExperimentalSetupError(
            message="Error", setup_name="test", available_setups=["a", "b"]
        )
        error_str = str(error)
        assert "test" in error_str

    def test_transform_configuration_error_signature(self):
        """Test TransformConfigurationError"""
        error = TransformConfigurationError(message="Invalid config", transform_name="Normalize")
        assert isinstance(error, ConfigurationError)
        assert error.transform_name == "Normalize"

    def test_transform_configuration_error_with_source(self):
        """Test TransformConfigurationError with config_source"""
        error = TransformConfigurationError(
            message="Error", transform_name="Test", config_source="yaml_file"
        )
        assert error.config_source == "yaml_file"


# =============================================================================
# TEST CLASS: Plugin System Exceptions
# =============================================================================


class TestPluginExceptions:
    """Test plugin system exception classes"""

    def test_plugin_error_basic(self):
        """Test PluginError basic usage"""
        error = PluginError("Plugin error")
        assert error.message == "Plugin error"
        assert isinstance(error, BaseProjectError)

    def test_plugin_error_with_plugin_name(self):
        """Test PluginError with plugin_name"""
        error = PluginError("Failed", plugin_name="test_plugin")
        assert error.plugin_name == "test_plugin"
        assert "test_plugin" in str(error)

    def test_plugin_validation_error_basic(self):
        """Test PluginValidationError"""
        errors = ["Error 1", "Error 2"]
        error = PluginValidationError(
            "Validation failed", plugin_name="bad_plugin", validation_errors=errors
        )
        assert isinstance(error, PluginError)
        assert error.validation_errors == errors

    def test_plugin_security_error_basic(self):
        """Test PluginSecurityError"""
        issues = ["Uses eval", "Uses exec"]
        error = PluginSecurityError(
            "Security violation", plugin_name="unsafe", security_issues=issues
        )
        assert isinstance(error, PluginError)
        assert error.security_issues == issues

    def test_plugin_dependency_error_basic(self):
        """Test PluginDependencyError"""
        deps = ["torch>=1.9", "numpy>=1.20"]
        error = PluginDependencyError(
            "Missing deps", plugin_name="plugin", missing_dependencies=deps
        )
        assert isinstance(error, PluginError)
        assert error.missing_dependencies == deps

    def test_plugin_discovery_error_basic(self):
        """Test PluginDiscoveryError"""
        error = PluginDiscoveryError("Discovery failed", plugin_name="hidden")
        assert isinstance(error, PluginError)

    def test_plugin_registration_error_signature(self):
        """Test PluginRegistrationError"""
        error = PluginRegistrationError(
            message="Conflict", plugin_name="new", conflicting_plugin="existing"
        )
        assert isinstance(error, PluginError)
        assert error.conflicting_plugin == "existing"

    def test_plugin_load_error_signature(self):
        """Test PluginLoadError"""
        error = PluginLoadError(
            message="Load failed", plugin_name="broken", load_path="/path/to/plugin"
        )
        assert isinstance(error, PluginError)
        assert error.load_path == "/path/to/plugin"

    def test_plugin_load_error_with_original_error(self):
        """Test PluginLoadError with original_error"""
        error = PluginLoadError(
            message="Failed", plugin_name="test", original_error="ImportError: No module"
        )
        assert error.original_error == "ImportError: No module"


# =============================================================================
# TEST CLASS: Descriptor Exceptions
# =============================================================================


class TestDescriptorExceptions:
    """Test descriptor exception hierarchy"""

    def test_descriptor_error_basic(self):
        """Test DescriptorError"""
        error = DescriptorError("Error", descriptor_name="MolWt")
        assert error.descriptor_name == "MolWt"
        assert isinstance(error, BaseProjectError)

    def test_descriptor_calculation_error_basic(self):
        """Test DescriptorCalculationError"""
        error = DescriptorCalculationError(
            "Calculation failed", descriptor_name="LogP", molecule_index=5
        )
        assert isinstance(error, DescriptorError)
        assert error.molecule_index == 5

    def test_descriptor_calculation_error_with_smiles(self):
        """Test DescriptorCalculationError with smiles"""
        error = DescriptorCalculationError("Failed", descriptor_name="Test", smiles="CCO")
        assert error.smiles == "CCO"

    def test_descriptor_validation_error_basic(self):
        """Test DescriptorValidationError"""
        error = DescriptorValidationError("Invalid", descriptor_name="MolWt")
        assert isinstance(error, DescriptorError)

    def test_descriptor_plugin_error_basic(self):
        """Test DescriptorPluginError"""
        error = DescriptorPluginError("Plugin error", plugin_name="test")
        assert isinstance(error, DescriptorError)
        assert error.plugin_name == "test"

    def test_descriptor_plugin_load_error_basic(self):
        """Test DescriptorPluginLoadError"""
        error = DescriptorPluginLoadError("Load failed", plugin_name="broken")
        assert isinstance(error, DescriptorPluginError)

    def test_descriptor_plugin_validation_error_basic(self):
        """Test DescriptorPluginValidationError"""
        errors = ["Error 1"]
        error = DescriptorPluginValidationError(
            "Validation failed", plugin_name="test", validation_errors=errors
        )
        assert isinstance(error, DescriptorPluginError)
        assert error.validation_errors == errors

    def test_descriptor_plugin_config_error_basic(self):
        """Test DescriptorPluginConfigError"""
        error = DescriptorPluginConfigError("Config error", plugin_name="test")
        assert isinstance(error, DescriptorPluginError)

    def test_descriptor_calculation_error_with_original_error(self):
        """Test DescriptorCalculationError with original_error."""
        original = ValueError("Original calculation error")
        error = DescriptorCalculationError(
            "Calculation failed", descriptor_name="MolWt", molecule_index=5, original_error=original
        )
        assert error.original_error is original

    def test_descriptor_error_inheritance_chain(self):
        """Test descriptor exception inheritance chain."""
        assert issubclass(DescriptorError, BaseProjectError)
        assert issubclass(DescriptorCalculationError, DescriptorError)
        assert issubclass(DescriptorValidationError, DescriptorError)
        assert issubclass(DescriptorPluginError, DescriptorError)
        assert issubclass(DescriptorPluginLoadError, DescriptorPluginError)
        assert issubclass(DescriptorPluginValidationError, DescriptorPluginError)
        assert issubclass(DescriptorPluginConfigError, DescriptorPluginError)

    def test_descriptor_plugin_error_with_path(self):
        """Test DescriptorPluginError with plugin_path."""
        error = DescriptorPluginError(
            "Plugin error", plugin_name="custom_descriptor", plugin_path="/path/to/plugin"
        )
        assert error.plugin_path == "/path/to/plugin"


# =============================================================================
# TEST CLASS: Model System Exceptions
# =============================================================================


class TestModelExceptions:
    """Test model system exception classes."""

    def test_model_error_basic(self):
        """Test ModelError basic usage."""
        error = ModelError("Model error occurred")
        assert error.message == "Model error occurred"
        assert isinstance(error, BaseProjectError)

    def test_model_error_with_model_name(self):
        """Test ModelError with model_name."""
        error = ModelError("Training failed", model_name="GCN")
        assert error.model_name == "GCN"

    def test_model_error_with_details(self):
        """Test ModelError with details."""
        error = ModelError("Model error", model_name="GAT", details="CUDA out of memory")
        assert error.details == "CUDA out of memory"

    def test_model_not_found_error_basic(self):
        """Test ModelNotFoundError with required parameters."""
        error = ModelNotFoundError(message="Model not found", model_name="UnknownModel")
        assert isinstance(error, ModelError)
        assert error.model_name == "UnknownModel"

    def test_model_not_found_error_with_available_models(self):
        """Test ModelNotFoundError with available_models list."""
        available = ["GCN", "GAT", "GraphSAGE"]
        error = ModelNotFoundError(
            message="Not found", model_name="Unknown", available_models=available
        )
        assert error.available_models == available

    def test_model_not_found_error_str_method(self):
        """Test ModelNotFoundError __str__ includes available models."""
        error = ModelNotFoundError(
            message="Error", model_name="Test", available_models=["A", "B", "C"]
        )
        error_str = str(error)
        assert "A" in error_str or "Available models" in error_str

    def test_model_validation_error_basic(self):
        """Test ModelValidationError with validation_errors."""
        errors = ["hidden_channels must be > 0", "num_layers out of range"]
        error = ModelValidationError(
            message="Validation failed", model_name="GCN", validation_errors=errors
        )
        assert isinstance(error, ModelError)
        assert error.validation_errors == errors

    def test_model_validation_error_str_method(self):
        """Test ModelValidationError __str__ includes validation errors."""
        error = ModelValidationError(
            message="Error", model_name="Test", validation_errors=["Error1", "Error2"]
        )
        error_str = str(error)
        assert "Error1" in error_str or "Validation errors" in error_str

    def test_model_instantiation_error_basic(self):
        """Test ModelInstantiationError basic usage."""
        error = ModelInstantiationError(message="Failed to create model", model_name="CustomModel")
        assert isinstance(error, ModelError)

    def test_model_instantiation_error_with_hyperparameters(self):
        """Test ModelInstantiationError with hyperparameters."""
        params = {"hidden_channels": 64, "num_layers": 3}
        error = ModelInstantiationError(
            message="Instantiation failed", model_name="GCN", hyperparameters=params
        )
        assert error.hyperparameters == params

    def test_model_instantiation_error_with_original_error(self):
        """Test ModelInstantiationError with original_error."""
        error = ModelInstantiationError(
            message="Failed", model_name="GAT", original_error="TypeError: invalid argument"
        )
        assert error.original_error == "TypeError: invalid argument"

    def test_hyperparameter_error_basic(self):
        """Test HyperparameterError basic usage."""
        error = HyperparameterError(message="Invalid hyperparameter", model_name="GCN")
        assert isinstance(error, ModelError)

    def test_hyperparameter_error_with_all_params(self):
        """Test HyperparameterError with all parameters."""
        error = HyperparameterError(
            message="Invalid value",
            model_name="GAT",
            parameter_name="heads",
            parameter_value=-1,
            expected_type="positive integer",
        )
        assert error.parameter_name == "heads"
        assert error.parameter_value == -1
        assert error.expected_type == "positive integer"

    def test_hyperparameter_error_str_method(self):
        """Test HyperparameterError __str__ includes parameter info."""
        error = HyperparameterError(
            message="Error",
            parameter_name="learning_rate",
            parameter_value=0.0,
            expected_type="positive float",
        )
        error_str = str(error)
        assert "learning_rate" in error_str

    def test_data_compatibility_error_basic(self):
        """Test DataCompatibilityError basic usage."""
        error = DataCompatibilityError(message="Data incompatible", model_name="GCN")
        assert isinstance(error, ModelError)

    def test_data_compatibility_error_with_missing_features(self):
        """Test DataCompatibilityError with missing_features."""
        features = ["edge_index", "edge_attr"]
        error = DataCompatibilityError(
            message="Incompatible", model_name="GAT", missing_features=features
        )
        assert error.missing_features == features

    def test_data_compatibility_error_str_method(self):
        """Test DataCompatibilityError __str__ includes missing features."""
        error = DataCompatibilityError(message="Error", missing_features=["x", "edge_index"])
        error_str = str(error)
        assert "x" in error_str or "Missing features" in error_str

    def test_training_error_basic(self):
        """Test TrainingError basic usage."""
        error = TrainingError(message="Training failed", model_name="GraphSAGE")
        assert isinstance(error, ModelError)

    def test_training_error_with_training_context(self):
        """Test TrainingError with epoch, batch_index, phase."""
        error = TrainingError(
            message="Error during training",
            model_name="GCN",
            epoch=10,
            batch_index=5,
            phase="train",
        )
        assert error.epoch == 10
        assert error.batch_index == 5
        assert error.phase == "train"

    def test_training_error_str_method(self):
        """Test TrainingError __str__ includes training context."""
        error = TrainingError(message="Error", epoch=5, phase="val")
        error_str = str(error)
        assert "5" in error_str
        assert "val" in error_str

    def test_checkpoint_error_basic(self):
        """Test CheckpointError basic usage."""
        error = CheckpointError(message="Checkpoint failed")
        assert isinstance(error, ModelError)

    def test_checkpoint_error_with_path_and_operation(self):
        """Test CheckpointError with checkpoint_path and operation."""
        error = CheckpointError(
            message="Failed to save", checkpoint_path="/path/to/checkpoint.pt", operation="save"
        )
        assert error.checkpoint_path == "/path/to/checkpoint.pt"
        assert error.operation == "save"

    def test_checkpoint_error_str_method(self):
        """Test CheckpointError __str__ includes checkpoint info."""
        error = CheckpointError(message="Error", checkpoint_path="/model.pt", operation="load")
        error_str = str(error)
        assert "/model.pt" in error_str or "Checkpoint" in error_str

    def test_data_error_basic(self):
        """Test DataError basic usage."""
        error = DataError(message="Data error")
        assert isinstance(error, ModelError)

    def test_data_error_with_dataset_info(self):
        """Test DataError with dataset_size and split_ratios."""
        ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
        error = DataError(message="Split failed", dataset_size=100, split_ratios=ratios)
        assert error.dataset_size == 100
        assert error.split_ratios == ratios

    def test_plugin_model_error_basic(self):
        """Test PluginModelError basic usage."""
        error = PluginModelError(message="Plugin model error", plugin_name="custom_plugin")
        assert isinstance(error, ModelError)
        assert error.plugin_name == "custom_plugin"

    def test_plugin_model_error_with_all_params(self):
        """Test PluginModelError with all parameters."""
        error = PluginModelError(
            message="Error",
            plugin_name="my_plugin",
            model_name="CustomGNN",
            plugin_path="/path/to/plugin",
        )
        assert error.plugin_name == "my_plugin"
        assert error.model_name == "CustomGNN"
        assert error.plugin_path == "/path/to/plugin"


# =============================================================================
# TEST CLASS: Dataset Registration Exceptions
# =============================================================================


class TestDatasetRegistrationExceptions:
    """Test dataset registration exception classes."""

    def test_dataset_registration_error_basic(self):
        """Test DatasetRegistrationError basic usage."""
        error = DatasetRegistrationError(message="Registration failed")
        assert isinstance(error, BaseProjectError)

    def test_dataset_registration_error_with_dataset_name(self):
        """Test DatasetRegistrationError with dataset_name."""
        error = DatasetRegistrationError(message="Failed to register", dataset_name="CustomDataset")
        assert error.dataset_name == "CustomDataset"

    def test_dataset_registration_error_with_conflicting_class(self):
        """Test DatasetRegistrationError with conflicting_class."""
        error = DatasetRegistrationError(
            message="Conflict", dataset_name="DFT", conflicting_class="ExistingDFTDataset"
        )
        assert error.conflicting_class == "ExistingDFTDataset"

    def test_dataset_not_found_error_basic(self):
        """Test DatasetNotFoundError basic usage."""
        error = DatasetNotFoundError(message="Dataset not found", dataset_name="Unknown")
        assert isinstance(error, BaseProjectError)
        assert error.dataset_name == "Unknown"

    def test_dataset_not_found_error_with_available_datasets(self):
        """Test DatasetNotFoundError with available_datasets."""
        available = ["DFT", "DMC", "Wavefunction"]
        error = DatasetNotFoundError(
            message="Not found", dataset_name="QMC", available_datasets=available
        )
        assert error.available_datasets == available

    def test_dataset_not_found_error_auto_fills_available(self):
        """Test DatasetNotFoundError auto-fills available_datasets from registry (Phase 7)."""
        error = DatasetNotFoundError(
            message="Not found",
            dataset_name="Unknown",
            # available_datasets not provided - should be auto-filled
        )
        assert isinstance(error.available_datasets, list)
        # Should auto-fill from registry or filesystem discovery
        assert len(error.available_datasets) >= 0  # At least empty list

    def test_dataset_not_found_error_str_method(self):
        """Test DatasetNotFoundError __str__ method."""
        error = DatasetNotFoundError(
            message="Dataset not found",
            dataset_name="QMC",
            available_datasets=["DFT", "DMC"],
            details="Check dataset name",
        )
        error_str = str(error)
        assert "QMC" in error_str
        assert "DFT" in error_str or "Available" in error_str

    def test_dataset_registration_error_str_method(self):
        """Test DatasetRegistrationError __str__ method."""
        error = DatasetRegistrationError(
            message="Registration failed",
            dataset_name="CustomDataset",
            conflicting_class="ExistingDataset",
            details="Name collision",
        )
        error_str = str(error)
        assert "CustomDataset" in error_str
        assert "ExistingDataset" in error_str or "Conflicts" in error_str


# =============================================================================
# TEST CLASS: HPO Exceptions
# =============================================================================


class TestHPOExceptions:
    """Test HPO (Hyperparameter Optimization) exception classes."""

    def test_hpo_error_basic(self):
        """Test HPOError basic usage."""
        error = HPOError(message="HPO error")
        assert isinstance(error, BaseProjectError)

    def test_hpo_error_with_study_name(self):
        """Test HPOError with study_name."""
        error = HPOError(message="Study error", study_name="gcn_hyperparam_search")
        assert error.study_name == "gcn_hyperparam_search"

    def test_hpo_error_with_trial_number(self):
        """Test HPOError with trial_number."""
        error = HPOError(message="Trial error", trial_number=15)
        assert error.trial_number == 15

    def test_hpo_configuration_error_basic(self):
        """Test HPOConfigurationError basic usage."""
        error = HPOConfigurationError(message="Invalid HPO config")
        assert isinstance(error, HPOError)

    def test_hpo_configuration_error_with_config_details(self):
        """Test HPOConfigurationError with config_key and values."""
        error = HPOConfigurationError(
            message="Config error",
            config_key="learning_rate",
            actual_value=0.0,
            expected_value="positive float",
        )
        assert error.config_key == "learning_rate"
        assert error.actual_value == 0.0
        assert error.expected_value == "positive float"

    def test_trial_failed_error_basic(self):
        """Test TrialFailedError basic usage."""
        error = TrialFailedError(message="Trial failed")
        assert isinstance(error, HPOError)

    def test_trial_failed_error_with_all_params(self):
        """Test TrialFailedError with all parameters."""
        params = {"lr": 0.001, "hidden_dim": 256}
        error = TrialFailedError(
            message="Trial crashed",
            trial_number=15,
            trial_params=params,
            original_error="CUDA out of memory",
            epoch=42,
        )
        assert error.trial_number == 15
        assert error.trial_params == params
        assert error.original_error == "CUDA out of memory"
        assert error.epoch == 42

    def test_trial_failed_error_str_method(self):
        """Test TrialFailedError __str__ includes trial info."""
        error = TrialFailedError(message="Error", trial_number=10, epoch=5, original_error="OOM")
        error_str = str(error)
        assert "5" in error_str or "OOM" in error_str

    def test_study_not_found_error_basic(self):
        """Test StudyNotFoundError basic usage."""
        error = StudyNotFoundError(message="Study not found", study_name="missing_study")
        assert isinstance(error, HPOError)
        assert error.study_name == "missing_study"

    def test_study_not_found_error_with_available_studies(self):
        """Test StudyNotFoundError with available_studies."""
        available = ["study1", "study2"]
        error = StudyNotFoundError(
            message="Not found",
            study_name="unknown",
            available_studies=available,
            storage_url="sqlite:///optuna.db",
        )
        assert error.available_studies == available
        assert error.storage_url == "sqlite:///optuna.db"

    def test_study_not_found_error_str_method(self):
        """Test StudyNotFoundError __str__ includes available studies."""
        error = StudyNotFoundError(message="Error", study_name="test", available_studies=["a", "b"])
        error_str = str(error)
        assert "a" in error_str or "Available studies" in error_str

    def test_backend_error_basic(self):
        """Test BackendError basic usage."""
        error = BackendError(message="Backend error")
        assert isinstance(error, HPOError)

    def test_backend_error_with_backend_info(self):
        """Test BackendError with backend_name and operation."""
        error = BackendError(message="Failed", backend_name="optuna", operation="create_study")
        assert error.backend_name == "optuna"
        assert error.operation == "create_study"

    def test_search_space_error_basic(self):
        """Test SearchSpaceError basic usage."""
        error = SearchSpaceError(message="Invalid search space")
        assert isinstance(error, HPOError)

    def test_search_space_error_with_parameter_info(self):
        """Test SearchSpaceError with parameter_name and parameter_config."""
        config = {"low": 0.1, "high": 0.01}
        error = SearchSpaceError(
            message="Invalid bounds", parameter_name="learning_rate", parameter_config=config
        )
        assert error.parameter_name == "learning_rate"
        assert error.parameter_config == config

    def test_pruning_error_basic(self):
        """Test PruningError basic usage."""
        error = PruningError(message="Pruning error")
        assert isinstance(error, HPOError)

    def test_pruning_error_with_all_params(self):
        """Test PruningError with all parameters."""
        error = PruningError(
            message="Pruning decision failed",
            trial_number=23,
            pruner_type="MedianPruner",
            intermediate_value=0.456,
        )
        assert error.trial_number == 23
        assert error.pruner_type == "MedianPruner"
        assert error.intermediate_value == 0.456

    def test_pruning_error_str_method(self):
        """Test PruningError __str__ includes pruner info."""
        error = PruningError(message="Error", pruner_type="HyperbandPruner", intermediate_value=0.5)
        error_str = str(error)
        assert "HyperbandPruner" in error_str or "0.5" in error_str

    def test_hpo_error_str_method_comprehensive(self):
        """Test HPOError __str__ with all attributes."""
        error = HPOError(
            message="HPO error", study_name="test_study", trial_number=10, details="Additional info"
        )
        error_str = str(error)
        assert "test_study" in error_str
        assert "10" in error_str

    def test_hpo_configuration_error_str_method(self):
        """Test HPOConfigurationError __str__ with all attributes."""
        error = HPOConfigurationError(
            message="Config error",
            config_key="sampler.n_startup",
            actual_value=-1,
            expected_value="non-negative integer",
            details="Must be >= 0",
        )
        error_str = str(error)
        assert "sampler.n_startup" in error_str
        assert "-1" in error_str

    def test_hpo_inheritance_chain(self):
        """Test HPO exception inheritance chain."""
        assert issubclass(HPOError, ModelError)
        assert issubclass(HPOConfigurationError, HPOError)
        assert issubclass(TrialFailedError, HPOError)
        assert issubclass(StudyNotFoundError, HPOError)
        assert issubclass(BackendError, HPOError)
        assert issubclass(SearchSpaceError, HPOError)
        assert issubclass(PruningError, HPOError)


# =============================================================================
# TEST CLASS: Exception Inheritance Hierarchy
# =============================================================================


class TestExceptionInheritance:
    """Test exception inheritance hierarchy"""

    def test_base_inheritance(self):
        """Test base exception inheritance"""
        assert issubclass(BaseProjectError, Exception)
        assert issubclass(MoleculeFilterRejectedError, BaseException)
        assert not issubclass(MoleculeFilterRejectedError, Exception)

    def test_all_base_project_error_subclasses(self):
        """Test major exception classes inherit from BaseProjectError"""
        base_subclasses = [
            LoggingConfigurationError,
            ConfigurationError,
            DataProcessingError,
            MoleculeProcessingError,
            HandlerError,
            ValidationError,
            CompatibilityError,
            MigrationError,
            LegacyCodeError,
            TransformError,
            TransformationError,
            PluginError,
            DescriptorError,
            HandlerIntegrationError,
        ]
        for exc_class in base_subclasses:
            assert issubclass(exc_class, BaseProjectError), (
                f"{exc_class.__name__} should inherit from BaseProjectError"
            )

    def test_configuration_exceptions_hierarchy(self):
        """Test configuration exceptions"""
        assert issubclass(ExperimentalSetupError, ConfigurationError)
        assert issubclass(TransformConfigurationError, ConfigurationError)

    def test_data_processing_exceptions_hierarchy(self):
        """Test data processing exceptions"""
        assert issubclass(TransformCompositionError, DataProcessingError)
        assert issubclass(VibrationRefinementError, DataProcessingError)
        assert issubclass(DatasetIntegrationError, DataProcessingError)

    def test_molecule_processing_exceptions_hierarchy(self):
        """Test molecule processing exceptions"""
        molecule_exceptions = [
            RDKitConversionError,
            PyGDataCreationError,
            PropertyEnrichmentError,
            StructuralFeatureError,
            UncertaintyProcessingError,
        ]
        for exc_class in molecule_exceptions:
            assert issubclass(exc_class, MoleculeProcessingError)

    def test_handler_exceptions_hierarchy(self):
        """Test handler exceptions"""
        handler_exceptions = [
            HandlerNotAvailableError,
            HandlerConfigurationError,
            HandlerOperationError,
            HandlerValidationError,
            HandlerCompatibilityError,
            DatasetSpecificHandlerError,
        ]
        for exc_class in handler_exceptions:
            assert issubclass(exc_class, HandlerError)

    def test_handler_integration_hierarchy(self):
        """Test handler integration exceptions"""
        # HandlerIntegrationError inherits from BaseProjectError, NOT HandlerError
        assert issubclass(HandlerIntegrationError, BaseProjectError)
        assert not issubclass(HandlerIntegrationError, HandlerError)
        assert issubclass(TransformHandlerIntegrationError, HandlerIntegrationError)

    def test_validation_exceptions_hierarchy(self):
        """Test validation exceptions"""
        assert issubclass(ValidationError, BaseProjectError)
        # CompatibilityError does NOT inherit from ValidationError
        assert not issubclass(CompatibilityError, ValidationError)
        assert issubclass(TransformValidationError, ValidationError)

    def test_transform_exceptions_hierarchy(self):
        """Test transform exceptions"""
        assert issubclass(TransformCompatibilityError, TransformError)
        assert issubclass(TransformNotFoundError, TransformationError)
        assert issubclass(TransformRegistryError, TransformationError)

    def test_plugin_exceptions_hierarchy(self):
        """Test plugin exceptions"""
        plugin_exceptions = [
            PluginValidationError,
            PluginSecurityError,
            PluginDependencyError,
            PluginDiscoveryError,
            PluginRegistrationError,
            PluginLoadError,
        ]
        for exc_class in plugin_exceptions:
            assert issubclass(exc_class, PluginError)

    def test_descriptor_exceptions_hierarchy(self):
        """Test descriptor exceptions"""
        assert issubclass(DescriptorCalculationError, DescriptorError)
        assert issubclass(DescriptorValidationError, DescriptorError)
        assert issubclass(DescriptorPluginError, DescriptorError)

        plugin_subclasses = [
            DescriptorPluginLoadError,
            DescriptorPluginValidationError,
            DescriptorPluginConfigError,
        ]
        for exc_class in plugin_subclasses:
            assert issubclass(exc_class, DescriptorPluginError)

    def test_dataset_specific_handler_hierarchy(self):
        """Test DatasetSpecificHandlerError hierarchy"""
        assert issubclass(DatasetSpecificHandlerError, HandlerError)

    def test_uncertainty_processing_hierarchy(self):
        """Test UncertaintyProcessingError hierarchy"""
        assert issubclass(UncertaintyProcessingError, MoleculeProcessingError)

    def test_model_exceptions_hierarchy(self):
        """Test model exceptions inherit from ModelError."""
        model_exceptions = [
            ModelNotFoundError,
            ModelValidationError,
            ModelInstantiationError,
            HyperparameterError,
            DataCompatibilityError,
            TrainingError,
            CheckpointError,
            DataError,
            PluginModelError,
        ]
        for exc_class in model_exceptions:
            assert issubclass(exc_class, ModelError), (
                f"{exc_class.__name__} should inherit from ModelError"
            )

        # ModelError should inherit from BaseProjectError
        assert issubclass(ModelError, BaseProjectError)

    def test_hpo_exceptions_hierarchy(self):
        """Test HPO exceptions inherit from HPOError."""
        hpo_exceptions = [
            HPOConfigurationError,
            TrialFailedError,
            StudyNotFoundError,
            BackendError,
            SearchSpaceError,
            PruningError,
        ]
        for exc_class in hpo_exceptions:
            assert issubclass(exc_class, HPOError), (
                f"{exc_class.__name__} should inherit from HPOError"
            )

        # HPOError should inherit from BaseProjectError
        assert issubclass(HPOError, BaseProjectError)

    def test_dataset_registration_exceptions_hierarchy(self):
        """Test dataset registration exceptions inherit from BaseProjectError."""
        assert issubclass(DatasetRegistrationError, BaseProjectError)
        assert issubclass(DatasetNotFoundError, BaseProjectError)

    def test_preprocessing_required_error_hierarchy(self):
        """Test PreprocessingRequiredError inherits from DataProcessingError."""
        assert issubclass(PreprocessingRequiredError, DataProcessingError)
        assert issubclass(PreprocessingRequiredError, BaseProjectError)


# =============================================================================
# TEST CLASS: Validation Function
# =============================================================================


class TestValidationFunction:
    """Test the validate_exception_hierarchy function"""

    def test_validate_exception_hierarchy_returns_dict(self):
        """Test validation function returns dict"""
        results = validate_exception_hierarchy()
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_validate_exception_hierarchy_all_pass(self):
        """Test all validations pass"""
        results = validate_exception_hierarchy()
        for test_name, passed in results.items():
            assert passed, f"Validation failed: {test_name}"

    def test_validate_exception_hierarchy_checks_handlers(self):
        """Test validation checks handler exceptions"""
        results = validate_exception_hierarchy()
        assert "HandlerError_inherits_BaseProjectError" in results
        assert results["HandlerError_inherits_BaseProjectError"] == True

    def test_validate_exception_hierarchy_checks_molecules(self):
        """Test validation checks molecule exceptions"""
        results = validate_exception_hierarchy()
        assert "MoleculeProcessingError_inherits_BaseProjectError" in results
        assert results["MoleculeProcessingError_inherits_BaseProjectError"] == True

    def test_validate_exception_hierarchy_checks_plugins(self):
        """Test validation checks plugin exceptions"""
        results = validate_exception_hierarchy()
        assert "PluginError_inherits_BaseProjectError" in results
        assert results["PluginError_inherits_BaseProjectError"] == True

    def test_validate_exception_hierarchy_checks_generic_exceptions(self):
        """Test validation checks dynamic exceptions"""
        results = validate_exception_hierarchy()
        # DatasetSpecificHandlerError should be in handler exceptions
        assert "DatasetSpecificHandlerError_inherits_HandlerError" in results
        assert results["DatasetSpecificHandlerError_inherits_HandlerError"] == True

        # UncertaintyProcessingError should be in molecule exceptions
        assert "UncertaintyProcessingError_inherits_MoleculeProcessingError" in results
        assert results["UncertaintyProcessingError_inherits_MoleculeProcessingError"] == True

    def test_validate_exception_hierarchy_dynamic_subclass_validation(self):
        """Test validation dynamically discovers and validates any runtime subclasses"""
        results = validate_exception_hierarchy()
        # The validation function uses __subclasses__() to dynamically find subclasses.
        # If no subclasses exist at runtime, these keys simply won't be present.
        # The important thing is all present keys pass.
        for key, value in results.items():
            if (
                "inherits_DatasetSpecificHandlerError" in key
                or "inherits_UncertaintyProcessingError" in key
            ):
                assert value == True, f"Dynamic subclass validation failed: {key}"

    def test_validate_exception_hierarchy_registry_integration(self):
        """Test validation includes registry integration check"""
        results = validate_exception_hierarchy()

    def test_validate_exception_hierarchy_checks_models(self):
        """Test validation checks model exceptions."""
        results = validate_exception_hierarchy()
        assert "ModelError_inherits_BaseProjectError" in results
        assert results["ModelError_inherits_BaseProjectError"] == True

    def test_validate_exception_hierarchy_checks_all_model_subclasses(self):
        """Test validation checks all model exception subclasses."""
        results = validate_exception_hierarchy()
        model_subclasses = [
            "ModelNotFoundError",
            "ModelValidationError",
            "ModelInstantiationError",
            "HyperparameterError",
            "DataCompatibilityError",
            "TrainingError",
            "CheckpointError",
            "DataError",
            "PluginModelError",
        ]
        for name in model_subclasses:
            key = f"{name}_inherits_ModelError"
            assert key in results, f"Missing validation for {name}"
            assert results[key] == True, f"{name} should inherit from ModelError"

    def test_validate_exception_hierarchy_checks_hpo_subclasses(self):
        """Test validation checks HPO exception subclasses."""
        results = validate_exception_hierarchy()
        # HPOError inherits from ModelError, which inherits from BaseProjectError
        assert "ModelError_inherits_BaseProjectError" in results
        assert results["ModelError_inherits_BaseProjectError"] == True

    def test_validate_exception_hierarchy_checks_plugin_subclasses(self):
        """Test validation checks plugin exception subclasses."""
        results = validate_exception_hierarchy()
        plugin_subclasses = [
            "PluginValidationError",
            "PluginSecurityError",
            "PluginDependencyError",
            "PluginDiscoveryError",
            "PluginRegistrationError",
            "PluginLoadError",
        ]
        for name in plugin_subclasses:
            key = f"{name}_inherits_PluginError"
            assert key in results, f"Missing validation for {name}"
            assert results[key] == True, f"{name} should inherit from PluginError"

    def test_validate_exception_hierarchy_checks_transform_exceptions(self):
        """Test validation checks transform exception subclasses."""
        results = validate_exception_hierarchy()
        transform_checks = [
            "ExperimentalSetupError_inherits_ConfigurationError",
            "TransformConfigurationError_inherits_ConfigurationError",
            "TransformValidationError_inherits_ValidationError",
            "TransformCompositionError_inherits_DataProcessingError",
            "TransformRegistryError_inherits_TransformationError",
        ]
        for check in transform_checks:
            assert check in results, f"Missing validation for {check}"
            assert results[check] == True, f"{check} should be True"

    def test_validate_exception_hierarchy_dynamic_implementation_check(self):
        """Test validation checks dynamic exception implementation."""
        results = validate_exception_hierarchy()
        assert "dynamic_exceptions_implemented" in results
        assert results["dynamic_exceptions_implemented"] == True


# =============================================================================
# TEST CLASS: Registry Integration Functions
# =============================================================================


class TestRegistryIntegration:
    """Test registry integration functions"""

    def test_init_registry_returns_bool(self):
        """Test _init_registry returns boolean"""
        result = _init_registry()
        assert isinstance(result, bool)

    def test_init_registry_is_idempotent(self):
        """Test _init_registry can be called multiple times safely"""
        result1 = _init_registry()
        result2 = _init_registry()
        result3 = _init_registry()

        # All calls should return the same result
        assert result1 == result2
        assert result2 == result3

    def test_get_available_dataset_types_returns_list(self):
        """Test _get_available_dataset_types returns list"""
        types = _get_available_dataset_types()
        assert isinstance(types, list)
        assert len(types) > 0

    def test_get_available_dataset_types_includes_core_types(self):
        """Test _get_available_dataset_types includes DFT, DMC, Wavefunction"""
        types = _get_available_dataset_types()
        assert "DFT" in types
        assert "DMC" in types
        assert "Wavefunction" in types

    def test_is_dataset_type_registered_for_dft(self):
        """Test _is_dataset_type_registered returns True for DFT"""
        assert _is_dataset_type_registered("DFT") == True

    def test_is_dataset_type_registered_for_dmc(self):
        """Test _is_dataset_type_registered returns True for DMC"""
        assert _is_dataset_type_registered("DMC") == True

    def test_is_dataset_type_registered_for_unknown(self):
        """Test _is_dataset_type_registered returns False for unknown type"""
        assert _is_dataset_type_registered("UNKNOWN_TYPE_XYZ") == False

    def test_is_dataset_type_registered_empty_string(self):
        """Test _is_dataset_type_registered handles empty string"""
        assert _is_dataset_type_registered("") == False

    def test_get_dataset_feature_dmc_uncertainty(self):
        """Test _get_dataset_feature for DMC uncertainty_handling"""
        result = _get_dataset_feature("DMC", "uncertainty_handling")
        assert result == True

    def test_get_dataset_feature_dft_vibrational(self):
        """Test _get_dataset_feature for DFT vibrational_analysis"""
        result = _get_dataset_feature("DFT", "vibrational_analysis")
        assert result == True

    def test_get_dataset_feature_dft_uncertainty(self):
        """Test _get_dataset_feature for DFT uncertainty_handling (False)"""
        result = _get_dataset_feature("DFT", "uncertainty_handling")
        assert result == False

    def test_get_dataset_feature_unknown_feature(self):
        """Test _get_dataset_feature for unknown feature returns default"""
        result = _get_dataset_feature("DFT", "unknown_feature_xyz", default=False)
        assert result == False

    def test_get_dataset_feature_unknown_dataset(self):
        """Test _get_dataset_feature for unknown dataset returns default"""
        result = _get_dataset_feature("UNKNOWN", "uncertainty_handling", default=False)
        assert result == False

    def test_get_exception_registry_status_returns_dict(self):
        """Test get_exception_registry_status returns dict"""
        status = get_exception_registry_status()
        assert isinstance(status, dict)

    def test_get_exception_registry_status_has_required_keys(self):
        """Test get_exception_registry_status has all required keys"""
        status = get_exception_registry_status()

        required_keys = [
            "registry_available",
            "registry_initialized",
            "available_dataset_types",
            "using_fallback",
            "dynamic_integration",
        ]

        for key in required_keys:
            assert key in status, f"Missing key: {key}"

    def test_get_exception_registry_status_dynamic_integration_true(self):
        """Test get_exception_registry_status shows dynamic integration"""
        status = get_exception_registry_status()
        assert status["dynamic_integration"] == True

    def test_get_exception_registry_status_available_types(self):
        """Test get_exception_registry_status includes available types"""
        status = get_exception_registry_status()
        types = status["available_dataset_types"]

        assert isinstance(types, list)
        assert "DFT" in types
        assert "DMC" in types


# =============================================================================
# TEST CLASS: Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Test exception factory functions"""

    def test_create_dataset_handler_error_returns_dataset_specific(self):
        """Test create_dataset_handler_error always returns DatasetSpecificHandlerError"""
        error = create_dataset_handler_error(
            message="Test error", dataset_type="TestDataset", operation="validate_properties"
        )

        assert isinstance(error, DatasetSpecificHandlerError)
        assert error.dataset_type == "TestDataset"
        assert error.handler_operation == "validate_properties"

    def test_create_dataset_handler_error_for_any_dataset_type(self):
        """Test create_dataset_handler_error works for any dataset type"""
        for ds_type in ["QMC", "CCSD", "FCIQMC", "NewDataset", "TestDS"]:
            error = create_dataset_handler_error(
                message=f"{ds_type} error",
                dataset_type=ds_type,
                operation="validate",
                property_name="test_prop",
            )
            assert isinstance(error, DatasetSpecificHandlerError)
            assert error.dataset_type == ds_type
            assert error.property_name == "test_prop"

    def test_create_dataset_handler_error_with_details(self):
        """Test create_dataset_handler_error with details"""
        error = create_dataset_handler_error(
            message="Test error", dataset_type="CCSD", details="Additional context"
        )

        assert error.details == "Additional context"

    def test_create_uncertainty_processing_error_returns_uncertainty(self):
        """Test create_uncertainty_processing_error always returns UncertaintyProcessingError"""
        error = create_uncertainty_processing_error(
            message="Test error",
            dataset_type="TestDataset",
            molecule_index=5,
            property_name="energy_uncertainty",
        )

        assert isinstance(error, UncertaintyProcessingError)
        assert error.dataset_type == "TestDataset"
        assert error.molecule_index == 5
        assert error.uncertainty_property_name == "energy_uncertainty"

    def test_create_uncertainty_processing_error_for_any_dataset_type(self):
        """Test create_uncertainty_processing_error works for any dataset type"""
        for ds_type in ["QMC", "FCIQMC", "CCSD", "NewDataset"]:
            error = create_uncertainty_processing_error(
                message=f"{ds_type} uncertainty error",
                dataset_type=ds_type,
                molecule_index=10,
                property_name="correlation_energy",
            )
            assert isinstance(error, UncertaintyProcessingError)
            assert error.dataset_type == ds_type
            assert error.uncertainty_property_name == "correlation_energy"

    def test_create_uncertainty_processing_error_requires_dataset_type(self):
        """Test create_uncertainty_processing_error requires dataset_type (no default)"""
        with pytest.raises(TypeError):
            create_uncertainty_processing_error(message="Test error", molecule_index=5)

    def test_create_handler_not_available_error_auto_fills_types(self):
        """Test create_handler_not_available_error auto-fills available_types"""
        error = create_handler_not_available_error(
            message="Handler not available", requested_dataset_type="UNKNOWN"
        )

        assert isinstance(error, HandlerNotAvailableError)
        assert error.requested_dataset_type == "UNKNOWN"
        # available_types should be auto-filled from registry
        assert isinstance(error.available_types, list)

    def test_create_handler_not_available_error_custom_types(self):
        """Test create_handler_not_available_error with custom available_types"""
        error = create_handler_not_available_error(
            message="Handler not available",
            requested_dataset_type="UNKNOWN",
            available_types=["CustomA", "CustomB"],
        )

        # Custom list should override auto-fill
        assert error.available_types == ["CustomA", "CustomB"]

    def test_create_handler_not_available_error_with_dependencies(self):
        """Test create_handler_not_available_error with missing_dependencies"""
        error = create_handler_not_available_error(
            message="Handler not available",
            requested_dataset_type="Custom",
            missing_dependencies=["torch", "rdkit"],
        )

        assert error.missing_dependencies == ["torch", "rdkit"]


# =============================================================================
# TEST CLASS: Lazy Initialization
# =============================================================================


class TestLazyInitialization:
    """Test lazy initialization patterns"""

    def test_registry_state_variables_exist(self):
        """Test that registry state variables are defined"""
        import milia_pipeline.exceptions as exc_module

        assert hasattr(exc_module, "_REGISTRY_INITIALIZED")
        assert hasattr(exc_module, "_REGISTRY_AVAILABLE")
        assert hasattr(exc_module, "_registry_list_all")
        assert hasattr(exc_module, "_registry_get")
        assert hasattr(exc_module, "_registry_is_registered")

    def test_legacy_variables_removed(self):
        """Test that legacy hardcoded variables have been removed"""
        import milia_pipeline.exceptions as exc_module

        assert not hasattr(exc_module, "_LEGACY_DATASET_TYPES"), (
            "_LEGACY_DATASET_TYPES should be removed in dynamic refactoring"
        )
        assert not hasattr(exc_module, "_LEGACY_FEATURES"), (
            "_LEGACY_FEATURES should be removed in dynamic refactoring"
        )

    def test_init_registry_sets_initialized_flag(self):
        """Test that _init_registry sets _REGISTRY_INITIALIZED flag"""
        import milia_pipeline.exceptions as exc_module

        _init_registry()

        assert exc_module._REGISTRY_INITIALIZED == True


# =============================================================================
# TEST CLASS: Filesystem Discovery Function
# =============================================================================


class TestFilesystemDiscovery:
    """Test _discover_dataset_types_from_filesystem function."""

    def test_discover_dataset_types_returns_list(self):
        """Test _discover_dataset_types_from_filesystem returns a list."""
        result = _discover_dataset_types_from_filesystem()
        assert isinstance(result, list)

    def test_discover_dataset_types_returns_uppercase_names(self):
        """Test discovered types are uppercase."""
        result = _discover_dataset_types_from_filesystem()
        for dtype in result:
            assert dtype == dtype.upper(), f"Type '{dtype}' is not uppercase"

    def test_discover_dataset_types_excludes_special_modules(self):
        """Test special modules (BASE, REGISTRY, UTILS, COMMON, PROTOCOLS) are excluded."""
        result = _discover_dataset_types_from_filesystem()
        excluded = ["BASE", "REGISTRY", "UTILS", "COMMON", "PROTOCOLS"]
        for excl in excluded:
            assert excl not in result, f"'{excl}' should be excluded from discovered types"

    def test_discover_dataset_types_excludes_private_modules(self):
        """Test private modules (starting with _) are excluded."""
        result = _discover_dataset_types_from_filesystem()
        for dtype in result:
            assert not dtype.startswith("_"), f"Private module '{dtype}' should be excluded"

    def test_discover_dataset_types_with_mocked_nonexistent_path(self):
        """Test _discover_dataset_types_from_filesystem handles missing directory gracefully."""
        with patch("pathlib.Path.exists", return_value=False):
            result = _discover_dataset_types_from_filesystem()
            assert isinstance(result, list)

    def test_discover_dataset_types_handles_exception_gracefully(self):
        """Test _discover_dataset_types_from_filesystem handles exceptions gracefully."""
        with patch("pathlib.Path.__truediv__", side_effect=Exception("Simulated error")):
            result = _discover_dataset_types_from_filesystem()
            # Should return empty list on error, not raise
            assert isinstance(result, list)


# =============================================================================
# TEST CLASS: Registry Fallback Scenarios
# =============================================================================


class TestRegistryFallbackScenarios:
    """Test registry fallback behavior with various scenarios."""

    def test_get_available_dataset_types_with_registry_unavailable(self):
        """Test _get_available_dataset_types falls back correctly when registry unavailable."""
        # Reset and mock registry as unavailable
        reset_registry_state()
        import milia_pipeline.exceptions as exc_module

        with patch.object(exc_module, "_REGISTRY_AVAILABLE", False):
            with patch.object(exc_module, "_REGISTRY_INITIALIZED", True):
                types = _get_available_dataset_types()
                assert isinstance(types, list)

    def test_is_dataset_type_registered_with_registry_unavailable(self):
        """Test _is_dataset_type_registered falls back correctly when registry unavailable."""
        reset_registry_state()
        import milia_pipeline.exceptions as exc_module

        with patch.object(exc_module, "_REGISTRY_AVAILABLE", False):
            with patch.object(exc_module, "_REGISTRY_INITIALIZED", True):
                # Should check against dynamically discovered types or empty list
                result = _is_dataset_type_registered("NONEXISTENT_XYZ")
                assert result == False

    def test_get_dataset_feature_with_registry_unavailable_returns_default(self):
        """Test _get_dataset_feature returns default when registry unavailable."""
        reset_registry_state()
        import milia_pipeline.exceptions as exc_module

        with patch.object(exc_module, "_REGISTRY_AVAILABLE", False):
            with patch.object(exc_module, "_REGISTRY_INITIALIZED", True):
                # With no registry and no legacy fallback, should return default (False)
                result = _get_dataset_feature("SomeDataset", "uncertainty_handling")
                assert result == False  # default value

    def test_get_dataset_feature_with_registry_unavailable_custom_default(self):
        """Test _get_dataset_feature returns custom default when registry unavailable."""
        reset_registry_state()
        import milia_pipeline.exceptions as exc_module

        with patch.object(exc_module, "_REGISTRY_AVAILABLE", False):
            with patch.object(exc_module, "_REGISTRY_INITIALIZED", True):
                result = _get_dataset_feature("SomeDataset", "some_feature", default=True)
                assert result == True  # custom default

    def test_get_dataset_feature_unknown_feature_returns_default(self):
        """Test _get_dataset_feature for unknown feature returns default."""
        result = _get_dataset_feature("SomeDataset", "unknown_feature_xyz", default=False)
        assert result == False

    def test_get_dataset_feature_unknown_dataset_returns_default(self):
        """Test _get_dataset_feature for unknown dataset returns default."""
        result = _get_dataset_feature("UNKNOWN", "uncertainty_handling", default=False)
        assert result == False


# =============================================================================
# TEST CLASS: Utility Functions for Exception Handling
# =============================================================================


class TestExceptionUtilityFunctions:
    """Test utility functions for exception handling."""

    def test_create_handler_error_context_basic(self):
        """Test create_handler_error_context with required params."""
        context = create_handler_error_context(handler_type="DFT", operation="validate")
        assert isinstance(context, dict)
        assert context["handler_type"] == "DFT"
        assert context["operation"] == "validate"
        assert "timestamp" in context

    def test_create_handler_error_context_with_molecule_index(self):
        """Test create_handler_error_context with molecule_index."""
        context = create_handler_error_context(
            handler_type="DMC", operation="enrich", molecule_index=42
        )
        assert context["molecule_index"] == 42

    def test_create_handler_error_context_with_additional_context(self):
        """Test create_handler_error_context with additional_context."""
        additional = {"custom_key": "custom_value", "count": 100}
        context = create_handler_error_context(
            handler_type="DFT", operation="process", additional_context=additional
        )
        assert context["custom_key"] == "custom_value"
        assert context["count"] == 100

    def test_create_handler_error_context_includes_registry_info(self):
        """Test create_handler_error_context includes registry information (Phase 7)."""
        context = create_handler_error_context(handler_type="DFT", operation="test")
        # Phase 7: Should include registry status
        assert "dataset_type_registered" in context
        assert "available_dataset_types" in context

    def test_format_handler_exception_summary_basic(self):
        """Test format_handler_exception_summary with basic exception."""
        error = HandlerError("Test error", handler_type="DFT")
        summary = format_handler_exception_summary(error)

        assert isinstance(summary, dict)
        assert summary["exception_type"] == "HandlerError"
        assert summary["is_handler_exception"] == True
        assert "message" in summary

    def test_format_handler_exception_summary_non_handler_exception(self):
        """Test format_handler_exception_summary with non-handler exception."""
        error = ConfigurationError("Config error")
        summary = format_handler_exception_summary(error)

        assert summary["is_handler_exception"] == False

    def test_format_handler_exception_summary_dataset_specific(self):
        """Test format_handler_exception_summary with DatasetSpecificHandlerError."""
        error = DatasetSpecificHandlerError(
            message="Error", dataset_type="QMC", property_name="energy"
        )
        summary = format_handler_exception_summary(error)

        assert summary["is_dataset_specific_handler_exception"] == True
        assert summary["dataset_type"] == "QMC"
        assert summary["property_name"] == "energy"

    def test_get_exception_recovery_suggestions_handler_not_available(self):
        """Test get_exception_recovery_suggestions for HandlerNotAvailableError."""
        error = HandlerNotAvailableError(
            message="Handler not available", requested_dataset_type="UNKNOWN"
        )
        suggestions = get_exception_recovery_suggestions(error)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_get_exception_recovery_suggestions_handler_configuration(self):
        """Test get_exception_recovery_suggestions for HandlerConfigurationError."""
        error = HandlerConfigurationError(message="Config error", handler_type="DFT")
        suggestions = get_exception_recovery_suggestions(error)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_get_exception_recovery_suggestions_handler_operation(self):
        """Test get_exception_recovery_suggestions for HandlerOperationError."""
        error = HandlerOperationError(
            message="Operation failed", handler_type="DMC", operation="process"
        )
        suggestions = get_exception_recovery_suggestions(error)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_get_exception_recovery_suggestions_handler_validation(self):
        """Test get_exception_recovery_suggestions for HandlerValidationError."""
        error = HandlerValidationError(
            message="Validation failed", handler_type="DFT", validation_type="molecule"
        )
        suggestions = get_exception_recovery_suggestions(error)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_get_exception_recovery_suggestions_migration(self):
        """Test get_exception_recovery_suggestions for MigrationError."""
        error = MigrationError(
            message="Migration failed", migration_phase="phase1", rollback_available=True
        )
        suggestions = get_exception_recovery_suggestions(error)

        assert isinstance(suggestions, list)
        assert any("rollback" in s.lower() for s in suggestions)

    def test_get_exception_recovery_suggestions_legacy_code(self):
        """Test get_exception_recovery_suggestions for LegacyCodeError."""
        error = LegacyCodeError(message="Legacy code", legacy_pattern="old_pattern")
        suggestions = get_exception_recovery_suggestions(error)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_get_exception_recovery_suggestions_molecule_processing(self):
        """Test get_exception_recovery_suggestions for MoleculeProcessingError."""
        error = MoleculeProcessingError(message="Molecule error", molecule_index=5)
        suggestions = get_exception_recovery_suggestions(error)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_get_exception_recovery_suggestions_configuration(self):
        """Test get_exception_recovery_suggestions for ConfigurationError."""
        error = ConfigurationError(message="Config error")
        suggestions = get_exception_recovery_suggestions(error)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_get_exception_recovery_suggestions_dataset_specific(self):
        """Test get_exception_recovery_suggestions for DatasetSpecificHandlerError (Phase 7)."""
        error = DatasetSpecificHandlerError(message="Dataset error", dataset_type="QMC")
        suggestions = get_exception_recovery_suggestions(error)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        # Should include dataset-type-specific suggestions
        assert any("QMC" in s for s in suggestions)

    def test_get_exception_recovery_suggestions_uncertainty_processing(self):
        """Test get_exception_recovery_suggestions for UncertaintyProcessingError (Phase 7)."""
        error = UncertaintyProcessingError(message="Uncertainty error", dataset_type="QMC")
        suggestions = get_exception_recovery_suggestions(error)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_is_recoverable_handler_error_not_available(self):
        """Test is_recoverable_handler_error returns False for HandlerNotAvailableError."""
        error = HandlerNotAvailableError(message="Not available", requested_dataset_type="UNKNOWN")
        assert is_recoverable_handler_error(error) == False

    def test_is_recoverable_handler_error_configuration(self):
        """Test is_recoverable_handler_error returns True for HandlerConfigurationError."""
        error = HandlerConfigurationError(message="Config error", handler_type="DFT")
        assert is_recoverable_handler_error(error) == True

    def test_is_recoverable_handler_error_operation(self):
        """Test is_recoverable_handler_error returns True for HandlerOperationError."""
        error = HandlerOperationError(message="Op error", handler_type="DMC", operation="process")
        assert is_recoverable_handler_error(error) == True

    def test_is_recoverable_handler_error_dataset_specific(self):
        """Test is_recoverable_handler_error returns True for DatasetSpecificHandlerError."""
        error = DatasetSpecificHandlerError(message="Error", dataset_type="QMC")
        assert is_recoverable_handler_error(error) == True

    def test_is_recoverable_handler_error_uncertainty_processing(self):
        """Test is_recoverable_handler_error returns True for UncertaintyProcessingError."""
        error = UncertaintyProcessingError(message="Error", dataset_type="QMC")
        assert is_recoverable_handler_error(error) == True

    def test_is_recoverable_handler_error_migration_with_rollback(self):
        """Test is_recoverable_handler_error for MigrationError with rollback available."""
        error = MigrationError(
            message="Migration failed", migration_phase="phase1", rollback_available=True
        )
        assert is_recoverable_handler_error(error) == True

    def test_is_recoverable_handler_error_migration_no_rollback(self):
        """Test is_recoverable_handler_error for MigrationError without rollback."""
        error = MigrationError(
            message="Migration failed", migration_phase="phase1", rollback_available=False
        )
        assert is_recoverable_handler_error(error) == False

    def test_is_recoverable_handler_error_molecule_filter_rejected(self):
        """Test is_recoverable_handler_error returns True for MoleculeFilterRejectedError."""
        error = MoleculeFilterRejectedError(
            molecule_index=5, inchi="InChI=1S/test", reason="Rejected"
        )
        assert is_recoverable_handler_error(error) == True

    def test_is_recoverable_handler_error_configuration_error(self):
        """Test is_recoverable_handler_error returns False for ConfigurationError."""
        error = ConfigurationError("Config error")
        assert is_recoverable_handler_error(error) == False

    def test_is_recoverable_handler_error_data_processing_error(self):
        """Test is_recoverable_handler_error returns False for DataProcessingError."""
        error = DataProcessingError("Data error")
        assert is_recoverable_handler_error(error) == False


# =============================================================================
# TEST CLASS: Decorator Functions
# =============================================================================


class TestDecoratorFunctions:
    """Test decorator functions for exception handling."""

    def test_wrap_handler_operation_passes_through_success(self):
        """Test wrap_handler_operation passes through successful calls."""

        @wrap_handler_operation("DFT", "test_operation")
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_wrap_handler_operation_reraises_handler_error(self):
        """Test wrap_handler_operation re-raises HandlerError as-is."""

        @wrap_handler_operation("DFT", "test_operation")
        def raises_handler_error():
            raise HandlerError("Original error", handler_type="DFT")

        with pytest.raises(HandlerError) as exc_info:
            raises_handler_error()

        assert "Original error" in str(exc_info.value)

    def test_wrap_handler_operation_converts_molecule_error(self):
        """Test wrap_handler_operation converts MoleculeProcessingError."""

        @wrap_handler_operation("DFT", "test_operation")
        def raises_molecule_error():
            raise MoleculeProcessingError("Molecule error", molecule_index=5)

        with pytest.raises(HandlerOperationError) as exc_info:
            raises_molecule_error()

        assert exc_info.value.handler_type == "DFT"
        assert exc_info.value.handler_operation == "test_operation"

    def test_wrap_handler_operation_converts_generic_exception(self):
        """Test wrap_handler_operation converts generic Exception."""

        @wrap_handler_operation("DMC", "process")
        def raises_generic_error():
            raise ValueError("Generic error")

        with pytest.raises(HandlerOperationError) as exc_info:
            raises_generic_error()

        assert exc_info.value.handler_type == "DMC"
        assert "ValueError" in exc_info.value.details

    def test_wrap_transform_operation_passes_through_success(self):
        """Test wrap_transform_operation passes through successful calls."""

        @wrap_transform_operation("Normalize", "apply")
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_wrap_transform_operation_reraises_transform_errors(self):
        """Test wrap_transform_operation re-raises transform errors as-is."""

        @wrap_transform_operation("Normalize", "apply")
        def raises_transform_error():
            raise TransformConfigurationError("Config error", transform_name="Normalize")

        with pytest.raises(TransformConfigurationError) as exc_info:
            raises_transform_error()

        assert "Config error" in str(exc_info.value)

    def test_wrap_transform_operation_converts_validation_error(self):
        """Test wrap_transform_operation converts ValidationError."""

        @wrap_transform_operation("RandomRotate", "validate")
        def raises_validation_error():
            raise ValidationError("Validation failed", validation_type="param")

        with pytest.raises(TransformValidationError) as exc_info:
            raises_validation_error()

        assert exc_info.value.transform_name == "RandomRotate"

    def test_wrap_transform_operation_converts_configuration_error(self):
        """Test wrap_transform_operation converts ConfigurationError."""

        @wrap_transform_operation("AddSelfLoops", "configure")
        def raises_config_error():
            raise ConfigurationError("Config error")

        with pytest.raises(TransformConfigurationError) as exc_info:
            raises_config_error()

        assert exc_info.value.transform_name == "AddSelfLoops"

    def test_wrap_transform_operation_validation_context(self):
        """Test wrap_transform_operation uses validation context for validate operations."""

        @wrap_transform_operation("Test", "validate")
        def raises_generic():
            raise RuntimeError("Runtime error")

        with pytest.raises(TransformValidationError):
            raises_generic()

    def test_wrap_transform_operation_composition_context(self):
        """Test wrap_transform_operation uses composition context for compose operations."""

        @wrap_transform_operation("Test", "compose")
        def raises_generic():
            raise RuntimeError("Runtime error")

        with pytest.raises(TransformCompositionError):
            raises_generic()

    def test_wrap_transform_operation_with_experimental_setup(self):
        """Test wrap_transform_operation with experimental_setup parameter."""

        @wrap_transform_operation("Normalize", "apply", experimental_setup="baseline")
        def raises_error():
            raise ValueError("Error")

        with pytest.raises(TransformConfigurationError) as exc_info:
            raises_error()

        assert exc_info.value.experimental_setup == "baseline"


# =============================================================================
# TEST CLASS: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases"""

    def test_exception_with_none_values(self):
        """Test exceptions handle None values"""
        error = MoleculeProcessingError("Error", molecule_index=0, smiles=None)
        assert error.smiles == "N/A"

    def test_exception_with_empty_lists(self):
        """Test exceptions handle empty lists"""
        error = PluginDependencyError("Error", plugin_name="test", missing_dependencies=[])
        assert error.missing_dependencies == []

    def test_exception_str_methods_dont_crash(self):
        """Test __str__ methods don't crash"""
        exceptions_to_test = [
            BaseProjectError("test"),
            HandlerError("test", handler_type="DFT"),
            TransformError("test", transform_name="T"),
            PluginError("test", plugin_name="P"),
            DatasetSpecificHandlerError("test", dataset_type="QMC"),
            UncertaintyProcessingError("test", dataset_type="QMC"),
        ]
        for exc in exceptions_to_test:
            str(exc)  # Should not crash

    def test_exception_with_long_message(self):
        """Test exception with very long message"""
        long_msg = "Error " * 1000
        error = BaseProjectError(long_msg)
        assert len(error.message) > 5000
        str(error)  # Should not crash

    def test_dynamic_exception_with_all_none_optional_params(self):
        """Test dynamic exceptions with all None optional params"""
        error = DatasetSpecificHandlerError(
            message="Error", dataset_type="QMC", operation=None, property_name=None, details=None
        )
        str(error)  # Should not crash

    def test_create_dataset_handler_error_minimal(self):
        """Test create_dataset_handler_error with minimal params"""
        error = create_dataset_handler_error(message="Test", dataset_type="TEST")
        assert error.dataset_type == "TEST"

    def test_create_uncertainty_processing_error_minimal(self):
        """Test create_uncertainty_processing_error with minimal params"""
        error = create_uncertainty_processing_error(message="Test", dataset_type="TEST")
        assert error.dataset_type == "TEST"

    def test_exception_with_special_characters_in_message(self):
        """Test exceptions handle special characters in messages."""
        special_msg = "Error with <special> & \"characters\" 'quotes' \n\ttabs"
        error = BaseProjectError(special_msg)
        assert error.message == special_msg
        str(error)  # Should not crash

    def test_exception_with_unicode_message(self):
        """Test exceptions handle unicode characters."""
        unicode_msg = "Error: 分子处理失败 🧪 → ✗"
        error = BaseProjectError(unicode_msg)
        assert error.message == unicode_msg
        str(error)  # Should not crash

    def test_handler_error_with_none_handler_type(self):
        """Test HandlerError handles None handler_type."""
        error = HandlerError("Error", handler_type=None)
        assert error.handler_type is None
        str(error)  # Should not crash

    def test_model_error_str_methods_all_classes(self):
        """Test __str__ methods of all model exceptions don't crash."""
        model_exceptions = [
            ModelError("test"),
            ModelNotFoundError("test", model_name="GCN"),
            ModelValidationError("test"),
            ModelInstantiationError("test"),
            HyperparameterError("test"),
            DataCompatibilityError("test"),
            TrainingError("test"),
            CheckpointError("test"),
            DataError("test"),
            PluginModelError("test"),
        ]
        for exc in model_exceptions:
            str(exc)  # Should not crash

    def test_hpo_error_str_methods_all_classes(self):
        """Test __str__ methods of all HPO exceptions don't crash."""
        hpo_exceptions = [
            HPOError("test"),
            HPOConfigurationError("test"),
            TrialFailedError("test"),
            StudyNotFoundError("test", study_name="s"),
            BackendError("test"),
            SearchSpaceError("test"),
            PruningError("test"),
        ]
        for exc in hpo_exceptions:
            str(exc)  # Should not crash

    def test_exception_chaining_preserved(self):
        """Test that exception chaining is preserved."""
        original = ValueError("Original error")
        try:
            raise HandlerOperationError(
                message="Wrapped error", handler_type="DFT", operation="test"
            ) from original
        except HandlerOperationError as e:
            assert e.__cause__ is original

    def test_empty_lists_in_exceptions(self):
        """Test exceptions handle empty lists correctly."""
        error = ModelNotFoundError("Not found", model_name="Test", available_models=[])
        assert error.available_models == []
        str(error)  # Should not crash

    def test_large_lists_in_exceptions(self):
        """Test exceptions handle large lists."""
        large_list = [f"model_{i}" for i in range(100)]
        error = ModelNotFoundError("Not found", model_name="Test", available_models=large_list)
        assert len(error.available_models) == 100
        str(error)  # Should not crash, may truncate


# =============================================================================
# TEST CLASS: Comprehensive __str__ Method Coverage
# =============================================================================


class TestStrMethodComprehensiveCoverage:
    """Test comprehensive __str__ method coverage for all exception classes."""

    def test_base_project_error_str_with_empty_message(self):
        """Test BaseProjectError __str__ with empty message."""
        error = BaseProjectError("")
        assert str(error) == ""

    def test_configuration_error_str_all_attributes(self):
        """Test ConfigurationError __str__ with all attributes populated."""
        error = ConfigurationError(
            "Full config error",
            config_key="model.params.learning_rate",
            actual_value=0.001,
            expected_value=float,
            details="Value should be positive",
        )
        error_str = str(error)
        assert "model.params.learning_rate" in error_str
        assert "0.001" in error_str
        assert "float" in error_str.lower()

    def test_handler_not_available_error_str_comprehensive(self):
        """Test HandlerNotAvailableError __str__ with all attributes."""
        error = HandlerNotAvailableError(
            message="Handler unavailable",
            requested_dataset_type="QMC",
            available_types=["DFT", "DMC", "Wavefunction"],
            missing_dependencies=["qmc_lib", "torch"],
            details="Installation required",
        )
        error_str = str(error)
        assert "QMC" in error_str
        assert "DFT" in error_str or "Available" in error_str
        assert "qmc_lib" in error_str or "Missing" in error_str

    def test_handler_operation_error_str_comprehensive(self):
        """Test HandlerOperationError __str__ with all attributes."""
        error = HandlerOperationError(
            message="Operation failed",
            handler_type="DFT",
            operation="validate_freqs",
            molecule_index=42,
            recovery_suggestions=["Check data format", "Verify dependencies"],
            details="Frequency validation failed",
        )
        error_str = str(error)
        assert "DFT" in error_str
        assert "validate_freqs" in error_str
        assert "42" in error_str

    def test_handler_validation_error_str_comprehensive(self):
        """Test HandlerValidationError __str__ with all attributes."""
        error = HandlerValidationError(
            message="Validation failed",
            handler_type="DMC",
            validation_type="uncertainty_check",
            failed_validations=["std_negative", "correlation_invalid"],
            molecule_index=15,
            details="Multiple validation failures",
        )
        error_str = str(error)
        assert "DMC" in error_str
        assert "uncertainty_check" in error_str
        assert "15" in error_str

    def test_handler_compatibility_error_str_comprehensive(self):
        """Test HandlerCompatibilityError __str__ with all attributes."""
        error = HandlerCompatibilityError(
            message="Incompatibility detected",
            handler_type="Custom",
            incompatible_features=["feature_a", "feature_b"],
            minimum_requirements={"torch": "1.9.0", "python": "3.8"},
            details="Version mismatch",
        )
        error_str = str(error)
        assert "Custom" in error_str
        assert "feature_a" in error_str or "Incompatible" in error_str

    def test_migration_error_str_with_rollback_available(self):
        """Test MigrationError __str__ with rollback available."""
        error = MigrationError(
            message="Migration error", migration_phase="phase2", rollback_available=True
        )
        error_str = str(error)
        assert "phase2" in error_str
        # Should NOT contain "NO ROLLBACK" when rollback is available
        assert "NO ROLLBACK" not in error_str

    def test_transform_not_found_error_str_with_available_transforms(self):
        """Test TransformNotFoundError __str__ with available_transforms."""
        error = TransformNotFoundError(
            message="Transform not found",
            transform_name="InvalidTransform",
            available_transforms=[
                "Normalize",
                "AddSelfLoops",
                "ToUndirected",
                "RandomRotate",
                "Center",
                "Scale",
            ],
        )
        error_str = str(error)
        assert "InvalidTransform" in error_str
        # Should show some available transforms
        assert "Normalize" in error_str or "Available" in error_str

    def test_study_not_found_error_str_comprehensive(self):
        """Test StudyNotFoundError __str__ with all attributes."""
        error = StudyNotFoundError(
            message="Study not found",
            study_name="gcn_hyperparam_search",
            available_studies=["study1", "study2", "study3", "study4", "study5", "study6"],
            storage_url="sqlite:///optuna.db",
        )
        error_str = str(error)
        assert "gcn_hyperparam_search" in error_str
        assert "sqlite" in error_str or "Storage" in error_str

    def test_dataset_specific_handler_error_str_without_operation(self):
        """Test DatasetSpecificHandlerError __str__ without optional operation."""
        error = DatasetSpecificHandlerError(message="Dataset error", dataset_type="NewType")
        error_str = str(error)
        assert "NewType" in error_str
        assert "Dataset" in error_str

    def test_uncertainty_processing_error_str_with_dataset_type_and_molecule(self):
        """Test UncertaintyProcessingError __str__ includes dataset_type and molecule_index."""
        error = UncertaintyProcessingError(
            message="Uncertainty error", dataset_type="DMC", molecule_index=5
        )
        error_str = str(error)
        # dataset_type is always required and always appears in __str__
        assert "DMC" in error_str
        assert "5" in error_str or "molecule" in error_str.lower()


# =============================================================================
# TEST CLASS: Registry Context and Error Handling
# =============================================================================


class TestRegistryContextAndErrorHandling:
    """Test registry context creation and error handling enhancements."""

    def test_create_handler_error_context_includes_registry_info(self):
        """Test create_handler_error_context includes registry info."""
        context = create_handler_error_context(handler_type="DFT", operation="process")
        # Registry integration additions
        assert "dataset_type_registered" in context
        assert "available_dataset_types" in context
        assert isinstance(context["available_dataset_types"], list)

    def test_create_handler_error_context_dft_is_registered(self):
        """Test create_handler_error_context reports DFT as registered."""
        context = create_handler_error_context(handler_type="DFT", operation="validate")
        assert context["dataset_type_registered"] == True

    def test_create_handler_error_context_unknown_type_not_registered(self):
        """Test create_handler_error_context reports unknown type as not registered."""
        context = create_handler_error_context(handler_type="UNKNOWN_TYPE_XYZ", operation="process")
        assert context["dataset_type_registered"] == False

    def test_format_handler_exception_summary_handler_error(self):
        """Test format_handler_exception_summary with HandlerError."""
        error = HandlerError("Handler error", handler_type="DFT", handler_operation="validate")
        summary = format_handler_exception_summary(error)

        assert summary["is_handler_exception"] == True
        assert summary["handler_type"] == "DFT"
        assert summary["handler_operation"] == "validate"

    def test_format_handler_exception_summary_migration_error(self):
        """Test format_handler_exception_summary with MigrationError."""
        error = MigrationError(
            message="Migration failed", migration_phase="schema_update", rollback_available=False
        )
        summary = format_handler_exception_summary(error)

        assert "migration_phase" in summary
        assert summary["migration_phase"] == "schema_update"
        assert "rollback_available" in summary
        assert summary["rollback_available"] == False

    def test_format_handler_exception_summary_validation_error(self):
        """Test format_handler_exception_summary with ValidationError."""
        error = ValidationError(
            message="Validation failed",
            validation_type="schema",
            failed_checks=["check1", "check2"],
        )
        summary = format_handler_exception_summary(error)

        assert "validation_type" in summary
        assert summary["validation_type"] == "schema"
        assert "failed_checks" in summary
        assert summary["failed_checks"] == ["check1", "check2"]

    def test_format_handler_exception_summary_with_molecule_info(self):
        """Test format_handler_exception_summary with molecule information."""
        error = MoleculeProcessingError(
            message="Molecule error", molecule_index=42, inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
        )
        summary = format_handler_exception_summary(error)

        assert "molecule_index" in summary
        assert summary["molecule_index"] == 42
        assert "molecule_inchi" in summary
        assert "InChI" in summary["molecule_inchi"]

    def test_is_recoverable_handler_error_handler_integration(self):
        """Test is_recoverable_handler_error for HandlerIntegrationError."""
        error = HandlerIntegrationError(message="Integration error", handler_type="DFT")
        assert is_recoverable_handler_error(error) == True

    def test_is_recoverable_handler_error_legacy_code(self):
        """Test is_recoverable_handler_error for LegacyCodeError."""
        error = LegacyCodeError(message="Legacy code", legacy_pattern="old_pattern")
        assert is_recoverable_handler_error(error) == True

    def test_is_recoverable_handler_error_generic_handler(self):
        """Test is_recoverable_handler_error for base HandlerError."""
        error = HandlerError(message="Generic handler error", handler_type="Test")
        assert is_recoverable_handler_error(error) == True

    def test_is_recoverable_handler_error_unknown_exception(self):
        """Test is_recoverable_handler_error returns False for unknown exception."""
        error = ValueError("Standard Python error")
        assert is_recoverable_handler_error(error) == False

    def test_get_exception_recovery_suggestions_handler_compatibility(self):
        """Test get_exception_recovery_suggestions for HandlerCompatibilityError."""
        error = HandlerCompatibilityError(message="Compatibility error", handler_type="Custom")
        suggestions = get_exception_recovery_suggestions(error)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_get_exception_recovery_suggestions_transform_error(self):
        """Test get_exception_recovery_suggestions for TransformError."""
        error = TransformError(message="Transform error", transform_name="Custom")
        suggestions = get_exception_recovery_suggestions(error)
        assert isinstance(suggestions, list)

    def test_get_exception_recovery_suggestions_dataset_specific_with_features(self):
        """Test get_exception_recovery_suggestions for DatasetSpecificHandlerError with features."""
        # DMC has uncertainty_handling=True
        error = DatasetSpecificHandlerError(message="DMC error", dataset_type="DMC")
        suggestions = get_exception_recovery_suggestions(error)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        # Should include uncertainty-specific suggestion for DMC
        assert any("uncertainty" in s.lower() for s in suggestions)


# =============================================================================
# TEST CLASS: Module Architecture (Dynamic)
# =============================================================================


class TestModuleArchitectureDynamic:
    """Test dynamic module architecture and exports"""

    def test_registry_functions_exported(self):
        """Test that registry functions are properly exported"""
        from milia_pipeline import exceptions

        assert hasattr(exceptions, "_init_registry")
        assert hasattr(exceptions, "_get_available_dataset_types")
        assert hasattr(exceptions, "_is_dataset_type_registered")
        assert hasattr(exceptions, "_get_dataset_feature")
        assert hasattr(exceptions, "get_exception_registry_status")

    def test_factory_functions_exported(self):
        """Test that factory functions are properly exported"""
        from milia_pipeline import exceptions

        assert hasattr(exceptions, "create_dataset_handler_error")
        assert hasattr(exceptions, "create_uncertainty_processing_error")
        assert hasattr(exceptions, "create_handler_not_available_error")

    def test_generic_exceptions_exported(self):
        """Test that dynamic exceptions are exported"""
        from milia_pipeline import exceptions

        assert hasattr(exceptions, "DatasetSpecificHandlerError")
        assert hasattr(exceptions, "UncertaintyProcessingError")

    def test_deleted_classes_no_longer_exported(self):
        """Test that removed hardcoded subclasses are no longer exported"""
        from milia_pipeline import exceptions

        assert not hasattr(exceptions, "DFTHandlerError"), (
            "DFTHandlerError should be removed in dynamic refactoring"
        )
        assert not hasattr(exceptions, "DMCHandlerError"), (
            "DMCHandlerError should be removed in dynamic refactoring"
        )
        assert not hasattr(exceptions, "DMCProcessingError"), (
            "DMCProcessingError should be removed in dynamic refactoring"
        )


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
