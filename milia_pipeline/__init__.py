# milia_pipeline/__init__.py

"""
Milia Pipeline - Molecular Graph Dataset Processing
====================================================

A comprehensive, production-ready molecular data processing pipeline for quantum
mechanical calculations and graph-based molecular representation learning.

The milia Pipeline provides modular tools for molecular conversion, feature
enrichment, dataset handling, preprocessing, and extensible transformation systems
with full plugin support.

Core Features
-------------
- **Molecular Processing**: Multi-format molecular conversion with RDKit integration
- **Feature Enrichment**: Automated structural and chemical feature extraction
- **Dataset Management**: PyTorch Geometric compatible dataset implementations
- **Preprocessing**: Modular wavefunction data preprocessing (MOLDEN, FCHK formats)
- **Transformations**: Extensible graph transformation system with experimental setup
- **Plugin System**: Three-tier plugin architecture for custom extensions
- **Handler Pattern**: Unified dataset handling with DFT/DMC support
- **Configuration**: Schema-validated YAML configuration system
- **CLI Interface**: Comprehensive command-line tools with interactive mode
- **Model Training**: GNN model training with HPO support (Phase 9)
- **Post-Training**: Prediction/inference workflow with checkpoint support (Phase 5b)

Quick Start
-----------
Basic usage with CLI:

    >>> from milia_pipeline import create_cli_manager, setup_logging
    >>> 
    >>> # Setup logging
    >>> logger = setup_logging(log_level="INFO")
    >>> 
    >>> # Create CLI manager and parse arguments
    >>> cli = create_cli_manager(logger=logger)
    >>> args = cli.parse_args(['--config', 'config.yaml', '--process'])
    >>> 
    >>> # Load and validate configuration
    >>> config = cli.load_and_merge_config(args)
    >>> cli.validate_args(args, config)

Prediction mode (Phase 5b):

    >>> # Run inference on new molecules using trained model
    >>> args = cli.parse_args([
    ...     '--predict',
    ...     '--model-path', './checkpoints/best_model.pt',
    ...     '--test-path', './molecules.csv',
    ...     '--preds-path', './predictions.csv'
    ... ])

Programmatic API usage:

    >>> from milia_pipeline.config import load_config
    >>> from milia_pipeline.handlers import create_handler
    >>> from milia_pipeline.datasets import miliaDataset
    >>> 
    >>> # Load configuration
    >>> config = load_config('config.yaml')
    >>> 
    >>> # Create dataset handler
    >>> handler = create_handler(
    ...     dataset_type='DFT',
    ...     config=config,
    ...     logger=logger
    ... )
    >>> 
    >>> # Process dataset
    >>> dataset = miliaDataset(
    ...     root='./data',
    ...     handler=handler,
    ...     transform=my_transforms
    ... )

Module Organization
-------------------
The package is organized into 7 core modules:

**config/**
    Configuration management, validation, and schema definitions
    - load_config, YAMLSchemaValidator, ValidationConfig
    - get_dataset_type, get_transformation_config, get_experimental_setup

**molecules/**
    Molecular processing, conversion, and feature enrichment
    - MoleculeConverter, MoleculeValidator, MoleculeFeatureEnricher
    - Property enrichment and structural feature extraction

**transformations/**
    Graph transformation system with experimental setup support
    - get_graph_transforms, PluginRegistry, PluginValidator
    - Custom transforms and research API

**datasets/**
    PyTorch Geometric dataset implementations
    - miliaDataset with handler integration

**handlers/**
    Dataset handler pattern implementation
    - create_handler, DFTDatasetHandler, DMCDatasetHandler
    - Unified processing interface

**preprocessing/**
    Wavefunction data preprocessing system
    - WavefunctionPreprocessor, PreprocessorRegistry
    - Multi-format support (MOLDEN, FCHK)

**descriptors/**
    Molecular descriptor calculation with plugin support
    - DescriptorCalculator, DescriptorRegistry
    - Extensible descriptor plugin system

**models/**
    Model training, evaluation, and post-training inference
    - ModelFactory, Trainer, DataSplitter
    - HPOManager for hyperparameter optimization
    
**models/post_training/** (Phase 5b)
    Post-training inference and prediction workflow
    - Predictor, ModelLoader, load_model
    - DataConverterRegistry for multi-format input support
    - FineTuner, FreezeStrategy for transfer learning

Exception Hierarchy
-------------------
The package provides a comprehensive three-tier exception hierarchy:

**Tier 1: Base Exception**
    BaseProjectError - Root exception for all project errors

**Tier 2: Domain Base Classes**
    - ConfigurationError - Configuration issues
    - DataProcessingError - Data processing failures
    - MoleculeProcessingError - Molecule processing failures
    - HandlerError - Handler pattern issues
    - TransformError - Transformation errors
    - PluginError - Plugin system errors
    - ValidationError - Validation failures
    - DescriptorError - Descriptor calculation errors

**Tier 3: Specialized Exceptions**
    50+ specific exception classes for precise error handling

**Phase 7 Additions (Registry Integration)**
    - UncertaintyProcessingError - Generic base for uncertainty-enabled datasets
    - DatasetSpecificHandlerError - Generic base for dataset-specific handlers
    - Factory functions for dynamic exception creation
    - Registry integration for dataset type validation
    - CLI registry status diagnostics via get_cli_registry_status()

Plugin System
-------------
Three plugin systems for extensibility:

1. **Descriptor Plugins**: Custom molecular descriptor calculations
2. **Transformation Plugins**: Custom data augmentation and preprocessing
3. **General Plugins**: Arbitrary functionality extensions

Plugin discovery is automatic via YAML-based configuration. See plugins/
directory for templates and examples.

Configuration System
--------------------
YAML-based configuration with Pydantic schema validation:

- Schema validation with detailed error messages
- Type-safe configuration containers
- Hierarchical configuration access
- CLI override support (CLI > Config File > Defaults)

Logging System
--------------
Comprehensive logging with specialized adapters:

- HandlerLoggerAdapter: Handler-specific logging
- MigrationLoggerAdapter: Migration operation tracking
- TransformLoggerAdapter: Transformation system logging
- Structured logging for systematic workflows

Architecture Patterns
---------------------
**Handler-First Architecture**
    All dataset operations use handler-based processing for consistency

**Plugin System Architecture**
    Three-tier plugin system with automatic discovery and validation

**Registry Pattern**
    Component registration for preprocessors, transforms, and descriptors

**Configuration Management**
    Multi-layered configuration with schema validation

**Fail-Fast Validation**
    Early error detection before processing begins

Examples
--------
See examples/ directory for:
- Preprocessing configurations
- Custom transform implementations
- Plugin development templates
- Dataset processing workflows

Testing
-------
Comprehensive test suite with 60+ tests:
- Unit tests for all core components
- Integration tests for cross-component interactions
- Performance benchmarks
- Validation scripts

Dependencies
------------
Core:
    - PyTorch Geometric (graph neural networks)
    - RDKit (molecular informatics)
    - NumPy (numerical computing)
    - Pydantic (data validation)
    - PyYAML (configuration)

Optional:
    - IOData (quantum chemistry formats)
    - Various plugins may have additional dependencies

Development Status
------------------
Version: 1.0.0
Status: Production-Ready
Python: 3.8+
License: See LICENSE file

For More Information
--------------------
- Documentation: See docs/ directory
- Examples: See examples/ directory
- API Reference: Module docstrings
- Plugins: See plugins/ directory templates

Notes
-----
- Handler-based architecture is mandatory (no legacy modes)
- Plugin security: Only enable trusted plugins
- Configuration validation is strongly recommended
- Memory-efficient binary formats (NPZ) for large datasets

"""

__version__ = "1.1.0"  # Phase 5b: Post-training inference support
__author__ = "milia Pipeline Development Team"
__license__ = "See LICENSE file"
__maintainer__ = "milia Pipeline Development Team"
__status__ = "Production"

# ==========================================
# CORE API EXPORTS
# ==========================================

# --- CLI Management ---
from milia_pipeline.cli_manager import (
    CLIManager,
    CLIValidationError,
    create_cli_manager,
    parse_cli_args,
    # PHASE 7: Registry integration diagnostics
    get_cli_registry_status,
)

# --- Post-Training Module (Phase 5b - Conditional) ---
# Prediction/inference workflow using trained checkpoints
_POST_TRAINING_AVAILABLE = False
try:
    from milia_pipeline.models.post_training import (
        # Model Loading
        ModelLoader,
        load_model,
        load_model_only,
        # Prediction
        Predictor,
        predict,
        # Checkpoint Management
        CheckpointManager,
        CHECKPOINT_FORMAT_VERSION,
    )
    _POST_TRAINING_AVAILABLE = True
except ImportError:
    # Post-training module not yet implemented or dependencies not available
    ModelLoader = None
    load_model = None
    load_model_only = None
    Predictor = None
    predict = None
    CheckpointManager = None
    CHECKPOINT_FORMAT_VERSION = None

# --- Post-Training Data Preparation (Conditional) ---
_DATA_PREPARATION_AVAILABLE = False
try:
    from milia_pipeline.models.post_training.data_preparation import (
        convert_to_pyg,
        convert_batch_to_pyg,
        list_available_formats,
        DataConverterRegistry,
    )
    _DATA_PREPARATION_AVAILABLE = True
except ImportError:
    convert_to_pyg = None
    convert_batch_to_pyg = None
    list_available_formats = None
    DataConverterRegistry = None

# --- Transfer Learning (Conditional) ---
_TRANSFER_LEARNING_AVAILABLE = False
try:
    from milia_pipeline.models.post_training.transfer_learning import (
        FineTuner,
        FreezeStrategy,
    )
    _TRANSFER_LEARNING_AVAILABLE = True
except ImportError:
    FineTuner = None
    FreezeStrategy = None

# --- Logging Configuration ---
from milia_pipeline.logging_config import (
    setup_logging,
    HandlerLoggerAdapter,
    MigrationLoggerAdapter,
    TransformLoggerAdapter,
    create_handler_logger,
    create_migration_logger,
    create_transform_logger,
    log_exception_with_context,
    configure_debug_logging_for_handlers,
    configure_debug_logging_for_transforms,
    disable_verbose_third_party_logging,
)

# ==========================================
# EXCEPTION EXPORTS
# ==========================================

# Base and Configuration Exceptions
from milia_pipeline.exceptions import (
    BaseProjectError,
    LoggingConfigurationError,
    ConfigurationError,
    DataProcessingError,
    PreprocessingRequiredError,
    MissingDependencyError,
)

# Molecule Processing Exceptions
from milia_pipeline.exceptions import (
    MoleculeProcessingError,
    MoleculeFilterRejectedError,
    AtomFilterError,
    RDKitConversionError,
    PyGDataCreationError,
    PropertyEnrichmentError,
    StructuralFeatureError,
    VibrationRefinementError,
)

# PHASE 7: Generic Uncertainty Processing Exception
from milia_pipeline.exceptions import (
    UncertaintyProcessingError,
)

# Handler System Exceptions
from milia_pipeline.exceptions import (
    HandlerError,
    HandlerNotAvailableError,
    HandlerConfigurationError,
    HandlerOperationError,
    HandlerValidationError,
    HandlerCompatibilityError,
    HandlerIntegrationError,
    TransformHandlerIntegrationError,
)

# PHASE 7: Generic Dataset-Specific Handler Exception
from milia_pipeline.exceptions import (
    DatasetSpecificHandlerError,
)

# Validation and Compatibility Exceptions
from milia_pipeline.exceptions import (
    ValidationError,
    CompatibilityError,
    MigrationError,
    LegacyCodeError,
)

# Transform System Exceptions
from milia_pipeline.exceptions import (
    TransformError,
    TransformCompatibilityError,
    TransformationError,
    DatasetIntegrationError,
    TransformValidationError,
    TransformCompositionError,
    TransformNotFoundError,
    TransformRegistryError,
    ExperimentalSetupError,
    TransformConfigurationError,
)

# Plugin System Exceptions
from milia_pipeline.exceptions import (
    PluginError,
    PluginValidationError,
    PluginSecurityError,
    PluginDependencyError,
    PluginDiscoveryError,
    PluginRegistrationError,
    PluginLoadError,
)

# Descriptor System Exceptions
from milia_pipeline.exceptions import (
    DescriptorError,
    DescriptorCalculationError,
    DescriptorValidationError,
    DescriptorPluginError,
    DescriptorPluginLoadError,
    DescriptorPluginValidationError,
    DescriptorPluginConfigError,
)

# PHASE 7: Exception Factory Functions and Registry Integration
from milia_pipeline.exceptions import (
    # Factory functions for dynamic exception creation
    create_dataset_handler_error,
    create_uncertainty_processing_error,
    create_handler_not_available_error,
    # Registry status diagnostics
    get_exception_registry_status,
    # Validation function
    validate_exception_hierarchy,
)

# ==========================================
# EXPLICIT PUBLIC API
# ==========================================

__all__ = [
    # Package Metadata
    "__version__",
    "__author__",
    "__license__",
    "__maintainer__",
    "__status__",
    
    # CLI Management
    "CLIManager",
    "CLIValidationError",
    "create_cli_manager",
    "parse_cli_args",
    # PHASE 7: Registry integration diagnostics
    "get_cli_registry_status",
    
    # Post-Training Module (Phase 5b - Conditional)
    # Model Loading
    "ModelLoader",
    "load_model",
    "load_model_only",
    # Prediction
    "Predictor",
    "predict",
    # Checkpoint Management
    "CheckpointManager",
    "CHECKPOINT_FORMAT_VERSION",
    # Data Preparation
    "convert_to_pyg",
    "convert_batch_to_pyg",
    "list_available_formats",
    "DataConverterRegistry",
    # Transfer Learning
    "FineTuner",
    "FreezeStrategy",
    # Availability Flags
    "_POST_TRAINING_AVAILABLE",
    "_DATA_PREPARATION_AVAILABLE",
    "_TRANSFER_LEARNING_AVAILABLE",
    
    # Logging Configuration
    "setup_logging",
    "HandlerLoggerAdapter",
    "MigrationLoggerAdapter",
    "TransformLoggerAdapter",
    "create_handler_logger",
    "create_migration_logger",
    "create_transform_logger",
    "log_exception_with_context",
    "configure_debug_logging_for_handlers",
    "configure_debug_logging_for_transforms",
    "disable_verbose_third_party_logging",
    
    # Base and Configuration Exceptions
    "BaseProjectError",
    "LoggingConfigurationError",
    "ConfigurationError",
    "DataProcessingError",
    "PreprocessingRequiredError",
    "MissingDependencyError",
    
    # Molecule Processing Exceptions
    "MoleculeProcessingError",
    "MoleculeFilterRejectedError",
    "AtomFilterError",
    "RDKitConversionError",
    "PyGDataCreationError",
    "PropertyEnrichmentError",
    "StructuralFeatureError",
    "VibrationRefinementError",
    
    # PHASE 7: Generic Uncertainty Processing Exception
    "UncertaintyProcessingError",
    
    # Handler System Exceptions
    "HandlerError",
    "HandlerNotAvailableError",
    "HandlerConfigurationError",
    "HandlerOperationError",
    "HandlerValidationError",
    "HandlerCompatibilityError",
    "HandlerIntegrationError",
    "TransformHandlerIntegrationError",
    
    # PHASE 7: Generic Dataset-Specific Handler Exception
    "DatasetSpecificHandlerError",
    
    # Validation and Compatibility Exceptions
    "ValidationError",
    "CompatibilityError",
    "MigrationError",
    "LegacyCodeError",
    
    # Transform System Exceptions
    "TransformError",
    "TransformCompatibilityError",
    "TransformationError",
    "DatasetIntegrationError",
    "TransformValidationError",
    "TransformCompositionError",
    "TransformNotFoundError",
    "TransformRegistryError",
    "ExperimentalSetupError",
    "TransformConfigurationError",
    
    # Plugin System Exceptions
    "PluginError",
    "PluginValidationError",
    "PluginSecurityError",
    "PluginDependencyError",
    "PluginDiscoveryError",
    "PluginRegistrationError",
    "PluginLoadError",
    
    # Descriptor System Exceptions
    "DescriptorError",
    "DescriptorCalculationError",
    "DescriptorValidationError",
    "DescriptorPluginError",
    "DescriptorPluginLoadError",
    "DescriptorPluginValidationError",
    "DescriptorPluginConfigError",
    
    # PHASE 7: Exception Factory Functions and Registry Integration
    "create_dataset_handler_error",
    "create_uncertainty_processing_error",
    "create_handler_not_available_error",
    "get_exception_registry_status",
    "validate_exception_hierarchy",
]

# ==========================================
# SUBMODULE DOCUMENTATION
# ==========================================

# Note: Individual submodules should be imported explicitly for full access
# to their APIs. The exports above provide the most commonly used components.
#
# Examples:
#   from milia_pipeline.config import load_config, get_dataset_type
#   from milia_pipeline.molecules import MoleculeConverter, MoleculeValidator
#   from milia_pipeline.transformations import get_graph_transforms, PluginRegistry
#   from milia_pipeline.datasets import miliaDataset
#   from milia_pipeline.handlers import create_handler, DFTDatasetHandler
#   from milia_pipeline.preprocessing import WavefunctionPreprocessor, PreprocessorRegistry
#   from milia_pipeline.descriptors import DescriptorCalculator, DescriptorRegistry
#
# Post-Training (Phase 5b):
#   from milia_pipeline.models.post_training import Predictor, load_model
#   from milia_pipeline.models.post_training.data_preparation import convert_to_pyg
#   from milia_pipeline.models.post_training.transfer_learning import FineTuner
#
# For more detailed API information, refer to the documentation of each submodule.

# ==========================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ==========================================

def get_version() -> str:
    """
    Get the current version of the milia Pipeline package.
    
    Returns:
        str: Version string in semantic versioning format (MAJOR.MINOR.PATCH)
    
    Example:
        >>> from milia_pipeline import get_version
        >>> print(get_version())
        1.0.0
    """
    return __version__


def get_package_info() -> dict:
    """
    Get comprehensive package information including version, status, and metadata.
    
    Returns:
        dict: Dictionary containing package metadata with keys:
            - version: Package version string
            - author: Package author/development team
            - license: License information
            - maintainer: Current maintainer
            - status: Development status
            - python_requires: Minimum Python version
    
    Example:
        >>> from milia_pipeline import get_package_info
        >>> info = get_package_info()
        >>> print(f"milia Pipeline v{info['version']} - {info['status']}")
        milia Pipeline v1.0.0 - Production
    """
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "maintainer": __maintainer__,
        "status": __status__,
        "python_requires": ">=3.8",
    }


def check_dependencies() -> dict:
    """
    Check availability of optional dependencies and features.
    
    Returns:
        dict: Dictionary with dependency availability status:
            - rdkit: RDKit molecular informatics (required)
            - torch_geometric: PyTorch Geometric (required)
            - iodata: IOData quantum chemistry formats (optional)
            - transforms_available: Transform system availability
            - plugins_available: Plugin system availability
            - config_validation_available: Schema validation availability
            - post_training_available: Post-training inference module (Phase 5b)
            - data_preparation_available: Data conversion for inference
            - transfer_learning_available: Transfer learning/fine-tuning support
    
    Example:
        >>> from milia_pipeline import check_dependencies
        >>> deps = check_dependencies()
        >>> if deps['post_training_available']:
        ...     print("Post-training inference available")
        >>> if not deps['iodata']:
        ...     print("Warning: IOData not available, quantum chemistry format support limited")
    """
    dependencies = {}
    
    # Check RDKit
    try:
        import rdkit
        dependencies['rdkit'] = True
    except ImportError:
        dependencies['rdkit'] = False
    
    # Check PyTorch Geometric
    try:
        import torch_geometric
        dependencies['torch_geometric'] = True
    except ImportError:
        dependencies['torch_geometric'] = False
    
    # Check IOData
    try:
        import iodata
        dependencies['iodata'] = True
    except ImportError:
        dependencies['iodata'] = False
    
    # Check transform system
    try:
        from milia_pipeline.transformations import get_graph_transforms
        dependencies['transforms_available'] = True
    except ImportError:
        dependencies['transforms_available'] = False
    
    # Check plugin system
    try:
        from milia_pipeline.transformations.plugin_system import PluginRegistry
        dependencies['plugins_available'] = True
    except ImportError:
        dependencies['plugins_available'] = False
    
    # Check config validation
    try:
        from milia_pipeline.config.config_schemas import YAMLSchemaValidator
        dependencies['config_validation_available'] = True
    except ImportError:
        dependencies['config_validation_available'] = False
    
    # Phase 5b: Check post-training module
    dependencies['post_training_available'] = _POST_TRAINING_AVAILABLE
    dependencies['data_preparation_available'] = _DATA_PREPARATION_AVAILABLE
    dependencies['transfer_learning_available'] = _TRANSFER_LEARNING_AVAILABLE
    
    return dependencies


def initialize_pipeline(
    config_path: str = None,
    log_level: str = "INFO",
    enable_plugins: bool = True,
    validate_config: bool = True
) -> tuple:
    """
    Initialize the milia Pipeline with configuration and logging.
    
    This is a convenience function that sets up the complete pipeline
    environment including logging, configuration loading, and optional
    plugin discovery.
    
    Args:
        config_path (str, optional): Path to YAML configuration file.
            If None, uses default configuration.
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
            Defaults to 'INFO'.
        enable_plugins (bool): Whether to discover and enable plugins.
            Defaults to True.
        validate_config (bool): Whether to validate configuration against schema.
            Defaults to True.
    
    Returns:
        tuple: (logger, config, cli_manager) where:
            - logger: Configured logger instance
            - config: Loaded configuration dictionary (or None if no config_path)
            - cli_manager: CLI manager instance for further operations
    
    Raises:
        ConfigurationError: If configuration loading or validation fails
        LoggingConfigurationError: If logging setup fails
        PluginError: If plugin discovery fails (when enable_plugins=True)
    
    Example:
        >>> from milia_pipeline import initialize_pipeline
        >>> 
        >>> # Basic initialization
        >>> logger, config, cli = initialize_pipeline(
        ...     config_path='config.yaml',
        ...     log_level='INFO'
        ... )
        >>> 
        >>> # Use the initialized components
        >>> logger.info("Pipeline initialized successfully")
        >>> dataset_type = config.get('dataset', {}).get('type', 'DFT')
    
    Notes:
        - This function is a high-level convenience wrapper
        - For more control, use individual initialization functions
        - Plugin discovery may take a few seconds on first run
        - Configuration validation is strongly recommended for production use
    """
    # Setup logging first
    logger = setup_logging(
        enable_handler_logging=True,
        enable_migration_logging=False,
        enable_transform_logging=True,
        log_level=log_level
    )
    
    logger.info(f"Initializing milia Pipeline v{__version__}")
    
    # Create CLI manager
    cli_manager = create_cli_manager(logger=logger)
    
    # Load configuration if provided
    config = None
    if config_path:
        try:
            from milia_pipeline.config.config_loader import load_config
            config = load_config(config_path)
            logger.info(f"Configuration loaded from: {config_path}")
            
            # Validate configuration if requested
            if validate_config:
                try:
                    from milia_pipeline.config.config_schemas import (
                        YAMLSchemaValidator,
                        ValidationConfig
                    )
                    validator = YAMLSchemaValidator()
                    validation_config = ValidationConfig(
                        strict_mode=True,
                        warn_on_unknown_keys=True
                    )
                    validator.validate(config, validation_config)
                    logger.info("Configuration validation: PASSED")
                except ImportError:
                    logger.warning("Configuration validation skipped (schemas not available)")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(
                message=f"Configuration loading failed: {e}",
                config_key=config_path,
                details=str(e)
            )
    
    # Discover plugins if enabled
    if enable_plugins:
        try:
            from milia_pipeline.transformations.plugin_system import PluginRegistry
            plugin_count = len(PluginRegistry.list_plugins())
            logger.info(f"Plugin discovery: Found {plugin_count} plugins")
        except ImportError:
            logger.warning("Plugin system not available")
        except Exception as e:
            logger.warning(f"Plugin discovery failed: {e}")
    
    logger.info("Milia Pipeline initialization complete")
    
    return logger, config, cli_manager


# ==========================================
# BACKWARD COMPATIBILITY
# ==========================================

# Provide backward compatibility for common import patterns
# These aliases ensure existing code continues to work

# Legacy exception imports (for compatibility with older code)
ProjectError = BaseProjectError  # Alias for legacy code
ProcessingError = DataProcessingError  # Alias for legacy code

# ==========================================
# MODULE INITIALIZATION
# ==========================================

# Disable verbose third-party logging by default
# This can be re-enabled by calling the function explicitly
try:
    disable_verbose_third_party_logging()
except Exception:
    # Silently ignore if logging setup fails during import
    pass

# ==========================================
# END OF MODULE
# ==========================================
