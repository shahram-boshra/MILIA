# MILIA (Machine Intellegent Learning Interface Assistant) Pipeline - Complete Structural Overview

## Project Overview

MILIA (Machine Intellegent Learning Interface Assistant) Pipeline is a comprehensive molecular data processing, and machine learning pipeline designed for quantum mechanical calculations, and graph-based molecular representation learning. The project provides modular tools for molecular conversion, feature enrichment, dataset handling, preprocessing, descriptor calculation, graph transformations, and a complete ML/DL model lifecycle management system including model selection, training, evaluation, hardware acceleration, deployment, and production monitoring.

---

## Project Root Structure

```
milia/
├── main.py                          # Main entry point
├── config.yaml.DEPRECATED           # Former monolithic config (superseded by configs/ — YAML Splitting v2.2)
├── configs/                         # Split configuration directory (YAML Splitting Architecture)
│   ├── main.yaml                    # Global settings, dataset_type, paths, data_config.common_settings
│   ├── descriptors.yaml             # Molecular descriptors configuration
│   ├── filter_config.yaml           # Filter settings
│   ├── models.yaml                  # Model configurations
│   ├── plugins.yaml                 # Plugin system configuration
│   ├── structural_features.yaml     # Structural features configuration
│   ├── transformations.yaml         # PyG transformations
│   └── datasets/                    # Dataset-specific configs (FULLY COLOCATED)
│       ├── dft.yaml                 # dft_config + data_config.property_selection.DFT + property_availability.DFT
│       ├── dmc.yaml                 # dmc_config + data_config.property_selection.DMC + property_availability.DMC
│       ├── wavefunction.yaml        # wavefunction_config + data_config.property_selection.Wavefunction + property_availability.Wavefunction
│       ├── qm9.yaml                 # qm9_config + data_config.property_selection.QM9 + property_availability.QM9
│       ├── ani1x.yaml               # ani1x_config + data_config.property_selection.ANI1X + property_availability.ANI1X
│       ├── ani1ccx.yaml             # ani1ccx_config + data_config.property_selection.ANI1CCX + property_availability.ANI1CCX
│       ├── rmd17.yaml               # rmd17_config + data_config.property_selection.RMD17 + property_availability.RMD17
│       ├── ani2x.yaml               # ani2x_config + data_config.property_selection.ANI2X + property_availability.ANI2X
│       ├── xxmd.yaml                # xxmd_config + data_config.property_selection.XXMD + property_availability.XXMD
│       └── qdpi.yaml                # qdpi_config + data_config.property_selection.QDPi + property_availability.QDPi
├── migrate_config.py.DEPRECATED     # One-time YAML splitting migration utility (completed, no longer needed)
├── research_experiments.yaml        # Research experiment configurations
├── setup.py                         # Package setup and installation
├── .gitignore                       # Git ignore rules
├── milia_pipeline/                  # Core package directory
├── tests/                           # Test suite
├── test_data/                       # Test datasets
├── scripts/                         # Utility scripts
├── docs/                            # Documentation
├── examples/                        # Usage examples
├── experiments/                     # Experimental code/configs
└── archive/                         # Archived documentation
```

---

## Core Package Structure (`milia_pipeline/`)

### Package Organization

```
milia_pipeline/
├── __init__.py                      # Package initialization
├── exceptions.py                    # Custom exception definitions
├── logging_config.py                # Logging configuration
├── cli_manager.py                   # Command-line interface manager
├── config/                          # Configuration management
├── molecules/                       # Molecular processing
├── transformations/                 # Data transformation system
├── datasets/                        # Dataset implementations
├── handlers/                        # Dataset handlers
├── preprocessing/                   # Data preprocessing
├── descriptors/                     # Molecular descriptor system
├── models/                          # Model training, deployment, and post-training
└── plugins/                         # Plugin system
```

---

### Root-Level Files

#### exceptions.py (~3809 lines)

**Purpose**: Comprehensive exception hierarchy for the entire pipeline with fully dynamic, registry-based dataset type support

**Architecture**: Three-tier exception hierarchy with registry integration — zero hardcoded dataset-specific names in any executable code, class definition, or fallback path

**Registry Integration**:
- `_REGISTRY_INITIALIZED`, `_REGISTRY_AVAILABLE`: Lazy initialization flags
- `_registry_list_all`, `_registry_get`, `_registry_is_registered`: Function placeholders
- `_discover_dataset_types_from_filesystem() -> List[str]`: Dynamic discovery fallback
- `_init_registry() -> bool`: Lazy registry initialization
- `_get_available_dataset_types() -> List[str]`: Get registered dataset types
- `_is_dataset_type_registered(dataset_type) -> bool`: Check if dataset type exists
- `_get_dataset_feature(dataset_type, feature_name) -> bool`: Query feature flags (registry-only, no legacy fallback)
- `get_exception_registry_status() -> Dict`: Registry diagnostics

**Tier 1 - Base Exception**:
- `BaseProjectError(Exception)`: Root exception for all project-specific errors
  - Attributes: `message`, `details`, `extra_info`

**Tier 2 - Domain-Specific Base Classes**:

| Class | Purpose | Key Attributes |
|-------|---------|----------------|
| `ConfigurationError` | Config file/parameter issues | config_key, actual_value, expected_value |
| `DataProcessingError` | General data processing failures | file_path, operation |
| `MoleculeProcessingError` | Individual molecule failures | molecule_index, smiles, inchi |
| `HandlerError` | Handler pattern errors | handler_type, handler_operation |
| `DatasetSpecificHandlerError` | Generic dataset-specific errors | dataset_type, property_name, operation |
| `UncertaintyProcessingError` | Generic uncertainty errors | dataset_type, uncertainty_property_name |
| `TransformError` | Transformation system errors | transform_name, experimental_setup |
| `PluginError` | Plugin system errors | plugin_name |
| `ValidationError` | Validation failures | validation_type, failed_checks, data_context |
| `ModelError` | Model training errors | model_name |
| `DescriptorError` | Descriptor calculation errors | descriptor_name |
| `HPOError` | Hyperparameter optimization errors | study_name, trial_number |

**Tier 3 - Specialized Exceptions** (60+ classes):

*Configuration Exceptions* (8 classes):
- `LoggingConfigurationError`
- `PreprocessingRequiredError` (with manual command generation)

*Molecule Processing Exceptions* (7 classes):
- `MoleculeFilterRejectedError` (inherits BaseException - expected rejection)
- `RDKitConversionError`
- `PyGDataCreationError`
- `PropertyEnrichmentError`
- `StructuralFeatureError`
- `VibrationRefinementError`

*Handler Exceptions* (12 classes):
- `HandlerNotAvailableError` (auto-populates available_types from registry)
- `HandlerConfigurationError`
- `HandlerOperationError`
- `HandlerValidationError`
- `HandlerCompatibilityError`
- `HandlerIntegrationError`
- `TransformHandlerIntegrationError`
- `DatasetSpecificHandlerError` (generic for all dataset types)

*Transform Exceptions* (10 classes):
- `TransformCompatibilityError`
- `TransformationError`
- `DatasetIntegrationError`
- `TransformValidationError` (inherits ValidationError)
- `TransformCompositionError`
- `TransformNotFoundError`
- `TransformRegistryError`
- `ExperimentalSetupError`
- `TransformConfigurationError` (inherits ConfigurationError)

*Plugin Exceptions* (7 classes):
- `PluginValidationError`
- `PluginSecurityError`
- `PluginDependencyError`
- `PluginDiscoveryError`
- `PluginRegistrationError`
- `PluginLoadError`

*Descriptor Exceptions* (6 classes):
- `DescriptorCalculationError`
- `DescriptorValidationError`
- `DescriptorPluginError`
- `DescriptorPluginLoadError`
- `DescriptorPluginValidationError`
- `DescriptorPluginConfigError`

*Model Exceptions* (10 classes):
- `ModelNotFoundError`
- `ModelValidationError`
- `ModelInstantiationError`
- `HyperparameterError`
- `DataCompatibilityError`
- `TrainingError`
- `CheckpointError`
- `DataError`
- `PluginModelError`

*Dataset Exceptions*:
- `DatasetNotFoundError` (auto-populates available_datasets from registry)

*HPO Exceptions* (6 classes):
- `HPOConfigurationError`
- `TrialFailedError`
- `StudyNotFoundError`
- `ObjectiveFunctionError`
- `SearchSpaceError`
- `PruningError`

*Validation/Compatibility Exceptions* (4 classes):
- `CompatibilityError`
- `MigrationError`
- `LegacyCodeError`
- `MissingDependencyError`
- `AtomFilterError`

**Factory Functions**:
- `create_dataset_handler_error(message, dataset_type, operation, property_name, **kwargs) -> DatasetSpecificHandlerError`: Creates generic handler exception for any dataset type
- `create_uncertainty_processing_error(message, dataset_type, molecule_index, ...) -> UncertaintyProcessingError`: Creates generic uncertainty processing exception for any dataset type
- `create_handler_not_available_error(message, requested_dataset_type, available_types, ...) -> HandlerNotAvailableError`: Auto-fills available_types from registry

**Utility Functions**:
- `create_handler_error_context(handler_type, operation, molecule_index, additional_context) -> Dict`: Standardized error context with registry info
- `wrap_handler_operation(handler_type, operation)`: Decorator for handler operations
- `wrap_transform_operation(transform_name, operation, experimental_setup)`: Decorator for transform operations
- `format_handler_exception_summary(exception) -> Dict`: Formats exception for logging

---

#### cli_manager.py (~3745 lines)

**Purpose**: Enhanced CLI management system with handler-first architecture

**Architecture**: 12 argument groups, multi-system integration, CLI-first override model

**Registry Integration**:
- `_REGISTRY_INITIALIZED`, `_REGISTRY_AVAILABLE`, `_REGISTRY_IMPORT_ERROR`: Lazy initialization
- `_registry_list_all`, `_registry_get`, `_registry_is_registered`: Function placeholders
- `_discover_dataset_types_from_filesystem() -> list`: Dynamic discovery
- `_init_registry() -> bool`: Lazy initialization
- `_get_available_dataset_types() -> list`: Get registered types (populates argparse choices)
- `_is_dataset_type_registered(dataset_type) -> bool`: Validate dataset type
- `_get_dataset_feature(dataset_type, feature_name) -> bool`: Query features (registry-only)
- `_get_dataset_input_format(dataset_type) -> str`: Get input format ('npz', 'tar.gz')
- `get_cli_registry_status() -> Dict`: Registry diagnostics

**CLIValidationError**:
- Custom exception for CLI validation failures

**CLIManager** Class:
- Constructor: `__init__(logger: Optional[logging.Logger] = None)`
- Attributes: `logger`, `parser`, `config`

*Core Methods*:
- `parse_args(args: Optional[List[str]] = None) -> argparse.Namespace`
- `load_and_merge_config(args: argparse.Namespace) -> Dict[str, Any]`
- `validate_args(args: argparse.Namespace, config: Dict[str, Any]) -> bool`
- `run(args: argparse.Namespace) -> int`

*Argument Groups* (12 total):
1. **Basic Options**: root-dir, config (file or directory path), force-reload, chunk-size
2. **Processing Modes**: process, quick-validation, stats-only, interactive
3. **Transformation System**: experimental-setup, validate-transforms, list-transforms, disable-transforms
4. **Plugin Management**: plugin-path, discover-plugins, list-plugins, validate-plugin, enable-plugin, disable-plugin, trust-plugin
5. **Research API**: run-experiment, validate-experiment, list-experiments
6. **Handler System**: validate-handler, test-handler, list-handlers
7. **Filter Options**: max-atoms, min-atoms, max-heavy-atoms, allowed-elements
8. **Validation Options**: validate-config, validate-schema, strict-validation
9. **Logging Options**: log-level, log-file, verbose, quiet
10. **Advanced Options**: num-workers, memory-limit, profile, debug
11. **Training System**: train, model-name, epochs, batch-size, learning-rate, optimizer, scheduler, checkpoint-dir, resume-from, early-stopping, hpo, hpo-trials
12. **Prediction System**: predict, model-checkpoint, output-format, batch-predict

*Plugin Management Methods*:
- `handle_plugin_operations(args) -> bool`
- `_list_plugins_operation(args)`
- `_discover_plugins_operation(args)`
- `_get_plugin_info_operation(plugin_name, args)`
- `_validate_plugin_operation(plugin_name, args)`
- `_validate_plugin_comprehensive_operation(plugin_name, args)`
- `_enable_plugin_operation(plugin_name)`
- `_disable_plugin_operation(plugin_name)`
- `_trust_plugin_operation(plugin_name)`

*Validation Methods*:
- `_validate_chunk_size(args) -> bool`
- `_validate_experimental_setup(args) -> bool`
- `_validate_plugin_paths(args) -> bool`
- `_validate_filter_options(args) -> bool`
- `_validate_preprocessing_options(args) -> bool`

*Interactive Mode*:
- `_run_interactive_wizard(args) -> argparse.Namespace`

**Module-Level Functions**:
- `create_cli_manager(logger: Optional[logging.Logger] = None) -> CLIManager`: Factory function
- `parse_cli_args(args: Optional[List[str]] = None, logger: Optional[logging.Logger] = None) -> Tuple[argparse.Namespace, CLIManager]`: Convenience function

**Feature Availability Flags**:
- `CONFIG_VALIDATION_AVAILABLE`: YAMLSchemaValidator, ValidationConfig
- `TRANSFORMS_AVAILABLE`: get_graph_transforms
- `PLUGIN_SYSTEM_AVAILABLE`: PluginRegistry, PluginValidator, PluginMetadata

**Integration Points**:
- Configuration System: load_config, config_accessors
- Validation System: YAMLSchemaValidator, ValidationConfig
- Transform System: get_graph_transforms
- Plugin System: PluginRegistry, PluginValidator
- Handlers: handler validation and testing
- Research API: experiment execution

---

## Module APIs and Usage Patterns

### Datasets Module API  + PYDANTIC V2

**Architecture**: Protocol + ABC + Explicit Registry | **Thread-Safe**: Yes

The datasets module has been comprehensively refactored to implement a zero-core-file-modification architecture for adding new dataset types. The refactoring follows the Open/Closed Principle with Protocol-based contracts, Abstract Base Classes, and explicit registry pattern.

**Module Constants**:
- `TRANSFORMATION_SYSTEM_VERSION`: "2.1" (Enhanced with Experimental Setup and Standard Transforms)

#### Module Structure

```
milia_pipeline/datasets/
├── __init__.py                      # Module initialization and public API
├── milia_dataset.py                 # miliaDataset class (PyG InMemoryDataset, ~7218 lines)
├── base.py                          # BaseDataset ABC, DatasetMetadata, DatasetSchema, DatasetFeatures ⭐ PYDANTIC V2
├── registry.py                      # DatasetRegistry (thread-safe, testable)
├── protocols.py                     # DatasetHandlerProtocol (11 methods), DatasetConverterProtocol, DatasetValidatorProtocol
└── implementations/                 # Concrete dataset implementations 
    ├── __init__.py                  # Dynamic discovery exports (7 datasets)
    ├── dft.py                       # DFTDataset (@register decorated)
    ├── dmc.py                       # DMCDataset (@register decorated)
    ├── wavefunction.py              # WavefunctionDataset (@register decorated)
    ├── qm9.py                       # QM9Dataset (@register decorated)
    ├── ani1x.py                     # ANI1xDataset (@register decorated)
    ├── ani1ccx.py                   # ANI1ccxDataset (@register decorated)
    └── rmd17.py                     # RMD17Dataset (@register decorated) 
```

#### Key API Functions

| Category | Functions | Description |
|----------|-----------|-------------|
| **Dataset Creation** | `miliaDataset(root, dataset_config, ...)`, `miliaDataset.from_config()` | Create dataset instances |
| **Registry Discovery** | `list_all()`, `get(name)`, `is_registered(name)`, `get_default_registry()` | Query registered datasets |
| **New Dataset Types** | `@register` decorator on `BaseDataset` subclass | Zero-modification extension |
| **Type Checking** | `DatasetHandlerProtocol`, `DatasetConverterProtocol`, `DatasetValidatorProtocol` | Runtime-checkable protocols |
| **Module Info** | `get_module_info()`, `check_dependencies()`, `get_registry_integration_status()` | Diagnostics and status |

*For complete usage examples, see `examples/datasets/` and `MILIA_Adding_New_Datasets_Implementation_Blueprint.md`*

#### Refactored Architecture Components

**Base Classes** (`datasets/base.py`) ⭐ PYDANTIC V2:
- **Import**: `from pydantic.dataclasses import dataclass` (drop-in replacement for runtime validation)
- **DatasetMetadata**: Immutable metadata (Pydantic frozen dataclass)
  - `name`, `version`, `description`, `author`, `license`
  - `__post_init__` validation preserved (called AFTER Pydantic validation)
- **DatasetSchema**: Immutable schema definition (Pydantic frozen dataclass)
  - `required_properties`, `optional_properties`, `identifier_keys`
  - `coordinate_units` ('angstrom' or 'bohr'), `energy_units`
  - `__post_init__` validation for valid coordinate/energy units preserved
- **DatasetFeatures**: Immutable feature flags (Pydantic frozen dataclass)
  - 8 feature flags: vibrational_analysis, uncertainty_handling, atomization_energy, rotational_constants, frequency_analysis, orbital_analysis, homo_lumo_gap, mo_energies
  - `to_dict()` for compatibility, `supports(feature_name)` for checking
- **BaseDataset**: Abstract base class with `__init_subclass__` validation
  - Class attributes: `metadata`, `schema`, `features`, `config_key`
  - Optional: `handler_class`, `converter_class`, `validator_class`
  - Abstract methods: `get_required_properties()`, `get_feature_support()`, `get_molecule_creation_strategy()`
  - Factory method: `create_handler()` with 5-parameter signature

**Protocols** (`datasets/protocols.py`):
- **DatasetHandlerProtocol**: 11-method contract (runtime_checkable)
  1. `get_dataset_type()` → str
  2. `validate_molecule_data(raw_properties_dict, molecule_index, identifier)` → None
  3. `get_required_properties()` → List[str]
  4. `process_property_value(key, value, molecule_index, identifier)` → Any
  5. `enrich_pyg_data(pyg_data, raw_properties_dict, molecule_index, identifier)` → Data
  6. `get_processing_statistics(processed_molecules)` → Dict[str, Any]
  7. `get_supported_structural_features()` → Dict[str, List[str]]
  8. `get_molecular_charge(raw_properties_dict, atomic_numbers, mol_identifier)` → int
  9. `get_molecule_creation_strategy()` → str
  10. `get_transform_recommendations()` → Dict[str, List[str]]
  11. `get_supported_descriptors()` → Dict[str, List[str]]
- **DatasetConverterProtocol**: `convert()`, `supports_format()`
- **DatasetValidatorProtocol**: `validate()`, `get_validation_rules()`

**Registry** (`datasets/registry.py`):
- **DatasetRegistry**: Thread-safe, testable registry class
  - NOT a singleton: isolated instances for testing
  - Methods: `register()`, `unregister()`, `get()`, `get_or_none()`, `list_all()`, `list_all_classes()`, `is_registered()`, `clear()`
  - Cache invalidation: `add_on_change_callback()`, `remove_on_change_callback()`
  - Supports `in` operator, iteration, `len()`
- **Convenience functions**: `register` (decorator), `get`, `list_all`, `is_registered`, `get_default_registry`

**Implementations** (`datasets/implementations/`):
- **DFTDataset**: DFT quantum chemistry with vibrational analysis
- **DMCDataset**: DMC quantum Monte Carlo with uncertainty handling
- **WavefunctionDataset**: Quantum mechanical wavefunction with orbital analysis
- **QM9Dataset**: QM9 quantum chemistry dataset (133,885 small organic molecules, B3LYP/6-31G(2df,p))
- **ANI1xDataset**: ANI-1x quantum chemistry dataset (~5 million DFT conformations, ωB97x/6-31G*, CHNO)
- **ANI1ccxDataset**: ANI-1ccx quantum chemistry dataset (~500k CCSD(T)/CBS conformations, subset of ANI-1x, CHNO)
- **ANI2xDataset**: ANI-2x quantum chemistry dataset (~10 million DFT conformations, ωB97X/6-31G(d), H/C/N/O/S/F/Cl) 
- **RMD17Dataset**: rMD17 quantum chemistry dataset (~1M conformations, PBE/def2-SVP, 10 molecules)

#### Key Features

**Zero-Modification Architecture**:
- Adding new dataset types requires creating only 1 file
- No modifications to core files (config_constants.py, config_containers.py, etc.)
- Explicit registration with `@register` decorator
- Full backward compatibility maintained

**Type Safety**:
- Compile-time validation via `__init_subclass__`
- Protocol-based contracts with runtime checking
- IDE support for autocomplete and refactoring
- Fail-fast: Registration errors caught at import time

**Thread Safety**:
- DatasetRegistry protected by RLock
- Thread-safe registration and lookup
- Cache invalidation callbacks for consumers

**Testability**:
- Non-singleton registry: create isolated instances for testing
- Dependency injection via factory methods
- `clear()` method for test cleanup

#### Registry Integration

The milia_dataset.py now supports registry-based feature queries:
- Feature-based queries replace hardcoded dataset type checks
- Generalized insight extraction (uncertainty, vibrational, orbital)
- Generalized metadata extraction
- Format-driven NPZ metadata extraction (no dataset type guards)
- Dynamic identifier key retrieval via handler/config
- Zero hardcoded dataset type checks in any executable code
- Zero modifications required to add new dataset types

**Registry Methods in milia_dataset.py**:
- `_init_registry()`: Lazy registry initialization
- `_get_dataset_feature(dataset_type, feature_name)`: Feature query for any dataset type (registry-only, no legacy fallback)
- `_get_available_dataset_types()`: Dynamic dataset type list from registry (exclusion list: `['BASE', 'REGISTRY', 'UTILS', 'COMMON', 'PROTOCOLS']`)
- `_is_dataset_type_registered(dataset_type)`: Dataset type validation
- `_get_dataset_specific_insight_types(dataset_type)`: Dynamic insight type determination
- `_extract_uncertainty_specific_insights()`: Generalized uncertainty insights extraction
- `_extract_vibrational_specific_insights()`: Generalized vibrational insights extraction
- `_extract_orbital_specific_insights()`: Generalized orbital insights extraction
- `_extract_uncertainty_metadata_fallback_enhanced()`: Generalized uncertainty metadata
- `_extract_vibrational_metadata_fallback_enhanced()`: Generalized vibrational metadata
- `_extract_orbital_metadata_fallback_enhanced()`: Generalized orbital metadata
- `get_registry_integration_status()`: Registry status reporting

**Dynamic Dataset Discovery**:
- `get_supported_dataset_types()`: Returns dynamically discovered dataset types from registry
- `SUPPORTED_DATASET_TYPES`: DEPRECATED - kept for backward compatibility only

**Module Exports** (`__all__`, 27 exports):
- Primary: `miliaDataset`
- Base: `BaseDataset`, `DatasetMetadata`, `DatasetSchema`, `DatasetFeatures`
- Registry: `DatasetRegistry`, `get_default_registry`, `register`, `get`, `list_all`, `is_registered`
- Protocols: `DatasetHandlerProtocol`, `DatasetConverterProtocol`, `DatasetValidatorProtocol`
- Exceptions: `DatasetRegistrationError`, `DatasetNotFoundError`
- Implementations: `DFTDataset`, `DMCDataset`, `WavefunctionDataset`, `QM9Dataset`, `ANI1xDataset`, `ANI1ccxDataset`, `ANI2xDataset`, `RMD17Dataset` 
- Utilities: `initialize_plugins`, `get_supported_dataset_types`, `SUPPORTED_DATASET_TYPES`

**NOTE - Dynamic Dataset Discovery**: The `datasets/implementations/__init__.py` now uses **dynamic discovery** - all dataset modules are automatically imported and registered. No manual import updates are required when adding new datasets - just create the file with `@register` decorator.

---

### Models Module API

**Version**: 1.6.0 (Checkpoint Featurization Config Persistence) | **Models**: ALL PyG models (dynamically discovered) | **Thread-Safe**: Yes

The models module provides a complete production-ready infrastructure for machine learning model lifecycle management, including model selection, training, evaluation, acceleration, deployment, monitoring, and **post-training inference/transfer learning**.

**⭐ v1.6.0 Checkpoint Featurization Config Persistence** (Fixes 16-21):
- Solves training/prediction feature dimension mismatch problem
- Training captures `structural_features_config` from dataset into checkpoint
- Prediction loads and applies same featurization for dimension compatibility
- `ModelLoader._load()` extracts `data_info` with featurization config
- `Predictor.structural_features_config` property exposes config to callers
- `convert_to_pyg(..., structural_features_config=)` applies training-time features
- `_apply_structural_features_if_available()` post-processing helper function

**⭐ Dependency Injection Refactoring**:
- **REMOVED** `path_utils.py` (Service Locator anti-pattern)
- All post_training components now require explicit `working_root_dir: Path` parameter
- Follows `CallbackFactory` pattern from `models/training/callbacks.py` (lines 851-855)
- Callers compute `working_root_dir` from config and pass it explicitly
- No hidden config loading - all dependencies visible in function signatures
- Breaking change: All public APIs updated from `config=` to `working_root_dir=`

**⭐ Post-Training Extensions**:
- Added `post_training/` submodule for inference and transfer learning
- Model loading from checkpoints with v2.0 self-contained format
- PyG-compatible `Predictor` class for molecular graph inference
- Multi-format data conversion via `DataConverterRegistry`
- Transfer learning via `FineTuner` with freeze strategies
- CLI `--predict` mode for post-training workflows

**⭐ v1.2.0 Dynamic Introspection Refactoring**:
- Replaced static `model_categories.py` (1770 lines) with dynamic `pyg_introspector.py` (1661 lines)
- Runtime discovery of ALL PyG models via signature introspection
- No more hardcoded model lists - supports any PyG model automatically
- Backward-compatible API: `get_model_metadata()`, `get_all_model_names()`, etc.
- Dynamic search space generation for HPO

#### Module Structure

```
milia_pipeline/models/
├── __init__.py                      # Module initialization and public API
├── registry/                        # Model discovery and management
│   ├── __init__.py                  # Registry exports
│   ├── model_registry.py            # ModelRegistry  singleton, dynamic discovery 
│   └── pyg_introspector.py          # PyGModelIntrospector  1661 lines 
├── factory/                         # Model creation and validation
│   ├── __init__.py                  # Factory exports
│   ├── model_factory.py             # ModelFactory  Wrappers, ModelValidator 
│   └── target_selection_config.py   # TargetSelectionConfig, 3 enums (713 lines) 
├── training/                        # Training infrastructure
│   ├── __init__.py                  # Training exports
│   ├── trainer.py                   # Trainer class with checkpoint support
│   ├── callbacks.py                 # Callbacks  6 types + CallbackFactory + structural_features_config 
│   ├── data_preparation.py          # TaskDataPreparer, task-specific data transforms 
│   ├── data_splitting.py            # DataSplitter  5 strategies incl. k-fold 
│   ├── loss_functions.py            # LossRegistry  18 losses, task-aware selection 
│   ├── optimizers.py                # OptimizerRegistry  12 optimizers, dynamic param filtering 
│   ├── schedulers.py                # SchedulerRegistry  13 schedulers, dynamic param filtering 
│   ├── metrics.py                   # MetricsRegistry  12 metrics, task-aware selection 
│   └── visualization.py             # TrainingVisualizer  4 plot types, config-driven 
├── post_training/                   # Post-Training Inference & Transfer Learning
│   ├── __init__.py                  # Public API exports (~350 lines) 
│   ├── checkpoint/                  # Checkpoint management
│   │   ├── __init__.py              # Checkpoint exports
│   │   └── checkpoint_manager.py    # CheckpointManager  DI pattern 
│   ├── inference/                   # Model loading & prediction
│   │   ├── __init__.py              # Inference exports (180 lines)
│   │   ├── model_loader.py          # ModelLoader  DI pattern, data_info extraction 
│   │   └── predictor.py             # Predictor  DI pattern, structural_features_config property 
│   ├── data_preparation/            # Data conversion for inference
│   │   ├── __init__.py              # Data preparation exports (259 lines, 17 exports)
│   │   └── data_converter.py        # DataConverterRegistry  DI pattern, structural_features_config post-processing 
│   └── transfer_learning/           # Fine-tuning infrastructure
│       ├── __init__.py              # Transfer learning exports (168 lines)
│       └── fine_tuner.py            # FineTuner  DI pattern 
├── hpo/                             # Hyperparameter Optimization (HPO) 
│   ├── __init__.py                  # HPO public API exports
│   ├── hpo_config.py                # HPOConfig  7 pruners, 7 samplers 
│   ├── hpo_manager.py               # HPOManager  task-aware data prep 
│   ├── backends/                    # HPO backend implementations
│   │   ├── __init__.py              # Backend exports
│   │   ├── base.py                  # HPOBackendProtocol, get_backend() 
│   │   ├── optuna_backend.py        # OptunaBackend, 7 pruners, 7 samplers 
│   │   └── ray_tune_backend.py      # RayTuneBackend (complete, inactive)
│   ├── callbacks/                   # HPO training callbacks
│   │   ├── __init__.py              # Callback exports
│   │   ├── optuna_callback.py       # OptunaPruningCallback, create_hpo_callback 
│   │   └── ray_tune_callback.py     # RayTuneReportCallback (complete, inactive)
│   ├── search_spaces/               # Search space definition and building
│   │   ├── __init__.py              # Search space exports
│   │   ├── param_types.py           # ParamType (7 types), SearchSpaceParamConfig 
│   │   └── search_space_builder.py  # SearchSpaceBuilder, dynamic spaces 
│   ├── transfer/                    # HPO transfer learning
│   │   ├── __init__.py              # Transfer learning exports
│   │   ├── transfer_manager.py      # HPOTransferManager  4 adaptation methods 
│   │   ├── meta_features.py         # MetaFeatureExtractor  6 categories 
│   │   └── warm_start.py            # WarmStartStrategy  4 methods 
│   ├── nas/                         # Neural Architecture Search
│   │   ├── __init__.py              # NAS exports
│   │   ├── search_space.py          # GNNArchitectureSpace  4 enums, LayerConfig 
│   │   └── nas_manager.py           # NASManager  NASConfig, HeterogeneousGNN 
│   └── analysis/                    # Study analysis and visualization
│       ├── __init__.py              # Analysis exports
│       └── study_analyzer.py        # StudyAnalyzer  AnalysisConfig, 2 enums 
├── builders/                        # Custom architecture building
│   ├── __init__.py                  # Builders exports
│   ├── layer_registry.py            # LayerCategory, FunctionalLayerWrapper, LayerMetadata, LayerRegistry (63+ layers), 3 convenience functions
│   ├── architecture_builder.py      # LayerConfig, ResidualConnection, ArchitectureConfig, ArchitectureBuilder, CustomArchitecture
│   ├── model_composer.py            # ModelSpec, EnsembleConfig, ModelComposer, ParallelEnsemble, SequentialStack, HierarchicalComposition
│   ├── templates.py                 # ArchitectureTemplates (10 templates, 2 utility methods)
│   ├── config_parser.py             # ArchitectureConfigParser, 4 convenience functions (parse/load/validate/save)
│   └── validation.py                # ArchitectureValidator, 2 convenience functions
├── acceleration/                    # Hardware acceleration
│   ├── __init__.py                  # Acceleration exports
│   ├── device_manager.py            # DeviceType, DeviceInfo, DeviceManager, 3 convenience functions
│   ├── distributed_strategies.py    # DistributedStrategy, DistributedBackend, DistributedConfig, DistributedManager, 4 convenience functions
│   ├── memory_optimization.py       # MemoryConfig, MemoryOptimizer, 2 convenience functions
│   └── computation_optimization.py  # ComputationConfig, ComputationOptimizer, 2 convenience functions, @optimize_inference
├── deployment/                      # Model deployment
│   ├── __init__.py                  # Deployment exports
│   ├── deployment_strategies.py     # DeploymentTarget, ServingMode, DeploymentConfig, DeploymentStrategy ABC, 6 strategy implementations, DeploymentManager
│   ├── model_optimization.py        # QuantizationType, PruningType, OptimizationConfig, ModelOptimizer, convenience functions
│   └── monitoring.py                # MetricType, AlertSeverity, DriftType, MonitoringConfig, Alert, ModelMonitor
├── utils/                           # Utilities
│   ├── __init__.py                  # Utils exports
│   ├── config_bridge.py             # ConfigBridge v1.1.0 → Pydantic  13 enums, 31 BaseModel classes ⭐ PYDANTIC
│   └── pyg_integration.py           # PyG utilities  validation, inference, stats 
└── plugins/                         # Model plugin system
    ├── __init__.py                  # Plugin exports
    └── model_plugin_system.py       # ModelPluginLoader, plugin discovery
```

#### Key API Functions

| Category | Functions | Description |
|----------|-----------|-------------|
| **Model Discovery** | `list_models(task_type)`, `get_model_info(name)`, `search_models(tags)` | Query available models |
| **Model Creation** | `create_model(name, hyperparameters, task_type, sample_data)` | Factory pattern creation |
| **Architecture Building** | `ArchitectureBuilder.add_layer()`, `ArchitectureBuilder.build()` | Custom architecture |
| **Ensemble Creation** | `ModelComposer.add_model()`, `ModelComposer.build()` | Ensemble models |
| **Target Selection** | `TargetSelectionConfig.from_config()`, `.resolve()` | Property/index selection |
| **Data Splitting** | `DataSplitter.random_split()`, `stratified_split()`, `scaffold_split()` | Train/val/test splits |
| **Task Preparation** | `TaskDataPreparer.prepare_for_task()`, `prepare_data_for_task()` | Task-specific transforms |
| **Training** | `Trainer(model, ...)`, `Trainer.fit(epochs)` | Training with callbacks |
| **Loss Functions** | `get_loss(name)`, `LossRegistry`, 18 built-in losses | Task-aware loss selection |
| **Optimizers** | `get_optimizer(name)`, `OptimizerRegistry`, 12 built-in | Parameter filtering |
| **Schedulers** | `get_scheduler(name)`, `create_warmup_scheduler()` | 13 PyTorch schedulers |
| **Acceleration** | `DeviceManager`, `DistributedTrainer`, `enable_mixed_precision()` | Hardware optimization |
| **Deployment** | `EdgeDeployer`, `CloudDeployer`, `quantize_model()`, `prune_model()` | Model optimization |
| **Monitoring** | `DriftDetector`, `PerformanceMonitor`, `RetrainingScheduler` | Production monitoring |
| **Plugins** | `discover_plugins()`, `load_plugin()`, `list_plugins()` | Model extension |
| **PyG Integration** | `validate_pyg_data()`, `compute_dataset_statistics()`, `create_dataloader()` | PyG utilities |
| **Post-Training** | `Predictor.from_checkpoint()`, `load_model()`, `predict()` | Inference |
| **Data Conversion** | `convert_to_pyg()`, `convert_batch_to_pyg()`, `DataConverterRegistry` | Multi-format support |
| **Transfer Learning** | `FineTuner.from_checkpoint()`, `FreezeStrategy` | Fine-tuning strategies |

**CLI Prediction**: `python main.py --predict --model-path <path> --test-path <path> --preds-path <path>`

*For complete usage examples, see `examples/models/` and `docs/api/models.md`*

#### Model Categories

The models module dynamically discovers ALL PyTorch Geometric models via runtime introspection, organized into 12 categories:

1. **BASIC_GNN** (8 models): Foundational GNN architectures
   - GCN, GAT, GraphSAGE, GIN, etc.

2. **CONVOLUTIONAL** (18 models): Convolution-based GNNs
   - GCNConv, GATConv, SAGEConv, GINConv, etc.

3. **ATTENTION** (12 models): Attention mechanism models
   - GAT, Transformer, SuperGAT, etc.

4. **SPECTRAL** (8 models): Spectral graph theory models
   - ChebNet, APPNP, ARMA, etc.

5. **MESSAGE_PASSING** (15 models): General message passing
   - MPNN, MetaLayer, NNConv, etc.

6. **POOLING** (10 models): Graph pooling architectures
   - DiffPool, TopKPool, SAGPool, etc.

7. **HIERARCHICAL** (8 models): Multi-scale architectures
   - U-Net, JK-Net, Jumping Knowledge, etc.

8. **RECURRENT** (6 models): RNN-based GNNs
   - GGNN, LSTM-GNN, GRU-GNN, etc.

9. **EQUIVARIANT** (12 models): Equivariant networks
   - SchNet, DimeNet, EGNN, etc.

10. **QUANTUM** (8 models): Quantum chemistry models
    - PaiNN, SpookyNet, ForceNet, etc.

11. **GEOMETRIC** (10 models): 3D geometry models
    - PointNet++, DGCNN, PointConv, etc.

12. **SPECIALIZED** (15 models): Domain-specific models
    - Molecular fingerprints, protein models, etc.

#### Submodule Details

**Registry Submodule** (`models/registry/`):
- **ModelRegistry**: Central registry for model discovery and management 
  - **Thread-safe singleton pattern** with RLock for reentrant locking
  - **Dynamic auto-discovery** via `auto_discover_pyg_models()`:
    - Uses `PyGModelIntrospector` for runtime model discovery
    - Graceful handling of missing/incompatible models
    - `_auto_discovered` flag prevents redundant discovery
  - **Discovery statistics tracking**:
    - `_discovery_stats`: total_attempted, successful, failed, last_discovery
    - `get_availability_report()`: Comprehensive availability report
    - `log_availability_summary()`: Human-readable console summary
  - **Model registration**:
    - `ModelRegistration` dataclass: name, model_class, metadata, is_builtin, plugin_name
    - `register_model()`: Register custom models with validation
    - `unregister_model()`: Remove models from registry
  - **Query methods**:
    - `get_model()`: Get model class by name
    - `get_metadata()`: Get metadata (with dynamic fallback)
    - `get_registration()`: Get full registration object
    - `list_available_models()`: Filter by category, task_type, tags, heterogeneous support
    - `search_models()`: Keyword search in name, description, tags
    - `list_by_category()`: Models organized by category
  - **Plugin management**:
    - `list_plugin_models()`: Models by plugin name
    - `get_builtin_models()`: PyG built-in models only
    - `get_custom_models()`: Custom/plugin models only
  - **Convenience functions**: `get_model()`, `has_model()`, `list_models()`, `get_model_info()`
  - **Global instance**: `registry = ModelRegistry()` (singleton)
- **PyGModelIntrospector**: Dynamic discovery of ALL PyG models 
  - **Dynamic model discovery** via `discover_pyg_models()`:
    - Scans `torch_geometric.nn.models` and submodules at runtime
    - Discovers 120+ models across all PyG versions
  - **Signature introspection**:
    - `ParameterInfo` dataclass: name, type, required, default, min/max, choices
    - `introspect_model_signature()`: __init__ parameter discovery
    - `introspect_forward_signature()`: forward() parameter discovery 
    - `get_required_data_attributes()`: Data compatibility detection (z, pos, x, edge_index)
  - **Conv layer kwargs handling**:
    - `model_accepts_kwargs()`: Detects **kwargs support for Conv passthrough
    - `KNOWN_CONV_KWARGS`: Set of valid conv layer parameters
    - `MODEL_SPECIFIC_CONV_KWARGS`: Model-specific kwargs (GCN, GAT, GraphSAGE, GIN, PNA)
    - `get_model_conv_kwargs()`: Get valid kwargs for specific model
  - **DynamicModelMetadata dataclass**:
    - Backward compatible with original ModelMetadata
    - NEW: `forward_parameters`, `required_data_attributes`, `accepts_kwargs`
  - **Intelligent defaults** via `_infer_intelligent_default()`:
    - Pattern-based defaults from original papers (DimeNet, SchNet, PaiNN)
    - Supports int, float, bool, str parameter types
  - **Singleton pattern**: `get_introspector()` returns cached instance
  - **Backward compatible API**: `get_all_model_names()`, `get_model_metadata()`, `get_models_by_category()`, `get_models_by_task()`, `get_models_by_tag()`, `search_models()`, `get_category_statistics()`
  - **Lazy loading**: `ALL_MODELS` dict for backward compatibility

**Factory Submodule** (`models/factory/`) :
- **ModelFactory**: Factory pattern for model creation 
  - Automatic hyperparameter validation
  - Channel inference from sample data
  - Device placement
  - Default value application
  - Custom architecture support (`_create_custom_model()`)
  - Ensemble model support (`_create_ensemble_model()`)
  - Dynamic introspection from pyg_introspector
  - **Methods**: `create_model()`, `get_model_info()`, `_count_parameters()`
  - **Convenience functions**: `get_factory()`, `create_model()`, `get_model_info()`
- **GraphLevelModelWrapper** (nn.Module): Wrapper for graph-level tasks
  - Task detection (`_is_graph_level_task()`)
  - Global pooling methods: mean, max, add
  - Output projection for fixed-output models (SchNet, DimeNet)
- **EdgeLevelModelWrapper** (nn.Module): Wrapper for edge-level tasks
  - Decoder methods: dot_product, concat_mlp, bilinear, etc.
  - Link prediction support with `edge_label_index`
  - VGAE/GAE autoencoder support via `encode()` method
- **ModelValidator**: Comprehensive validation
  - Hyperparameter schema validation
  - Data compatibility checking (`validate_data_compatibility()`)
  - Architecture validation
- **Dynamic dependency validation**:
  - `validate_model_dependencies()`: Check PyG optional packages
  - `_detect_model_dependencies()`: Inspect model source code
  - PyG packages: torch_cluster, torch_sparse, torch_scatter, torch_spline_conv
- **TargetSelectionConfig** (713 lines): Dynamic target/property selection 
  - **SelectionMode enum** (4 types): PROPERTIES, INDICES, RANGE, ALL
  - **TargetLevel enum** (3 types): GRAPH, NODE, EDGE (task-level inference)
  - **TargetSource enum** (6 types): Y, X, EDGE_ATTR, EDGE_LABEL, EDGE_Y, CUSTOM
  - **Dataclass fields**: mode, properties, indices, range_spec, strict, config_level, config_source
  - **Resolved fields**: resolved_level, resolved_source, resolved_source_attr, resolved_indices, resolved_names
  - **Factory method**: `from_config()` for config dict parsing
  - **Resolution methods**:
    - `resolve()`: Resolve against dataset properties
    - `resolve_for_task()`: Resolve level and source for specific task type
    - `infer_level_from_task_type()`: Pattern-based task detection (node_, edge_, graph_)
    - `infer_source_from_level()`: Fallback source inference
    - `_resolve_source_auto()`: Auto-resolve with shape validation
  - **Internal resolution**: `_resolve_properties()`, `_resolve_indices()`, `_resolve_range()`
  - **Serialization**: `to_dict()` for logging and model_info
  - Supports negative indices (Python-style)

**Training Submodule** (`models/training/`):
- **Trainer**: Full-featured training loop
  - Automatic train/validation/test split handling
  - Callback system integration with HPO support
  - Progress tracking and logging
  - **Checkpoint format**:
    - `save_checkpoint()`: Saves `hyper_parameters`, `data_info`, `version_info`
    - `load_checkpoint()`: Auto-detects v1.0 vs v2.0 format, restores `model_info`
    - `get_checkpoint_info()`: Static method - inspect without loading model
    - `is_v2_checkpoint()`: Static method - verify checkpoint format
    - Self-contained checkpoints enable inference without external config
  - Results saving with `save_results()` (JSON serialization)
  - Early stopping support
  - Learning rate scheduling with ReduceLROnPlateau support
  - Device management (auto-detect CUDA/CPU)
  - Gradient clipping and accumulation
  - **Dynamic forward signature introspection** 
    - `_get_forward_signature_params()`: Introspects model.forward() at runtime
    - `_model_accepts_3d_params()`: Detects 3D model support (SchNet, DimeNet)
    - `_forward_with_dynamic_signature()`: Signature-based forward dispatch
    - Supports ANY PyG model regardless of forward signature
  - **Task-aware target handling** 
    - `_is_edge_level_task()`: Detects link_prediction, edge_regression
    - `_is_graph_level_task()`: Detects graph_regression, graph_classification
    - `_get_target()`: Intelligent target extraction per task type
    - `_apply_target_selection()`: Column selection for multi-target tasks
    - Automatic reshape for flattened graph-level multi-targets
  - **model_info integration** 
    - `uses_edge_features`: Whether to pass edge_attr
    - `task_type`: For intelligent target handling
    - `is_classification`: Skip reshape for classification
    - `out_channels`: For multi-target reshape
    - `target_selection`: For column selection
  - **HPO callback integration** 
    - `hpo_callback` parameter for OptunaPruningCallback
    - Auto-appends to callbacks list
    - Proper TrialPruned exception handling
- **Callbacks** (6 types + factory): 
  - **Base class**: `Callback` (ABC) with hooks:
    - `set_trainer()`: Attach to trainer
    - `on_train_begin()`: Called at training start
    - `on_epoch_end()`: Called after each epoch with metrics
    - `on_train_end()`: Called at training end
  - **6 callback implementations**:
    - `EarlyStopping`: Stop training on plateau (monitor, patience, mode, min_delta)
    - `ModelCheckpoint`: Save best/periodic models 
      - **Parameters**: dirpath (required), monitor, mode, save_top_k, save_last, save_best, filename_pattern, verbose
      - `save_best: bool = True`: Save dedicated `best.pt` file for easy post-training access
      - **Properties** (PyTorch Lightning compatible API):
        - `best_model_path`: Path to best checkpoint (returns `dirpath/best.pt` when `save_best=True`)
        - `best_model_score`: Score of best checkpoint (auto-updated during training)
      - **Checkpoint contents**: Includes `is_best` marker for checkpoint identification
      - `_save_checkpoint()` saves `structural_features_config` in `checkpoint['data_info']`
      - `data_info` structure: `{requires_edge_features, uses_edge_features, structural_features_config}`
    - `TensorBoardLogger`: TensorBoard integration (requires tensorboard)
    - `LearningRateMonitor`: Track learning rate per param group
    - `ProgressBar`: Training progress display with metrics
    - `GradientMonitor`: Monitor gradient norms for vanishing/exploding detection
  - **CallbackFactory** : Config-based callback creation
    - `from_config()`: Create callbacks from config dictionary
    - Dynamic parameter introspection via `_filter_params()`
    - Auto path resolution for checkpoint/tensorboard directories
    - `register_custom_callback()`: Register custom callback types
    - `list_available()`, `get_callback_class()`: Discovery methods
- **TaskDataPreparer** : Task-specific data preparation
  - Dispatcher pattern for 7 task types:
    - `graph_regression`: No transform needed (float targets)
    - `graph_classification`: Auto-discretization for float targets
    - `node_regression`: Node-level target extraction from x/y
    - `node_classification`: Node-level extraction + discretization
    - `edge_regression`: Edge-level target extraction from edge_attr
    - `edge_classification`: Edge-level extraction + discretization
    - `link_prediction`: RandomLinkSplit for edge_label generation
  - TargetSelectionConfig integration for source/indices specification
  - Consistent discretization: fits on train, applies to all splits
  - Convenience function: `prepare_data_for_task()`
  - Task listing: `list_supported_tasks()`
- **DataSplitter**: Multiple splitting strategies 
  - **5 splitting strategies**:
    - `random_split()`: Simple random shuffling with seed control
    - `stratified_split()`: Maintains class distribution (requires scikit-learn)
    - `temporal_split()`: Chronological ordering for time-series data
    - `scaffold_split()`: Murcko scaffold-based for molecular data (requires rdkit)
    - `k_fold_split()`: K-fold cross-validation 
  - Configurable train/val/test ratios with validation
  - Custom label/time/mol getter functions for flexible data access
  - Reproducible splits via random_seed parameter
  - Returns `torch.utils.data.Subset` objects
  - **Convenience functions**: `random_split()`, `stratified_split()`, `temporal_split()`, `scaffold_split()`, `k_fold_split()`
- **LossRegistry**: Loss functions 
  - **18 loss functions organized by category**:
    - Regression: `mse`, `mae`/`l1`, `huber`, `smooth_l1`, `rmse`, `weighted_mse`
    - Classification: `cross_entropy`/`ce`, `nll`, `bce`, `bce_with_logits`, `focal`
    - Multi-label: `multilabel_soft_margin`
    - Ranking: `margin_ranking`, `triplet_margin`
    - Other: `kl_div`, `poisson_nll`, `cosine_embedding`
  - **Custom loss implementations**:
    - `FocalLoss`: For imbalanced classification (α, γ parameters)
    - `WeightedMSELoss`: Per-sample/per-feature weighted regression
    - `RMSELoss`: Root mean squared error
  - **Task-aware loss selection**  via `get_loss_for_task()`:
    - Auto-selects appropriate loss based on task_type
    - Prevents dtype mismatch errors (Long vs Float targets)
    - Overrides incompatible losses with warning
    - Default mappings for all 7 task types
  - **Dynamic parameter introspection** via `_filter_params()` - auto-filters invalid params
  - **Loss-task compatibility** via `is_loss_compatible_with_task()`
  - **Introspection methods**:
    - `get_valid_params()`: Discover loss parameters via inspect.signature()
    - `get_loss_info()`: Complete loss metadata
    - `get_default_loss_for_task()`: Task-to-loss mapping
  - **Custom loss registration** via `register_custom_loss()`
  - **Convenience functions**: `get_loss()`, `list_losses()`, `get_loss_for_task()`
- **OptimizerRegistry**: PyTorch optimizers 
  - **12 PyTorch optimizers organized by category**:
    - Adaptive: `adam`, `adamw`, `adamax`, `adadelta`, `adagrad`, `rmsprop`
    - SGD variants: `sgd`, `asgd`
    - Second-order: `lbfgs`
    - Other: `rprop`, `nadam`, `radam`
  - **Dynamic parameter introspection** via `_filter_params()` - auto-filters invalid params
  - **Registry defaults** for common optimizers (adam, adamw, sgd, rmsprop, adagrad)
  - **Introspection methods**:
    - `get_valid_params()`: Discover optimizer parameters via inspect.signature()
    - `get_optimizer_info()`: Complete optimizer metadata
    - `get_default_params()`: Registry-defined defaults
  - **Custom optimizer registration** via `register_custom_optimizer()`
  - **Convenience functions**: `get_optimizer()`, `list_optimizers()`
- **SchedulerRegistry**: Learning rate schedulers 
  - **13 PyTorch schedulers organized by category**:
    - Adaptive: `reduce_on_plateau` (metric-based)
    - Step-based: `step_lr`, `multistep_lr`, `exponential_lr`
    - Cosine annealing: `cosine_annealing`, `cosine_annealing_warm_restarts`
    - Cyclic: `cyclic_lr`, `one_cycle`
    - Polynomial: `polynomial_lr`
    - Linear: `linear_lr`
    - Chained/Sequential: `chained`, `sequential`
    - Constant: `constant_lr`
  - **Dynamic parameter introspection** via `_filter_params()` - auto-filters invalid params
  - **Registry defaults** for common configurations
  - **Metric-based detection** via `is_metric_based()` for ReduceLROnPlateau
  - **Introspection methods**:
    - `get_valid_params()`: Discover scheduler parameters via inspect.signature()
    - `get_scheduler_info()`: Complete scheduler metadata
    - `get_default_params()`: Registry-defined defaults
  - **Custom scheduler registration** via `register_custom_scheduler()`
  - **Warmup support** via `create_warmup_scheduler()` helper function
  - **Convenience functions**: `get_scheduler()`, `list_schedulers()`
- **MetricsRegistry**: TorchMetrics-based evaluation metrics 
  - **12 metrics organized by category**:
    - Regression: `mse`, `mae`, `rmse`, `r2`, `mape`, `explained_variance`
    - Classification: `accuracy`, `precision`, `recall`, `f1`, `auroc`, `auprc`
  - **Task-aware metric selection** via `get_metrics_for_task()`:
    - Auto-selects appropriate metrics based on task_type
    - `link_prediction` → binary classification metrics (AUROC, AUPRC, Accuracy)
    - `graph_regression` → regression metrics (MAE, MSE, RMSE, R2)
    - Validates config metrics against task type
  - **Dynamic parameter introspection** via `_filter_params()`:
    - Inspects BOTH `__new__` and `__init__` (TorchMetrics v1.0+ compatibility)
    - Auto-filters invalid params for any metric class
  - **TorchMetrics v1.0+ task parameter handling**:
    - Automatic `task='binary'` for link_prediction
    - Automatic `task='multiclass'` when num_classes > 2
  - **Introspection methods**:
    - `get_valid_params()`: Discover metric parameters via inspect.signature()
    - `get_metric_info()`: Complete metric metadata
    - `is_metric_compatible_with_task()`: Task-metric compatibility check
  - **Custom metric registration** via `register_custom_metric()`
  - **Convenience functions**: `get_metric()`, `get_metrics_for_task()`, `list_available()`
- **TrainingVisualizer**: Training visualization utilities 
  - **4 plot types**:
    - `plot_loss_curves()`: Train/Val loss over epochs (matplotlib)
    - `plot_metrics()`: All metrics in subplots (matplotlib)
    - `plot_learning_rate()`: Learning rate schedule (matplotlib)
    - `plot_interactive()`: Combined interactive dashboard (plotly)
  - **Config-driven visualization** via `evaluation.visualization` in config.yaml:
    - Master switch: `enabled: true/false`
    - Format selection: `formats: [png, html, pdf]`
    - Individual plot toggles: `plots.loss_curves`, `plots.metrics`, etc.
    - Style configuration: `style.figure_size`, `style.dpi`, `style.colors`
  - **Multiple output formats**:
    - PNG: Static matplotlib plots (default)
    - HTML: Interactive plotly plots (default)
    - PDF: Export via kaleido (optional)
  - **Graceful dependency handling**: Works with/without matplotlib, plotly, kaleido
  - **Convenience functions**: `plot_training_summary()`, `create_visualizer()`

**Post-Training Submodule** (`models/post_training/`):
- **Public API** (`__init__.py`,  ~350 lines):
  - Unified exports for checkpoint, inference, data_preparation, transfer_learning
  - **path_utils.py REMOVED** - Dependency Injection pattern now used
  - Conditional imports with graceful fallback
  - `get_available_components()`: List all available components by category
  - `get_implementation_status()`: Implementation status tracking
  - 24 total exports (2 checkpoint + 5 inference + 17 data_preparation + 2 transfer_learning)
- **CheckpointManager** (`checkpoint/checkpoint_manager.py`, v2.0.0) :
  - `__init__(working_root_dir: Path)`: **Required** working_root_dir parameter (DI pattern)
  - `_resolve_path()`: Private method for path resolution against working_root_dir
  - `_resolve_checkpoint_path()`: Private method for intelligent checkpoint search
  - `save()`: Resolves filepath via `_resolve_path()`, auto-creates directories
  - `load()`: Resolves filepath via `_resolve_checkpoint_path()` with intelligent search
  - `get_checkpoint_dir(subdir)`: Returns working_root_dir / subdir (default: 'checkpoints')
  - `is_v2_checkpoint()`: Format verification
  - `get_hyper_parameters()`: Extract hyperparameters without full load
  - `get_model_name()`: Extract model name from checkpoint
  - `create_version_info()`: Generate version metadata
  - **CHECKPOINT_FORMAT_VERSION**: `'2.0'` constant
- **ModelLoader** (`inference/model_loader.py`,  ~500 lines) :
  - `__init__(working_root_dir: Path)`: **Required** working_root_dir parameter (DI pattern)
  - `load_from_checkpoint(checkpoint_path, working_root_dir, ...)`: **Required** working_root_dir
  - Uses `CheckpointManager._resolve_checkpoint_path()` for path resolution
  - `FileNotFoundError` includes all searched locations for debugging
  - Uses `ModelFactory.create_model_with_info()` with COMPLETE hyperparameters
  - Handles v1.0 backward compatibility with override parameters
  - Handles wrapped models (GraphLevelModelWrapper, EdgeLevelModelWrapper)
  - `_load()` extracts `data_info` from checkpoint and includes in `final_model_info`
  - Logs `structural_features_config` when present in checkpoint
  - `get_checkpoint_info(checkpoint_path, working_root_dir)`: Inspection without loading
  - **Convenience functions**: `load_model(path, working_root_dir)`, `load_model_only(path, working_root_dir)`
- **Predictor** (`inference/predictor.py`,  ~450 lines) :
  - `__init__(model, working_root_dir: Path, ..., model_info: Optional[Dict])`: **Required** working_root_dir parameter (DI pattern)
  - New `model_info` parameter stores checkpoint metadata including featurization config
  - `structural_features_config` property exposes featurization config from `model_info['data_info']`
  - `from_checkpoint(checkpoint_path, working_root_dir, ...)`: **Required** working_root_dir, passes `model_info` to `__init__`
  - `_resolve_path()`: Private method for output path resolution
  - PyG-compatible inference (unlike `deployment_strategies.py` which expects Tensor)
  - `predict()`: Single Data or Batch prediction
  - `predict_batch()`: Entire datasets with DataLoader
  - Auto-detects edge_attr, batch, pos attributes
  - Post-processing based on task_type (classification → argmax)
  - `save_predictions()`: Uses `_resolve_path()` for output
  - **Convenience function**: `predict(checkpoint_path, data, working_root_dir)`
- **DataConverterRegistry** (`data_preparation/data_converter.py`,  ~970 lines) :
  - File-based converters (XYZ, SDF) accept optional `working_root_dir` parameter
  - `XYZConverter(cutoff, working_root_dir=None)`: Optional working_root_dir for path resolution
  - `SDFConverter(working_root_dir=None)`: Optional working_root_dir for path resolution, preserves SMILES in output
  - `_resolve_path()`: Private method in file-based converters
  - Singleton registry with `@register_converter` decorator
  - **7 built-in converters**: PyGData, Dict, SMILES, InChI, XYZ, ASEAtoms, SDF
  - `BaseDataConverter` ABC for custom converters
  - `auto_detect()`: Content-based format detection
  - `SMILESConverter` accepts optional `structural_features_config` parameter
  - `SMILESConverter._use_structural_features` flag for checkpoint featurization mode
  - `_apply_structural_features_if_available(data, config)` helper function for post-processing
  - `convert_to_pyg(..., structural_features_config=None)` applies featurization via post-processing
  - Post-processing reconstructs mol from SMILES/InChI and applies `add_structural_features()`
  - **Zero-modification extension**: New formats via `@register_converter`
  - **Convenience functions**: `convert_to_pyg()`, `convert_batch_to_pyg()`, `list_available_formats()`, `list_all_formats()`
- **FineTuner** (`transfer_learning/fine_tuner.py`,  ~260 lines) :
  - `__init__(model, hyper_parameters, working_root_dir: Path)`: **Required** working_root_dir (DI pattern)
  - `from_checkpoint(checkpoint_path, working_root_dir, ...)`: **Required** working_root_dir
  - Uses `CheckpointManager._resolve_checkpoint_path()` for path resolution
  - `prepare_for_finetuning()`: Apply freeze strategy, replace head
  - **FreezeStrategy** enum: `NONE`, `ENCODER`, `ENCODER_PARTIAL`, `ALL_BUT_LAST`
  - Pattern-based layer freezing via `_freeze_encoder_layers()`
  - Dynamic output head replacement via `_replace_output_head()`

**Builders Submodule** (`models/builders/`):
- **layer_registry.py**: Thread-safe singleton layer catalog ⭐ DOCUMENTED
  - **LayerCategory** (Enum, 8 values): CONVOLUTIONAL, POOLING, NORMALIZATION, ACTIVATION, AGGREGATION, LINEAR, DROPOUT, CUSTOM
  - **FunctionalLayerWrapper** (nn.Module): Wraps functional operations into nn.Module
    - Attributes: func, func_name, requires_batch, requires_edge_index, requires_edge_attr
    - `forward()`: Handles argument combinations based on requirements
  - **LayerMetadata** (dataclass, 12 attributes):
    - name, category, class_path, description
    - requires_edge_index, requires_edge_attr, requires_batch
    - has_in_channels, has_out_channels, modifies_graph_structure
    - supported_task_levels, is_functional
    - `to_dict()`: Serialization method
  - **LayerNotFoundError**: Exception with available_layers suggestion
  - **LayerRegistry**: Thread-safe singleton with RLock
    - **63+ built-in layers organized by category**:
      - Convolutional (33): GCNConv, GATConv, SAGEConv, GINConv, ChebConv, TransformerConv, GATv2Conv, etc.
      - Pooling (9): global_mean_pool, global_max_pool, global_add_pool, TopKPooling, SAGPooling, etc.
      - Normalization (7): BatchNorm, LayerNorm, InstanceNorm, GraphNorm, PairNorm, etc.
      - Activation (9): ReLU, LeakyReLU, ELU, PReLU, GELU, Tanh, Sigmoid, etc.
      - Aggregation (3): MeanAggregation, MaxAggregation, SumAggregation
      - Standard (2): Linear, Dropout
    - **Registration methods**:
      - `_register_builtin_layers()`: Auto-registers all PyG layers on init
      - `register_custom_layer()`: Register custom layer with auto-wrapping for functions
      - `_create_functional_wrapper()`: Create wrapper class for functional operations
    - **Retrieval methods**:
      - `get_layer()`: Get layer class by name
      - `get_layer_metadata()`: Get LayerMetadata
      - `has_layer()`: Check existence
      - `list_layers()`: List with optional category filter
      - `list_categories()`: List all categories
    - **Utility methods**:
      - `get_statistics()`: Returns {total_layers, by_category, functional_layers, class_layers}
  - **Global Instance**: `registry = LayerRegistry()` singleton
  - **Convenience Functions**:
    - `get_layer()`: Get layer from global registry
    - `list_layers()`: List layers from global registry
    - `get_layer_metadata()`: Get metadata from global registry
- **architecture_builder.py**: Dynamic GNN architecture composition ⭐ DOCUMENTED
  - **Dataclasses**:
    - `LayerConfig` (6 attributes): type, params, position, in_channels, out_channels, input_from
      - `to_dict()`, `from_dict()` classmethod for serialization
    - `ResidualConnection` (3 attributes): start_layer, end_layer, connection_type (add/concat)
      - `to_dict()`, `from_dict()` classmethod for serialization
    - `ArchitectureConfig` (6 attributes): name, task_type, in_channels, out_channels, layers, residual_connections
      - `to_dict()`, `from_dict()` classmethod for serialization
  - **Exceptions**:
    - `ArchitectureError`: Base exception for architecture building errors
    - `ChannelMismatchError`: Detailed channel dimension mismatch with layer info and suggestions
  - **ArchitectureBuilder**: Fluent builder pattern for custom architectures
    - **Layer manipulation methods** (all return self for chaining):
      - `add_layer()`: Add layer at position with params
      - `remove_layer()`: Remove layer at position
      - `insert_layer()`: Insert at specific position
      - `replace_layer()`: Replace layer at position
      - `swap_layers()`: Swap two layers
    - **Residual connections**:
      - `add_residual_connection()`: Add skip connections (add or concat types)
    - **Channel inference**:
      - `_infer_channels()`: Automatic channel inference with multi-head attention handling (heads, concat)
    - **Validation**:
      - `validate_architecture()`: Returns {valid, errors, warnings, suggestions} with task-specific checks
    - **Building**:
      - `build()`: Build into CustomArchitecture nn.Module
    - **Config import/export**:
      - `to_config()`: Export to ArchitectureConfig
      - `from_config()`: Classmethod to create from config dict or ArchitectureConfig
  - **CustomArchitecture** (nn.Module): Built model from ArchitectureBuilder
    - `_build_layers()`: Instantiate all layers from configurations
    - `_create_projection_if_needed()`: Dynamic projection layers for residual dimension mismatch
    - `forward()`: Forward pass supporting both class-based and functional layers
    - Automatic handling of edge_index, edge_attr, batch based on layer metadata
- **model_composer.py**: Multi-model ensemble composition ⭐ DOCUMENTED
  - **Dataclasses**:
    - `ModelSpec` (4 attributes): model, weight, name, level
      - `to_dict()`: Serialization method
    - `EnsembleConfig` (5 attributes): name, task_type, models, strategy, fusion
      - `to_dict()`: Serialization method
  - **CompositionError**: Exception with strategy, num_models, details attributes
  - **ModelComposer**: Fluent builder for ensembles (all methods return self)
    - **Model management**:
      - `add_model()`: Add model with weight, name, level
      - `remove_model()`: Remove by index
      - `clear_models()`: Clear all models
    - **Configuration**:
      - `set_strategy()`: parallel, sequential, or hierarchical
      - `set_fusion()`: mean, weighted, attention, or voting
    - **Validation**:
      - `validate_composition()`: Returns {valid, errors, warnings, suggestions}
    - **Building**:
      - `build()`: Returns ParallelEnsemble, SequentialStack, or HierarchicalComposition
    - **Config import/export**:
      - `to_config()`: Export to EnsembleConfig
      - `from_config()`: Classmethod to create from config
    - **Utility**:
      - `summary()`: Detailed formatted summary string
  - **ParallelEnsemble** (nn.Module): Models run in parallel, outputs aggregated
    - Attributes: models (ModuleList), weights (buffer), attention (for attention fusion)
    - `_is_graph_level_task()`: Dynamic task type detection
    - `_apply_global_pooling()`: PyG native pooling (mean, max, add)
    - `_get_innermost_model()`: Recursive wrapper unwrapping
    - `_model_supports_edge_attr()`: Check PyG model edge feature support
    - `_call_model_with_signature()`: Signature-aware model calling for heterogeneous ensembles
    - `forward()`: With DataBatch extraction, 2D/3D model support
  - **SequentialStack** (nn.Module): Output of model N → input of model N+1
    - Same helper methods as ParallelEnsemble
    - `forward()`: Sequential processing with edge_label_index for last model
    - Supports: GAE/VGAE encode() method, 3D models (SchNet, DimeNet)
  - **HierarchicalComposition** (nn.Module): Models organized by level
    - `levels`: Sorted level indices
    - `level_ensembles`: ModuleList of ParallelEnsemble for each level
    - `forward()`: Level-by-level processing with edge_label_index for last level
- **templates.py**: Pre-built architecture templates ⭐ DOCUMENTED
  - **ArchitectureTemplates**: Collection of 10 parameterized templates (all @staticmethod)
    - **Basic Templates** (8):
      - `simple_gcn()`: (Conv → ReLU → Dropout) × N → Pool → Linear
        - Parameters: in_channels, out_channels, num_layers, hidden_channels, dropout, task_type
      - `attention_network()`: (GAT → ELU → Dropout) × N → Pool → Linear
        - Parameters: + heads (multi-head attention)
      - `deep_residual()`: (Conv → ReLU → Conv → ReLU + Skip) × N → Pool → Linear
        - Parameters: depth, uses add_residual_connection()
      - `hybrid_conv_attention()`: (GCN → ReLU) × N → (GAT → ELU) × M → Pool → Linear
        - Parameters: conv_layers, attention_layers, heads
      - `hierarchical_pooling()`: (Conv → ReLU → TopKPool) × N → Global Pool → Linear
        - Parameters: num_levels, pooling_ratio
      - `graph_sage_network()`: (SAGE → ReLU → Dropout) × N → Pool → Linear
        - Parameters: aggr (mean, max, lstm)
      - `gin_network()`: (GIN → ReLU → Dropout) × N → Sum Pool → Linear
        - Parameters: train_eps
      - `molecular_network()`: Conv → (Conv/GAT alternating) × N → Pool → MLP
        - Parameters: optimized for drug discovery, chemical prediction
    - **Task-Specific Templates** (2):
      - `node_classification_network()`: (Conv → ReLU → Dropout) × N → Linear
        - Parameters: num_classes instead of out_channels
      - `graph_classification_network()`: (Conv → ReLU → Dropout) × N → Pool → Linear
        - Parameters: num_classes instead of out_channels
    - **Utility Methods**:
      - `list_templates()`: Returns list of 10 template names
      - `get_template_info()`: Returns {name, description, parameters, suitable_for, best_use_cases}
    - All templates support 6 task types: node_regression, node_classification, graph_regression, graph_classification, link_prediction, edge_regression
    - All templates return ArchitectureBuilder instances for further customization
- **config_parser.py**: YAML/JSON configuration parsing ⭐ DOCUMENTED
  - **ArchitectureConfigParser**: Main parser class
    - **Constructor**: `__init__(validator, templates, strict_validation)`
    - **Custom architecture parsing**:
      - `parse_custom_architecture()`: Parse dict/Path/string into ArchitectureBuilder
      - `_parse_template_based()`: Template-based parsing with modifications
        - Supports: additional_layers, modifications (insert, remove, replace)
    - **Ensemble parsing**:
      - `parse_ensemble()`: Parse ensemble configuration into ModelComposer
      - Supports multiple config formats: top-level strategy/fusion, nested composition
    - **Configuration loading**:
      - `_load_config()`: Load from YAML/JSON file or string with auto-detection
      - Supports: .yaml, .yml, .json extensions, or string content
    - **Validation**:
      - `validate_config()`: Validate config structure for custom_architecture or ensemble
      - Returns: {valid, errors, warnings, suggestions}
      - Validates: layers, template, residual_connections, composition, strategy, fusion
    - **Export methods**:
      - `export_builder_config()`: Export ArchitectureBuilder to YAML/JSON string
      - `export_composer_config()`: Export ModelComposer to YAML/JSON string
    - **Utility methods**:
      - `save_config()`: Save config to file with format auto-detection from extension
  - **Configuration Formats**:
    - Custom architecture: name, task_type, in_channels, out_channels, layers[], residual_connections[]
    - Ensemble: name, task_type, composition (strategy, fusion), models[]
    - Template-based: template, task_type, params, additional_layers, modifications
  - **Convenience Functions**:
    - `parse_custom_architecture()`: Quick architecture parsing
    - `parse_ensemble()`: Quick ensemble parsing
    - `load_config()`: Load config file to dict
    - `validate_config()`: Validate config structure
- **validation.py**: Comprehensive architecture validation ⭐ DOCUMENTED
  - **ArchitectureValidator**: Main validator class
    - **Constructor**: `__init__(registry)` - optional LayerRegistry, uses global if None
    - **Main validation**:
      - `validate()`: Validates layers, task_type, in_channels, out_channels
      - Returns: {valid, errors, warnings, suggestions}
    - **Specific validations**:
      - `validate_channel_flow()`: Validates channel dimensions flow correctly
        - Checks in_channels, out_channels consistency
        - Handles multi-head attention (heads, concat parameters)
        - Detects mismatches and suggests Linear layer fixes
      - `validate_task_compatibility()`: Validates architecture matches task type
        - Graph-level tasks: requires pooling layer
        - Node-level tasks: warns if pooling present
      - `validate_layer_ordering()`: Validates sensible layer ordering
        - Warns: pooling before convolution
        - Warns: consecutive activation layers
      - `validate_data_compatibility()`: Validates architecture works with sample data
        - Checks required fields (x, edge_index)
        - Tests forward pass with torch.no_grad()
    - **Suggestion generation**:
      - `suggest_fixes()`: Generates actionable fix suggestions from validation result
  - **Convenience Functions**:
    - `validate_architecture()`: Quick architecture validation
    - `validate_data_compatibility()`: Quick data compatibility check

**Acceleration Submodule** (`models/acceleration/`):
- **device_manager.py**: Device detection, selection, and monitoring ⭐ PYDANTIC V2
  - **DeviceType** (Enum, 5 values): CPU, CUDA, MPS, TPU, AUTO
  - **DeviceInfo** (Pydantic BaseModel, mutable, 8 attributes):
    - device_type, device_id, name, total_memory, available_memory
    - compute_capability (CUDA major, minor) - `Optional[Tuple[int, int]]`
    - is_available, is_default
    - `to_dict()`: Backward-compatible method wrapping `model_dump()`
    - `memory_summary()`: Human-readable memory display
    - `model_config = {'arbitrary_types_allowed': True}`: Tuple compatibility
  - **DeviceManager**: Main device management class
    - **Constructor**: device, allow_fallback, verbose
    - **Device priority**: _DEVICE_PRIORITY = [CUDA, MPS, TPU, CPU]
    - **Auto-detection**:
      - `_auto_detect_device()`: Priority-based detection
      - `_validate_and_set_device()`: Validation with configurable fallback
      - `_is_mps_available()`: Apple Silicon detection
      - `_is_tpu_available()`: TPU detection via torch_xla
    - **Public API**:
      - `get_device()`: Get current torch.device
      - `get_device_info()`: Get DeviceInfo for current device
      - `get_available_devices()`: List all devices with optional type filter
      - `move_to_device()`: Move model with non_blocking option
    - **Memory management** (CUDA):
      - `get_memory_info()`: Returns {total, allocated, reserved, free}_memory in bytes and GB
      - `reset_peak_memory_stats()`: Reset CUDA peak memory stats
      - `empty_cache()`: Empty CUDA cache
    - **Synchronization**:
      - `synchronize()`: Device sync (CUDA or TPU via xm.mark_step)
    - **Context manager**:
      - `device_context()`: Temporary device switching context
    - **Utility**:
      - `print_device_summary()`: Formatted summary of all devices
  - **Convenience Functions**:
    - `get_default_device()`: Quick device selection
    - `list_available_devices()`: List all DeviceInfo
    - `get_device_capabilities()`: Returns {cuda_available, cuda_device_count, mps_available, tpu_available, cudnn_available, cudnn_enabled}
- **distributed_strategies.py**: Comprehensive distributed training ⭐ DOCUMENTED
  - **DistributedStrategy** (Enum, 6 values): NONE, DP, DDP, FSDP, DEEPSPEED, HOROVOD
  - **DistributedBackend** (Enum, 4 values): GLOO, NCCL, MPI, AUTO
  - **DistributedConfig** (dataclass, 12 attributes):
    - strategy, backend, world_size, rank, local_rank
    - master_addr, master_port, find_unused_parameters
    - gradient_as_bucket_view, static_graph, cpu_offload, mixed_precision
    - `to_dict()`: Configuration serialization
  - **DistributedManager**: Main distributed training manager
    - **Constructor**: 8 parameters (strategy, backend, find_unused_parameters, etc.)
    - **Setup**:
      - `_load_env_variables()`: Load WORLD_SIZE, RANK, LOCAL_RANK, MASTER_ADDR/PORT
      - `_get_backend()`: Auto-detect (nccl for CUDA, gloo for CPU)
      - `setup()`: Initialize process group for DDP/FSDP/Horovod
    - **Model wrapping**:
      - `wrap_model()`: Wrap with DataParallel, DDP, FSDP, or Horovod broadcast
    - **Lifecycle**:
      - `cleanup()`: Destroy process group
    - **Process info**:
      - `is_main_process()`, `get_world_size()`, `get_rank()`, `get_local_rank()`
    - **Synchronization**:
      - `barrier()`: Synchronize all processes
    - **Collective operations**:
      - `all_reduce()`: Reduce with op (sum, avg, min, max)
      - `all_gather()`: Gather tensors from all processes
    - **Checkpointing**:
      - `save_checkpoint()`: Save on main process only, includes optimizer state
      - `load_checkpoint()`: Load model and optimizer state
    - **Utility**:
      - `print_distributed_summary()`: Formatted summary
  - **Convenience Functions**:
    - `is_distributed_available()`: Check dist.is_available()
    - `get_world_size()`: Get world size (1 if not distributed)
    - `get_rank()`: Get process rank (0 if not distributed)
    - `is_main_process()`: Check if rank 0
- **memory_optimization.py**: Comprehensive memory optimization ⭐ DOCUMENTED
  - **MemoryConfig** (dataclass, 9 attributes):
    - mixed_precision, precision (fp16, bf16, fp32, fp8)
    - gradient_checkpointing, pin_memory, non_blocking
    - empty_cache_interval, garbage_collect_interval, max_memory_allocated, growth_interval
    - `to_dict()`: Configuration serialization
  - **MemoryOptimizer**: Main memory optimization manager
    - **Constructor**: 10 parameters (mixed_precision, precision, gradient_checkpointing, etc.)
    - **Mixed precision training (AMP)**:
      - `autocast()`: Context manager for automatic mixed precision (CUDA + CPU)
      - `_get_autocast_dtype()`: Get dtype based on precision config
      - `get_grad_scaler()`: Get GradScaler for loss scaling
      - `scale_loss()`: Scale loss for mixed precision
      - `step_optimizer()`: Step with gradient unscaling and update
    - **Gradient checkpointing**:
      - `enable_gradient_checkpointing()`: Auto-detect PyG, Transformers, or wrap custom models
      - `checkpoint_sequential()`: Checkpoint sequential operations
    - **Memory monitoring**:
      - `get_memory_stats()`: Returns {allocated, reserved, max_allocated, max_reserved, total, utilization}
      - `get_memory_summary()`: Human-readable formatted summary
      - `reset_peak_memory_stats()`: Reset CUDA peak stats
      - `empty_cache()`: Empty CUDA cache
      - `run_garbage_collection()`: Python GC + cache clear
      - `step()`: Periodic cache/GC based on configured intervals
      - `check_memory_usage()`: Check against threshold (0.0-1.0)
    - **Memory profiling**:
      - `profile_memory()`: Context manager with torch.profiler (shapes, memory, stack)
      - `get_memory_snapshot()`: Detailed CUDA memory snapshot
    - **Data loading optimization**:
      - `optimize_dataloader()`: Optimize with num_workers, prefetch_factor, pin_memory, persistent_workers
    - **Memory leak detection**:
      - `detect_memory_leaks()`: Test model for memory leaks over iterations
    - **Utility**:
      - `print_memory_summary()`: Formatted summary output
  - **Convenience Functions**:
    - `get_memory_efficient_settings()`: Recommended settings for small/medium/large/xlarge models
    - `estimate_model_memory()`: Estimate parameters, activations, gradients, optimizer memory in MB/GB
- **computation_optimization.py**: Comprehensive computation optimization ⭐ DOCUMENTED
  - **ComputationConfig** (dataclass, 10 attributes):
    - compile_model, compile_mode (default, reduce-overhead, max-autotune), compile_dynamic
    - cudnn_benchmark, cudnn_deterministic, use_tf32
    - channels_last, fusion_strategy (none, default, aggressive), jit_compile, operator_fusion
    - `to_dict()`: Configuration serialization
  - **ComputationOptimizer**: Main optimizer class
    - **Constructor**: 12 parameters, applies global optimizations on init
    - **torch.compile methods** (PyTorch 2.0+):
      - `compile_model()`: Compile with mode, dynamic, fullgraph, backend (inductor, aot_eager, cudagraphs)
    - **JIT compilation**:
      - `jit_script_model()`: Script or trace model with example inputs
      - `jit_freeze_model()`: Freeze for inference optimization
    - **Memory format**:
      - `convert_to_channels_last()`: Channels-last format for convolution performance
    - **Main entry point**:
      - `optimize_model()`: Applies all enabled optimizations (device, channels-last, JIT, compile)
    - **Performance profiling**:
      - `profile_performance()`: Context manager with CPU/CUDA profiling, shapes, stacks, FLOPs
      - `benchmark_model()`: Returns {avg_time_ms, min_time_ms, max_time_ms, throughput_fps, std_time_ms}
      - `compare_optimizations()`: Compare multiple optimization configs
    - **Kernel fusion**:
      - `enable_fusion()`: Default or aggressive fusion (texpr_fuser, nvfuser)
      - `disable_fusion()`: Disable all fusion
    - **Graph optimization**:
      - `optimize_graph()`: JIT graph passes (inline, constant_prop, peephole, fuse) with levels 0-3
    - **Utility**:
      - `print_optimization_summary()`: Formatted summary output
  - **Convenience Functions**:
    - `get_optimal_settings()`: Get optimal settings for training/inference
    - `auto_optimize_model()`: Auto-optimize with best settings for task
  - **Decorators**:
    - `@optimize_inference`: Wraps function with torch.no_grad() + inference_mode()

**Deployment Submodule** (`models/deployment/`):
- **deployment_strategies.py**: Comprehensive deployment strategies ⭐ DOCUMENTED
  - **Enums**:
    - `DeploymentTarget` (9 values): CLOUD_AWS, CLOUD_GCP, CLOUD_AZURE, EDGE_MOBILE, EDGE_IOT, FEDERATED, SERVERLESS, CONTAINER, LOCAL
    - `ServingMode` (3 values): ONLINE (real-time), BATCH, STREAMING
  - **DeploymentConfig** (dataclass, 13 attributes):
    - `target`, `serving_mode`, `instance_type`, `num_instances`
    - `auto_scaling`, `min_instances`, `max_instances`
    - `api_type` (rest, grpc, batch), `enable_monitoring`, `enable_logging`, `enable_caching`
    - `timeout_seconds`, `max_batch_size`
    - `to_dict()`: Configuration serialization
  - **DeploymentStrategy** (ABC): Base class for all deployment strategies
    - Abstract methods: `prepare_model()`, `deploy()`, `predict()`, `teardown()`
    - `get_deployment_info()`: Returns deployment metadata
  - **6 Deployment Strategy Implementations**:
    - `AWSDeploymentStrategy`: AWS SageMaker deployment
      - `_create_sagemaker_inference_script()`: Generates inference.py with model_fn, input_fn, predict_fn, output_fn
    - `GCPDeploymentStrategy`: Google Cloud AI Platform deployment
    - `AzureDeploymentStrategy`: Azure Machine Learning deployment
    - `EdgeDeploymentStrategy`: Mobile/IoT edge deployment with mobile optimization (torch.jit.trace, optimize_for_mobile)
    - `ContainerDeploymentStrategy`: Docker/Kubernetes deployment
      - `_create_dockerfile()`: Generates Dockerfile
      - `_create_serving_script()`: Generates Flask serving script with /predict and /health endpoints
    - `LocalDeploymentStrategy`: Local development/testing deployment
  - **DeploymentManager**: Unified interface for all deployment strategies
    - `_strategies`: Class attribute mapping target names to strategy classes (aws, gcp, azure, mobile, iot, container, local)
    - Methods: `prepare_model()`, `deploy()`, `predict()`, `teardown()`, `get_deployment_info()`
    - `list_available_targets()`: Classmethod returning available deployment targets
  - **Convenience Functions**:
    - `deploy_locally()`: Quick local deployment helper
    - `list_deployment_targets()`: List all available targets
- **model_optimization.py**: Comprehensive model optimization ⭐ DOCUMENTED
  - **Enums**:
    - `QuantizationType` (4 values): DYNAMIC, STATIC, QAT, FP16
    - `PruningType` (4 values): UNSTRUCTURED, STRUCTURED, MAGNITUDE, GRADIENT
  - **OptimizationConfig** (dataclass, 11 attributes):
    - `quantization_enabled`, `quantization_type`, `quantization_backend` (fbgemm, qnnpack)
    - `pruning_enabled`, `pruning_type`, `pruning_amount` (0.0-1.0)
    - `distillation_enabled`, `distillation_temperature`, `distillation_alpha`
    - `export_onnx`, `optimize_for_mobile`
    - `to_dict()`: Configuration serialization
  - **ModelOptimizer**: Main optimization manager class
    - **Quantization methods**:
      - `quantize_model()`: Main quantization entry point
      - `_dynamic_quantization()`: Quantizes Linear, LSTM, GRU layers to qint8
      - `_static_quantization()`: Static quantization with calibration data and module fusion
      - `_prepare_qat()`: Prepare model for quantization-aware training
      - `finalize_qat()`: Convert QAT-trained model to quantized model
    - **Pruning methods**:
      - `prune_model()`: Main pruning with iterative steps support
      - `_magnitude_pruning()`: L1 unstructured magnitude-based pruning
      - `_unstructured_pruning()`: Global unstructured pruning across all layers
      - `_structured_pruning()`: Channel-wise structured pruning (dim=0)
      - `_remove_pruning_reparameterization()`: Make pruning permanent
      - `get_sparsity()`: Calculate sparsity statistics (zero_parameters, global_sparsity, compression_ratio)
    - **Knowledge Distillation methods**:
      - `distillation_loss()`: KL divergence loss with temperature scaling
      - `create_student_model()`: Placeholder for student architecture creation
    - **Export methods**:
      - `export_to_onnx()`: ONNX export with dynamic_axes, opset_version, constant_folding
      - `optimize_for_mobile()`: Mobile optimization via torch.jit.trace and _save_for_lite_interpreter
    - **Metrics methods**:
      - `get_model_size()`: Model size in MB (parameters, buffers, total)
      - `compare_models()`: Compare original vs optimized (size_reduction, compression_ratio)
      - `print_optimization_summary()`: Formatted optimization summary output
  - **Convenience Functions**:
    - `quantize_for_inference()`: Quick dynamic quantization helper
    - `prune_for_deployment()`: Quick pruning helper with configurable amount
- **monitoring.py**: Comprehensive production monitoring ⭐ DOCUMENTED
  - **Enums**:
    - `MetricType` (8 values): LATENCY, THROUGHPUT, ERROR_RATE, ACCURACY, LOSS, MEMORY, CPU, GPU
    - `AlertSeverity` (4 values): INFO, WARNING, ERROR, CRITICAL
    - `DriftType` (3 values): DATA_DRIFT, CONCEPT_DRIFT, MODEL_DRIFT
  - **MonitoringConfig** (dataclass, 12 attributes):
    - `enable_performance_tracking`, `enable_drift_detection`, `enable_health_checks`, `enable_alerting`
    - `drift_detection_method` (ks_test, psi, wasserstein), `drift_threshold`
    - `alert_threshold`, `health_check_interval`, `metrics_window_size`
    - `log_predictions`, `log_metrics_interval`, `retraining_trigger_threshold`
    - `to_dict()`: Configuration serialization
  - **Alert** (dataclass, 6 attributes):
    - `severity`, `message`, `metric_type`, `metric_value`, `threshold`, `timestamp`
    - `to_dict()`: Alert serialization
  - **ModelMonitor**: Unified production monitoring class
    - **Performance tracking methods**:
      - `log_prediction()`: Log prediction with latency, accuracy, memory usage
      - `log_error()`: Log errors with automatic error rate calculation
      - `_check_for_anomalies()`: Detect latency spikes and accuracy degradation
    - **Drift detection methods**:
      - `set_reference_data()`: Set reference data for drift detection
      - `detect_drift()`: Main drift detection entry point
      - `_calculate_drift()`: Calculate drift score using configured method
      - `_ks_test_drift()`: Kolmogorov-Smirnov test (scipy.stats.ks_2samp)
      - `_psi_drift()`: Population Stability Index calculation
      - `_wasserstein_drift()`: Wasserstein distance (scipy.stats.wasserstein_distance)
    - **Health check methods**:
      - `health_check()`: Perform health check with status levels (healthy, degraded, unhealthy)
    - **Alerting methods**:
      - `_trigger_alert()`: Trigger alert with severity and callback invocation
      - `register_alert_callback()`: Register callback function for alerts
      - `get_alerts()`: Get alerts with severity and time filtering
      - `clear_alerts()`: Clear all alerts
    - **Metrics methods**:
      - `get_metrics_summary()`: Summary statistics (mean, std, min, max, p50, p95, p99)
      - `get_metric()`: Get specific metric values
      - `export_metrics()`: Export metrics to JSON file
      - `reset_metrics()`: Reset all metrics and counters
      - `print_monitoring_summary()`: Formatted monitoring summary output
  - **Convenience Functions**:
    - `create_monitor()`: Create monitor with default settings (enable_all option)

**Utils Submodule** (`models/utils/`):
- **ConfigBridge** (Pydantic): Configuration integration ⭐ PYDANTIC
  - **Migration**: 31 dataclasses → Pydantic BaseModel with `@field_validator` and `@model_validator`
  - **13 Configuration Enums** (unchanged):
    - Core: `TaskType` (6 tasks), `DataSplitMethod` (4), `LossFunction` (9), `OptimizerType` (6), `SchedulerType` (6)
    - Acceleration: `DeviceType` (5), `DistributedStrategy` (5), `MixedPrecision` (4)
    - Deployment: `DeploymentStrategy` (5)
    - HPO: `HPOParamType` (4), `HPOPrunerType` (4), `HPOSamplerType` (4), `HPODirection` (2)
  - **31 Pydantic BaseModel Classes** with runtime validation:
    - Selection: `ModelSelectionConfig`
    - Training: `DataSplitConfig`, `LossConfig`, `OptimizerConfig`, `SchedulerConfig`, `CallbackConfig`, `CallbacksConfig`, `ValidationConfig`, `LoggingConfig`, `TrainingConfig`
    - Evaluation: `EvaluationConfig`
    - Acceleration: `DeviceConfig`, `DDPConfig`, `FSDPConfig`, `DeepSpeedConfig`, `DistributedConfig`, `MemoryConfig`, `DataLoaderConfig`, `ComputationConfig`, `AccelerationConfig`
    - Deployment: `QuantizationConfig`, `PruningConfig`, `DistillationConfig`, `OptimizationConfig`, `EdgeDeploymentConfig`, `CloudDeploymentConfig`, `FederatedConfig`, `DriftDetectionConfig`, `RetrainingConfig`, `MonitoringConfig`, `DeploymentConfig`
    - Plugins: `PluginsConfig`
    - HPO: `HPOSearchSpaceParamBridge`, `HPOPrunerConfigBridge`, `HPOSamplerConfigBridge`, `HPOStudyConfigBridge`, `HPOConfigBridge`
  - **Validation Patterns**:
    - `@field_validator`: Enum validation (TaskType, LossFunction, etc.)
    - `@model_validator(mode='after')`: Cross-field validation (ratios sum to 1.0), conditional validation (enabled flag)
    - Backward-compatible `validate()` methods preserved as pass-through
  - **Main Container**: `ModelConfig` (Pydantic BaseModel) - complete models module configuration
    - `from_dict()`, `from_yaml()`: Factory methods (unchanged API)
    - `model_dump()`, `model_dump_json()`: Pydantic serialization
    - `is_phase_enabled()`: Phase status checking
  - **Accessor Functions**: `get_models_config()`, `get_training_config()`, `get_acceleration_config()`, `get_deployment_config()`, `get_plugins_config()`, `get_hpo_config()`, `is_hpo_enabled()`, `validate_models_config()`
- **PyGIntegration**: PyTorch Geometric utilities 
  - **Data Validation**:
    - `validate_pyg_data()`: Comprehensive validation with strict mode
    - `check_data_compatibility()`: Model requirement checking
    - Checks: node features, edge_index, edge_attr, edge_weight, labels, 3D coords
    - NaN/Inf detection, dimension consistency, index bounds
  - **Feature Dimension Inference**:
    - `infer_num_features()`: Auto-detect node/edge features, num_classes, output_dim
    - `infer_out_channels()`: **Single source of truth** for model output layer
    - Task-aware inference (classification → num_classes, regression → output_dim)
    - Supports Data, Dataset, and DataLoader inputs
  - **Dataset Statistics**:
    - `compute_dataset_statistics()`: Node/edge counts, feature presence, class info
    - `print_dataset_summary()`: Formatted console output
  - **Graph Statistics**:
    - `compute_graph_statistics()`: Density, degree distribution, min/max/avg
  - **Batch Processing**:
    - `create_dataloader()`: PyG DataLoader with sensible defaults
    - `get_batch_info()`: Batch size, node/edge counts per graph
  - **Utilities**:
    - `to_device()`: Move PyG data to device
    - `detach_data()`: Detach tensors from computation graph
    - `clone_data()`: Deep copy Data objects

**Plugins Submodule** (`models/plugins/`):
- **ModelPluginLoader**: Plugin system
  - Plugin discovery from YAML metadata
  - Automatic model registration
  - Validation (dependencies, security, functional)
  - Version management
  - Thread-safe registry

**HPO Submodule** (`models/hpo/`) :
- **HPOManager**: Main orchestrator for hyperparameter optimization 
  - **Backend support**: Optuna (primary), Ray Tune (optional)
  - **Factory methods**: `from_config()`, `from_yaml()`
  - **Core method**: `optimize()` - runs full HPO pipeline
  - **Task-specific data preparation**:
    - `_prepare_data_for_task_hpo()`: Routes to task-specific handlers
    - `_prepare_classification_data_hpo()`: DiscretizeTargets integration
    - `_prepare_node_level_data_hpo()`: TargetSelectionConfig support
    - `_prepare_edge_regression_data_hpo()`: Edge target extraction
    - `_prepare_link_prediction_data_hpo()`: Edge label handling
  - **`infer_task_type()`**: Auto-detect from metric/data characteristics
  - **Registry integration**:
    - `_create_optimizer_from_registry()`: Uses OptimizerRegistry
    - `_create_scheduler_from_registry()`: Uses SchedulerRegistry
    - `_create_loss_from_registry()`: Uses LossRegistry with task-aware selection
  - **Search space filtering**: `_filter_search_space_for_model()` validates params against model
  - **Cross-validation**: `_run_cross_validation()` with k-fold support
  - **Convenience functions**: `is_hpo_enabled()`, `get_best_params()`, `create_hpo_manager()`
  - Uses DynamicModelMetadata from pyg_introspector
- **HPOConfig**: Configuration dataclasses (frozen, validated) 
  - **HPOConfig**: Master switch and complete settings
    - `enabled`: MASTER SWITCH for HPO
    - `backend`: "optuna" or "ray_tune"
    - `n_trials`, `timeout`, `n_jobs`: Trial execution settings
    - `search_space`: Dict[str, Dict[str, SearchSpaceParamConfig]]
    - `cv_folds`, `cv_metric_aggregation`, `task_type`: CV settings
    - `from_dict()`: Factory method from config dictionary
  - **PrunerType enum** (7 types):
    - MEDIAN, PERCENTILE, HYPERBAND, SUCCESSIVE_HALVING, THRESHOLD, PATIENT, NONE
  - **PrunerConfig** (frozen dataclass):
    - `type`, `n_startup_trials`, `n_warmup_steps`, `interval_steps`
    - `percentile` (for PERCENTILE), `n_brackets` (for HYPERBAND)
  - **SamplerType enum** (7 types):
    - TPE, RANDOM, CMAES, GRID, NSGAII, MOTPE, QMCSAMPLER
  - **SamplerConfig** (frozen dataclass):
    - `type`, `n_startup_trials`, `seed`, `multivariate`, `constant_liar`
  - **OptimizationDirection enum**: MINIMIZE, MAXIMIZE
  - **StudyConfig** (frozen dataclass):
    - `direction`, `metric`, `study_name`, `storage`, `load_if_exists`
    - `is_multi_objective` property
  - **MultiObjectiveStudyConfig** (frozen dataclass):
    - `directions` tuple, `metrics` tuple (Pareto optimization)
    - `reference_point` for hypervolume calculation
- **Backends** (`hpo/backends/`) :
  - **HPOBackendProtocol** (@runtime_checkable Protocol): Abstract interface (7 methods)
    - `create_study()`, `optimize()`, `get_best_params()`, `get_best_value()`
    - `get_all_trials()`, `create_pruner()`, `create_sampler()`
  - **Factory function**: `get_backend()` returns backend by name (optuna, ray_tune)
  - **OptunaBackend** (497 lines): Primary implementation 
    - **7 Pruner types**: median, percentile, hyperband, successive_halving, threshold, patient, none
    - **7 Sampler types**: tpe, random, cmaes, grid, nsgaii, motpe, qmc
    - **Methods**: `suggest_params()` maps search space to Optuna suggest calls
    - **Storage**: SQLite, PostgreSQL persistence
  - RayTuneBackend: Distributed optimization (complete, inactive)
- **Callbacks** (`hpo/callbacks/`) :
  - **OptunaPruningCallback** (extends Callback ABC): Integrates with Trainer callback system
    - **Attributes**: trial, monitor, report_every, _reported_steps (duplicate prevention)
    - **Methods**: `set_trainer()`, `on_train_begin()`, `on_epoch_end()`, `on_train_end()`, `should_stop()`
    - **Metric resolution**: Alternative name checking (val_*, validation_*)
  - **Factory function**: `create_hpo_callback()` for backend selection (optuna, ray_tune)
  - RayTuneReportCallback: Ray Tune metric reporting (complete, inactive)
- **Search Spaces** (`hpo/search_spaces/`) :
  - **ParamType enum** (7 types): INT, FLOAT, CATEGORICAL, LOGUNIFORM, UNIFORM, INT_UNIFORM, DISCRETE_UNIFORM
  - **SearchSpaceParamConfig** (frozen dataclass):
    - `type`: ParamType enum value
    - `low`, `high`: Bounds for numeric types
    - `step`: Step size for int types (optional)
    - `choices`: List for categorical type
    - `log`: Whether to use log scale
    - `__post_init__` validation: Validates required fields based on type
  - **SearchSpaceBuilder** (1140 lines): Fluent builder with dynamic model spaces 
    - **VALID_CATEGORIES** (7): hyperparameters, model, optimizer, scheduler, loss, training, architecture
    - **Fluent builder methods**: `add_int()`, `add_float()`, `add_loguniform()`, `add_categorical()`, `add_uniform()`, `add_discrete_uniform()`, `add_param()`, `add_category()`, `remove_param()`, `build()`, `to_dict()`
    - **Dynamic introspection**: `for_model()` uses pyg_introspector for ANY PyG model
    - **Class methods**: `for_model()`, `for_optimizer()`, `for_scheduler()`, `merge()`, `validate()`, `get_param_count()`, `estimate_search_space_size()`, `list_available_models()`
    - **Legacy fallback**: Hardcoded spaces for GCN, GAT, GraphSAGE, GIN, SchNet, DimeNet, MPNN
    - **Convenience functions**: `build_search_space()`, `get_model_search_space()`, `validate_search_space()`
- **Transfer Learning** (`hpo/transfer/`) :
  - **HPOTransferManager** (1255 lines): Cross-study knowledge transfer ⭐ PYDANTIC V2
    - **MetaFeatureMethod enum** (3 types): STATISTICAL, LEARNED, LANDMARK
    - **AdaptationMethod enum** (4 types): WEIGHTED, FILTERED, FULL, ADAPTIVE
    - **TransferConfig** (frozen BaseModel, Pydantic V2):
      - `n_warm_start_trials`, `similarity_threshold` (0-1)
      - `meta_feature_method`, `adaptation_method`
      - `weight_by_performance`, `scale_to_bounds`, `add_noise`, `noise_scale`
      - `persist_meta_db`, `meta_db_path`
      - `@field_validator`: n_warm_start_trials (>=1), similarity_threshold (0-1), noise_scale (0-1)
      - `@model_validator(mode='before')`: enum conversion, cross-field validation
      - `to_dict()` wraps `model_dump()`
    - **RegisteredStudyInfo** (mutable BaseModel, Pydantic V2): study_name, meta_features, best_params, best_value, n_trials, n_completed, direction, model_name, dataset_info, registered_at
      - `@model_validator(mode='before')`: auto-sets registered_at timestamp
      - `to_dict()` wraps `model_dump()`, `from_dict()` uses `model_validate()`
    - **Study registration**: `register_study()`, `unregister_study()`, `get_registered_studies()`, `get_study_info()`
    - **Similarity computation**: `find_similar_studies()`, `compute_dataset_similarity()`
    - **Warm-start**: `warm_start_study()`, `_transfer_trials()`, `_filter_params()`, `_add_noise_to_params()`
    - **Persistence**: `export_meta_db()`, `import_meta_db()`, `_save_meta_db()`, `_load_meta_db()`
    - **Utilities**: `get_transfer_summary()`, `clear()`
  - **MetaFeatureExtractor** (1157 lines): Dataset similarity computation ⭐ PYDANTIC V2
    - **MetaFeatureCategory enum** (7 types): STATISTICAL, GRAPH, MOLECULAR, TARGET, NODE_FEATURES, EDGE_FEATURES, ALL
    - **MetaFeatureConfig** (frozen BaseModel, Pydantic V2): categories, max_samples, normalize, include_molecular, compute_expensive
    - **6 Meta-Feature Categories**:
      - Statistical: n_samples, n_features, n_edge_features
      - Graph: mean_nodes/edges, density, degree distribution, clustering_coefficient
      - Target: mean, std, range, skewness, dim
      - Node Features: mean, std, min, max, sparsity
      - Edge Features: has_edge_features, mean, std
      - Molecular: atom_frac_*, bond_frac_*, mol_weight_*, ring_count_*, heavy_atom_ratio
    - **API**: `extract()` static, `extract_features()` instance, `compute_similarity()` cosine
    - **Utilities**: `get_feature_names()`, `get_category_for_feature()`, normalization
    - **Lazy imports**: RDKit, torch, torch_geometric
  - **WarmStartStrategy** (820 lines): Trial transfer mechanisms 
    - **WarmStartMethod enum** (4 types): WEIGHTED, FILTERED, FULL, ADAPTIVE
    - **WarmStartConfig** (frozen dataclass):
      - `method`, `n_trials`, `min_similarity` (0-1)
      - `weight_by_performance`, `filter_invalid`, `scale_to_bounds`
      - `add_noise`, `noise_scale`
    - **TransferredTrial dataclass**: params, value, source_study, similarity, weight, is_valid
    - **Static transfer methods**:
      - `weighted_transfer()`: Weight by similarity scores
      - `filtered_transfer()`: Filter by target search space compatibility
      - `full_transfer()`: Top-k without modification
    - **Instance methods**: `transfer()`, `apply_to_study()`
    - **Trial processing**: `_filter_invalid_trials()`, `_add_noise_to_trials()`, `_select_adaptive_method()`
    - **Utilities**: `get_transfer_summary()`, `create_from_best_trials()`
- **Neural Architecture Search** (`hpo/nas/`) :
  - **LayerType enum** (7 types): GCN, GAT, SAGE, GIN, GATV2, TRANSFORMER, PNA
  - **PoolingType enum** (6 types): MEAN, MAX, SUM, ATTENTION, SET2SET, TOPK
  - **AggregationType enum** (5 types): MEAN, MAX, SUM, LSTM, MULTI
  - **ActivationType enum** (7 types): RELU, GELU, ELU, LEAKY_RELU, SILU, TANH, PRELU
  - **LayerConfig** (frozen dataclass): type, hidden_channels, heads, dropout, activation, batch_norm, residual
  - **GNNArchitectureSpace** (1018 lines): Searchable GNN components 
    - **Search space fields**: min_layers, max_layers, layer_types, hidden_channels, heads, dropout_range
    - **Architecture options**: allow_skip_connections, allow_dense_connections, allow_mixed_layers
    - **Component options**: pooling_types, aggregation_types, activation_types, batch_norm_options
    - **Methods**: `to_optuna_search_space()`, `to_dict()`, `from_dict()`, `get_attention_layer_types()`, `has_attention_layers()`, `get_search_dimensions()`, `estimate_search_space_size()`, `create_default_layer_config()`
  - **Factory functions**: `create_gnn_search_space()` with presets (gcn, gat, sage, gin, transformer, pna, mixed)
  - **NASConfig dataclass**: n_trials, timeout, metric, direction, cv_folds, study_name, storage
  - **NASManager** (1138 lines): Architecture search orchestrator 
    - **Core methods**: `search()`, `build_model()`, `get_best_architecture()`, `get_best_params()`
    - **Model building**: Homogeneous (single layer type) or heterogeneous (mixed types)
    - **Internal methods**: `_convert_arch_space_to_hpo_format()`, `_create_hpo_config_from_nas_config()`, `_merge_search_spaces()`, `_extract_architecture()`
    - **Integration**: Uses HPOManager for optimization
  - **HeterogeneousGNN** (nn.Module): Dynamic GNN model from architecture config
    - **Layer support**: GCN, GAT, GATv2, SAGE, GIN, Transformer
    - **Methods**: `forward()`, `_create_layer()`, `_get_activation()`, `_create_pooling()`
    - **Features**: batch_norm, skip connections, attention heads, global pooling
  - **Convenience functions**: `create_nas_manager()`, `get_default_gnn_search_space()`
- **Analysis** (`hpo/analysis/`) :
  - **ImportanceMethod enum** (2 types): FANOVA, MDI
  - **ExportFormat enum** (4 types): JSON, CSV, DATAFRAME, DICT
  - **AnalysisConfig** (frozen dataclass): importance_method, n_importance_trials, convergence_window, include_pruned, include_failed, percentile_thresholds
  - **StudyAnalyzer** (1704 lines): Comprehensive study analysis 
    - **Factory methods**: `from_manager()`, `from_storage()`
    - **Trial data**: `get_trials()`, `get_completed_trials()`, `get_trial_count()`
    - **Parameter analysis**: `get_parameter_importance()`, `get_parameter_importance_ranking()`, `get_parameter_statistics()`, `get_parameter_correlations()`
    - **Convergence**: `get_convergence_data()`, `get_optimization_trajectory()`
    - **Statistics**: `get_value_statistics()` with percentiles
    - **Multi-objective**: `get_pareto_front()`, `get_hypervolume()`
    - **Visualization data**: `get_optimization_history_data()`, `get_slice_plot_data()`, `get_contour_plot_data()`, `get_parallel_coordinate_data()`
    - **Export**: `export_results()`, `to_dataframe()`
    - **Comparison**: `compare_with()` for multi-study analysis
    - **Comprehensive**: `get_comprehensive_analysis()`

#### Key Features

**Key Extensions**:
- Custom architecture building via ArchitectureBuilder
- Multi-model ensembles via ModelComposer
- Layer registry with 50+ GNN layers
- Architecture templates for quick prototyping
- YAML/JSON configuration parsing
- Architecture validation
- Backward compatibility (100%)

**Core Capabilities**:
- ALL PyTorch Geometric models via dynamic introspection 
- Automatic hyperparameter validation
- Channel inference from sample data
- Multiple data splitting strategies
- Comprehensive callback system
- Custom loss functions
- Hardware acceleration support
- Distributed training
- Memory optimization
- Model deployment utilities
- Production monitoring
- Plugin architecture

**Training Infrastructure**:
- Full-featured Trainer class
- 6 callback types for training control
- Multiple optimizer support
- Learning rate schedulers with warmup
- Loss function registry
- Progress tracking and logging

**Performance Optimization**:
- Mixed precision training
- Gradient checkpointing
- JIT compilation
- Distributed training (DP/DDP)
- Memory profiling
- Computation optimization

**Production Features**:
- Model quantization
- Pruning
- Knowledge distillation
- Edge deployment
- Cloud deployment
- Drift detection
- Performance monitoring
- Automated retraining

**HPO Features** :
- Optuna-based hyperparameter optimization (primary backend)
- Ray Tune backend for distributed scale-out (complete, inactive)
- TPE, CMA-ES, Random samplers for hyperparameter suggestion
- Median, Hyperband, Percentile pruners for early trial termination
- Cross-validation integration (k-fold via DataSplitter)
- Search space builder with dynamic model spaces (ANY PyG model) 
- HPO transfer learning across studies/datasets
- Meta-feature extraction for dataset similarity
- Warm-start strategies for faster convergence
- Neural Architecture Search (NAS) for GNN architectures
- Mixed layer type architectures (heterogeneous GNNs)
- Comprehensive study analysis (parameter importance, convergence)
- Multi-objective optimization (Pareto front, hypervolume)
- Export to DataFrame, JSON, CSV for analysis

#### Thread Safety

- **Thread-Safe Components**:
  - ModelRegistry: Singleton with thread locks
  - ModelFactory: Thread-safe instantiation
  - LayerRegistry: Thread-safe layer access
  - Plugin system: Thread-safe loading/registration
  - HPOManager: Thread-safe study creation and access 
  - SearchSpaceBuilder: Immutable frozen configs 

- **Thread-Unsafe Components** (require external synchronization):
  - Model training state modifications
  - Callback state updates during training
  - Plugin loading/unloading operations
  - HPO trial execution (each trial runs single-threaded) 

#### Integration

The models module integrates with:
- **PyTorch Geometric**: For GNN implementations
- **Config module**: Via config_bridge for configuration access
- **Datasets module**: For data loading and processing
- **Handlers module**: For dataset-specific processing
- **Descriptors module**: For molecular feature enrichment
- **Transformations module**: For data augmentation
- **Optuna**: For hyperparameter optimization backend 
- **Ray Tune**: For distributed scale-out (optional) 

#### Error Handling

Custom exceptions for comprehensive error management:
- ModelError: Base exception for model-related errors
- ModelNotFoundError: Model not found in registry
- ModelValidationError: Validation failures
- ModelInstantiationError: Model creation failures
- HyperparameterError: Invalid hyperparameters
- DataCompatibilityError: Data incompatibility
- TrainingError: Training failures
- CheckpointError: Checkpoint loading/saving failures
- PluginModelError: Plugin-related errors
- HPOError: Base exception for HPO-related errors 
- HPOConfigurationError: HPO configuration validation failures 
- TrialFailedError: HPO trial execution failures 
- StudyNotFoundError: Study not found in storage 
- SearchSpaceError: Invalid search space configuration 
- PruningError: Pruning-related failures 
- BackendError: HPO backend initialization/operation failures 

---

### HPO Module API 

**Version**: 1.0.0 | **Backend**: Optuna (primary), Ray Tune (optional) | **Thread-Safe**: Yes

The HPO module provides comprehensive hyperparameter optimization capabilities integrated with the MILIA training infrastructure. It supports single-objective and multi-objective optimization, neural architecture search, and transfer learning across studies.

#### Module Structure

```
milia_pipeline/models/hpo/
├── __init__.py                      # Public API exports
├── hpo_config.py                    # HPOConfig (7 pruners, 7 samplers) ⭐ PYDANTIC V2
├── hpo_manager.py                   # HPOManager v1.1.0 (2593 lines) 
├── backends/                        # HPO backend implementations
│   ├── __init__.py                  # Backend exports
│   ├── base.py                      # HPOBackendProtocol 
│   ├── optuna_backend.py            # OptunaBackend 
│   └── ray_tune_backend.py          # RayTuneBackend (inactive)
├── callbacks/                       # Training callbacks for HPO
│   ├── __init__.py                  # Callback exports
│   ├── optuna_callback.py           # OptunaPruningCallback 
│   └── ray_tune_callback.py         # RayTuneReportCallback (inactive)
├── search_spaces/                   # Search space definition
│   ├── __init__.py                  # Search space exports
│   ├── param_types.py               # ParamType (7), SearchSpaceParamConfig ⭐ PYDANTIC V2
│   └── search_space_builder.py      # SearchSpaceBuilder v2.0.0 
├── transfer/                        # HPO transfer learning
│   ├── __init__.py                  # Transfer exports
│   ├── transfer_manager.py          # HPOTransferManager v1.0.0 
│   ├── meta_features.py             # MetaFeatureExtractor v1.0.0 
│   └── warm_start.py                # WarmStartStrategy v1.0.0 
├── nas/                             # Neural Architecture Search
│   ├── __init__.py                  # NAS exports
│   ├── search_space.py              # GNNArchitectureSpace v1.0.0 
│   └── nas_manager.py               # NASManager v1.0.0 
└── analysis/                        # Study analysis
    ├── __init__.py                  # Analysis exports
    └── study_analyzer.py            # StudyAnalyzer v1.0.0 
```

#### Key API Functions

| Category | Functions | Description |
|----------|-----------|-------------|
| **HPO Manager** | `HPOManager(config)`, `.optimize()`, `.get_best_value()`, `.from_yaml()` | Main optimization interface |
| **Configuration** | `HPOConfig`, `PrunerConfig`, `SamplerConfig`, `PrunerType`, `SamplerType` | HPO configuration (Pydantic BaseModel, frozen=True) |
| **Search Space** | `SearchSpaceBuilder.add_int()/.add_float()/.add_categorical()/.build()` | Fluent builder pattern |
| **Predefined Spaces** | `get_model_search_space(model_name)`, `get_optimizer_search_space()` | Model-specific spaces |
| **Study Analysis** | `StudyAnalyzer.from_manager()`, `.get_parameter_importance()`, `.to_dataframe()` | Post-optimization analysis |
| **NAS** | `NASManager`, `GNNArchitectureSpace`, `LayerType`, `PoolingType` | Neural architecture search |
| **Transfer Learning** | `HPOTransferManager`, `TransferConfig`, `.register_study()`, `.warm_start_study()` | Cross-study transfer |

*For complete usage examples, see `examples/hpo/` and `config.yaml` models.hpo section.*

#### Configuration Schema (config.yaml)

```yaml
models:
  hpo:
    # MASTER SWITCH - enables HPO when true
    enabled: false
    
    # Backend selection
    backend: "optuna"           # optuna or ray_tune
    
    # Trial settings
    n_trials: 100               # Number of optimization trials
    timeout: null               # Max time in seconds (null = no limit)
    n_jobs: 1                   # Parallel jobs (1 = sequential)
    
    # Search space (per-category)
    search_space:
      model:
        hidden_channels:
          type: "int"
          low: 32
          high: 256
          step: 32
        num_layers:
          type: "int"
          low: 2
          high: 6
        dropout:
          type: "float"
          low: 0.0
          high: 0.5
      optimizer:
        lr:
          type: "loguniform"
          low: 1e-5
          high: 1e-2
        weight_decay:
          type: "loguniform"
          low: 1e-6
          high: 1e-3
    
    # Pruner configuration
    pruner:
      type: "median"            # median, hyperband, percentile, none
      n_startup_trials: 5
      n_warmup_steps: 10
    
    # Sampler configuration
    sampler:
      type: "tpe"               # tpe, random, cmaes, grid
      n_startup_trials: 10
      multivariate: true
    
    # Study configuration
    study:
      direction: "minimize"     # minimize or maximize
      metric: "val_loss"
      study_name: "milia_hpo"
      storage: null             # null = in-memory, or "sqlite:///hpo.db"
    
    # Cross-validation
    cv_folds: 0                 # 0 = no CV, >0 = k-fold CV
    cv_metric_aggregation: "mean"
```

#### HPO Backend Protocol

All backends implement the `HPOBackendProtocol` with 7 methods:

```python
class HPOBackendProtocol(Protocol):
    def create_study(self, study_name, direction, storage, ...) -> Any: ...
    def optimize(self, study, objective_fn, n_trials, ...) -> None: ...
    def get_best_params(self, study) -> Dict[str, Any]: ...
    def get_best_value(self, study) -> float: ...
    def get_all_trials(self, study) -> List[Dict[str, Any]]: ...
    def create_pruner(self, pruner_type, ...) -> Any: ...
    def create_sampler(self, sampler_type, ...) -> Any: ...
```

#### Key Features

**Optimization Capabilities**:
- Single-objective optimization (minimize/maximize)
- Multi-objective optimization with Pareto front
- Cross-validation integration (k-fold)
- Early trial termination (pruning)
- Study persistence (SQLite, in-memory)
- Resume interrupted studies

**Search Algorithms**:
- TPE (Tree-structured Parzen Estimator)
- CMA-ES (Covariance Matrix Adaptation)
- Random search
- Grid search (exhaustive)
- NSGA-II (multi-objective)

**Pruning Strategies**:
- Median pruner (prune below median)
- Hyperband pruner (successive halving)
- Percentile pruner (configurable percentile)
- Patient pruner (with tolerance)
- Threshold pruner (fixed threshold)

**Transfer Learning**:
- Meta-feature extraction for dataset similarity
- Cross-study warm-starting
- Few-shot optimization on new tasks
- Knowledge transfer between related datasets

**Neural Architecture Search**:
- GNN-specific search space (7 layer types)
- Heterogeneous architectures (mixed layer types)
- Dynamic model building from configurations
- Pooling, aggregation, activation search

---

### Descriptors Module API

**Version**: 1.0.0 | **Descriptors**: 400+ across 6 categories | **Thread-Safe**: Yes

**Key API Functions**:

| Category | Functions | Description |
|----------|-----------|-------------|
| **Calculation** | `DescriptorCalculator.calculate_batch(mol, descriptors)` | Compute descriptors |
| **Categories** | `get_category_descriptor_names(category)`, `DescriptorCategory` enum | Category-based selection |
| **Validation** | `DescriptorValidator.filter_by_requirements(mol, descriptors)` | Requirement filtering |
| **PyG Integration** | `add_descriptors_to_pyg_data(data, descriptors, create_feature_vector)` | Add to PyG Data |
| **Plugins** | `DescriptorPluginLoader.discover_plugins(paths, auto_validate)` | Plugin discovery |

**Descriptor Categories**: Constitutional (35), Topological (350+), Electronic (8), Geometric (10), Drug-likeness (4), Fragments (85)

**Key Features**: Auto-discovery of RDKit descriptors, caching support, conformer generation for 3D, batch processing with statistics.

*For complete usage examples, see `examples/descriptors/`*

---

### Handlers Module API

**Version**: 2.0.0 | **Handler Types**: 3 (DFT, DMC, Wavefunction) | **Circular Import Resolution**: Lazy Loading

**Key API Functions**:

| Category | Functions | Description |
|----------|-----------|-------------|
| **Handler Creation** | `create_dataset_handler(dataset_config, filter_config, processing_config, logger)` | Factory pattern (recommended) |
| **Transform Integration** | `TransformAwareHandlerIntegrator(configs, experimental_setup, enable_caching)` | Transform-aware processing |
| **Query** | `get_available_handlers()`, `get_handler_info(handler_type)` | Handler discovery |

**Handler Characteristics**:

| Handler | Structural Features | Uncertainty | Typical Properties |
|---------|-------------------|-------------|-------------------|
| DFT | All supported | No | Etot, U0, H, G, Cv, gap, dipole |
| DMC | Limited (safe subset) | Yes | Etot, std, Eatomization |
| Wavefunction | All supported | No | orbital_energies, HOMO, LUMO |
| QM9 | All supported | No | U0, U, H, G, zpve, homo, lumo, gap |
| ANI1x | All supported | No | energy, forces, charges, dipole |
| ANI1ccx | All supported | No | ccsd_energy, dft_energy, forces, charges, dipole |
| RMD17 | All supported | No | energies, forces, molecule_name |

**Architectural Note**: Uses `__getattr__` lazy loading to break circular import chain. Error handling via `handle_transform_errors` decorator.

*For complete usage examples, see `examples/handlers/`*

---

### Molecules Module API

**Architecture**: Handler-Only

**Key API Functions**:

| Category | Functions | Description |
|----------|-----------|-------------|
| **Conversion** | `MoleculeDataConverter(dataset_config, logger, handler).convert()` | Raw data → PyG Data |
| **Molecule Creation** | `create_rdkit_mol(mol_identifier, coords, atoms, handler)` | Handler-determined strategy |
| **Feature Extraction** | `add_structural_features(mol, pyg_data, feature_config)`, `get_available_features()` | Atom/bond features |
| **Validation** | `validate_molecular_structure(atoms, coords, handler)`, `check_dataset_compatibility()` | Structure validation |
| **Enrichment** | `enrich_pyg_data_with_properties(pyg_data, properties, handler)` | Handler-orchestrated |
| **Filtering** | `create_molecule_filter(configs, handler, transform_config)`, `.apply_filters()` | Transform-aware filtering |
| **Registry Status** | `get_registry_integration_status()`, `get_validator_registry_status()` | Registry diagnostics |

**Processing Workflow**: Raw Data → `create_dataset_handler()` → `create_rdkit_mol()` → `add_structural_features()` → `validate_molecular_structure()` → `mol_to_pyg_data()` → `enrich_pyg_data_with_properties()` → `apply_filters()` → PyG Data

*For complete usage examples, see `examples/molecules/`*

---

### Transformations Module API

**Version**: 1.0.0 | **Plugin System**: Yes | **Transform Types**: 40+ built-in, unlimited via plugins

**Key API Functions**:

| Category | Functions | Description |
|----------|-----------|-------------|
| **Composition** | `TransformComposer(registry, cache_enabled).add(transform).compose()` | Transform chaining |
| **Research API** | `TransformResearchAPI(dataset, registry).run_experiment(experiment)` | Ablation studies |
| **Plugins** | `discover_transform_plugins(paths, auto_validate)` | Plugin discovery |

**Transform Categories**: Node transforms (15+), Edge transforms (10+), Graph transforms (10+), Structural transforms (5+)

**Key Features**: Caching for expensive transforms, plugin-based extensibility, research API for experiments, validation and error handling, composition and chaining.

*For complete usage examples, see `examples/transformations/`*

---

### Preprocessing Module API

**Version**: 1.5 | **Preprocessor Types**: Wavefunction, QM9, ANI1x, ANI1ccx, ANI2x, RMD17, XXMD, QDPi | **Registry-Based**: Yes

**Module Exports** (`__all__`, 9 items):
- Core: `BasePreprocessor`, `PreprocessorRegistry`
- Archive: `extract_from_targz`
- Parsers: `parse_molden_files`
- Builders: `build_npz`, `validate_npz_structure`
- Convenience: `get_preprocessing_info`, `list_available_preprocessors`, `supports_dataset`

**Utils Exports** (`preprocessing/utils/__init__.py`, 10 functions):
- Archive: `extract_from_targz`, `extract_from_archive`, `get_supported_formats`
- Parsers: `parse_molden_files`, `parse_qm9_xyz_files`, `get_qm9_property_info`
- Builders: `build_npz`, `validate_npz_structure`

**Key API Functions**:

| Category | Functions | Description |
|----------|-----------|-------------|
| **Module Info** | `get_preprocessing_info()`, `list_available_preprocessors()`, `supports_dataset()` | Query available preprocessors |
| **Registry** | `PreprocessorRegistry.get_preprocessor()`, `.list_preprocessors()`, `.supports_preprocessing()` | Registry-based access |
| **Custom Preprocessors** | `@PreprocessorRegistry.register()` decorator on `BasePreprocessor` subclass | Zero-modification extension |
| **Preprocessors** | `WavefunctionPreprocessor`, `QM9Preprocessor`, `ANI1xPreprocessor`, `ANI1ccxPreprocessor`, `ANI2xPreprocessor`, `RMD17Preprocessor`, `XXMDPreprocessor`, `QDPiPreprocessor` | Dataset-specific preprocessing |
| **Archive Utils** | `extract_from_archive()`, `extract_from_targz()`, `get_supported_formats()` | Multi-format extraction |
| **Parsers** | `parse_molden_files()`, `parse_qm9_xyz_files()`, `get_qm9_property_info()` | Format-specific parsing |
| **NPZ Utils** | `build_npz()`, `validate_npz_structure()` | NPZ file construction |

**Preprocessor Features**: Automatic format detection, archive extraction (tar.gz/bz2/xz, zip, HDF5), parallel processing, progress tracking, error recovery, NPZ output with metadata.

*For complete usage examples, see `examples/preprocessing/` and `MILIA_Adding_New_Datasets_Implementation_Blueprint.md`*

---

### Config Module API

**Architecture**: Registry Integration + Pydantic V2 | **Thread-Safe**: Yes

**Module Structure**:
```
config/
├── __init__.py                  # Module exports (~1402 lines, 200+ exports)
├── config_loader.py             # YAML loading with caching and splitting (~2755 lines)
├── config_containers.py         # Pydantic BaseModel containers (~4479 lines) ⭐ PYDANTIC V2
├── config_accessors.py          # Accessor functions (~4353 lines, 60+ functions)
├── config_constants.py          # Constants with registry (~2444 lines)
├── config_schemas.py            # Schema validation (~3461 lines)
├── validators.py                # Data validation (~5069 lines)
└── data_refining.py             # Data refinement (~3190 lines)
```

**Key API Functions**:

| Category | Functions | Description |
|----------|-----------|-------------|
| **Loading** | `load_config()`, `load_config_with_validation()`, `reload_config()`, `clear_config_cache()` | Thread-safe YAML loading (single-file and split-file modes) |
| **YAML Splitting** | `_discover_config_files()`, `_collect_yaml_files()`, `_deep_merge_configs()`, `_load_and_merge_yaml_files()` | Split configuration support |
| **Containers** | `DatasetConfig`, `FilterConfig`, `ProcessingConfig`, `TransformationConfig`, `DescriptorConfig` | Pydantic frozen BaseModel ⭐ |
| **Factory** | `create_dataset_config_from_global()`, `create_transformation_config_from_global()` | Config creation |
| **Accessors** | `get_dataset_type()`, `get_dataset_config()`, `get_filter_config()`, `get_processing_config()` | 60+ accessor functions |
| **Uncertainty** | `get_uncertainty_config()`, `is_uncertainty_enabled()` | DMC-specific |
| **Transforms** | `get_experimental_setup()`, `list_experimental_setups()`, `get_standard_transforms()` | Transform config |
| **Validation** | `YAMLSchemaValidator`, `TransformValidator`, `validate_molecular_structure()`, `validate_config_file()` | Schema/data validation (split-file aware) |
| **Migration** | `check_migration_status()`, `migrate_legacy_config()` | Config migration |
| **Refinement** | `refine_molecular_data()`, `refine_molecular_vibrations()`, `create_refinement_handler()` | Data refinement |
| **Registry** | `get_supported_handler_types()`, `get_handler_feature_support()`, `get_validators_registry_status()` | Registry diagnostics |

**Thread Safety**: Configuration cache protected by `RLock`, statistics updates protected by `Lock`.

*For complete usage examples, see `examples/config/` and API reference documentation.*

---

## Detailed Module Breakdown

### 1. Config Module (`config/`)

**Purpose**: Comprehensive configuration management with registry integration, Pydantic V2 validation, and type safety

**Architecture**: Handler-Only with Registry Integration + Pydantic V2 | **Thread-Safe**: Yes

**Components**:

```
config/
├── __init__.py                  # Package exports (~1402 lines)
├── config_loader.py             # YAML loading with thread-safe caching and splitting (~2755 lines)
├── config_containers.py         # Pydantic BaseModel containers (~4479 lines) ⭐ PYDANTIC V2
├── config_accessors.py          # 60+ accessor functions (~4353 lines)
├── config_constants.py          # Constants with registry integration (~2444 lines)
├── config_schemas.py            # Schema validation & migration (~3461 lines)
├── validators.py                # Data & transform validation (~5069 lines)
└── data_refining.py             # Data refinement with VQM24 support (~3190 lines)
```

**config_loader.py** (~2755 lines):
- **YAML Splitting Architecture**: Supports both single-file (`config.yaml`) and split-file (`configs/`) modes
- `_discover_config_files()`: Detect single-file vs directory mode, returns (is_split_mode, files_list)
- `_collect_yaml_files()`: Discover YAML files in merge order (main.yaml first, then alphabetical, then datasets/)
- `_deep_merge_configs()`: Recursive dict merge with copy.deepcopy (thread-safe, no input mutation)
- `_load_and_merge_yaml_files()`: Load multiple YAML files and merge in order
- `_get_default_config_path()`: Priority: config.yaml > config.yml > ./configs/ > ./configs/config.yaml
- `load_config()`: Thread-safe YAML loading with intelligent caching (supports split-file mode with hash-based cache keys)
- `load_config_with_validation()`: Loading with validation levels
- `reload_config()`: Force cache invalidation and reload
- `clear_config_cache()`: Clear configuration cache
- `validate_config_file()`: Validation with split-file support (returns split_mode, config_files_count metadata)
- `migrate_legacy_config()`: Migrate legacy format configs
- `check_migration_status()`: Check if migration needed
- `_get_default_dataset_type()`: Dynamic default from registry
- Registry flags: `_REGISTRY_AVAILABLE`, `_REGISTRY_INITIALIZED`

**config_containers.py** (⭐ PYDANTIC V2):
- **Architecture**: All 10 frozen dataclasses migrated to Pydantic V2 `BaseModel` with `frozen=True`
- **Import**: `from pydantic import BaseModel, field_validator, model_validator, Field`
- **Pattern**: Uses `model_validator(mode='before')` for field initialization (NOT `mode='after'` with `model_copy`)
- **DatasetConfig**: Dataset configuration (Pydantic frozen BaseModel)
  - Fields: `dataset_type`, `uncertainty_config`, `is_uncertainty_enabled`, `handler_config`, `validation_config`, `migration_config`
  - Validators: `@field_validator('dataset_type')` for dynamic registry validation
  - Methods: `is_compatible_with_handler()`, `get_handler_config()`, `get_required_properties()`, `validate_handler_compatibility()`, `to_dict()` (wraps `model_dump()`)
  - Uses `_is_valid_dataset_type()` for dynamic validation
- **FilterConfig**: Filtering rules (Pydantic frozen BaseModel)
  - Fields: `max_atoms`, `min_atoms`, `heavy_atom_filter`, `dmc_uncertainty_filter`, `handler_filters`, `filter_validation`
  - Methods: `get_handler_filters()`, `validate_filter_config()`, `to_dict()`
- **StructuralFeaturesConfig**: Feature extraction (Pydantic frozen BaseModel)
  - Fields: `atom_features`, `bond_features`, `preprocessing`, `handler_features`, `feature_validation`
  - Methods: `get_handler_features()`, `to_dict()`
- **ProcessingConfig**: Processing parameters (Pydantic frozen BaseModel)
  - Fields: `scalar_graph_targets`, `node_features`, `vector_graph_properties`, `variable_len_graph_properties`, `calculate_atomization_energy_from`, `vibration_refinement`, `test_molecule_limit`, `handler_processing`, `migration_settings`
  - Methods: `get_handler_processing_config()`, `is_migration_enabled()`, `to_dict()`
- **HandlerConfig**: Handler configuration (Pydantic frozen BaseModel)
  - Fields: `handler_type`, `validation_settings`, `processing_settings`, `error_handling`, `performance_settings`, `migration_mode`, `compatibility_layer`
  - Validators: `@field_validator('handler_type')` for dynamic registry validation
  - Methods: `get_validation_setting()`, `get_processing_setting()`, `is_strict_validation_enabled()`, `to_dict()`
- **TransformSpec**: Transform specification (Pydantic frozen BaseModel)
  - Fields: `name`, `kwargs`, `enabled`, `description`, `validation_config`
  - Validators: `@field_validator('name')` for non-empty string, `@field_validator('kwargs')` for dict type
  - Methods: `get_cache_key()`, `validate_kwargs()`, `to_dict()`
- **ExperimentalSetup**: Experimental configuration (Pydantic frozen BaseModel)
  - Fields: `name`, `transforms`, `description`, `enabled`, `research_context`, `expected_effects`, `validation_config`, `dataset_compatibility`
  - Validators: `@field_validator('name')` for non-empty string, `@field_validator('transforms')` for list of TransformSpec
  - Methods: `get_transform_names()`, `get_enabled_transforms()`, `to_dict()`
- **TransformationConfig**: Complete transform system (Pydantic frozen BaseModel)
  - Fields: `experimental_setups`, `default_setup`, `standard_transforms`, `validation`, `performance_settings`, `migration_metadata`, `research_metadata`
  - Validators: `@field_validator('experimental_setups')` for dict of ExperimentalSetup, `@model_validator(mode='after')` for `default_setup` existence check
  - Methods: `get_default_setup()`, `get_setup()`, `list_setup_names()`, `get_all_transforms()`, `to_dict()`
- **DescriptorConfig**: Descriptor settings (Pydantic frozen BaseModel)
  - Fields: `enabled`, `default_categories`, `categories`, `cache_descriptors`, `cache_path`, `parallel_computation`, `num_workers`, `error_handling`, `validation_mode`
  - Validators: `@field_validator('error_handling')`, `@field_validator('validation_mode')`, `@field_validator('num_workers')` for >= 1
  - Auto-adjustment: `num_workers` set to 2 if `parallel_computation=True` and `num_workers=1`
  - Methods: `is_category_enabled()`, `get_category_descriptors()`, `to_dict()`
- **DescriptorCategoryConfig**: Category configuration (Pydantic frozen BaseModel)
  - Fields: `category_name`, `enabled`, `descriptors`, `options`
  - Validators: `@field_validator('category_name')` for valid category names
  - Methods: `to_dict()`, `from_dict()`
- Factory functions: `create_dataset_config_from_global()`, `create_filter_config_from_global()`, `create_processing_config_from_global()`, `create_transformation_config_from_global()`, `create_descriptor_config_from_yaml()`
- Registry helpers: `_get_valid_dataset_types()`, `_is_valid_dataset_type()`, `verify_container_registry_integration()`

**config_accessors.py**:
- **Core Accessors**: `get_dataset_type()`, `get_config_value()`, `get_dataset_config()`, `get_data_config()`, `get_property_availability()`
- **Uncertainty**: `get_uncertainty_config()`, `is_uncertainty_enabled()`
- **Structural Features**: `get_structural_features_config()`, `is_structural_features_enabled()`, `get_atom_features()`, `get_bond_features()`, `get_dataset_appropriate_structural_features()`, `validate_structural_features_for_dataset()`
- **Transformations**: `get_transformation_config()`, `get_experimental_setup()`, `list_experimental_setups()`, `get_default_experimental_setup()`, `get_standard_transforms()`, `get_combined_transforms()`, `has_standard_transforms()`, `validate_transformation_config()`
- **Descriptors**: `is_descriptors_enabled()`, `get_descriptor_config()`, `get_selected_descriptors()`
- **Registry Functions**: `registry_list_all()`, `registry_get()`, `registry_is_registered()`, `validate_dataset_type()`, `is_valid_dataset_type()`, `get_valid_dataset_types()`
- Class: `EnhancedConfigAccessor` for advanced access patterns
- Registry flags: `_REGISTRY_AVAILABLE`, `_REGISTRY_INITIALIZED`, `_REGISTRY_IMPORT_ERROR`

**config_constants.py**:
- **Dataset Constants**: `RAW_NPZ_FILENAME`, `RAW_DATA_DOWNLOAD_URL`, `DATASET_ROOT_DIR`, `PROCESSED_DATA_FILENAME`
- **Lazy-loaded Constants**: `ATOMIC_ENERGIES_HARTREE`, `HEAVY_ATOM_SYMBOLS_TO_Z`, `HAR2EV`, `BOHR_TO_ANGSTROM`
- **Dynamic Registry Functions**:
  - `get_supported_handler_types()`: Dynamic list from registry
  - `get_default_handler_type()`: Default type ('DFT')
  - `is_handler_type_supported()`: Check if registered
  - `get_handler_feature_support()`: Feature dict from registry
  - `get_handler_required_properties()`: Required properties from registry
  - `get_handler_molecule_creation_strategy()`: Strategy from registry
- **Handler Utilities**: `get_handler_constants()`, `get_handler_identifier_keys()`, `validate_handler_configuration()`, `get_handler_compatibility_info()`
- **Transform Utilities**: `get_transformation_constants()`, `get_compatible_transforms_for_handler()`, `get_incompatible_transforms_for_handler()`
- **Cache Management**: `clear_all_caches()`, `get_all_cache_info()`, `clear_handler_caches()`, `clear_transformation_caches()`
- Registry flags: `_REGISTRY_AVAILABLE`, `_CACHE_INVALIDATION_REGISTERED`, `_REGISTRY_IMPORT_ERROR`
- Lazy loading via `__getattr__()` to avoid circular imports

**config_schemas.py**:
- **Schema Classes**:
  - `TransformationSchema`: Transform config schema
  - `PluginConfigSchema`: Plugin configuration
  - `ExperimentSchema`: Research experiment schema
  - `DescriptorConfigSchema`: Descriptor settings
  - `WavefunctionConfigSchema`: Wavefunction-specific
  - `ValidationConfig`: Validation settings
- **Validator Classes**:
  - `YAMLSchemaValidator`: Complete YAML validation
  - `PluginSchemaValidator`: Plugin config validation
  - `ExperimentSchemaValidator`: Experiment validation
  - `DescriptorSchemaValidator`: Descriptor validation
- **Migration**: `ConfigMigration` class with format detection and migration
- Feature flags: `YAML_AVAILABLE`, `PLUGIN_SYSTEM_AVAILABLE`, `RESEARCH_API_AVAILABLE`
- Registry helpers: `_registry_list_all_safe()`, `_registry_is_registered_safe()`

**validators.py** (Pydantic Integration):
- **Pydantic V2 Integration**:
  - `PYDANTIC_AVAILABLE`: Runtime flag for Pydantic availability
  - `wrap_pydantic_validation_error()`: Converts Pydantic ValidationError → MILIA ValidationResult
  - `validate_with_pydantic_model()`: Validates dict against Pydantic BaseModel, returns ValidationResult or tuple
  - Aliased import: `from pydantic import ValidationError as PydanticValidationError`
- **Classes**:
  - `ValidationResult`: Wrapper with is_valid, errors, warnings
  - `ValidationSeverity`: Enum (ERROR, WARNING, INFO)
  - `ValidationIssueDetail`: Rich error reporting
  - `TransformValidator`: Advanced transform validation
- **Decorators**: `@must_check` - Enforces result checking
- **Molecular Validation**:
  - `validate_molecular_structure()`: Atoms and coordinates
  - `validate_molecular_data_dict()`: Full data dict
  - `validate_uncertainty_data()`: Uncertainty values
  - `validate_coordinates_3d()`: 3D coordinate arrays
  - `validate_atomic_numbers()`: Atomic number arrays
  - `validate_handler_molecular_batch()`: Batch validation
- **Value Validation**:
  - `is_value_valid_and_not_nan()`: NaN checking
  - `validate_array_shape()`: Array dimensions
  - `validate_numeric_range()`: Value ranges
  - `validate_property_value()`: Property type checking
- **Transform Validation**:
  - `validate_transform_spec()`: Transform specification
  - `validate_experimental_setup()`: Experimental setup
  - `validate_transformation_config()`: Complete config (supports standard_transforms)
  - `validate_transform_composition_rules()`: Ordering rules
- **Handler Validation**:
  - `validate_handler_compatibility()`: Handler-config compatibility
  - `run_handler_validation_tests()`: Comprehensive handler tests
- **Registry Functions**: `_validators_get_dataset_feature()` (registry-only), `_validators_is_dataset_type_registered()`, `_validators_get_handler_compatibility_checks()` (registry-only), `get_validators_registry_status()`

**data_refining.py**:
- **Core Refinement**:
  - `refine_molecular_data()`: Main refinement function
  - `refine_molecular_vibrations()`: DFT vibrational data (VQM24 compatible)
  - `validate_refined_data_quality()`: Post-refinement validation
  - `apply_dataset_specific_refinement()`: Handler-delegated refinement
- **DMC-Specific**:
  - `detect_dmc_statistical_outliers()`: Outlier detection
  - `calculate_dmc_uncertainty_weights()`: Uncertainty weighting
- **Handler-Based**:
  - `create_refinement_handler()`: Create handler for refinement
  - `refine_molecular_data_with_handler()`: Handler-based refinement
  - `validate_refined_data_with_handler()`: Handler validation
- **VQM24 Compatibility**:
  - Enhanced vibrational mode processing for nested structures
  - `diagnose_vibrational_data_structure()`: Debug complex data
- **Registry Functions**: `_get_dataset_feature()` (registry-only), `_get_dataset_refinement_category()`, `_is_dataset_type_registered()`, `get_refining_registry_status()`

**Supported Dataset Features (Registry Queries)**:
- `uncertainty_handling`: DMC datasets
- `vibrational_analysis`: DFT datasets
- `atomization_energy`: DFT datasets
- `rotational_constants`: DFT datasets
- `frequency_analysis`: DFT datasets
- `orbital_analysis`: Wavefunction datasets
- `homo_lumo_gap`: Wavefunction datasets
- `mo_energies`: Wavefunction datasets

**Exception Integration**:
- `ConfigurationError`: General config errors
- `ValidationError`: Validation failures
- `HandlerConfigurationError`: Handler config errors
- `HandlerCompatibilityError`: Compatibility issues
- `VibrationRefinementError`: Vibrational data errors
- `DataProcessingError`: Data processing failures

---

### 2. Molecules Module (`molecules/`)

**Purpose**: Molecular structure processing, validation, filtering, and PyG data conversion

**Architecture**: Handler-Only

**Components**:

```
molecules/
├── __init__.py                      # Package exports (~250 lines)
├── molecule_converter_core.py       # MoleculeDataConverter class (~3813 lines)
├── mol_conversion_utils.py          # RDKit conversion utilities (~1286 lines)
├── molecule_validator.py            # Validation with registry integration (~1638 lines)
├── mol_structural_features.py       # Feature extraction (~914 lines)
├── molecule_feature_enricher.py     # Feature enrichment (~1771 lines)
├── property_enrichment.py           # Property calculation (~2142 lines)
└── molecule_filters.py              # MoleculeFilter class (~2717 lines)
```

**Architecture Pattern**: Handler-Only (Zero Compatibility Layer)
- Handlers are NEVER created within this package
- All handler-dependent functions accept handlers as parameters
- Zero backward compatibility mechanisms or fallback patterns
- All modules refactored with registry integration

**Core Classes**:

**MoleculeDataConverter** (`molecule_converter_core.py`):
- Main class for raw molecular data → PyG Data conversion
- Orchestrates entire conversion pipeline
- Full integration with transformation system (TransformationConfig, ExperimentalSetup)
- Handler-only architecture (STEP 3 cleanup complete)
- Registry integration (8 hardcoded location replacements)
- Imports: DatasetConfig, FilterConfig, ProcessingConfig, TransformationConfig, ExperimentalSetup
- Key methods: `convert()`, `get_transform_capabilities()`, `validate_transform_compatibility()`
- Registry methods: `_init_registry()`, `_get_available_dataset_types()`, `_is_dataset_type_registered()`, `_get_dataset_feature()`, `get_registry_integration_status()`

**MoleculeFilter** (`molecule_filters.py`):
- Filtering with handler integration and transform awareness
- Transform compatibility validation with parameter introspection
- Severity-based conflict categorization (low/medium/high)
- Factory function: `create_molecule_filter()`
- Registry integration for dynamic dataset filtering
- Key methods: `apply_filters()`, `validate_configuration()`, `get_registry_integration_status()`

**Key Functions**:

**mol_conversion_utils.py** (Handler-Only):
- `create_rdkit_mol()`: Creates RDKit molecule using handler-determined strategy
  - Strategies: `identifier_coordinate_based` (DFT/DMC), `coordinate_based` (Wavefunction)
  - Handler parameter REQUIRED (raises ValueError if None)
- `create_mol_with_dataset_support()`: Dataset-aware molecule creation
- `mol_to_pyg_data()`: Convert RDKit mol to PyG Data
- `validate_handler_for_conversion()`: Handler validation
- `validate_conversion_prerequisites()`: Prerequisites validation
- `get_handler_conversion_statistics()`: Handler-specific statistics
- `apply_handler_specific_rdkit_processing()`: Handler-specific RDKit processing

**molecule_validator.py**:
- `validate_molecular_structure()`: Validate atoms and coordinates
- `check_dataset_compatibility()`: Dataset-specific validation
- `validate_pyg_data_completeness()`: PyG Data completeness check
- `validate_uncertainty_data()`: Uncertainty validation (raises `DatasetSpecificHandlerError` for datasets with `uncertainty_handling` feature)
- `create_validation_context()`: Comprehensive validation context
- `validate_with_detailed_feedback()`: Detailed validation with diagnostics
- `get_registry_status()`: Registry integration diagnostics
- Registry functions: `_init_registry()`, `_get_available_dataset_types()`, `_is_dataset_type_registered()`, `_get_dataset_feature()`
- `_create_handler_specific_error()`: Dynamic error creation via `DatasetSpecificHandlerError` — routes `uncertainty_handling` and `vibrational_analysis` features to dataset-typed errors, falls back to `HandlerValidationError` for unrecognized feature sets
- Phase 6.3: Removed stale `DMCHandlerError`/`DFTHandlerError` imports; all dataset-specific handler errors now use `DatasetSpecificHandlerError(dataset_type=...)` with runtime dataset type from handler

**mol_structural_features.py**:
- `add_structural_features()`: Add atom/bond features to PyG Data
- `get_available_features()`: Returns available atom/bond features
- Atom features: degree, total_degree, hybridization, total_valence, is_aromatic, is_in_ring, partial_charge, mulliken_charge, num_aromatic_bonds, chirality
- Bond features: bond_type, is_conjugated, is_aromatic, is_in_any_ring, stereo, bond_length, bond_length_binned
- milia integration: QM-optimized coordinates, Mulliken charges

**molecule_feature_enricher.py**:
- `estimate_molecular_properties()`: Property estimation
- `get_molecule_identifiers()`: Extract molecular identifiers
- `get_structural_feature_summary()`: Structural feature analysis
- `get_feature_extraction_diagnostics()`: Feature extraction diagnostics
- `analyze_structural_feature_capabilities()`: Capability analysis
- Handler-only functions: `estimate_properties_with_handler()`, `analyze_capabilities_with_handler()`, `create_handler_compatible_fingerprint()`, `validate_feature_extraction_with_handler()`
- `get_registry_integration_status()`: Registry diagnostics
- Registry integration: `_get_dataset_feature()` uses registry-only pattern (no legacy fallback), `_get_dataset_enrichment_category()` routes via feature queries

**property_enrichment.py**:
- `enrich_pyg_data_with_properties()`: Main enrichment function (handler-orchestrated)
- `calculate_atomization_energy()`: Atomization energy calculation
- `add_scalar_graph_targets()`: Add scalar targets to PyG Data
- `add_node_features()`: Add node-level features
- `add_vector_graph_properties()`: Add vector properties
- `add_variable_len_graph_properties()`: Add variable-length properties
- `validate_handler_compatibility()`: Handler compatibility validation
- `get_handler_integration_status()`: Handler integration status
- `get_registry_integration_status()`: Registry diagnostics

**molecule_filters.py**:
- `create_molecule_filter()`: Factory function with validation and introspection
- `get_default_molecule_filter()`: Default filter singleton
- `apply_pre_filters()`: Apply all pre-conversion filters
- `apply_atom_count_filters()`: Min/max atom count filtering
- `apply_heavy_atom_filters()`: Element inclusion/exclusion
- `apply_dataset_specific_filters()`: Handler-delegated filtering
- `validate_filter_configuration()`: Filter config validation
- `validate_filter_compatibility_with_transforms()`: Transform compatibility
- `introspect_transform_filter_parameters()`: Parameter introspection

**Module Exports** (`__all__`, 34 items):
- Core classes: `MoleculeDataConverter`, `MoleculeFilter`
- Conversion: `create_rdkit_mol`, `mol_to_pyg_data`, `create_mol_with_dataset_support`
- Features: `add_structural_features`, `get_available_features`
- Validation: `validate_molecular_structure`, `check_dataset_compatibility`, `validate_pyg_data_completeness`
- Enrichment: `enrich_pyg_data_with_properties`, `calculate_atomization_energy`, `estimate_molecular_properties`, `get_molecule_identifiers`, `get_structural_feature_summary`, `get_feature_extraction_diagnostics`, `analyze_structural_feature_capabilities`
- Handler-only: `estimate_properties_with_handler`, `analyze_capabilities_with_handler`, `create_handler_compatible_fingerprint`, `validate_feature_extraction_with_handler`
- Filtering: `create_molecule_filter`, `get_default_molecule_filter`, `apply_pre_filters`, `apply_atom_count_filters`, `apply_heavy_atom_filters`, `apply_dataset_specific_filters`, `validate_filter_configuration`, `validate_filter_compatibility_with_transforms`, `introspect_transform_filter_parameters`, `create_handler_aware_filter_stats`
- Diagnostics: `get_registry_integration_status`, `get_enricher_registry_status`, `get_validator_registry_status`, `get_filter_registry_status`

**Registry Integration**:
All modules include registry integration infrastructure:
- Lazy initialization via `_init_registry()` to avoid circular imports
- Feature-based processing via `_get_dataset_feature(dataset_type, feature_name)` — registry-only, no legacy fallback
- Dynamic dataset discovery via `_get_available_dataset_types()` — exclusion list: `['BASE', 'REGISTRY', 'UTILS', 'COMMON', 'PROTOCOLS']`
- Dataset validation via `_is_dataset_type_registered(dataset_type)`
- Registry status reporting via `get_registry_integration_status()`

**Supported Features for Registry Queries**:
- `uncertainty_handling`: DMC datasets
- `vibrational_analysis`: DFT datasets
- `atomization_energy`: DFT datasets
- `rotational_constants`: DFT datasets
- `frequency_analysis`: DFT datasets
- `orbital_analysis`: Wavefunction datasets
- `homo_lumo_gap`: Wavefunction datasets
- `mo_energies`: Wavefunction datasets

**Exception Integration**:
- `MoleculeProcessingError`: General processing errors
- `RDKitConversionError`: RDKit conversion failures
- `PyGDataCreationError`: PyG Data creation errors
- `PropertyEnrichmentError`: Property enrichment errors
- `StructuralFeatureError`: Feature extraction errors
- `MoleculeFilterRejectedError`: Filter rejection errors
- `HandlerError`, `HandlerOperationError`, `HandlerValidationError`: Handler-specific errors
- `DatasetSpecificHandlerError`: Dynamic dataset-typed handler errors (used by `molecule_validator.py` for uncertainty/vibrational validation)
- `TransformConfigurationError`, `TransformValidationError`: Transform errors

---

### 3. Transformations Module (`transformations/`)

**Purpose**: Production-ready graph transformation system for molecular graph data processing with research-grade features

**Version**: 1.0.0 | **Architecture**: Multi-layered (Discovery → Registry → Validation → Composition → Integration → Recovery) | **PyG**: 2.1+

**Components**:

```
transformations/
├── __init__.py                  # Package exports (~686 lines, circular dependency resolution)
├── graph_transforms.py          # Core transform system (~9192 lines)
├── custom_transforms.py         # Custom transform base classes (~2640 lines)
├── plugin_system.py             # Plugin discovery and loading (~2045 lines)
└── research_api.py              # Research experiment API (~1359 lines)
```

**__init__.py** (~686 lines):
- Circular dependency resolution with `_INITIALIZING`, `_INITIALIZED` flags
- `_ensure_initialized()`: Thread-safe cross-module dependency resolution
- Feature availability flags: `GRAPH_TRANSFORMS_AVAILABLE`, `CUSTOM_TRANSFORMS_AVAILABLE`, `PLUGIN_SYSTEM_AVAILABLE`, `RESEARCH_API_AVAILABLE`
- Module-level convenience functions:
  - `get_available_transforms() -> List[str]`
  - `create_transform_sequence(configs, dataset_type) -> Compose`
  - `validate_transform_config(configs, dataset_type, validation_level) -> Dict`
  - `register_custom_transform(transform_class, force) -> bool`
  - `discover_and_register_plugins(plugin_paths) -> Dict`
  - `get_system_status() -> Dict`
  - `get_module_info() -> Dict`
- 70+ exports in `__all__`

**graph_transforms.py** (~9192 lines):
- **Architecture**: 7-layer system for production-ready transforms
- **Dynamic Dataset Discovery**: `_discover_available_dataset_types()`, `_is_molecular_dataset_type()` (registry-first, filesystem fallback)

**Enums**:
- `ValidationLevel`: STRICT, STANDARD, LENIENT
- `ValidationScope`: BASIC, PARAMETER, SEMANTIC, DATASET_SPECIFIC, PRODUCTION
- `ValidationSeverity`: CRITICAL, ERROR, WARNING, INFO

**Metadata Classes** (dataclasses):
- `TransformCompatibility`: Version compatibility (min_version, max_version, deprecated_in, removed_in, replacement)
- `TransformDependency`: Inter-transform dependencies (depends_on, conflicts_with, recommended_before, recommended_after, required_graph_attributes, modifies_attributes)
- `TransformInfo`: Enhanced metadata (name, transform_class, category, description, complexity_score, dependencies, compatibility, research_applicability)
- `ParameterConstraint`: Constraint validation (type, description, constraint_value, inferred, confidence)
- `ParameterMetadata`: Parameter introspection (name, type_hint, default_value, required, description, constraints)
- `ValidationIssue`: Issue tracking (severity, category, message, location, suggestion, auto_fixable)
- `ValidationContext`: Validation state (level, scope, dataset_type, strict_mode, issues, fixes_applied)

**Edge-Attr Aware System**:
- `EdgeAttrAwareTransformConfig`: Configuration for edge_attr-aware transforms
- `EdgeAttrAwareParameterInjector`: Runtime parameter injection
- `EDGE_ATTR_AWARE_TRANSFORMS`: Registry (AddSelfLoops, AddRemainingSelfLoops)
- Problem: AddSelfLoops adds edges but doesn't handle edge_attr → shape mismatch
- Solution: Auto-inject `attr='edge_attr'`, `fill_value=0.0` when data has edge_attr

**DynamicTransformDiscovery**:
- Scans PyG modules for available transforms
- Methods:
  - `discover_transforms() -> Dict[str, Type]`
  - `get_discovery_metadata(name) -> Dict`
  - `get_compatibility_info(name) -> TransformCompatibility`
  - `is_transform_available(name, version) -> bool`
  - `get_all_discovered_transforms() -> List[str]`
- Version compatibility database for PyG 2.1+

**TransformRegistry**:
- Central repository of transform metadata
- Thread-safe with usage tracking
- Constructor initializes with dynamic discovery
- Key methods:
  - `register_custom(name, transform_class, metadata)`
  - `get_transform_info(name) -> TransformInfo`
  - `list_available_transforms() -> List[str]`
  - `list_custom_transforms(category, author) -> List[str]`
  - `is_custom_transform(name) -> bool`
  - `get_custom_metadata(name) -> TransformMetadata`
- 30+ pre-registered PyG transforms with comprehensive metadata

**TransformValidator**:
- Parameter validation with type checking and range constraints
- Methods:
  - `validate_parameters(name, kwargs) -> ValidationContext`
  - `validate_basic(configs) -> ValidationContext`
  - `validate_comprehensive(configs, dataset_type, level, scope) -> Dict`

**SemanticValidator**:
- Context-aware validation beyond syntax
- Semantic rules: ordering_dependencies, data_flow_integrity, resource_requirements, semantic_conflicts, transformation_completeness
- Anti-patterns: destructive_before_essential, redundant_normalization, excessive_augmentation, incompatible_spatial_transforms
- Best practices: structural_first, normalization_after_features, spatial_before_global

**DatasetAwareValidator**:
- Dataset-specific validation (DFT, DMC, Wavefunction)
- DFT checks: precision, determinism
- DMC checks: uncertainty preservation, sampling compatibility
- Wavefunction checks: orbital preservation, precision

**ValidationReporter**:
- Report generation in text, JSON, markdown formats
- Methods:
  - `generate_report(context) -> Dict`
  - `format_report(report, format_type) -> str`

**TransformComposer**:
- Transform sequence creation with caching
- Methods:
  - `compose_transforms(configs, sample_data) -> Compose`
  - `get_composition_stats() -> Dict`
- Intelligent caching with `IntelligentCacheManager`
- Edge-attr aware injection when sample_data provided

**ConfigurationBridge**:
- milia dataset-specific recommendations
- Configuration migration: v1 (list) → v2 (dict) → v3 (research-grade)
- Methods:
  - `load_transforms_from_config(experimental_setup) -> List`
  - `get_research_recommendations(research_type, dataset_type) -> Dict`
  - `validate_configuration(config, dataset_type) -> Dict`

**TransformErrorRecovery**:
- Multi-level error recovery strategies
- Automatic fallback to safe configurations

**ProductionMetricsCollector**:
- Prometheus/DataDog metrics export
- Methods:
  - `export_metrics(format_type) -> str`
  - `record_usage(transform_name, duration)`

**IntelligentCacheManager**:
- Memory-aware caching with pressure management
- Constructor: `IntelligentCacheManager(max_memory_mb, max_age_seconds, logger)`
- Methods:
  - `get(key) -> Optional[Any]`
  - `set(key, value)`
  - `clear()`
  - `get_stats() -> Dict`

**GraphTransforms** (Main Public API):
- Singleton accessor via `get_graph_transforms()`
- Key methods:
  - `list_available_transforms() -> List[str]`
  - `get_transform_info(name) -> TransformInfo`
  - `create_transform_sequence(configs, dataset_type, sample_data) -> Compose`
  - `validate_config_comprehensive(configs, dataset_type, level, scope) -> Dict`
  - `get_validation_report(configs, dataset_type, format_type) -> str`
  - `get_research_recommendations(research_type, dataset_type) -> Dict`
  - `get_milia_experimental_setups(research_focus) -> Dict`
  - `export_metrics(format_type) -> str`
  - `perform_health_check() -> Dict`
  - `optimize_performance(target_cache_hit_rate) -> Dict`
  - `register_custom_transform(transform_class, force) -> bool`
  - `set_sample_data_for_transforms(sample_data)`

**Module-Level Functions**:
- `get_graph_transforms() -> GraphTransforms`
- `list_available_transforms() -> List[str]`
- `get_transform_info(name) -> TransformInfo`
- `validate_v3_configuration(config, dataset_type) -> Dict`
- `validate_comprehensive(configs, dataset_type, level) -> Dict`
- `get_configuration_format_help() -> str`
- `export_metrics(format_type) -> str`
- `optimize_performance(target_cache_hit_rate) -> Dict`
- `get_milia_setups(research_focus) -> Dict`
- `perform_system_health_check() -> Dict`
- `get_validation_report_text(configs, dataset_type) -> str`
- `discover_custom_transforms() -> Dict[str, type]`
- `register_all_custom_transforms()`
- `set_sample_data_for_edge_attr_detection(sample_data)`
- `get_edge_attr_aware_transform_info() -> Dict`

---

**custom_transforms.py** (~2640 lines):
- Extensible base classes for domain-specific molecular transformations

**TransformMetadata** (dataclass):
- Fields: name, version, author, category, description, paper_reference, github_url, validated_datasets, required_node_features, required_edge_features, required_graph_attributes
- Methods: `to_dict() -> Dict`

**CustomTransformBase** (ABC, extends BaseTransform):
- Abstract methods:
  - `transform(data: Data) -> Data`: Core transformation logic
  - `get_metadata() -> TransformMetadata`: Provide metadata
- Concrete methods:
  - `__call__(data) -> Optional[Data]`: Wrapper with error handling
  - `get_parameter_constraints() -> Dict`: Parameter constraints
  - `get_required_node_attributes() -> Set[str]`
  - `get_required_edge_attributes() -> Set[str]`
  - `get_required_graph_attributes() -> Set[str]`
  - `validate_compatibility(data, validation_level) -> Tuple[bool, List[str]]`
  - `get_usage_statistics() -> Dict`
- Instance tracking: `_call_count`, `_error_count`, `_validation_cache`

**MolecularTransformBase** (extends CustomTransformBase):
- Chemistry-specific validation
- Constants: `VALID_ATOMIC_NUMBERS` (1-118), `VALID_BOND_TYPES` (1-4), `MIN_ATOMS` (1), `MAX_ATOMS` (10000)
- Methods:
  - `validate_molecular_structure(data) -> Tuple[bool, List[str]]`: Validates atomic numbers, bond types, atom count, coordinates

**QuantumTransformBase** (extends MolecularTransformBase):
- milia quantum properties handling
- Methods:
  - `validate_quantum_properties(data) -> Tuple[bool, List[str]]`: Validates energy, dmc_uncertainty, dmc_energy, charges, vibmodes, forces

**Example Transforms**:
- `NormalizeVibrationalModes`: Normalize milia vibmodes [n_modes, n_atoms, 3]
- `FilterByDMCUncertainty`: Filter molecules by DMC uncertainty threshold
- `ScaleMullikenCharges`: Scale Mulliken atomic charges

**Target Transforms**:
- `StandardizeTargets`: Standardize graph-level targets (z-score)
- `NormalizeTargets`: Normalize targets to [0, 1] range
- `DiscretizeTargets`: Discretize continuous targets for classification
  - Static methods: `compute_class_weights(dataset, attr, method)`, `get_num_classes(data_or_dataset, attr)`

---

**plugin_system.py** (~2045 lines):
- Secure, extensible plugin system for custom transforms

**TransformDeclaration** (dataclass):
- Fields: name, class_name, module_path, category, description, version, required_node_features, required_edge_features, required_graph_attributes, parameter_constraints
- Methods: `to_dict()`, `from_dict()`

**PluginMetadata** (dataclass):
- Identity: plugin_name, version, author, plugin_type ('pyg_fallback' or 'user_experimental')
- Version deps: milia_version, pyg_version, python_version, dependencies
- Declarations vs Registrations:
  - `transform_declarations`: List[TransformDeclaration] (from plugin.yaml)
  - `registered_transforms`: Set[str] (runtime state)
- Properties: `declared_count`, `registered_count`, `missing_implementations`, `undeclared_implementations`
- Methods: `to_dict()`, `from_dict()`

**PluginRegistry** (Thread-safe Singleton):
- Pattern: `_instance`, `_lock` for thread safety
- Storage: `_plugins`, `_plugin_paths`, `_enabled_plugins`, `_disabled_plugins`
- Class methods:
  - `add_plugin_path(path)`: Add directory to search
  - `discover_plugins(paths, auto_validate) -> List[str]`: Unified discovery with 3-tier fallback
  - `validate_plugin(plugin_name) -> Dict`: Basic validation
  - `get_plugin(plugin_name) -> PluginMetadata`
  - `list_plugins() -> List[str]`
  - `enable_plugin(plugin_name)`, `disable_plugin(plugin_name)`
- 3-tier fallback for transform registration:
  - Tier 1: PyG Native Implementation
  - Tier 2: Plugin Python Implementation
  - Tier 3: Declaration only (warning)

**PluginValidator**:
- Comprehensive validation (dependencies, security, functional, performance)
- Static methods:
  - `validate_comprehensive(plugin_name, test_data_path) -> Dict`
  - `_check_code_quality(metadata) -> Dict`
  - `_check_documentation(metadata) -> Dict`
  - `_run_functional_tests(metadata, test_data_path) -> Dict`
  - `_benchmark_performance(metadata) -> Dict`
  - `_analyze_security(metadata) -> Dict`
  - `_calculate_score(sections) -> float`
  - `_generate_recommendation(report) -> str`
- Score thresholds: ≥0.95 APPROVED (excellent), ≥0.85 APPROVED (good), ≥0.70 CONDITIONAL, ≥0.50 NOT APPROVED, <0.50 REJECTED

**Exceptions**:
- `PluginError`: Base plugin exception
- `PluginValidationError`: Validation failures
- `PluginSecurityError`: Security concerns
- `PluginDependencyError`: Missing dependencies

---

**research_api.py** (~1359 lines):
- Systematic experimentation framework for reproducible research

**ExperimentConfiguration** (dataclass):
- Fields: name, description, base_transforms (List[TransformSpec]), ablations, parameter_sweeps, paper_reference, hypothesis, expected_outcome, num_runs (default 3), random_seed (default 42), results, metadata
- Methods:
  - `to_dict() -> Dict`
  - `from_dict(data) -> ExperimentConfiguration`
  - `save_to_yaml(path)`
  - `load_from_yaml(path) -> ExperimentConfiguration`
  - `get_total_runs() -> int`
  - `get_variant_names() -> List[str]`

**AblationStudyBuilder** (Fluent API):
- Constructor: `AblationStudyBuilder(study_name)`
- Methods:
  - `with_baseline(transforms) -> self`
  - `remove_transform(transform_name, variant_name) -> self`
  - `keep_only(transform_names, variant_name) -> self`
  - `add_variant(variant_name, additional_transforms, position) -> self`
  - `replace_transform(old_transform, new_transform, variant_name) -> self`
  - `with_metadata(hypothesis, expected_outcome, paper_section) -> self`
  - `build() -> ExperimentConfiguration`

**ParameterSweepBuilder** (Fluent API):
- Constructor: `ParameterSweepBuilder(sweep_name)`
- Methods:
  - `for_transform(transform_name) -> self`
  - `sweep_parameter(param_name, values) -> self`
  - `with_baseline_transforms(transforms) -> self`
  - `with_metadata(hypothesis, expected_outcome) -> self`
  - `build() -> ExperimentConfiguration`

**ComparativeStudyBuilder** (Fluent API):
- Constructor: `ComparativeStudyBuilder(study_name)`
- Methods:
  - `add_approach(name, transforms) -> self`
  - `with_evaluation_metric(metric_name) -> self`
  - `with_metadata(**kwargs) -> self`
  - `build() -> ExperimentConfiguration`

**ExperimentRunner**:
- Constructor: `ExperimentRunner(config, output_dir)`
- Methods:
  - `run() -> Dict`
  - `run_variant(variant_name) -> Dict`
  - `get_results() -> Dict`
  - `save_results(path)`
  - `generate_report() -> str`

**Convenience Functions**:
- `create_ablation_study(study_name, baseline_transforms, transforms_to_ablate, **metadata) -> ExperimentConfiguration`
- `create_parameter_sweep(sweep_name, transform_name, parameter_ranges, baseline_transforms, **metadata) -> ExperimentConfiguration`
- `create_comparative_study(study_name, approaches, evaluation_metrics, **metadata) -> ExperimentConfiguration`
- `load_experiments_from_config(config_path) -> Dict[str, ExperimentConfiguration]`
- `get_experiment(experiment_name, config_path) -> ExperimentConfiguration`
- `list_available_experiments(config_path) -> List[str]`

---

**Exception Integration**:
- `TransformValidationError`, `TransformExecutionError`, `TransformConfigurationError`
- `TransformCompositionError`, `TransformNotFoundError`, `TransformRegistryError`
- `PluginError`, `PluginValidationError`, `PluginSecurityError`, `PluginDependencyError`

---

### 4. Datasets Module (`datasets/`)  + PYDANTIC V2

**Purpose**: PyTorch Geometric dataset implementations with registry-based architecture

**Components**:

```
datasets/
├── __init__.py                      # Module exports
├── milia_dataset.py                 # miliaDataset class (PyG InMemoryDataset)
├── base.py                          # BaseDataset ABC, DatasetMetadata, DatasetSchema, DatasetFeatures ⭐ PYDANTIC V2
├── registry.py                      # DatasetRegistry (thread-safe, testable)
├── protocols.py                     # DatasetHandlerProtocol (11 methods), DatasetConverterProtocol, DatasetValidatorProtocol
└── implementations/                 # Concrete dataset implementations
    ├── __init__.py                  # Implementation exports (dynamic discovery)
    ├── dft.py                       # DFTDataset (@register decorated)
    ├── dmc.py                       # DMCDataset (@register decorated)
    ├── wavefunction.py              # WavefunctionDataset (@register decorated)
    ├── xxmd.py                      # XXMDDataset (@register decorated)
    └── qdpi.py                      # QDPiDataset (@register decorated) 
```

**Key Classes**:
- `miliaDataset`: Main dataset class for molecular data (PyG InMemoryDataset)
- `BaseDataset`: Abstract base class with compile-time validation
- `DatasetMetadata`: Immutable metadata (frozen dataclass)
- `DatasetSchema`: Immutable schema definition (frozen dataclass)
- `DatasetFeatures`: Immutable feature flags (frozen dataclass)
- `DatasetRegistry`: Thread-safe registry (non-singleton, testable)
- `DatasetHandlerProtocol`: 11-method contract (runtime_checkable)
- `DFTDataset`, `DMCDataset`, `WavefunctionDataset`, `XXMDDataset`, `QDPiDataset`: Concrete implementations

---

### 5. Handlers Module (`handlers/`)

**Purpose**: Dataset-specific processing with transform integration

**Version**: 3.0.0 | **Architecture**: Handler Pattern + Registry Integration + Modular Design | **Lazy Loading**: Yes

**Components (Refactored Structure)**:

```
handlers/
├── __init__.py                          # Package exports (~680 lines, lazy loading + recursion guard)
├── base_handler.py                      # DatasetHandler ABC + factory functions (~1,527 lines) 
├── handler_registry.py                  # HandlerRegistry + @register_handler (~326 lines)
├── dataset_handler_integration.py       # Transform-aware integration (~3014 lines)
└── implementations/                     # Individual handler implementations
    ├── __init__.py                      # Dynamic discovery pattern (~78 lines)
    ├── dft.py                           # DFTDatasetHandler (~1,280 lines)
    ├── dmc.py                           # DMCDatasetHandler (~979 lines)
    ├── wavefunction.py                  # WavefunctionDatasetHandler (~1,059 lines)
    ├── qm9.py                           # QM9DatasetHandler (~871 lines)
    ├── ani1x.py                         # ANI1xDatasetHandler (~1,015 lines)
    ├── ani1ccx.py                       # ANI1ccxDatasetHandler (~1,009 lines)
    ├── ani2x.py                         # ANI2xDatasetHandler (~922 lines)
    ├── rmd17.py                         # RMD17DatasetHandler (~947 lines)
    ├── xxmd.py                          # XXMDDatasetHandler (~950 lines)
    └── qdpi.py                          # QDPiDatasetHandler (~950 lines) 
```

**Handler Module Refactoring**:
- Monolithic `dataset_handlers.py` (~9,713 lines) **REMOVED** ✅
- Each handler in separate file under `implementations/`
- Factory functions (`create_dataset_handler`, etc.) migrated to `base_handler.py`
- Dynamic discovery via `implementations/__init__.py` - no manual registration needed
- `@register_handler` decorator for automatic registration
- Thread-safe `HandlerRegistry` with `RLock`
- 100% backward compatibility via lazy loading in `__init__.py`
- Recursion guard in `__init__.py` prevents infinite loops during import

**__init__.py** (Pure Lazy Loading + Recursion Guard):
- Uses `__getattr__()` for lazy imports to resolve circular dependencies
- Import priority: (1) implementations/, (2) base_handler.py, (3) handler_registry.py
- `_DISCOVERING_HANDLERS` flag prevents infinite recursion during `implementations/` import
- Chain: handlers/__init__.py → implementations/ → base_handler.py → config/config_containers.py
- Exports via `__all__`: 44 items organized by category
- Helper functions: `get_available_handlers()`, `get_handler_info(handler_type)`

**base_handler.py** (~1,527 lines):
- **DatasetHandler** (Abstract Base Class)
- **Factory functions** (migrated from dataset_handlers.py):
  - `create_dataset_handler()`: Dynamic handler factory via registry + implementations/
  - `validate_dataset_handler_compatibility()`: Handler-config validation
  - `filter_descriptors_by_handler_support()`: Descriptor filtering helper
  - `verify_handler_abstraction()`: Handler abstraction verification
  - `get_handler_abstraction_summary()`: Comprehensive summary
- Shared utilities: `_ensure_tensor()`, `_is_valid_property()`, `_extract_charge_from_inchi()`
- Lazy registry initialization: `_init_registry()`, `_REGISTRY_AVAILABLE`
- `@handle_transform_errors` decorator for transform error handling

**handler_registry.py** (~326 lines):
- **HandlerRegistry** class with thread-safe registration (RLock)
- `@register_handler` decorator for automatic handler registration
- Methods: `register()`, `get()`, `list_all()`, `is_registered()`, `unregister()`
- Validation: handler must be subclass of DatasetHandler
- `get_default_registry()` function for global registry access

**implementations/__init__.py** (~78 lines):
- Dynamic discovery pattern (same as `datasets/implementations/`)
- Auto-imports all .py files (excluding `__init__.py`, base, registry, utils)
- Finds classes ending with 'DatasetHandler' or 'Handler'
- Triggers `@register_handler` decorators on import
- Builds `__all__` dynamically - no manual updates needed

**DatasetHandler** (Abstract Base Class, in `base_handler.py`):
- Constructor: `__init__(dataset_config, filter_config, processing_config, logger, experimental_setup=None)`
- Abstract methods (12):
  - `get_dataset_type() -> str`
  - `validate_molecule_data(raw_properties_dict, molecule_index, identifier)`
  - `get_required_properties() -> List[str]`
  - `get_identifier_keys() -> List[Tuple[str, str]]`
  - `process_property_value(key, value, molecule_index, identifier) -> Any`
  - `enrich_pyg_data(pyg_data, raw_properties_dict, molecule_index, identifier) -> Data`
  - `get_processing_statistics(processed_molecules) -> Dict`
  - `get_supported_structural_features() -> Dict[str, List[str]]`
  - `get_molecular_charge(raw_properties_dict, atomic_numbers, mol_identifier) -> int`
  - `get_molecule_creation_strategy() -> str` ('identifier_coordinate_based' or 'coordinate_based')
  - `get_transform_recommendations() -> Dict[str, List[str]]`
  - `get_supported_descriptors() -> Dict[str, List[str]]`
- Concrete methods:
  - `validate_configuration()`: Validate handler config
  - `validate_transform_compatibility(transform_sequence, experimental_setup)`: Transform validation
  - `get_experimental_setup_info() -> Dict`: Experimental context
  - `get_common_required_properties() -> List[str]`: From config
  - `_extract_charge_from_inchi(inchi) -> int`: Parse /q layer
- Transform validation helpers (abstract):
  - `_get_dataset_suitable_transforms(available_transforms) -> List[str]`
  - `_validate_dataset_specific_transforms(transform_names) -> List[str]`
  - `_check_transform_incompatibilities(transform_names) -> List[str]`
  - `_get_transform_recommendations(transform_names) -> List[str]`

**DFTDatasetHandler** (in `implementations/dft.py`, ~1,280 lines):
- `get_dataset_type() -> "DFT"`
- `get_molecule_creation_strategy() -> "identifier_coordinate_based"` (InChI parsing)
- `get_identifier_keys() -> [('inchi', 'inchi'), ('graphs', 'smiles')]`
- Supports ALL structural features (has complete quantum data)
- Transform recommendations: GCNNorm, AddSelfLoops, RandomRotate, Distance
- Transform warnings: VirtualNode + Qmulliken, DropNode + vibmodes
- Internal methods:
  - `_add_scalar_targets_internal()`
  - `_add_variable_length_properties_internal()`
  - `_process_vibrational_data_internal()` (VQM24 compatible)
  - `_calculate_atomization_energy_internal()` (Hartree → eV)
  - `_validate_vibrational_data()`: Deferred to refinement phase

**DMCDatasetHandler** (in `implementations/dmc.py`, ~979 lines):
- `get_dataset_type() -> "DMC"`
- `get_molecule_creation_strategy() -> "identifier_coordinate_based"` (InChI parsing)
- `get_identifier_keys() -> [('inchi', 'inchi'), ('graphs', 'smiles')]`
- LIMITED structural features (excludes partial_charge, mulliken_charge, bond_length)
- Uncertainty handling: Validates `std` field, threshold checking
- Transform recommendations: Minimal transforms, avoid heavy augmentation
- Internal methods:
  - `_add_uncertainty_metadata_internal()`
  - `_validate_uncertainty_data()`
  - `_add_scalar_targets_internal()`

**WavefunctionDatasetHandler** (in `implementations/wavefunction.py`, ~1,059 lines):
- `get_dataset_type() -> "Wavefunction"`
- `get_molecule_creation_strategy() -> "coordinate_based"` (rdDetermineBonds)
- `get_identifier_keys() -> [('compounds', 'compound_id')]`
- `get_molecular_charge()`: Calculates from n_electrons - sum(atomic_numbers)
- Supports ALL structural features with wavefunction data
- Validates: mo_energies, mo_occupations, homo_lumo_gap
- Internal methods:
  - `_validate_wavefunction_features()`
  - `_extract_homo_lumo_properties()`

**QM9DatasetHandler** (in `implementations/qm9.py`, ~871 lines):
- `get_dataset_type() -> "QM9"`
- `get_molecule_creation_strategy() -> "identifier_coordinate_based"` (SMILES parsing)
- `get_identifier_keys() -> [('smiles', 'smiles'), ('inchi', 'inchi')]`
- Supports ALL structural features (B3LYP/6-31G(2df,p) geometry)
- 17 QM9 properties: U0, U, H, G, zpve, homo, lumo, gap, mu, alpha, Cv, A, B, C, r2, freqs, Qmulliken
- Internal methods:
  - `_add_scalar_targets_internal()`
  - `_add_variable_length_properties_internal()`
  - `_calculate_atomization_energy_internal()`

**ANI1xDatasetHandler** (in `implementations/ani1x.py`, ~1,015 lines) :
- `get_dataset_type() -> "ANI1x"`
- `get_molecule_creation_strategy() -> "coordinate_based"` (rdDetermineBonds - NO InChI/SMILES available)
- `get_identifier_keys() -> []` (empty - no parseable identifiers in HDF5)
- `get_molecular_charge() -> 0` (all ANI-1x molecules are neutral)
- Supports ALL structural features (ωB97x/6-31G* geometry)
- Properties: energy, forces, hirshfeld_charges, cm5_charges, dipole
- Internal methods:
  - `_add_scalar_targets_internal()`
  - `_add_variable_length_properties_internal()` (forces, charges)
  - `_calculate_atomization_energy_internal()`

**ANI1ccxDatasetHandler** (in `implementations/ani1ccx.py`, ~1,009 lines) :
- `get_dataset_type() -> "ANI1ccx"`
- `get_molecule_creation_strategy() -> "coordinate_based"` (rdDetermineBonds - NO InChI/SMILES available)
- `get_identifier_keys() -> []` (empty - no parseable identifiers in HDF5)
- `get_molecular_charge() -> 0` (all ANI-1ccx molecules are neutral)
- Supports ALL structural features (ωB97x/6-31G* geometry)
- Properties: ccsd_energy (primary), dft_energy, forces, hirshfeld_charges, cm5_charges, dipole
- Internal methods:
  - `_add_scalar_targets_internal()` (handles both ccsd_energy and dft_energy)
  - `_add_variable_length_properties_internal()` (forces, charges)
  - `_calculate_atomization_energy_internal()`
  - `_add_scalar_targets_internal()`
  - `_add_variable_length_properties_internal()` (forces, charges)
  - `_calculate_atomization_energy_internal()`

**ANI2xDatasetHandler** (in `implementations/ani2x.py`, ~922 lines) :
- `get_dataset_type() -> "ANI2x"`
- `get_molecule_creation_strategy() -> "coordinate_based"` (rdDetermineBonds - NO InChI/SMILES available)
- `get_identifier_keys() -> []` (empty - no parseable identifiers in HDF5)
- `get_molecular_charge() -> 0` (all ANI-2x molecules are neutral)
- Supports ALL structural features (ωB97X/6-31G(d) geometry)
- Elements: H, C, N, O, S, F, Cl (7 elements - ~90% of drug-like molecules)
- Properties: energy (Hartree), forces (Hartree/Angstrom)
- Reference: Devereux et al., J. Chem. Theory Comput. 2020, 16, 4192-4202
- Internal methods:
  - `_add_scalar_targets_internal()` (energy)
  - `_add_node_features_internal()` (forces as node features)
  - `_add_variable_length_properties_internal()`
  - `_calculate_atomization_energy_internal()`
  - `get_processing_statistics()` with forces_configured tracking

**RMD17DatasetHandler** (in `implementations/rmd17.py`, ~947 lines) :
- `get_dataset_type() -> "RMD17"`
- `get_molecule_creation_strategy() -> "coordinate_based"` (rdDetermineBonds - NO InChI/SMILES in NPZ)
- `get_identifier_keys() -> []` (empty - uses nuclear_charges + coords for molecule creation)
- `get_molecular_charge() -> 0` (all rMD17 molecules are neutral)
- Supports ALL structural features (PBE/def2-SVP geometry from ORCA)
- Properties: energies (Hartree), forces (Hartree/Angstrom), molecule_name
- 10 molecules: aspirin, azobenzene, benzene, ethanol, malonaldehyde, naphthalene, paracetamol, salicylic, toluene, uracil
- Internal methods:
  - `_add_scalar_targets_internal()` (energies)
  - `_add_variable_length_properties_internal()` (forces)
  - `_calculate_atomization_energy_internal()`
  - `_normalize_dtype()` (handles object array → native dtype conversion)

**XXMDDatasetHandler** (in `implementations/xxmd.py`, ~950 lines) :
- `get_dataset_type() -> "XXMD"`
- `get_molecule_creation_strategy() -> "coordinate_based"` (rdDetermineBonds - NO InChI/SMILES in XYZ)
- `get_identifier_keys() -> []` (empty - uses atomic_numbers + coords for molecule creation)
- `get_molecular_charge() -> 0` (all xxMD molecules are neutral)
- Supports ALL structural features (UKS/def2-SVP geometry from photochemistry simulations)
- Properties: energy (Hartree), forces (Hartree/Angstrom), molecule_name, split
- 4 molecules: azobenzene (azo, 24 atoms), dithiophene (dia, 16 atoms), malonaldehyde (mal, 9 atoms), stilbene (sti, 26 atoms)
- Multi-state DFT (excited states) and ground-state DFT conformations
- Internal methods:
  - `_add_scalar_targets_internal()` (energy)
  - `_add_node_features_internal()` (forces as node features)
  - `_add_variable_length_properties_internal()` (forces)
  - `_calculate_atomization_energy_internal()`
  - `_normalize_dtype()` (handles object array → native dtype conversion)

**QDPiDatasetHandler** (in `implementations/qdpi.py`, ~950 lines) :
- `get_dataset_type() -> "QDPi"`
- `get_molecule_creation_strategy() -> "coordinate_based"` (rdDetermineBonds - NO InChI/SMILES in HDF5)
- `get_identifier_keys() -> []` (empty - uses atomic_numbers + coords for molecule creation)
- `get_molecular_charge()`: Returns charge from NPZ 'molecular_charge' field (supports BOTH neutral and charged molecules)
- Supports ALL structural features (ωB97M-D3(BJ)/def2-TZVPPD geometry)
- Elements: H, Li, C, N, O, F, Na, P, S, Cl, K, Br, I (13 elements - drug-like molecules + ions)
- Properties: energy (Hartree), forces (Hartree/Angstrom), formula, molecular_charge, charge_type, subset
- Source datasets: SPICE, ANI, GEOM, FreeSolv, RE, COMP6 (selected via query-by-committee active learning)
- **CRITICAL DIFFERENCE**: Supports charged molecules (ion pairs, protonated amino acids, deprotonated species)
- Reference: Zeng et al., Scientific Data 12, 693 (2025), DOI: 10.1038/s41597-025-04972-3
- Internal methods:
  - `_add_scalar_targets_internal()` (energy)
  - `_add_node_features_internal()` (forces as node features)
  - `_add_variable_length_properties_internal()` (forces)
  - `_calculate_atomization_energy_internal()`
  - `_normalize_dtype()` (handles object array → native dtype conversion)

**create_dataset_handler()** (Factory Function, in `base_handler.py`):
- Uses `HandlerRegistry` for dynamic handler resolution
- Falls back to legacy `dataset_handlers.py` if registry unavailable
- Signature: `create_dataset_handler(dataset_config, filter_config, processing_config, logger, experimental_setup=None) -> DatasetHandler`
- Raises `HandlerNotAvailableError` for unknown types

**@register_handler** (Decorator, in `handler_registry.py`):
- Registers handler class with global `HandlerRegistry`
- Used by all handler implementations in `implementations/`
- Enables dynamic discovery without manual registration

**@handle_transform_errors** (Decorator, in `base_handler.py`):
- Wraps handler methods for transform error handling
- Converts `TransformConfigurationError`, `TransformValidationError`, `TransformCompositionError` to `TransformHandlerIntegrationError`
- Preserves `HandlerError`, `PropertyEnrichmentError`, `MoleculeProcessingError`

**Verification Functions** (in `base_handler.py`):
- `verify_handler_abstraction() -> Dict`: Comprehensive handler verification
- `get_handler_abstraction_summary() -> Dict`: Handler summary
- `get_registry_status() -> Dict`: Handler registry diagnostics

**dataset_handler_integration.py** (~3014 lines):
- Registry integration for feature-based transform compatibility
- **Lazy Registry**: `_init_registry()`, `_REGISTRY_AVAILABLE`
- **Feature Queries**: `_get_dataset_feature(dataset_type, feature_name, default=False)`
- **Transform Requirements**: `_get_dataset_transform_requirements(dataset_type)`
- **Registry Status**: `get_registry_integration_status() -> Dict`

**TransformAwareHandlerIntegrator** (Main Class):
- Constructor:
  ```python
  TransformAwareHandlerIntegrator(
      dataset_config: DatasetConfig,
      filter_config: FilterConfig,
      processing_config: ProcessingConfig,
      logger: logging.Logger,
      experimental_setup: Optional[str] = None,
      enable_caching: bool = True
  )
  ```
- Components initialized:
  - `handler`: Created via `create_dataset_handler()`
  - `transform_registry`: `TransformRegistry()`
  - `transform_discovery`: `DynamicTransformDiscovery()`
  - `transform_validator`: `TransformValidator(registry, logger)`
  - `semantic_validator`: `SemanticValidator(validator)`
  - `dataset_validator`: `DatasetAwareValidator(semantic_validator)`
  - `transform_composer`: `TransformComposer(registry, validator, logger)`
  - `config_bridge`: `ConfigurationBridge(registry)`
  - `cache_manager`: `IntelligentCacheManager()` (if caching enabled)
- Key methods:
  - `_load_and_validate_experimental_setup(setup_name)`: Load transforms from config
  - `_perform_multi_level_validation() -> Dict`: Basic → Semantic → Dataset-Aware
  - `process_molecule_with_transforms(mol_data) -> Data`: Process with validated transforms
  - `get_composed_transforms()`: Get cached transform composition
- Attributes:
  - `validation_results`: Dict with `basic`, `semantic`, `dataset_aware`, `overall_passed`
  - `transforms`: List of loaded transforms
  - `experimental_setup`: Setup name

**Demonstration Functions**:
- `demonstrate_experimental_setup_workflow()`: Multiple setups
- `demonstrate_multi_level_validation_complete()`: Full validation workflow
- `demonstrate_dynamic_transform_discovery_workflow()`: Transform discovery
- `demonstrate_transform_error_handling()`: Error recovery patterns
- `demonstrate_config_migration_complete()`: v1 → v2 → v3 migration
- `demonstrate_complete_phase2_workflow()`: End-to-end example
- `demonstrate_testing_patterns()`: Test patterns

**Helper Functions**:
- `create_integration_checklist() -> Dict`: Integration checklist
- `generate_benefits() -> Dict`: Generate benefits summary
- `create_performance_guide() -> Dict`: Performance optimization guide
- `generate_quick_reference_guide() -> str`: Quick reference

**Dataset Features (Registry)**:
| Dataset | vibrational_analysis | uncertainty_handling | atomization_energy | orbital_analysis | homo_lumo_gap |
|---------|---------------------|---------------------|-------------------|------------------|---------------|
| DFT | ✓ | ✗ | ✓ | ✗ | ✗ |
| DMC | ✗ | ✓ | ✗ | ✗ | ✗ |
| Wavefunction | ✗ | ✗ | ✗ | ✓ | ✓ |
| QM9 | ✓ | ✗ | ✓ | ✗ | ✓ |
| ANI1x | ✗ | ✗ | ✓ | ✗ | ✗ |
| ANI1ccx | ✗ | ✗ | ✓ | ✗ | ✗ |
| ANI2x | ✗ | ✗ | ✓ | ✗ | ✗ |
| RMD17 | ✗ | ✗ | ✓ | ✗ | ✗ |
| XXMD | ✗ | ✗ | ✓ | ✗ | ✗ |
| QDPi | ✗ | ✗ | ✓ | ✗ | ✗ |

**Molecule Creation Strategies**:
- `identifier_coordinate_based`: Parse InChI/SMILES → map atoms → assign QM coordinates (DFT, DMC, QM9)
- `coordinate_based`: Infer connectivity from 3D positions via rdDetermineBonds (Wavefunction, ANI1x, ANI1ccx, ANI2x, RMD17, XXMD, QDPi)

**Exception Integration**:
- `HandlerError`, `HandlerNotAvailableError`, `HandlerConfigurationError`
- `HandlerOperationError`, `HandlerValidationError`, `HandlerCompatibilityError`
- `HandlerIntegrationError`, `DatasetSpecificHandlerError`
- `TransformHandlerIntegrationError`, `TransformConfigurationError`, `TransformValidationError`

**Key Classes** (Refactored Architecture):
- `DatasetHandler`: Abstract base class (in `base_handler.py`)
- `HandlerRegistry`: Thread-safe handler registry (in `handler_registry.py`)
- `DFTDatasetHandler`: DFT dataset processing (in `implementations/dft.py`)
- `DMCDatasetHandler`: DMC dataset processing (in `implementations/dmc.py`)
- `WavefunctionDatasetHandler`: Wavefunction dataset processing (in `implementations/wavefunction.py`)
- `QM9DatasetHandler`: QM9 dataset processing (in `implementations/qm9.py`)
- `ANI1xDatasetHandler`: ANI-1x dataset processing (in `implementations/ani1x.py`)
- `ANI1ccxDatasetHandler`: ANI-1ccx dataset processing (in `implementations/ani1ccx.py`)
- `ANI2xDatasetHandler`: ANI-2x dataset processing (in `implementations/ani2x.py`)
- `RMD17DatasetHandler`: rMD17 dataset processing (in `implementations/rmd17.py`)
- `XXMDDatasetHandler`: xxMD dataset processing (in `implementations/xxmd.py`)
- `QDPiDatasetHandler`: QDπ dataset processing (in `implementations/qdpi.py`) 
- `TransformAwareHandlerIntegrator`: Transform integration (in `dataset_handler_integration.py`)

---

### 6. Preprocessing Module (`preprocessing/`)

**Purpose**: Modular preprocessing with registry pattern for multiple dataset types

**Version**: 1.5 | **Preprocessors**: Wavefunction, QM9, ANI1x, ANI1ccx, ANI2x, RMD17, XXMD, QDPi | **Registry-Based**: Yes

**Components**:

```
preprocessing/
├── __init__.py                      # Module exports (~500 lines, 9 exports)
├── registry.py                      # PreprocessorRegistry (decorator-based)
├── base_preprocessor.py             # BasePreprocessor ABC
├── preprocessors/                   # Dataset-specific preprocessors
│   ├── __init__.py                  # Dynamic discovery exports (auto-registration)
│   ├── wavefunction.py              # WavefunctionPreprocessor (@register("Wavefunction"))
│   ├── qm9.py                       # QM9Preprocessor (@register("QM9"))
│   ├── ani1x.py                     # ANI1xPreprocessor (@register("ANI1x"))
│   ├── ani1ccx.py                   # ANI1ccxPreprocessor (@register("ANI1ccx"))
│   ├── rmd17.py                     # RMD17Preprocessor (@register("RMD17"))
│   ├── xxmd.py                      # XXMDPreprocessor (@register("XXMD"))
│   └── qdpi.py                      # QDPiPreprocessor (@register("QDPi")) 
└── utils/                           # Shared utility functions
    ├── __init__.py                  # Utility exports (10 functions)
    ├── archive_handlers.py          # Archive extraction (multi-format)
    ├── format_parsers.py            # Molden parsing (feature tiers)
    ├── npz_builders.py              # NPZ construction
    └── qm9_xyz_parser.py            # QM9 XYZ parser
```

**Core Classes**:

**BasePreprocessor** (`base_preprocessor.py`, v1.1):
- Abstract base class for all preprocessors
- Constructor: `__init__(config: Dict[str, Any], logger: logging.Logger)`
- Abstract methods:
  - `_validate_config()`: Validate preprocessor-specific configuration
  - `preprocess() -> Path`: Execute preprocessing logic
- Concrete methods:
  - `run() -> Path`: Execute full pipeline with timing and validation
  - `_validate_output(output_path)`: Validate .npz structure (requires 'compounds', 'metadata')

**PreprocessorRegistry** (`registry.py`, v1.1):
- Class-based registry with decorator pattern
- Class attribute: `_preprocessors: Dict[str, Type[BasePreprocessor]]`
- Class methods:
  - `@register(dataset_type)`: Decorator for auto-registration
  - `get_preprocessor(dataset_type) -> Type[BasePreprocessor]`: Get preprocessor class
  - `list_preprocessors() -> List[str]`: List registered types
  - `supports_preprocessing(dataset_type) -> bool`: Check if supported
  - `clear_registry()`: Clear all registrations (for testing)

**Module Exports** (`__init__.py`, `__all__` with 9 items):
- Core: `BasePreprocessor`, `PreprocessorRegistry`
- Archive: `extract_from_targz`
- Parsers: `parse_molden_files`
- Builders: `build_npz`, `validate_npz_structure`
- Convenience: `get_preprocessing_info`, `list_available_preprocessors`, `supports_dataset`

**Convenience Functions**:
- `get_preprocessing_info() -> dict`: Returns version, registered preprocessors, available utilities, import errors
- `list_available_preprocessors() -> List[str]`: Wrapper for `PreprocessorRegistry.list_preprocessors()`
- `supports_dataset(dataset_type) -> bool`: Wrapper for `PreprocessorRegistry.supports_preprocessing()`

**Preprocessor Classes**:

**WavefunctionPreprocessor** (`preprocessors/wavefunction.py`, v1.1):
- Registered as: `@PreprocessorRegistry.register("Wavefunction")`
- Pipeline: Extract .molden → Parse with IOData → Build .npz → Cleanup
- Config keys: `raw_tar_path`, `output_npz_path`, `num_molecules`, `feature_tier`, `cleanup_temp`
- Feature tiers: 'basic', 'standard', 'complete'
- Auto-skip if output exists

**QM9Preprocessor** (`preprocessors/qm9.py`, v1.0):
- Registered as: `@PreprocessorRegistry.register("QM9")`
- Pipeline: Extract .xyz from tar.bz2 → Parse QM9 XYZ → Build .npz → Cleanup
- Config keys: `raw_archive_path`, `output_npz_path`, `num_molecules`, `cleanup_temp`
- Supports: tar.bz2, tbz2, tar.gz, tgz archives
- Source: Figshare (https://figshare.com/ndownloader/files/3195389)
- Reference: Ramakrishnan et al., Scientific Data 1, 140022 (2014)
- Auto-skip if output exists

**RMD17Preprocessor** (`preprocessors/rmd17.py`, v1.0) :
- Registered as: `@PreprocessorRegistry.register("RMD17")`
- Pipeline: Extract tar.bz2 → Parse 10 molecular NPZ files → Convert kcal/mol → Hartree → Build unified .npz → Cleanup
- Config keys: `raw_archive_path`, `output_npz_path`, `molecules_to_include`, `max_conformers_per_molecule`, `include_old_data`, `cleanup_temp`
- Unit conversion: kcal/mol → Hartree (factor: 0.00159360143764)
- 10 molecules: aspirin, azobenzene, benzene, ethanol, malonaldehyde, naphthalene, paracetamol, salicylic, toluene, uracil (~100k conformers each)
- Source: Materials Cloud Archive (https://archive.materialscloud.org/records/pfffs-fff86/files/rmd17.tar.bz2?download=1)
- Reference: Christensen & von Lilienfeld, Mach. Learn.: Sci. Technol. 1, 045018 (2020)
- Auto-skip if output exists
- NOTE: Structure differs from previous datasets - uses `max_conformers_per_molecule` instead of `num_molecules` (10 conformers × 10 molecules = 100 total for testing)

**XXMDPreprocessor** (`preprocessors/xxmd.py`, v1.0) :
- Registered as: `@PreprocessorRegistry.register("XXMD")`
- Pipeline: Extract ZIP → Extract nested molecule ZIPs → Parse extended XYZ with ASE → Convert eV → Hartree → Build unified .npz → Cleanup
- Config keys: `raw_archive_path`, `output_npz_path`, `molecules_to_include`, `max_conformers_per_molecule`, `include_splits`, `cleanup_temp`
- Unit conversion: eV → Hartree (factor: 0.0367493)
- 4 molecules: azo (azobenzene, 24 atoms), dia (dithiophene, 16 atoms), mal (malonaldehyde, 9 atoms), sti (stilbene, 26 atoms)
- Archive structure: Nested ZIPs (xxMD-main.zip/xxMD-DFT/{mol}/{mol}.zip containing train.xyz, val.xyz, test.xyz)
- XYZ filename pattern: `{mol}_{split}_uks.xyz` (e.g., azo_train_uks.xyz)
- Source: Zenodo (https://zenodo.org/api/records/10393859/files/xxMD-main.zip/content)
- Reference: Axelrod & Gómez-Bombarelli, Scientific Data 9, 185 (2022)
- Auto-skip if output exists
- NOTE: Uses `max_conformers_per_molecule` and `molecules_to_include` similar to rMD17 pattern

**QDPiPreprocessor** (`preprocessors/qdpi.py`, v1.0) :
- Registered as: `@PreprocessorRegistry.register("QDPi")`
- Pipeline: Extract tar.gz → Parse DeePMD-kit HDF5 files → Convert eV → Hartree → Build unified .npz → Cleanup
- Config keys: `raw_archive_path`, `output_npz_path`, `num_molecules`, `property_keys`, `include_charged`, `include_neutral`, `cleanup_temp`
- Unit conversion: eV → Hartree (factor: 0.0367493), eV/Å → Hartree/Å (forces)
- DeePMD-kit HDF5 structure: Groups by formula, each with `type.raw`, `type_map.raw`, `set.XXX/coord.npy`, `set.XXX/energy.npy`, `set.XXX/force.npy`
- 10 HDF5 files: 7 neutral (geom.hdf5, ani.hdf5, spice.hdf5, etc.) + 3 charged (charged_amino_acid.hdf5, charged_ion.hdf5, charged_freesolv.hdf5)
- **CRITICAL**: Tracks molecular_charge from file path (data/neutral/ vs data/charged/)
- 13 elements: H, Li, C, N, O, F, Na, P, S, Cl, K, Br, I (drug-like molecules + ions)
- Source: Zenodo (https://zenodo.org/api/records/14970869/files/QDpiDataset-main.tar.gz/content)
- Reference: Zeng et al., Scientific Data 12, 693 (2025), DOI: 10.1038/s41597-025-04972-3
- Auto-skip if output exists
- NOTE: First preprocessor to support BOTH neutral AND charged molecules via charge_type tracking

**Utility Modules** (`utils/`):

**archive_handlers.py**:
- `extract_from_archive()`: Generic extractor with auto-detection
  - Supports: .tar.gz, .tgz, .tar.bz2, .tbz2, .tar.xz, .txz, .tar
  - Memory-efficient streaming extraction
  - `COMPRESSION_MODES` dict maps extensions to tarfile modes
- `extract_from_targz()`: Backward-compatible wrapper (calls extract_from_archive)
- `get_supported_formats()`: Returns dict of supported archive formats

**format_parsers.py**:
- `parse_molden_files()`: Parse .molden files with IOData
  - Returns: `Tuple[Dict[str, np.ndarray], Dict[str, Any]]`
  - Feature tiers: 'basic' (7 features), 'standard' (12 features), 'complete' (18+ features)
  - Complete tier includes: MO statistics, quantum descriptors, electrophilicity
- `FEATURE_TIERS` dict defines available features per tier

**qm9_xyz_parser.py** :
- `parse_qm9_xyz_files()`: Parse QM9 extended XYZ files
  - Returns: `Tuple[Dict[str, np.ndarray], Dict[str, Any]]`
  - Extracts 15 scalar properties + atoms + coordinates + Mulliken charges + frequencies
  - Handles QM9's `*^` scientific notation (e.g., `1.234*^-5`)
- `get_qm9_property_info()`: Returns property metadata (units, descriptions)
- `QM9_PROPERTY_NAMES`: List of 17 property names in QM9 order
- `ELEMENT_TO_Z`: Element symbol to atomic number mapping

**QM9 Properties Extracted** (from readme.txt):
| Property | Unit | Description |
|----------|------|-------------|
| A, B, C | GHz | Rotational constants |
| mu | Debye | Dipole moment |
| alpha | Bohr³ | Isotropic polarizability |
| homo, lumo | Hartree | Orbital energies |
| gap | Hartree | HOMO-LUMO gap |
| r2 | Bohr² | Electronic spatial extent |
| zpve | Hartree | Zero point vibrational energy |
| U0, U, H, G | Hartree | Thermodynamic energies |
| Cv | cal/(mol·K) | Heat capacity |

**npz_builders.py**:
- `build_npz()`: Create compressed .npz files
  - Required keys: 'compounds', 'atoms', 'coordinates'
  - Adds 'metadata' array with enhanced metadata
- `validate_npz_structure()`: Validate .npz file structure and return summary

---

### 7. Descriptors Module (`descriptors/`)

**Purpose**: Molecular descriptor calculation with 400+ descriptors

**Components**:

```
descriptors/
├── __init__.py                      # Module exports
├── descriptor_registry.py           # DescriptorRegistry (singleton)
├── descriptor_categories.py         # 6 descriptor categories
├── descriptor_calculator.py         # DescriptorCalculator
├── descriptor_validator.py          # Validation and filtering
├── descriptor_integration.py        # PyG integration utilities
└── descriptor_plugin_system.py      # Plugin system
```

**Key Classes**:
- `DescriptorRegistry`: Singleton registry with 400+ descriptors
- `DescriptorCalculator`: Batch calculation with caching
- `DescriptorValidator`: Requirement-based filtering
- `DescriptorPluginLoader`: Plugin management

---

### 8. Models Module (`models/`)

**Purpose**: Complete ML/DL model lifecycle management

**Components**:

```
models/
├── __init__.py                      # Module exports
├── registry/
│   ├── __init__.py
│   ├── model_registry.py            # ModelRegistry v1.1.0 (singleton) 
│   └── pyg_introspector.py          # PyGModelIntrospector v2.0.0 
├── factory/
│   ├── __init__.py
│   ├── model_factory.py             # ModelFactory v1.2.0 
│   └── target_selection_config.py   # TargetSelectionConfig 
├── training/
│   ├── __init__.py                  # Training exports
│   ├── trainer.py                   # Trainer class with checkpoint support 
│   ├── callbacks.py                 # 6 callback types
│   ├── data_splitting.py            # 5 splitting strategies
│   ├── loss_functions.py            # Custom losses
│   ├── optimizers.py                # Optimizer registry
│   └── schedulers.py                # Scheduler registry
├── post_training/                   # Post-Training Inference
│   ├── __init__.py                  # Public API (~350 lines) 
│   ├── checkpoint/
│   │   ├── __init__.py
│   │   └── checkpoint_manager.py    # CheckpointManager v2.0.0 
│   ├── inference/
│   │   ├── __init__.py              # Inference exports (180 lines)
│   │   ├── model_loader.py          # ModelLoader v2.1.0 
│   │   └── predictor.py             # Predictor v2.1.0 
│   ├── data_preparation/
│   │   ├── __init__.py              # Data prep exports (259 lines)
│   │   └── data_converter.py        # DataConverterRegistry v2.1.0 
│   └── transfer_learning/
│       ├── __init__.py              # Transfer exports (168 lines)
│       └── fine_tuner.py            # FineTuner v2.0.0 
├── builders/                        # Custom architecture building
│   ├── __init__.py
│   ├── layer_registry.py            # LayerRegistry
│   ├── architecture_builder.py      # ArchitectureBuilder
│   ├── model_composer.py            # ModelComposer
│   ├── templates.py                 # ArchitectureTemplates (10 templates)
│   ├── config_parser.py             # Configuration parsing
│   └── validation.py                # ArchitectureValidator
├── acceleration/
│   ├── __init__.py
│   ├── device_manager.py            # DeviceType, DeviceInfo, DeviceManager ⭐ PYDANTIC
│   ├── distributed_strategies.py    # DistributedManager (DP, DDP, FSDP, Horovod) ⭐ PYDANTIC
│   ├── memory_optimization.py       # MemoryOptimizer (AMP, checkpointing) ⭐ PYDANTIC
│   └── computation_optimization.py  # ComputationOptimizer
├── deployment/
│   ├── __init__.py
│   ├── deployment_strategies.py     # Edge/Cloud/Federated
│   ├── model_optimization.py        # Quantization/Pruning
│   └── monitoring.py                # Drift/Performance monitoring
├── utils/
│   ├── __init__.py
│   ├── config_bridge.py             # ConfigBridge v1.2.0 (Pydantic) ⭐ PYDANTIC
│   └── pyg_integration.py           # PyG utilities v1.0.0 
└── plugins/
    ├── __init__.py
    └── model_plugin_system.py       # ModelPluginLoader
```

**Key Classes**:
- **Registry**: ModelRegistry  ModelCategory, ModelMetadata, PyGModelIntrospector 
- **Factory**: ModelFactory, ModelValidator, create_model
- **Training**: Trainer with checkpoint support, DataSplitter, callbacks (6 types)
- **Post-Training**: CheckpointManager, ModelLoader, Predictor, DataConverterRegistry, FineTuner
- **Builders**: ArchitectureBuilder, ModelComposer, LayerRegistry
- **Acceleration**: DeviceManager, DistributedTrainer, memory/computation optimization
- **Deployment**: EdgeDeployer, CloudDeployer, quantization, pruning, monitoring
- **Utils**: ConfigBridge v1.2.0 (31 Pydantic BaseModel classes), PyGIntegration ⭐ PYDANTIC
- **Plugins**: ModelPluginLoader, plugin discovery/validation

---

### 9. Plugins Directory (`plugins/`)

**Purpose**: Plugin storage and examples

**Structure**:

```
plugins/
├── descriptors/
│   ├── example_descriptors/         # Example descriptor plugin
│   │   ├── plugin.yaml
│   │   ├── descriptors.py
│   │   └── README.md
│   ├── user_template/               # User template
│   │   ├── plugin.yaml.template
│   │   ├── descriptors.py.template
│   │   └── README.md
│   └── README.md
├── pyg_augmentation/                # PyG augmentation plugin
│   ├── plugin.yaml
│   ├── transforms.py
│   └── __init__.py
└── myplugins/                       # Custom plugin example
    ├── plugin.yaml
    ├── transforms/
    └── examples/
```

---

## Testing Infrastructure

### Test Suite Organization

```
tests/
├── Unit Tests (Component-level)
│   ├── test_config_*.py             # Config module tests (7 files)
│   ├── test_molecule_*.py           # Molecules module tests (6 files)
│   ├── test_descriptor_*.py         # Descriptors module tests (10 files)
│   ├── test_transforms_*.py         # Transformations tests (3 files)
│   ├── test_dataset_*.py            # Datasets module tests (13 files)
│   ├── test_handler_*.py            # Handlers module tests (12 files)
│   ├── test_preprocessor_*.py       # Preprocessing tests (11 files)
│   ├── test_model_*.py              # Models module tests (32 files)
│   ├── test_hpo_*.py                # HPO module tests (14 files)
│   ├── test_plugin_system_unit.py   # Plugin system tests
│   └── test_exceptions_unit.py      # Exception handling tests
│
├── Integration Tests (Cross-component)
│   ├── test_cli_preprocessing_integration.py
│   ├── test_descriptor_integration_unit.py
│   ├── test_descriptor_plugins_integration.py
│   ├── test_hpo_integration.py
│   ├── test_main_hpo_manager_integration.py
│   ├── test_model_composer_integration.py
│   ├── test_models_module_integration.py
│   ├── test_pyg_integration_unit.py
│   └── test_training_integration.py
│
├── Performance Tests
│   └── test_descriptor_performance.py
│
├── Standalone / Root-Level Tests
│   ├── test_main_unit.py                    # Main entry point tests
│   ├── test_cli_manager_unit.py             # CLI manager tests
│   ├── test_logging_config_unit.py          # Logging config tests
│   ├── test_milia_dataset_unit.py           # Milia dataset tests
│   ├── test_npz_builders_unit.py            # NPZ builders tests
│   ├── test_qm9_xyz_parser_unit.py          # QM9 XYZ parser tests
│   ├── test_format_parsers_unit.py          # Format parsers tests
│   ├── test_archive_handlers_unit.py        # Archive handlers tests
│   ├── test_property_enrichment_unit.py     # Property enrichment tests
│   ├── test_research_api_unit.py            # Research API tests
│   ├── test_templates_unit.py               # Templates tests
│   └── test_validators_unit.py              # Validators tests
│
└── Validation Scripts
    ├── VQM24_npz_chk.py
    ├── verification_wavfunc_npz_features.py
    └── test_IOData.py
```

### Test Categories

**127 Test Files** covering:

1. **Unit Tests** (103 files):
   - Configuration management (7 files)
   - Molecular processing (6 files)
   - Descriptors system (10 files)
   - Transformations (3 files)
   - Datasets (13 files)
   - Handlers (12 files)
   - Preprocessing (11 files)
   - **Models module (32 files)**:
     - Registry tests
     - Factory tests
     - Training tests
     - Builders tests
     - Acceleration tests
     - Deployment tests
     - Utils tests
     - Plugin tests
     - Prediction tests
     - Visualization tests
   - **HPO module (14 files)**:
     - Config tests
     - Manager tests
     - Backend tests (base, Optuna)
     - NAS tests (search space, manager)
     - Search space tests (builder, param types)
     - Transfer learning tests (meta features, transfer manager, warm start)
     - Callback tests
     - Analysis tests (study analyzer)
   - Plugin system (1 file)
   - Standalone/root-level tests (12 files)

2. **Integration Tests** (9 files):
   - CLI preprocessing integration
   - Descriptor integration
   - Descriptor plugins integration
   - HPO integration
   - Main HPO manager integration
   - Model composer integration
   - Models module integration
   - PyG integration
   - Training integration

3. **Performance Tests** (1 file):
   - Descriptor calculation performance

4. **Validation Scripts** (3 files):
   - NPZ file validation
   - Wavefunction feature verification
   - IOData testing

---

## Documentation

### Documentation Structure

```
docs/
├── INDEX.md                                        # Documentation index
├── README.md                                       # Main documentation
├── ARCHITECTURE_DIAGRAMS.md                        # System architecture
├── DELIVERABLES_SUMMARY.md                         # Project deliverables
├── MODELS_MODULE_LIFECYCLE_GUIDE.md                # Models module guide
│
├── Plugin System Documentation
│   ├── Plugin_system_KEY_INFO.md                   # Plugin system overview
│   ├── Plugin_Configuration_Quick_Reference.md     # Quick config reference
│   ├── Plugin_Configuration_Schema_Technical_Reference.md
│   ├── Plugin_Development_Checklist.md             # Development guide
│   ├── Plugin_Distribution_Quick_Reference.md      # Distribution guide
│   ├── Plugin_Exceptions_Quick_Reference.md        # Error handling
│   └── Plugin_Implementation_Summary.md            # Implementation details
│
├── Module Documentation
│   ├── CUSTOM_TRANSFORMS_QUICK_REFERENCE.md        # Transform system
│   └── DESCRIPTOR_REFERENCE.md                     # Descriptor catalog
│
└── Examples and Templates
    └── examples/preprocessing/                     # Preprocessing examples
```

### Key Documentation

1. **Architecture Documentation**:
   - System design and patterns
   - Module interactions
   - Data flow diagrams

2. **Plugin System Documentation**:
   - Comprehensive plugin guides
   - Configuration schemas
   - Development checklists
   - Distribution guidelines

3. **API Reference**:
   - Descriptor catalog (400+ descriptors)
   - Custom transforms guide
   - Model catalog (dynamic discovery of ALL PyG models) 
   - Module-specific references

4. **Examples**:
   - Preprocessing workflows
   - Plugin development examples
   - Configuration templates
   - Model training examples

---

## Scripts and Utilities

### Utility Scripts

```
scripts/
├── VQM24_npz_chk.py                # NPZ file validation
├── VQM24_npz_slicing.py            # NPZ file slicing
├── generate_descriptor_docs.py     # Auto-generate descriptor docs
├── generate_pyg_models_docs.py     # Auto-generate model docs
└── analyze_imports.py              # Import analysis
```

### Examples

```
examples/
└── preprocessing/
    ├── README.md
    ├── quick_preprocess.yaml        # Quick preprocessing example
    └── wavefunction_preprocess.yaml # Wavefunction preprocessing
```

---

## Entry Points and CLI

### Main Entry Point (`main.py`)

**Purpose**: Primary execution entry and orchestration layer for the pipeline

**Lines**: ~5,280 | **Processing Modes**: 12+

**Capabilities**:
- Configuration loading and validation
- Dataset initialization with handler pattern
- Handler integration (DFT, DMC, Wavefunction, extensible via registry)
- Transform application with experimental setups
- Model training (standard and HPO modes)
- **Post-training prediction**
- **Checkpoint featurization config persistence**
- Preprocessing workflows
- Plugin system initialization
- Research API operations
- Descriptor operations

**Post-Training Integration**:
- `handle_predict_mode()`: Complete prediction workflow (~180 lines) 
- Retrieves `structural_features_config` from `predictor.structural_features_config`
- Passes `structural_features_config` to `convert_to_pyg()` for dimension-compatible featurization
- Logs featurization config for debugging: `atom=[...], bond=[...]`
- Warns if no config in checkpoint (potential dimension mismatch)
- `_detect_predict_input_type()`: Auto-detect dataset vs molecular file
- `_load_from_dataset_pt()`: Load from PyG InMemoryDataset .pt files
- `_load_from_dataset_dir()`: Load from miliaDataset directories
- `_load_from_molecular_file()`: Load via DataConverterRegistry
- `_save_predictions()`: Multi-format output (CSV, JSON, NPY, PT)
- POST_TRAINING conditional imports with availability flag
- Early exit pattern (no config.yaml needed for prediction)

**Registry Integration**:
- Dynamic dataset type discovery via `_get_available_dataset_types()`
- Feature-based validation via `_get_dataset_feature()`
- Schema attribute lookup via `_get_dataset_schema_attribute()`
- Config key resolution via `_get_dataset_config_key()`
- Lazy initialization to avoid circular imports
- Legacy fallback for backward compatibility
- Diagnostics via `get_main_registry_status()`

**Processing Modes**:
1. `--validate-config`: Configuration validation only
2. `--validate-transforms-only`: Transform system validation
3. `--test-handlers-only`: Handler creation testing
4. `--list-experimental-setups`: Show available setups
5. `--list-transforms`: Show available transforms
6. `--quick-validation`: Load existing data without reprocessing
7. `--stats-only`: Generate statistics from existing data
8. `--dry-run`: Validate configuration without processing
9. `--train`: Execute training workflow (standard or HPO)
10. `--evaluate-only`: Skip training, only run evaluation
11. `--predict`: Execute post-training inference workflow
12. Default: Full dataset processing workflow

**Training System Integration** :
- `handle_training_mode()`: Main training workflow entry point
- Captures `dataset.structural_features_config` into `model_info` before training
- Logs captured featurization config: `atom_features=[...], bond_features=[...]`
- `_run_standard_training()`: Standard (non-HPO) training execution
- `_run_hpo_training()`: HPO optimization workflow
- `prepare_data_for_task()`: Task-specific data preparation via TaskDataPreparer
- `_get_loss_function()`: Dynamic loss function selection via LossRegistry
- `_get_optimizer()`: Dynamic optimizer creation via OptimizerRegistry
- `_get_scheduler()`: Dynamic scheduler creation via SchedulerRegistry
- `_create_callbacks()`: Callback factory integration
- `_save_training_results()`: Training results persistence
- `_save_hpo_results()`: HPO study results persistence

**System Initialization Flow**:
1. CLI Parsing (CLIManager) → Parse and validate arguments
2. Logging Setup → Initialize comprehensive logging system
3. Custom Transform Registration → Auto-discover custom transforms
4. Plugin System Initialization (8-step) → Discover and validate plugins
5. Configuration Loading → Load config (single-file `config.yaml` or split-file `configs/`) and merge CLI overrides
6. Configuration Validation → Validate handlers, transforms, and datasets
7. Processing Mode Execution → Execute requested operation
8. Training Mode (if enabled) → Execute training/HPO workflow

**Error Handling Strategy**:
- Handler errors: Categorized as recoverable/non-recoverable with suggestions
- Transform errors: Graceful degradation to legacy transforms
- Plugin errors: Non-blocking, system continues without problematic plugins
- Configuration errors: Fail-fast with actionable error messages
- Data processing errors: Chunk-level recovery with partial results
- Model/HPO errors: Detailed error context with recovery suggestions
- Training errors: Checkpoint recovery support

### CLI Manager (`cli_manager.py`)

**Purpose**: Command-line interface for common operations

**Lines**: ~3,601 | **Argument Groups**: 12

**Commands**:
- Dataset preprocessing
- Configuration validation
- Plugin management
- Descriptor calculation
- Model training and evaluation
- Transform experiments
- **Post-training prediction**

**Prediction System Arguments**:
- `--predict`: Enable prediction mode (master switch)
- `--model-path`: Path to trained model checkpoint
- `--test-path`: Path to input data (molecules or dataset)
- `--preds-path`: Path for output predictions
- `--predict-batch-size`: Batch size for inference (default: 32)
- `--predict-device`: Device selection (auto/cpu/cuda/mps)
- `--predict-format`: Input format override (auto-detected by default)
- `--predict-split`: Dataset split selection (train/val/test/all)
- `--predict-num-samples`: Limit number of samples for prediction
- `--predict-output-format`: Output format (csv/json/npy/pt)
- `--predict-include-inputs`: Include input features in output
- `--predict-uncertainty`: Enable uncertainty estimation

---

## Configuration System

### Configuration Files

**Configuration Modes** (YAML Splitting Architecture):
- **Single-file mode**: `config.yaml` (backward compatible, ~2900 lines)
- **Split-file mode**: `configs/` directory (modular, recommended for large projects)

**Primary Configuration** (`config.yaml` or `configs/`):
- Dataset configuration
- Filter settings
- Processing options
- Feature extraction
- Transform pipelines
- **Model training configuration**
- **Acceleration settings**
- **Deployment configuration**

**Split Configuration Directory** (`configs/`):
```
configs/
├── main.yaml                    # Global settings: dataset_type, global_paths, constants, data_config.common_settings
├── descriptors.yaml             # Molecular descriptors configuration
├── filter_config.yaml           # Filter settings
├── models.yaml                  # Model configurations
├── plugins.yaml                 # Plugin system configuration
├── structural_features.yaml     # Structural features configuration
├── transformations.yaml         # PyG transformations
└── datasets/                    # Dataset-specific configs (FULLY COLOCATED)
    ├── dft.yaml                 # dft_config + data_config.property_selection.DFT + property_availability.DFT
    ├── dmc.yaml                 # dmc_config + data_config.property_selection.DMC + property_availability.DMC
    ├── wavefunction.yaml        # wavefunction_config + data_config.property_selection.Wavefunction + property_availability.Wavefunction
    ├── qm9.yaml                 # qm9_config + data_config.property_selection.QM9 + property_availability.QM9
    ├── ani1x.yaml               # ani1x_config + data_config.property_selection.ANI1X + property_availability.ANI1X
    ├── ani1ccx.yaml             # ani1ccx_config + data_config.property_selection.ANI1CCX + property_availability.ANI1CCX
    ├── rmd17.yaml               # rmd17_config + data_config.property_selection.RMD17 + property_availability.RMD17
    ├── ani2x.yaml               # ani2x_config + data_config.property_selection.ANI2X + property_availability.ANI2X
    ├── xxmd.yaml                # xxmd_config + data_config.property_selection.XXMD + property_availability.XXMD
    └── qdpi.yaml                # qdpi_config + data_config.property_selection.QDPi + property_availability.QDPi
```

**YAML Splitting Merge Order**:
1. `main.yaml` (loaded first - contains `data_config.common_settings`)
2. Root-level `*.yaml` files (alphabetical order)
3. `datasets/*.yaml` files (alphabetical order - contain `data_config.property_selection.{DATASET}`)

**Deep Merge Strategy**: Nested dicts are recursively merged; lists and scalars are overridden.

> **NOTE**: The `data_config` key appears in both `main.yaml` (with `common_settings`) and `datasets/*.yaml` 
> (with `property_selection.{DATASET}`). The deep merge combines these into a single `data_config` dictionary 
> containing both `common_settings` and `property_selection` sub-keys.

**Research Configuration** (`research_experiments.yaml`):
- Experiment definitions
- Transform ablation studies
- Hyperparameter sweeps
- Model comparison studies

### Configuration Structure

**Single-file mode** (`config.yaml`):
```yaml
# config.yaml structure (monolithic, backward compatible)

dataset_type: "DFT"  # Options: "DFT", "DMC", "Wavefunction", "QM9", "ANI1X", "ANI1CCX", "RMD17", "ANI2X", "XXMD", "QDPi"

global_paths:
  working_root_dir: ~/Chem_Data/Milia_PyG_Dataset

# All configuration in single file...
# ... dataset settings, filter settings, processing settings, etc.
```

**Split-file mode** (`configs/`):
```yaml
# configs/main.yaml (global settings only)

dataset_type: "DFT"  # Options: "DFT", "DMC", "Wavefunction", "QM9", "ANI1X", "ANI1CCX", "RMD17", "ANI2X", "XXMD", "QDPi"

global_paths:
  working_root_dir: ~/Chem_Data/Milia_PyG_Dataset

global_constants:
  har2ev: 27.211386245988
  bohr_to_angstrom: 0.529177210903
```

```yaml
# configs/datasets/dft.yaml (dataset config with colocated property_availability)

dft_config:
  raw_npz_filename: DFT_all_sliced.npz
  raw_data_download_url: https://zenodo.org/records/15442257/files/DFT_all.npz?download=1

property_availability:
  DFT:
    molecular_identifiers: [compounds, inchi, graphs, frags]
    atomic_structure: [atoms, coordinates]
    scalar_graph_targets: [Etot, U0, U298, zpves, gap, ...]
    node_features: [Qmulliken, Vesp]
    # ... other properties
```

**CLI Usage**:
```bash
# Use split config (recommended for large projects)
python main.py --config configs/

# Use single file (backward compatible)
python main.py --config config.yaml

# Auto-detection (config.yaml takes priority if exists)
python main.py
```

**Detailed Single-file Structure** (`config.yaml`):
```yaml
# config.yaml structure

dataset:
  dataset_type: "DFT"
  root_dir: "./data"
  # ... dataset settings

filter:
  # ... filter settings

processing:
  # ... processing settings

transformations:
  # ... transform settings

# Models configuration
models:
  enabled: true
  
  selection:
    model_name: "GCN"
    task_type: "graph_regression"
    hyperparameters:
      hidden_channels: 64
      num_layers: 3
  
  training:
    epochs: 100
    batch_size: 32
    learning_rate: 0.001
    # ... training settings
  
  # Prediction configuration
  prediction:
    batch_size: 32
    device: "auto"
    format: "auto"
    split: "all"
    num_samples: null
    output_path: "./predictions.csv"
    output_format: "csv"
    include_inputs: false
    uncertainty: false
    uncertainty_method: "dropout"
    inverse_transform: true
  
  acceleration:
    device: "cuda"
    mixed_precision: true
    # ... acceleration settings
  
  deployment:
    strategy: "edge"
    quantization: true
    # ... deployment settings
```

---

## Architectural Patterns

The Milia Pipeline employs several key architectural patterns:

### 1. Modularity
- Independent, reusable components
- Clear separation of concerns
- Plugin-based extensibility

### 2. Flexibility
- Multiple input formats supported
- Configurable preprocessing pipelines
- Custom transformation chains
- Multiple model architectures
- **YAML Splitting Architecture**: Single-file (`config.yaml`) or split-file (`configs/`) configuration modes

### 3. Scalability
- Efficient binary data storage (NPZ)
- Batch processing capabilities
- Registry-based component discovery
- Distributed training support
- Memory optimization

### 4. Maintainability
- Comprehensive test coverage
- Extensive documentation
- Type hints and validation
- Clear error hierarchies
- **Modular configuration**: Split YAML files with colocated dataset properties for easier maintenance

### 5. Research-Ready
- Experiment configuration support
- Research API for transformations
- Performance benchmarking tools
- Model training and evaluation utilities
- Hyperparameter optimization

### 6. Production-Ready
- Model deployment utilities
- Performance monitoring
- Inference optimization
- Multi-device support
- Edge and cloud deployment

### 7. Extensibility (Datasets Module) 
- Zero-modification architecture for new dataset types
- Protocol-based contracts
- Explicit registry pattern
- Compile-time validation via `__init_subclass__`

---

## Performance and Thread Safety

### Thread Safety Considerations

**Thread-Safe Components**:

1. **Dataset Registry** (datasets module) :
   - Non-singleton pattern with RLock
   - Thread-safe registration and lookup
   - Cache invalidation callbacks
   - Isolated instances for testing

2. **Descriptor Registry** (descriptors module):
   - Singleton pattern with thread locks
   - Safe concurrent access to descriptor registry
   - Thread-safe plugin registration

3. **Descriptor Calculator** (descriptors module):
   - Thread-safe when using separate calculator instances
   - Cache implemented with thread-safe mechanisms
   - Batch processing supports parallel execution

4. **Handler Integration** (handlers module):
   - Lazy loading mechanism is thread-safe
   - Handler instances can be used concurrently
   - Transform-aware integration supports parallel processing

5. **Model Registry** (models module):
   - Thread-safe model registration
   - Concurrent model lookup support
   - Singleton pattern with locks

6. **Layer Registry** (models.builders module):
   - Thread-safe layer access
   - Concurrent layer lookup
   - Thread-safe registration

7. **HPO Manager** (models.hpo module) :
   - Thread-safe study creation and access
   - Frozen immutable configuration objects
   - Thread-safe backend initialization
   - SearchSpaceBuilder produces immutable configurations

**Thread-Unsafe Components** (require external synchronization):
- RDKit molecule modifications
- PyG Data object mutations
- Plugin loading/unloading operations
- Model training state modifications
- Callback state updates
- HPO trial execution (each trial runs single-threaded) 
- HPO transfer manager meta-database modifications 

### Performance Optimizations

**Caching Strategies**:

1. **Descriptor Caching** (descriptors module):
   - Enable with `DescriptorCalculator(enable_cache=True)`
   - Caches results per molecule
   - Statistics available: cache hit rate, total calculations
   - Significant speedup for repeated calculations

2. **Handler Caching** (handlers module):
   - Transform result caching in `TransformAwareHandlerIntegrator`
   - Enable with `enable_caching=True`
   - Reduces repeated transform computation

3. **Model Training Caching** (models module):
   - Dataset caching
   - Transform result caching
   - Checkpoint caching

4. **Registry Cache Invalidation** (datasets module) :
   - Callback-based cache invalidation
   - Consumers notified on registry changes
   - LRU cache integration support

**Batch Processing**:

1. **Descriptor Calculation**:
   - `calculate_for_molecules()` for batch processing
   - Automatic conformer generation batching
   - Progress tracking and statistics

2. **Molecular Conversion**:
   - Batch molecule creation
   - Vectorized feature extraction where possible

3. **Model Training**:
   - Mini-batch processing
   - DataLoader optimization
   - Prefetching and pinned memory
   - Gradient accumulation

**Lazy Loading**:

1. **Handlers Module**:
   - Pure lazy loading via `__getattr__`
   - Deferred imports until attribute access
   - Reduces initial import time
   - Breaks circular dependency chains

2. **Plugin System**:
   - Plugins loaded on-demand
   - Auto-discovery can be deferred
   - Selective plugin loading

**Memory Efficiency**:

1. **NPZ Storage**:
   - Binary format for wavefunction data
   - Compressed storage
   - Memory-mapped access for large datasets

2. **Descriptor Storage**:
   - Sparse storage for unused descriptors
   - Selective descriptor calculation
   - Requirement-based filtering

3. **Model Memory Optimization** (models module):
   - Gradient checkpointing
   - Mixed precision training (FP16/BF16)
   - Memory-efficient attention mechanisms
   - Activation recomputation
   - Memory profiling

**Computation Optimization** (models module):

1. **JIT Compilation**:
   - TorchScript compilation
   - Kernel fusion
   - Operator optimization

2. **Distributed Training**:
   - Data parallel training (DP)
   - Distributed data parallel (DDP)
   - Model parallelism
   - Pipeline parallelism
   - Gradient accumulation

3. **Hardware Acceleration**:
   - CUDA optimization
   - Multi-GPU support
   - MPS support (Apple Silicon)
   - TPU support (via XLA)

**Performance Best Practices**:

- Enable caching for repeated descriptor calculations
- Use batch processing for multiple molecules
- Filter descriptors by requirements before calculation
- Generate conformers only when 3D descriptors needed
- Use factory functions for proper initialization
- Employ lazy loading for large plugin systems
- **Enable mixed precision training for faster training**
- **Use gradient checkpointing for large models**
- **Leverage distributed training for multi-GPU setups**
- **Profile memory usage to identify bottlenecks**
- **Use JIT compilation for inference optimization**

---

## Production Considerations

### Dependencies Management
- `setup.py`: Package installation and dependency specification
- Modular imports for selective loading
- Optional dependencies for advanced features

### Logging
- Centralized logging configuration (`logging_config.py`)
- Per-module logging capabilities
- Plugin system logging
- Training progress logging
- Performance monitoring logging

### Error Handling
- Custom exception hierarchy (`exceptions.py`)
- Validation at configuration and runtime
- Graceful error recovery
- Training error handling and recovery
- Deployment error handling
- **Dataset registration error handling** 
  - `DatasetRegistrationError`: Registration failures
  - `DatasetNotFoundError`: Dataset lookup failures

### Performance
- Efficient data structures (NPZ, binary formats)
- Lazy loading where applicable
- Performance testing infrastructure
- Model performance monitoring
- Inference optimization

### Model Management
- Model versioning
- Checkpoint management
- Model registry for organization
- Deployment strategies
- A/B testing support
- Rollback capabilities

### Monitoring and Observability
- Training metrics logging
- TensorBoard integration
- Performance monitoring
- Drift detection
- Alert system
- Resource utilization tracking

---

## Summary

The Milia Pipeline is a production-ready, research-oriented molecular data processing and machine learning framework with:

- **11 core modules** for specialized functionality
  - **Config**: Multi-layered configuration management
  - **Molecules**: Comprehensive molecular processing pipeline
  - **Transformations**: Extensible graph transformation system
  - **Datasets**: Registry-based PyTorch Geometric implementations
  - **Handlers**: 10 dataset handler types with transform integration
  - **Preprocessing**: Modular wavefunction preprocessing
  - **Descriptors**: 400+ molecular descriptors across 6 categories
  - **Models**: Complete training and deployment infrastructure
  - **HPO**: Hyperparameter optimization with Optuna/Ray Tune backends
  - **CLI Manager**: Command-line interface
  - **Exceptions**: Comprehensive error handling

- **3 plugin systems** for extensibility
  - Descriptor plugins with YAML configuration
  - Transformation plugins
  - **Model plugins**
  - Plugin templates and examples

- **127 unit and integration tests** for reliability
  - Component-level unit tests
  - Cross-component integration tests
  - Performance benchmarking
  - Validation scripts
  - **Model training tests**
  - **Acceleration tests**
  - **HPO module tests (14 files)**
  - **Handler tests (12 files)**
  - **Dataset implementation tests (13 files)**
  - **Preprocessing tests (11 files)**

- **Advanced architectural patterns**
  - Lazy loading for circular dependency resolution (handlers module)
  - Thread-safe singleton registry pattern (descriptors, models)
  - **Thread-safe non-singleton registry pattern (datasets)** 
  - Factory pattern for handler and model creation
  - Auto-discovery mechanisms (descriptors, preprocessors, models)
  - Plugin discovery and validation systems
  - Configuration bridge pattern (models)
  - **Protocol + ABC + Explicit Registry pattern (datasets)** 
  - **Builder pattern for custom architectures**
  - **Composer pattern for ensembles**
  - **`__init_subclass__` for compile-time validation** 
  - **Backend Protocol pattern for HPO** (HPO module) 
  - **Fluent builder pattern for search spaces** (HPO module) 

- **Comprehensive APIs**
  - Minimal public interfaces for end users
  - Advanced submodule access for power users
  - Consistent error handling with exception hierarchies
  - PyTorch Geometric integration utilities
  - **Complete training API**
  - **Deployment API**
  - **Registry-based dataset discovery API** 
  - **HPO configuration and management API** 

- **Multiple entry points** for diverse use cases
  - CLI interface (cli_manager.py)
  - Main execution entry (main.py)
  - Programmatic access via package imports
  - **Training scripts**
  - **Deployment scripts**
  - **HPO optimization scripts** 

**Key Capabilities**:
- 400+ RDKit molecular descriptors with caching
- 10 specialized dataset handlers (DFT, DMC, Wavefunction, QM9, ANI1X, ANI1CCX, RMD17, ANI2X, XXMD, QDPi)
- Conformer generation for 3D descriptors
- Transform-aware handler integration
- Structural feature filtering per dataset type
- Comprehensive validation at multiple levels
- Plugin-based extensibility without core modifications
- **Zero-modification dataset extension architecture** 
- **Protocol-based contracts with 11-method handler interface** 
- **Compile-time validation for dataset implementations** 
- **ALL PyTorch Geometric models via dynamic introspection** 
- **Custom architecture building**
- **Multi-model ensembles**
- **Factory-based model creation and management**
- **Dynamic target selection with multiple modes (properties/indices/range)** 
- **Task-specific data preparation with 7 task types (graph/node/edge/link)** 
- **Multiple data splitting strategies (4 types)**
- **Comprehensive callback system (7 types including HPO callback)** 
- **Multi-device training (CPU/GPU/MPS/TPU)**
- **Distributed training strategies (DP/DDP/Model Parallel)**
- **Memory and computation optimization**
- **Model deployment (Edge/Cloud/Federated)**
- **Production monitoring (Drift/Performance)**
- **Training callbacks and schedulers**
- **Loss function customization**
- **Model quantization and pruning**
- **Hyperparameter optimization with Optuna backend** 
- **Neural Architecture Search for GNNs** 
- **HPO transfer learning and warm-starting** 
- **Search space builder with dynamic model spaces** 
- **Study analysis and visualization** 
- **Complete main.py training integration with registry** 
- **Dynamic forward signature introspection in Trainer (supports ANY PyG model)** 
- **Task-aware target handling for all 7 task types** 
- **MetricsRegistry with 12 TorchMetrics-based evaluation metrics** 
- **TrainingVisualizer with 4 config-driven plot types (loss, metrics, LR, interactive)** 
- **Config-driven visualization settings in evaluation.visualization** 

The architecture emphasizes modularity, extensibility, thread safety, performance optimization, and scientific rigor, making it suitable for both research and production molecular machine learning workflows with comprehensive model training, deployment, and hyperparameter optimization capabilities.

---


**Document Version**: 1.1.0 
**Last Updated**: February 2026 
**Maintained By**: MILIA Team
