# cli_manager.py - CLI Manager Enhancement (Enhanced Command-Line Interface)

"""
Enhanced CLI management system for milia dataset processing.

This module provides a comprehensive command-line interface with handler-first
architecture, organized argument groups, robust validation, and multi-system
integration capabilities for dataset processing workflows.

Core Components
---------------
CLIValidationError : Exception
    Raised when CLI argument validation fails.
CLIManager : class
    Main CLI manager providing argument parsing, validation, and orchestration.
create_cli_manager : function
    Factory function for creating CLIManager instances.
parse_cli_args : function
    Convenience function for parsing and validating CLI arguments.

Key Features
------------
- **Handler-First Architecture**: All operations require properly configured handlers
- **Structured Argument Parsing**: 10 logical argument groups for clarity
- **Multi-System Integration**: Seamless integration with configuration, transformation,
  plugin, and research API systems
- **CLI Override Model**: Priority hierarchy (CLI > Config File > Defaults)
- **Interactive Mode**: Wizard-based workflow for complex configurations
- **Comprehensive Validation**: Multi-level validation before processing
- **Graceful Degradation**: Optional dependencies handled via feature flags

Architecture Design
-------------------
The module follows a handler-first, validation-heavy approach:

1. **No Legacy Modes**: All operations use handler-based processing exclusively
2. **Fail-Fast Validation**: Errors detected before any data processing
3. **CLI-First Override**: Command-line arguments override configuration files
4. **Optional Features**: Gracefully handles missing optional dependencies

Integration Points
------------------
- **Configuration System**: Load and merge YAML configurations
- **Validation System**: Schema-based configuration validation
- **Transform System**: Experimental setup and transform management
- **Plugin System**: Plugin discovery, validation, and lifecycle management
- **Research API**: Experiment orchestration and validation

Argument Groups
---------------
The CLI organizes arguments into 10 logical groups:
1. Basic Options (root-dir, config, force-reload, chunk-size)
2. Processing Modes (process, interactive, stats-only, quick-validation)
3. Transformation System (experimental-setup, validate-transforms)
4. Plugin Management (plugin operations and configuration)
5. Research API (experiment execution and validation)
6. Handler System (handler validation and testing)
7. Filter Options (molecule filtering criteria)
8. Validation Options (configuration and schema validation)
9. Logging Options (log levels and file output)
10. Advanced Options (performance tuning and expert features)
11. Training System (model training, HPO, evaluation)
12. Prediction System (post-training inference, Phase 5b)

Usage Examples
--------------
Basic usage with factory function:
    >>> from milia_pipeline.cli_manager import create_cli_manager
    >>> cli = create_cli_manager(logger=my_logger)
    >>> args = cli.parse_args()
    >>> config = cli.load_and_merge_config(args)
    >>> cli.validate_args(args, config)

Convenience function for quick parsing:
    >>> from milia_pipeline.cli_manager import parse_cli_args
    >>> args, cli_manager = parse_cli_args()
    >>> # args is ready to use, cli_manager available for additional operations

Interactive mode workflow:
    >>> cli = create_cli_manager()
    >>> args = cli.parse_args(['--interactive'])
    >>> # Launches interactive wizard for guided configuration

Plugin management:
    >>> args = cli.parse_args(['--list-plugins'])
    >>> cli.handle_plugin_operations(args)
    >>> # Lists all available plugins with status

Notes
-----
- The module requires Python 3.8+ for type hints and pathlib support
- Optional dependencies (transforms, plugins) are detected at runtime
- All validation errors raise CLIValidationError with detailed messages
- Interactive mode is recommended for first-time users or complex setups

PHASE 7 ENHANCEMENTS:
--------------------
- Registry integration for dynamic dataset type support
- Dynamic argparse choices populated from registry
- Feature-based validation for dataset-specific requirements
- Automatic support for new dataset types when registered
- Backward compatibility with legacy fallback when registry unavailable

Registry Integration Functions (Internal):
-----------------------------------------
- _init_registry(): Lazy initialization of registry imports
- _get_available_dataset_types(): Get list of registered dataset types
- _is_dataset_type_registered(): Check if dataset type is registered
- _get_dataset_feature(): Query feature flags for dataset types
- _get_dataset_input_format(): Get expected input format for dataset
- get_cli_registry_status(): Get registry integration diagnostics

Adding New Dataset Types:
------------------------
After Phase 7, adding a new dataset type that is automatically supported by CLI:
1. Create dataset class with @register decorator (Phase 2 pattern)
2. Define features attribute with appropriate flags (e.g., requires_archive_input=True)
3. Add ONE import line in datasets/__init__.py
4. CLI automatically:
   - Includes new type in --preprocess-dataset choices
   - Validates input format based on features
   - Shows type in help text

See Also
--------
milia_pipeline.config.config_loader : Configuration loading system
milia_pipeline.config.config_schemas : Configuration validation schemas
milia_pipeline.transformations.plugin_system : Plugin management system
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from milia_pipeline.config.config_accessors import (
    list_experimental_setups,
)

# Configuration system
from milia_pipeline.config.config_loader import load_config

# Validation system
try:
    from milia_pipeline.config.config_schemas import ValidationConfig, YAMLSchemaValidator

    CONFIG_VALIDATION_AVAILABLE = True
except ImportError:
    CONFIG_VALIDATION_AVAILABLE = False
    YAMLSchemaValidator = None
    ValidationConfig = None

# Transformation system
try:
    from milia_pipeline.transformations.graph_transforms import get_graph_transforms

    TRANSFORMS_AVAILABLE = True
except ImportError:
    TRANSFORMS_AVAILABLE = False

# Plugin system (Plugin management support)
try:
    from milia_pipeline.exceptions import (
        PluginDependencyError,
        PluginDiscoveryError,
        PluginError,
        PluginSecurityError,
        PluginValidationError,
    )
    from milia_pipeline.transformations.plugin_system import (
        PluginMetadata,
        PluginRegistry,
        PluginValidator,
    )

    PLUGIN_SYSTEM_AVAILABLE = True
except ImportError:
    PLUGIN_SYSTEM_AVAILABLE = False
    PluginRegistry = None
    PluginValidator = None
    PluginMetadata = None
    # Set exception classes to None as well for completeness
    PluginError = None
    PluginValidationError = None
    PluginSecurityError = None
    PluginDependencyError = None
    PluginDiscoveryError = None


# ============================================================================
# Module-Level Logger
# ============================================================================
# Required for module-level functions (e.g., _discover_dataset_types_from_filesystem,
# _get_available_dataset_types, _is_dataset_type_registered) that execute before
# any CLIManager instance is created. Uses __name__ for proper logger hierarchy.
logger = logging.getLogger(__name__)

# ============================================================================
# PHASE 7: Registry Integration for Dynamic Dataset Type Support
# ============================================================================

# Registry availability flags - set during lazy initialization
_REGISTRY_INITIALIZED = False
_REGISTRY_AVAILABLE = False
_REGISTRY_IMPORT_ERROR = None

# Registry function placeholders (populated by _init_registry)
_registry_list_all = None
_registry_get = None
_registry_is_registered = None

# Legacy fallback dataset types (used when registry unavailable)
# DEPRECATED: Use _get_available_dataset_types() which does dynamic discovery
_LEGACY_DATASET_TYPES = [
    "DFT",
    "DMC",
    "Wavefunction",
]  # LEGACY - kept for backward compatibility only


def _discover_dataset_types_from_filesystem() -> list:
    """
    Dynamically discover dataset types from implementations directory.

    DYNAMIC APPROACH: Scans the filesystem to find available dataset implementations
    instead of using hardcoded fallback lists.

    Returns:
        List of discovered dataset type names (uppercase)
    """
    try:
        from pathlib import Path

        # Find the implementations directory relative to this file
        implementations_dir = Path(__file__).parent / "datasets" / "implementations"
        if implementations_dir.exists():
            discovered_types = []
            for py_file in implementations_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                # Extract dataset name from filename (e.g., dft.py -> DFT, qm9.py -> QM9)
                dataset_name = py_file.stem.upper()
                # Exclude non-dataset modules
                if dataset_name not in ["BASE", "REGISTRY", "UTILS", "COMMON"]:
                    discovered_types.append(dataset_name)
            if discovered_types:
                logger.debug(f"Dynamically discovered dataset types: {discovered_types}")
                return discovered_types
    except Exception as e:
        logger.debug(f"Dynamic dataset type discovery failed: {e}")

    # Final fallback: return empty list with warning
    logger.warning(
        "No dataset types available - registry not initialized and dynamic discovery failed"
    )
    return []


# Legacy feature fallback (used when registry unavailable)
_LEGACY_FEATURES = {
    "DFT": {
        "vibrational_analysis": True,
        "uncertainty_handling": False,
        "atomization_energy": True,
        "rotational_constants": True,
        "frequency_analysis": True,
        "orbital_analysis": False,
        "requires_archive_input": False,
        "input_file_format": "npz",
    },
    "DMC": {
        "vibrational_analysis": False,
        "uncertainty_handling": True,
        "atomization_energy": False,
        "rotational_constants": False,
        "frequency_analysis": False,
        "orbital_analysis": False,
        "requires_archive_input": False,
        "input_file_format": "npz",
    },
    "Wavefunction": {
        "vibrational_analysis": False,
        "uncertainty_handling": False,
        "atomization_energy": False,
        "rotational_constants": False,
        "frequency_analysis": False,
        "orbital_analysis": True,
        "homo_lumo_gap": True,
        "mo_energies": True,
        "requires_archive_input": True,
        "input_file_format": "tar.gz",
    },
}


def _init_registry() -> bool:
    """
    Lazily initialize registry imports to avoid circular import at module load time.

    The datasets/__init__.py imports implementations which may import this module
    indirectly. By deferring the registry import until first use, we allow both
    modules to fully load first.

    Returns:
        True if registry is available, False otherwise

    ADDED Phase 7: Lazy initialization following Phase 3/6/7 pattern.
    """
    global _REGISTRY_INITIALIZED, _REGISTRY_AVAILABLE, _REGISTRY_IMPORT_ERROR
    global _registry_list_all, _registry_get, _registry_is_registered

    if _REGISTRY_INITIALIZED:
        return _REGISTRY_AVAILABLE

    _REGISTRY_INITIALIZED = True

    try:
        from milia_pipeline.datasets.registry import get, is_registered, list_all

        _registry_list_all = list_all
        _registry_get = get
        _registry_is_registered = is_registered
        _REGISTRY_AVAILABLE = True
        return True
    except ImportError as e:
        _REGISTRY_IMPORT_ERROR = str(e)
        _REGISTRY_AVAILABLE = False
        return False


def _get_available_dataset_types() -> list:
    """
    Get list of available dataset types from registry or dynamic discovery.

    DYNAMIC APPROACH: Instead of hardcoded fallback, this function:
    1. First tries the registry (primary source of truth)
    2. If registry fails, dynamically discovers dataset implementations from filesystem
    3. Never uses hardcoded _LEGACY_DATASET_TYPES list

    Returns:
        List of dataset type names

    ADDED Phase 7: Dynamic dataset type retrieval for CLI.
    """
    _init_registry()

    if _REGISTRY_AVAILABLE and _registry_list_all is not None:
        try:
            return _registry_list_all()
        except Exception as e:
            logger.debug(f"Registry list_all() failed: {e}")

    # DYNAMIC FALLBACK: Use filesystem discovery instead of hardcoded list
    return _discover_dataset_types_from_filesystem()


def _is_dataset_type_registered(dataset_type: str) -> bool:
    """
    Check if a dataset type is registered in the registry or dynamically discovered.

    DYNAMIC APPROACH: Instead of hardcoded fallback check, this function:
    1. First tries the registry (primary source of truth)
    2. If registry fails, uses _get_available_dataset_types() which does dynamic discovery
    3. Never uses hardcoded _LEGACY_DATASET_TYPES list

    Args:
        dataset_type: Dataset type name to check

    Returns:
        True if registered or dynamically discovered, False otherwise

    ADDED Phase 7: Dynamic dataset type validation.
    """
    _init_registry()

    if _REGISTRY_AVAILABLE and _registry_is_registered is not None:
        try:
            return _registry_is_registered(dataset_type)
        except Exception as e:
            logger.debug(f"Registry is_registered() failed for '{dataset_type}': {e}")

    # DYNAMIC FALLBACK: Check against dynamically discovered types
    available_types = _get_available_dataset_types()
    return dataset_type in available_types


def _get_dataset_feature(dataset_type: str, feature_name: str, default: bool = False) -> bool:
    """
    Query a feature flag for a dataset type from registry or legacy fallback.

    Args:
        dataset_type: Dataset type name (e.g., 'DFT', 'DMC')
        feature_name: Feature to query (e.g., 'uncertainty_handling')
        default: Default value if feature not found

    Returns:
        Feature value (True/False)

    ADDED Phase 7: Feature-based validation for CLI.
    """
    _init_registry()

    if _REGISTRY_AVAILABLE and _registry_get is not None:
        try:
            dataset_class = _registry_get(dataset_type)
            if hasattr(dataset_class, "features"):
                return getattr(dataset_class.features, feature_name, default)
        except Exception:
            pass

    # Legacy fallback
    return _LEGACY_FEATURES.get(dataset_type, {}).get(feature_name, default)


def _get_dataset_input_format(dataset_type: str) -> str:
    """
    Get the expected input file format for a dataset type.

    Args:
        dataset_type: Dataset type name

    Returns:
        Input format string (e.g., 'npz', 'tar.gz')

    ADDED Phase 7: Input format validation support.
    """
    _init_registry()

    if _REGISTRY_AVAILABLE and _registry_get is not None:
        try:
            dataset_class = _registry_get(dataset_type)
            if hasattr(dataset_class, "schema"):
                return getattr(dataset_class.schema, "input_file_format", "npz")
            if hasattr(dataset_class, "features"):
                if getattr(dataset_class.features, "requires_archive_input", False):
                    return "tar.gz"
        except Exception:
            pass

    # Legacy fallback
    return _LEGACY_FEATURES.get(dataset_type, {}).get("input_file_format", "npz")


def get_cli_registry_status() -> dict:
    """
    Get registry integration status for CLI diagnostics.

    Returns:
        Dict with registry status information

    ADDED Phase 7: Diagnostic function for CLI registry integration.
    """
    _init_registry()

    return {
        "registry_available": _REGISTRY_AVAILABLE,
        "registry_initialized": _REGISTRY_INITIALIZED,
        "registry_import_error": _REGISTRY_IMPORT_ERROR,
        "available_dataset_types": _get_available_dataset_types(),
        "using_legacy_fallback": not _REGISTRY_AVAILABLE,
        "phase_7_integration": True,
    }


def _get_default_config_path() -> str:
    """
    Get the default configuration file or directory path.

    YAML Splitting Architecture Enhancement:
    - Supports both single-file (backward compatible) and directory (split-file) modes
    - Dynamically detects available configuration at runtime

    Priority Order:
    1. config.yaml (single file in CWD) - HIGHEST, backward compatible
    2. config.yml (single file in CWD)
    3. ./configs/ (directory) - triggers split-file mode
    4. ./configs/config.yaml (single file inside configs/) - fallback
    5. Return 'config.yaml' as default (will trigger helpful error if missing)

    Returns:
        Path to configuration file or directory

    Evidence: Blueprint Section 3.1.3 specifies this priority order for
    backward compatibility while enabling new split-file mode.
    """
    # Priority 1: Single file in CWD (backward compatible)
    for file_path in ["config.yaml", "config.yml"]:
        if Path(file_path).is_file():
            return file_path

    # Priority 2: Configs directory (NEW - split-file mode)
    # NOTE: 'configs/' (plural) avoids confusion with milia_pipeline/config/ (Python code module)
    configs_dir = Path("./configs")
    if configs_dir.is_dir():
        return str(configs_dir)

    # Priority 3: config.yaml inside configs/ directory (legacy layout)
    config_in_dir = Path("./configs/config.yaml")
    if config_in_dir.is_file():
        return str(config_in_dir)

    # Default fallback - return 'config.yaml' which will trigger a helpful error
    # message if it doesn't exist (validation will catch this)
    return "config.yaml"


class CLIValidationError(Exception):
    """Raised when CLI argument validation fails."""

    pass


class CLIManager:
    """
    Enhanced CLI manager for milia dataset processing.

    Provides structured argument parsing, validation, and user guidance
    for all dataset processing operations. All operations use the
    handler-based architecture for consistent and reliable processing.

    Handler-First Architecture:
        The CLIManager enforces handler-based processing across all
        operations. No legacy or fallback modes are supported. All
        dataset operations must use properly configured handlers.

    Key Features:
        - Comprehensive argument parsing with validation
        - Interactive configuration wizard
        - Plugin management integration
        - Research API integration
        - Transform system integration
        - Configuration file override support
        - Detailed error messages and user guidance

    Usage:
        >>> cli = CLIManager(logger=my_logger)
        >>> args = cli.parse_args()
        >>> config = cli.load_and_merge_config(args)
        >>> cli.validate_args(args, config)

    Attributes:
        logger: Logger instance for CLI operations
        parser: ArgumentParser instance with all CLI options
        config: Loaded configuration dictionary (after load_and_merge_config)
    """

    def __init__(self, logger: logging.Logger | None = None):
        """
        Initialize CLI manager.

        Args:
            logger: Optional logger instance (created if not provided)
        """
        self.logger = logger or self._create_basic_logger()
        self.parser = self._create_parser()
        self.config = None

    def _create_basic_logger(self) -> logging.Logger:
        """Create basic logger for CLI operations."""
        logger = logging.getLogger("CLI_Manager")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
        return logger

    def _create_parser(self) -> argparse.ArgumentParser:
        """
        Create comprehensive argument parser with organized groups.

        Returns:
            Configured ArgumentParser instance
        """
        parser = argparse.ArgumentParser(
            prog="milia_process",
            description="milia Dataset Processing System - Enhanced CLI\n"
            "Defaults to --process mode when no mode flag is specified.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_usage_examples(),
        )

        # Add all argument groups
        self._add_basic_arguments(parser)
        self._add_processing_arguments(parser)
        self._add_transformation_arguments(parser)
        self._add_plugin_arguments(parser)
        self._add_research_api_arguments(parser)
        self._add_handler_arguments(parser)
        self._add_filter_arguments(parser)
        self._add_validation_arguments(parser)
        self._add_logging_arguments(parser)
        self._add_advanced_arguments(parser)
        self._add_preprocessing_arguments(parser)
        self._add_descriptor_arguments(parser)
        self._add_training_arguments(parser)
        self._add_prediction_arguments(parser)  # Phase 5b: Post-training inference

        return parser

    def _add_basic_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add basic operation arguments."""
        basic = parser.add_argument_group("Basic Options", "Core dataset processing options")

        basic.add_argument(
            "--root-dir",
            type=str,
            default=None,
            metavar="PATH",
            help="Root directory for dataset (overrides config.yaml)",
        )

        basic.add_argument(
            "--force-reload",
            action="store_true",
            help="Force reprocessing even if processed data exists",
        )

        basic.add_argument(
            "--chunk-size",
            type=int,
            default=5000,
            metavar="N",
            help="Molecules per processing chunk (default: 5000, range: 100-50000)",
        )

        basic.add_argument(
            "--config",
            type=str,
            default="config.yaml",
            metavar="PATH",
            help="Configuration file or directory path. "
            "Auto-detects: config.yaml → config.yml → configs/ directory. "
            "If a directory is specified (e.g., configs/), all YAML files within are merged "
            "(YAML Splitting Architecture).",
        )

    def _add_processing_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add processing mode arguments."""
        processing = parser.add_argument_group(
            "Processing Modes",
            "Different operation modes for dataset processing. "
            "If no mode is specified, --process mode is used by default.",
        )

        mode_group = processing.add_mutually_exclusive_group()

        mode_group.add_argument(
            "--process",
            action="store_true",
            help="Full dataset processing (default when no other mode specified)",
        )

        mode_group.add_argument(
            "--quick-validation",
            action="store_true",
            help="Validate existing processed data without reprocessing",
        )

        mode_group.add_argument(
            "--stats-only",
            action="store_true",
            help="Generate statistics for existing dataset only",
        )

        mode_group.add_argument(
            "--interactive", action="store_true", help="Launch interactive configuration wizard"
        )

    def _add_transformation_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add transformation system arguments."""
        transforms = parser.add_argument_group(
            "Transformation System", "Graph transformation and experimental setup options"
        )

        transforms.add_argument(
            "--experimental-setup",
            type=str,
            default=None,
            metavar="NAME",
            help="Experimental setup for transformations (e.g., baseline, augmented)",
        )

        transforms.add_argument(
            "--list-experimental-setups",
            action="store_true",
            help="List available experimental setups and exit",
        )

        transforms.add_argument(
            "--switch-experimental-setup",
            type=str,
            default=None,
            metavar="NAME",
            help="Switch to different experimental setup after dataset creation",
        )

        transforms.add_argument(
            "--validate-transforms-only",
            action="store_true",
            help="Validate transformation system without dataset processing",
        )

        transforms.add_argument(
            "--disable-transforms",
            action="store_true",
            help="Disable transformation system (no transforms applied)",
        )

        transforms.add_argument(
            "--list-transforms",
            action="store_true",
            help="List available transforms by category and exit",
        )

    def _add_plugin_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Add plugin management arguments.

        CLI integration for plugin system operations handling
        discovery, validation, management, and comprehensive testing.
        """
        plugins = parser.add_argument_group(
            "Plugin Management", "Custom transform plugin operations"
        )

        # Plugin discovery and paths
        plugins.add_argument(
            "--plugin-path",
            type=str,
            action="append",
            default=None,
            metavar="PATH",
            help="Add plugin search path (can be used multiple times)",
        )

        plugins.add_argument(
            "--discover-plugins", action="store_true", help="Discover plugins in registered paths"
        )

        plugins.add_argument(
            "--auto-validate",
            action="store_true",
            help="Automatically validate plugins during discovery",
        )

        # Plugin listing and information
        plugins.add_argument(
            "--list-plugins", action="store_true", help="List all discovered plugins and exit"
        )

        plugins.add_argument(
            "--plugin-info",
            type=str,
            default=None,
            metavar="NAME",
            help="Show detailed information for specific plugin and exit",
        )

        # Plugin validation
        plugins.add_argument(
            "--validate-plugin",
            type=str,
            default=None,
            metavar="NAME",
            help="Validate specific plugin (basic validation) and exit",
        )

        plugins.add_argument(
            "--validate-plugin-comprehensive",
            type=str,
            default=None,
            metavar="NAME",
            help="Run comprehensive validation with all tests and exit",
        )

        plugins.add_argument(
            "--run-performance-tests",
            action="store_true",
            help="Include performance benchmarks in comprehensive validation",
        )

        # Plugin management
        plugins.add_argument(
            "--enable-plugin",
            type=str,
            action="append",
            default=None,
            metavar="NAME",
            help="Enable specific plugin (can be used multiple times)",
        )

        plugins.add_argument(
            "--disable-plugin",
            type=str,
            action="append",
            default=None,
            metavar="NAME",
            help="Disable specific plugin (can be used multiple times)",
        )

        plugins.add_argument(
            "--trust-plugin",
            type=str,
            action="append",
            default=None,
            metavar="NAME",
            help="Mark plugin as trusted (relaxed security checks)",
        )

        # Plugin filtering options
        plugins.add_argument(
            "--validated-only", action="store_true", help="List/use only validated plugins"
        )

        plugins.add_argument(
            "--enabled-only", action="store_true", help="List/use only enabled plugins"
        )

        # Plugin system control
        plugins.add_argument(
            "--disable-plugin-system",
            action="store_true",
            help="Disable entire plugin system (use built-in transforms only)",
        )

    def _add_research_api_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Add research API CLI arguments.

        Systematic experimentation support for ablation studies,
        parameter sweeps, and comparative analyses.
        """
        research_group = parser.add_argument_group(
            "Research Workflows", "Systematic experimentation and ablation studies"
        )

        # Experiment execution
        research_group.add_argument(
            "--run-experiment",
            type=str,
            metavar="EXPERIMENT_NAME",
            help="Run experiment from research_experiments.yaml or config experiments section",
        )

        # Experiment discovery
        research_group.add_argument(
            "--list-experiments",
            action="store_true",
            help="List all configured experiments with details and exit",
        )

        # Configuration
        research_group.add_argument(
            "--experiment-config",
            type=str,
            metavar="PATH",
            help="Path to experiments configuration file (default: research_experiments.yaml)",
        )

        # Output configuration
        research_group.add_argument(
            "--experiment-output",
            type=str,
            default="./experiments",
            metavar="DIR",
            help="Output directory for experiment results (default: ./experiments)",
        )

        # Execution parameters
        research_group.add_argument(
            "--num-runs",
            type=int,
            metavar="N",
            help="Number of runs per variant for statistical significance (overrides config)",
        )

        # Validation
        research_group.add_argument(
            "--validate-experiment",
            type=str,
            metavar="EXPERIMENT_NAME",
            help="Validate experiment configuration without running and exit",
        )

    def _add_handler_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add dataset handler arguments.

        All operations in milia pipeline require handlers. These options
        control handler validation and configuration.
        """
        handlers = parser.add_argument_group(
            "Handler System", "Handler configuration (required for all operations)"
        )

        handlers.add_argument(
            "--validate-handlers",
            action="store_true",
            help="Validate handler system configuration before processing",
        )

        handlers.add_argument(
            "--handler-strict-validation",
            action="store_true",
            help="Enable strict handler validation (fail on warnings)",
        )

        handlers.add_argument(
            "--handler-compatibility-check",
            action="store_true",
            help="Check handler compatibility with dataset and transforms",
        )

        handlers.add_argument(
            "--test-handlers-only",
            action="store_true",
            help="Test handler system integration without full processing",
        )

    def _add_filter_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add molecule filter arguments."""
        filters = parser.add_argument_group(
            "Molecule Filters", "Override filter settings from config.yaml"
        )

        filters.add_argument(
            "--max-atoms",
            type=int,
            default=None,
            metavar="N",
            help="Maximum atoms per molecule (overrides config, 0 = no limit)",
        )

        filters.add_argument(
            "--min-atoms",
            type=int,
            default=None,
            metavar="N",
            help="Minimum atoms per molecule (overrides config, 0 = no limit)",
        )

        filters.add_argument(
            "--max-uncertainty",
            type=float,
            default=None,
            metavar="VALUE",
            help="Maximum DMC uncertainty threshold (overrides config)",
        )

        filters.add_argument(
            "--no-filters",
            action="store_true",
            help="Disable all molecular filters (process entire dataset without filtering)",
        )

    def _add_validation_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add validation and testing arguments."""
        validation = parser.add_argument_group(
            "Validation & Testing", "Configuration and data validation options"
        )
        # ---
        validation.add_argument(
            "--validate-config", action="store_true", help="Validate configuration file and exit"
        )

        validation.add_argument(
            "--test-limit",
            type=int,
            default=None,
            metavar="N",
            help="Limit processing to N molecules for testing",
        )

        validation.add_argument(
            "--dry-run", action="store_true", help="Validate configuration without processing"
        )

    def _add_logging_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add logging configuration arguments."""
        logging_group = parser.add_argument_group(
            "Logging & Output", "Control logging verbosity and output files"
        )

        logging_group.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help="Logging verbosity level (default: INFO)",
        )

        logging_group.add_argument(
            "--log-file",
            type=str,
            default=None,
            metavar="FILE",
            help="Log file path (default: console only)",
        )

        # Create mutually exclusive group for verbosity
        verbosity_group = logging_group.add_mutually_exclusive_group()

        verbosity_group.add_argument(
            "--quiet", action="store_true", help="Suppress non-error output (ERROR level only)"
        )

        verbosity_group.add_argument(
            "--verbose", action="store_true", help="Enable verbose output (DEBUG level)"
        )

    def _add_advanced_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add advanced processing arguments.

        These options provide fine-grained control over processing behavior,
        performance tuning, and system resource utilization. Use with caution
        as incorrect settings may impact performance or results.
        """
        advanced = parser.add_argument_group(
            "Advanced Options", "Performance tuning and expert configuration"
        )

        advanced.add_argument(
            "--skip-validation",
            action="store_true",
            help="Skip configuration validation (not recommended)",
        )

        advanced.add_argument(
            "--debug-handlers", action="store_true", help="Enable detailed handler debugging output"
        )

        advanced.add_argument(
            "--debug-transforms",
            action="store_true",
            help="Enable detailed transform debugging output",
        )

    def _add_preprocessing_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Add preprocessing system arguments.

        Preprocessing is a ONE-TIME operation that transforms raw dataset files
        (e.g., .tar.gz archives, .molden files) into the .npz format expected by
        miliaDataset. This happens BEFORE dataset creation and loading.

        Pattern Notes:
            - Follows existing validation pattern (--validate-config, --validate-transforms-only)
            - Uses standalone flags, not subcommands
            - Includes both operational and validation modes
            - Supports CLI overrides of config file settings
        """
        preprocessing = parser.add_argument_group(
            "Preprocessing Options",
            "Dataset preprocessing system (one-time transformation of raw data)",
        )

        # Main preprocessing flag
        preprocessing.add_argument(
            "--preprocess",
            action="store_true",
            help="Run dataset preprocessing before other operations",
        )

        # Preprocessing configuration
        preprocessing.add_argument(
            "--preprocess-config",
            type=str,
            metavar="PATH",
            help="Path to preprocessing configuration file (YAML)",
        )

        # PHASE 7: Dynamic dataset type selection from registry
        available_types = _get_available_dataset_types()
        preprocessing.add_argument(
            "--preprocess-dataset",
            type=str,
            choices=available_types,
            metavar="TYPE",
            help=f"Dataset type to preprocess (choices: {', '.join(available_types)})",
        )

        # Input/output paths (CLI overrides)
        preprocessing.add_argument(
            "--preprocess-input",
            type=str,
            metavar="PATH",
            help="Path to raw dataset file(s) (overrides config)",
        )

        preprocessing.add_argument(
            "--preprocess-output",
            type=str,
            metavar="PATH",
            help="Path for output .npz file (overrides config)",
        )

        # Processing control
        preprocessing.add_argument(
            "--preprocess-num-molecules",
            type=int,
            metavar="N",
            help="Number of molecules to process (default: all)",
        )

        preprocessing.add_argument(
            "--preprocess-feature-tier",
            type=str,
            choices=["basic", "standard", "complete"],
            metavar="TIER",
            help="Feature extraction tier (choices: basic, standard, complete)",
        )

        # Operational flags
        preprocessing.add_argument(
            "--preprocess-force",
            action="store_true",
            help="Force preprocessing even if output exists",
        )

        preprocessing.add_argument(
            "--preprocess-cleanup",
            action="store_true",
            default=True,
            help="Clean up temporary files after preprocessing (default: True)",
        )

        preprocessing.add_argument(
            "--preprocess-no-cleanup",
            action="store_false",
            dest="preprocess_cleanup",
            help="Keep temporary files after preprocessing",
        )

        preprocessing.add_argument(
            "--preprocess-progress",
            action="store_true",
            help="Show detailed progress during preprocessing",
        )

        # Validation modes (mutually exclusive with --preprocess)
        validation_group = preprocessing.add_mutually_exclusive_group()

        validation_group.add_argument(
            "--validate-preprocessing-only",
            action="store_true",
            help="Validate preprocessing configuration and exit",
        )

        validation_group.add_argument(
            "--test-preprocessor-only",
            action="store_true",
            help="Test preprocessor with small dataset and exit",
        )

        validation_group.add_argument(
            "--list-preprocessors",
            action="store_true",
            help="List available preprocessors and exit",
        )

    def _add_descriptor_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Add descriptor management arguments.

        Phase 3 Integration: Provides command-line interface for descriptor
        calculation, validation, and configuration.
        """
        descriptor = parser.add_argument_group(
            "🧪 Descriptor Management", "Molecular descriptor calculation and configuration"
        )

        descriptor.add_argument(
            "--enable-descriptors",
            action="store_true",
            help="Enable descriptor calculation (overrides config)",
        )

        descriptor.add_argument(
            "--disable-descriptors",
            action="store_true",
            help="Disable descriptor calculation (overrides config)",
        )

        descriptor.add_argument(
            "--descriptor-mode",
            choices=["explicit", "category", "all"],
            help="Descriptor selection mode (explicit/category/all)",
        )

        descriptor.add_argument(
            "--descriptor-categories",
            nargs="+",
            choices=[
                "constitutional",
                "topological",
                "electronic",
                "geometric",
                "drug_likeness",
                "fragments",
            ],
            help="Descriptor categories to calculate (for category mode)",
        )

        descriptor.add_argument(
            "--list-descriptors",
            action="store_true",
            help="List all available descriptors and exit",
        )

        descriptor.add_argument(
            "--validate-descriptors",
            action="store_true",
            help="Validate descriptor configuration and exit",
        )

        descriptor.add_argument(
            "--descriptor-stats",
            action="store_true",
            help="Show descriptor calculation statistics after processing",
        )

    def _add_training_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Add training and HPO arguments.

        Phase 9 Integration: Training system arguments for model training,
        hyperparameter optimization, and evaluation workflows.
        """
        training = parser.add_argument_group(
            "Training System", "Model training, HPO, and evaluation options"
        )

        # =================================================================
        # CORE TRAINING ARGUMENTS
        # =================================================================
        training.add_argument(
            "--train",
            action="store_true",
            default=False,
            help="Enable training mode (MASTER SWITCH for training workflow)",
        )

        training.add_argument(
            "--mode",
            type=str,
            default=None,
            choices=["single", "custom", "ensemble"],
            help="Model selection mode: single (registry model), custom (ArchitectureBuilder), ensemble (ModelComposer). Overrides config.yaml models.selection.mode",
        )

        training.add_argument(
            "--task-type",
            type=str,
            default=None,
            choices=[
                "graph_regression",
                "graph_classification",
                "node_regression",
                "node_classification",
                "link_prediction",
                "edge_regression",
                "edge_classification",
            ],
            help="Task type (overrides config.yaml models.selection.task_type)",
        )

        training.add_argument(
            "--epochs",
            type=int,
            default=None,
            help="Number of training epochs (overrides config.yaml)",
        )

        training.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help="Training batch size (overrides config.yaml)",
        )

        training.add_argument(
            "--learning-rate",
            type=float,
            default=None,
            help="Base learning rate (overrides config.yaml)",
        )

        training.add_argument(
            "--checkpoint", type=str, default=None, help="Path to checkpoint for resuming training"
        )

        training.add_argument(
            "--evaluate-only",
            action="store_true",
            default=False,
            help="Only evaluate model (skip training)",
        )

        # =================================================================
        # SINGLE MODEL MODE ARGUMENTS
        # =================================================================
        training.add_argument(
            "--model-name",
            type=str,
            default=None,
            help="Model name from registry (e.g., GCN, GAT, SchNet). Used with --mode single",
        )

        # =================================================================
        # CUSTOM ARCHITECTURE MODE ARGUMENTS
        # =================================================================
        training.add_argument(
            "--custom-architecture",
            action="store_true",
            default=False,
            help="Enable custom architecture mode (alternative to --mode custom)",
        )

        training.add_argument(
            "--architecture-config",
            type=str,
            default=None,
            help="Path to YAML/JSON file with custom architecture definition",
        )

        training.add_argument(
            "--builder-type",
            type=str,
            default="sequential",
            choices=["sequential", "parallel", "hierarchical"],
            help="Architecture builder type for custom mode",
        )

        # =================================================================
        # ENSEMBLE MODE ARGUMENTS
        # =================================================================
        training.add_argument(
            "--ensemble",
            action="store_true",
            default=False,
            help="Enable ensemble mode (alternative to --mode ensemble)",
        )

        training.add_argument(
            "--ensemble-config",
            type=str,
            default=None,
            help="Path to YAML/JSON file with ensemble definition",
        )

        training.add_argument(
            "--ensemble-strategy",
            type=str,
            default="parallel",
            choices=["parallel", "sequential", "hierarchical"],
            help="Ensemble composition strategy",
        )

        training.add_argument(
            "--fusion-method",
            type=str,
            default="weighted",
            choices=["mean", "weighted", "attention", "voting"],
            help="Ensemble output fusion method",
        )

        # =================================================================
        # HPO ARGUMENTS
        # =================================================================
        # HPO trigger: --hpo / --no-hpo (mutually exclusive)
        # Priority: CLI > Config > Default
        # None = not passed (use config), True = --hpo, False = --no-hpo
        hpo_group = training.add_mutually_exclusive_group()
        hpo_group.add_argument(
            "--hpo",
            action="store_true",
            dest="hpo",
            help="Enable hyperparameter optimization (overrides config)",
        )
        hpo_group.add_argument(
            "--no-hpo",
            action="store_false",
            dest="hpo",
            help="Disable hyperparameter optimization (overrides config)",
        )
        training.set_defaults(hpo=None)

        training.add_argument(
            "--n-trials",
            type=int,
            default=None,
            help="Number of HPO trials (overrides config.yaml models.hpo.n_trials)",
        )

        training.add_argument(
            "--hpo-timeout",
            type=int,
            default=None,
            help="HPO timeout in seconds (overrides config.yaml models.hpo.timeout)",
        )

        training.add_argument(
            "--hpo-backend",
            type=str,
            default="optuna",
            choices=["optuna", "ray_tune"],
            help="HPO backend to use",
        )

        training.add_argument(
            "--cv-folds",
            type=int,
            default=None,
            help="Cross-validation folds (0 = no CV, >0 = k-fold CV)",
        )

        training.add_argument(
            "--resume-study", type=str, default=None, help="Resume existing HPO study by name"
        )

        training.add_argument(
            "--sampler",
            type=str,
            default="tpe",
            choices=["tpe", "random", "cmaes", "grid"],
            help="HPO sampler type",
        )

        training.add_argument(
            "--pruner",
            type=str,
            default="median",
            choices=["median", "hyperband", "percentile", "none"],
            help="HPO pruner type",
        )

    def _add_prediction_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Add prediction/inference arguments.

        Phase 5b Integration: Post-training prediction workflow.
        Follows Chemprop's pattern with MILIA's argument naming conventions.

        CRITICAL DESIGN DECISIONS:
        1. Use --model-path (not --checkpoint) because --checkpoint is for resume
        2. Use --test-path (following Chemprop) for input data
        3. Use --predict-* prefix for runtime args to avoid conflicts
        4. Support BOTH molecular files AND full datasets

        Evidence (Chemprop CLI):
            chemprop predict --test-path data.csv --model-path model.pt --preds-path preds.csv

        Evidence (MILIA cli_manager.py line 1206-1210):
            --checkpoint is already used for resuming training, so we use --model-path
        """
        prediction = parser.add_argument_group(
            "Prediction System", "Post-training inference and prediction options (Phase 5b)"
        )

        # =================================================================
        # CORE PREDICTION ARGUMENTS
        # =================================================================
        prediction.add_argument(
            "--predict",
            action="store_true",
            default=False,
            help="Enable prediction mode (MASTER SWITCH for inference workflow)",
        )

        prediction.add_argument(
            "--model-path",
            type=str,
            default=None,
            metavar="PATH",
            dest="model_path",
            help="Path to trained model checkpoint (.pt file) for inference. "
            "Note: Use --checkpoint for resuming training, --model-path for prediction.",
        )

        prediction.add_argument(
            "--test-path",
            type=str,
            default=None,
            metavar="PATH",
            dest="test_path",
            help="Path to input data: molecular file (CSV, XYZ, SDF), "
            "processed dataset (.pt), or miliaDataset root directory",
        )

        prediction.add_argument(
            "--preds-path",
            type=str,
            default="./predictions.csv",
            metavar="PATH",
            dest="preds_path",
            help="Path to save predictions (default: ./predictions.csv)",
        )

        # =================================================================
        # RUNTIME CONFIGURATION (--predict-* prefix to avoid conflicts)
        # =================================================================
        prediction.add_argument(
            "--predict-batch-size",
            type=int,
            default=32,
            metavar="N",
            dest="predict_batch_size",
            help="Batch size for inference (default: 32)",
        )

        prediction.add_argument(
            "--predict-device",
            type=str,
            default="auto",
            choices=["cpu", "cuda", "mps", "auto"],
            dest="predict_device",
            help="Device for inference (default: auto-detect)",
        )

        prediction.add_argument(
            "--predict-format",
            type=str,
            default="auto",
            choices=["auto", "smiles", "inchi", "xyz", "sdf", "csv", "dataset"],
            dest="predict_format",
            help="Force input format (default: auto-detect from path)",
        )

        prediction.add_argument(
            "--predict-uncertainty",
            action="store_true",
            default=False,
            dest="predict_uncertainty",
            help="Enable uncertainty estimation (if model supports)",
        )

        # =================================================================
        # DATASET-SPECIFIC ARGUMENTS
        # =================================================================
        prediction.add_argument(
            "--predict-split",
            type=str,
            default="all",
            choices=["train", "val", "test", "all"],
            dest="predict_split",
            help="Dataset split to predict on (default: all). "
            "Only applies when --test-path is a miliaDataset.",
        )

        prediction.add_argument(
            "--predict-num-samples",
            type=int,
            default=None,
            metavar="N",
            dest="predict_num_samples",
            help="Limit number of samples for prediction (default: all)",
        )

        # =================================================================
        # OUTPUT OPTIONS
        # =================================================================
        prediction.add_argument(
            "--predict-output-format",
            type=str,
            default="csv",
            choices=["csv", "json", "npy", "pt"],
            dest="predict_output_format",
            help="Output format for predictions (default: csv)",
        )

        prediction.add_argument(
            "--predict-include-inputs",
            action="store_true",
            default=False,
            dest="predict_include_inputs",
            help="Include input identifiers (SMILES, etc.) in output",
        )

    def _get_usage_examples(self) -> str:
        """Get usage examples for help text."""
        # PHASE 7: Get available types for dynamic documentation
        available_types = _get_available_dataset_types()
        types_str = ", ".join(available_types)

        return f"""
Examples:
  Examples:
  # Basic processing (--process is default, but shown explicitly for clarity)
  python main.py --process

  # Same as above - process mode is automatic without flags
  python main.py

  # Use specific experimental setup
  python main.py --experimental-setup baseline

  # Validate handler configuration before processing
  python main.py --validate-handlers

  # Force reprocessing with custom chunk size
  python main.py --force-reload --chunk-size 10000

  # Quick validation of existing data
  python main.py --quick-validation

  # List available experimental setups
  python main.py --list-experimental-setups

  # Interactive configuration wizard
  python main.py --interactive

  # Debug mode with detailed logging
  python main.py --verbose --log-file debug.log

  # Test with limited dataset
  python main.py --test-limit 100 --quick-validation

  # Override filters from command line
  python main.py --max-atoms 50 --min-atoms 3

  # Validate configuration without processing
  python main.py --validate-config

  # Plugin Management
  python main.py --plugin-path ./my_plugins --discover-plugins
  python main.py --list-plugins
  python main.py --validate-plugin my_custom_plugin
  python main.py --validate-plugin-comprehensive my_custom_plugin --run-performance-tests
  python main.py --enable-plugin my_custom_plugin
  python main.py --disable-plugin old_plugin
  python main.py --plugin-info my_custom_plugin
  python main.py --trust-plugin verified_plugin
  python main.py --plugin-path ./plugins --process --experimental-setup custom_setup

  # Research API
  python main.py --list-experiments
  python main.py --validate-experiment transform_ablation
  python main.py --run-experiment transform_ablation --experiment-output ./results
  python main.py --list-experiments --experiment-config my_experiments.yaml

  # Basic processing (--process is default, but shown explicitly for clarity)
  python main.py --process

  # Same as above - process mode is automatic without flags
  python main.py

  # Use specific experimental setup
  python main.py --experimental-setup baseline

  # Validate handler configuration before processing
  python main.py --validate-handlers

  # Force reprocessing with custom chunk size
  python main.py --force-reload --chunk-size 10000

  # Quick validation of existing data
  python main.py --quick-validation

  # List available experimental setups
  python main.py --list-experimental-setups

  # Interactive configuration wizard
  python main.py --interactive

  # Debug mode with detailed logging
  python main.py --verbose --log-file debug.log

  # NEW: Preprocessing examples
  # Available dataset types: {types_str}

  # Preprocess wavefunction dataset
  python main.py --preprocess --preprocess-config examples/preprocessing/wavefunction_preprocess.yaml

  # Preprocess with CLI overrides
  python main.py --preprocess --preprocess-dataset Wavefunction --preprocess-input raw/wavefunctions.tar.gz

  # Validate preprocessing configuration
  python main.py --validate-preprocessing-only --preprocess-config examples/preprocessing/wavefunction_preprocess.yaml

  # Test preprocessor with small dataset
  python main.py --test-preprocessor-only --preprocess-dataset Wavefunction --preprocess-num-molecules 10

  # List available preprocessors
  python main.py --list-preprocessors

  # =================================================================
  # PREDICTION MODE (Phase 5b - Post-Training Inference)
  # =================================================================

  # Predict on molecular file (SMILES CSV)
  python main.py --predict --model-path ./checkpoints/best_model.pt --test-path ./molecules.csv --preds-path ./predictions.csv

  # Predict on XYZ file
  python main.py --predict --model-path ./checkpoints/schnet_model.pt --test-path ./molecule.xyz

  # Predict on processed dataset (.pt file)
  python main.py --predict --model-path ./checkpoints/model.pt --test-path ./data/QM9/processed/data.pt

  # Predict on miliaDataset directory (auto-detects processed/data.pt)
  python main.py --predict --model-path ./checkpoints/model.pt --test-path ./data/QM9/

  # Predict with custom batch size and device
  python main.py --predict --model-path ./model.pt --test-path ./data.csv --predict-batch-size 64 --predict-device cuda

  # Predict on specific dataset split
  python main.py --predict --model-path ./model.pt --test-path ./data/QM9/ --predict-split test

  # Predict with sample limit
  python main.py --predict --model-path ./model.pt --test-path ./data.csv --predict-num-samples 1000

  # Predict with JSON output
  python main.py --predict --model-path ./model.pt --test-path ./data.csv --predict-output-format json


For more information, see: https://docs.example.com/milia-cli
    """

    def parse_args(self, args: list[str] | None = None) -> argparse.Namespace:
        """
        Parse command-line arguments with validation.

        Args:
            args: Optional argument list (uses sys.argv if None)

        Returns:
            Parsed and validated arguments

        Raises:
            CLIValidationError: If validation fails
        """
        parsed_args = self.parser.parse_args(args)

        # Apply argument processing
        parsed_args = self._process_arguments(parsed_args)

        # Validate arguments
        self._validate_arguments(parsed_args)

        return parsed_args

    def _process_arguments(self, args: argparse.Namespace) -> argparse.Namespace:
        """
        Process and normalize arguments.

        Args:
            args: Parsed arguments

        Returns:
            Processed arguments
        """
        # Handle verbose flag
        if args.verbose:
            args.log_level = "DEBUG"

        # Handle quiet flag
        if args.quiet:
            args.log_level = "ERROR"

        # YAML Splitting Architecture: Resolve config path dynamically
        # If user didn't specify --config, use dynamic default detection
        if args.config == "config.yaml":
            # User used the default - check if we should use split-file mode
            resolved_path = _get_default_config_path()
            if resolved_path != "config.yaml":
                args.config = resolved_path
                self.logger.debug(f"Using detected config path: {resolved_path}")

        # NEW: Handle preprocessing modes
        if hasattr(args, "preprocess") and args.preprocess:
            self.logger.info("Preprocessing mode enabled")
            # Preprocessing can run before or with other modes
            if not hasattr(args, "preprocess_config") or not args.preprocess_config:
                self.logger.warning(f"No preprocessing config specified, will use: {args.config}")

        # Set default processing mode if none specified
        if not any(
            [
                args.process,
                args.quick_validation,
                args.stats_only,
                args.interactive,
                getattr(args, "preprocess", False),
                getattr(args, "validate_preprocessing_only", False),
                getattr(args, "test_preprocessor_only", False),
                getattr(args, "list_preprocessors", False),
                getattr(args, "predict", False),
            ]
        ):  # Phase 5b: prediction mode
            args.process = True
            self.logger.info("No mode specified - defaulting to --process mode")

        # Generate log file if needed
        if args.log_level == "DEBUG" and not args.log_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.log_file = f"milia_debug_{timestamp}.log"

        # Handle plugin system modes
        if not any(
            [
                args.process,
                args.quick_validation,
                args.stats_only,
                args.interactive,
                args.list_experimental_setups,
                args.validate_transforms_only,
                args.test_handlers_only,
                args.validate_config,
                args.list_transforms,
                # Add plugin modes to the check
                args.list_plugins,
                args.plugin_info,
                args.validate_plugin,
                args.validate_plugin_comprehensive,
                args.discover_plugins,
                getattr(args, "list_experiments", False),
                getattr(args, "validate_experiment", None),
                getattr(args, "run_experiment", None),
                getattr(args, "predict", False),  # Phase 5b: prediction mode
            ]
        ):
            args.process = True

        # Initialize plugin paths BEFORE any discovery operations
        # This ensures paths are registered when discover_plugins executes
        if hasattr(args, "plugin_path") and args.plugin_path:
            if PLUGIN_SYSTEM_AVAILABLE:
                for path in args.plugin_path:
                    try:
                        plugin_path = Path(path)
                        if plugin_path.exists():
                            PluginRegistry.add_plugin_path(plugin_path)
                            self.logger.info(f"Registered plugin path: {plugin_path}")
                        else:
                            self.logger.warning(f"Plugin path does not exist: {plugin_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to add plugin path {path}: {e}")
            else:
                self.logger.warning("Plugin system not available - ignoring --plugin-path")

        return args

    def _validate_arguments(self, args: argparse.Namespace) -> None:
        """
        Validate argument values and combinations.

        Args:
            args: Parsed arguments

        Raises:
            CLIValidationError: If validation fails
        """
        # Validate chunk size
        if args.chunk_size < 100 or args.chunk_size > 50000:
            raise CLIValidationError(
                f"Invalid chunk size: {args.chunk_size}. Must be between 100 and 50000."
            )

        # Validate test limit
        if args.test_limit is not None and args.test_limit < 1:
            raise CLIValidationError(f"Invalid test limit: {args.test_limit}. Must be at least 1.")

        # Validate filter combinations
        if args.min_atoms is not None and args.max_atoms is not None:
            if args.min_atoms > args.max_atoms:
                raise CLIValidationError(
                    f"Invalid filter range: min_atoms ({args.min_atoms}) > "
                    f"max_atoms ({args.max_atoms})"
                )

        # Validate mutually exclusive filter options
        if args.no_filters and any([args.max_atoms, args.min_atoms, args.max_uncertainty]):
            active_filters = []
            if args.max_atoms:
                active_filters.append("--max-atoms")
            if args.min_atoms:
                active_filters.append("--min-atoms")
            if args.max_uncertainty:
                active_filters.append("--max-uncertainty")
            raise CLIValidationError(
                f"Cannot use --no-filters with specific filter overrides: {', '.join(active_filters)}\n"
                f"Either remove --no-filters to apply filters, or remove filter overrides."
            )

        # Validate path arguments
        if args.root_dir and not Path(args.root_dir).exists():
            self.logger.warning(f"Root directory does not exist: {args.root_dir}")

        if args.config and not Path(args.config).exists():
            # YAML Splitting Architecture: Config path can be file OR directory
            # Check if it's neither a file nor a directory
            raise CLIValidationError(
                f"Configuration path not found: {args.config}\n"
                f"Expected either:\n"
                f"  - A configuration file (e.g., config.yaml)\n"
                f"  - A configuration directory (e.g., configs/)"
            )
        # Validate plugin arguments
        if PLUGIN_SYSTEM_AVAILABLE:
            # Validate plugin names if specified
            if hasattr(args, "validate_plugin") and args.validate_plugin:
                # Will be validated when operation runs
                pass

            if hasattr(args, "enable_plugin") and args.enable_plugin:
                # Will be validated during enable operation
                pass

            # Check for conflicting plugin options
            if args.disable_plugin_system:
                if any(
                    [
                        args.list_plugins,
                        args.validate_plugin,
                        args.validate_plugin_comprehensive,
                        args.enable_plugin,
                        args.disable_plugin,
                    ]
                ):
                    raise CLIValidationError(
                        "Cannot use --disable-plugin-system with other plugin operations"
                    )

        # Validate skip-validation conflicts
        if args.skip_validation:
            conflicting_validation = []
            if args.validate_config:
                conflicting_validation.append("--validate-config")
            if args.validate_handlers:
                conflicting_validation.append("--validate-handlers")
            if args.validate_transforms_only:
                conflicting_validation.append("--validate-transforms-only")

            if conflicting_validation:
                raise CLIValidationError(
                    f"Cannot use --skip-validation with validation operations: {', '.join(conflicting_validation)}\n"
                    f"Remove --skip-validation to perform validation."
                )

        # Validate preprocessing arguments
        if hasattr(args, "preprocess") and args.preprocess:
            # Validate preprocessing configuration
            if hasattr(args, "preprocess_config") and args.preprocess_config:
                preprocess_config_path = Path(args.preprocess_config)
                if not preprocess_config_path.exists():
                    raise CLIValidationError(
                        f"Preprocessing config file not found: {args.preprocess_config}"
                    )

            # PHASE 7: Validate dataset type using registry
            if hasattr(args, "preprocess_dataset") and args.preprocess_dataset:
                if not _is_dataset_type_registered(args.preprocess_dataset):
                    available_types = _get_available_dataset_types()
                    raise CLIValidationError(
                        f"Invalid preprocessing dataset type: {args.preprocess_dataset}. "
                        f"Must be one of: {available_types}"
                    )

            # Validate input path if specified
            if hasattr(args, "preprocess_input") and args.preprocess_input:
                input_path = Path(args.preprocess_input)
                if not input_path.exists():
                    raise CLIValidationError(
                        f"Preprocessing input file not found: {args.preprocess_input}"
                    )
                # PHASE 7: Feature-based input format validation
                if _get_dataset_feature(args.preprocess_dataset, "requires_archive_input"):
                    expected_format = _get_dataset_input_format(args.preprocess_dataset)
                    if expected_format == "tar.gz":
                        if not (input_path.suffix == ".gz" and input_path.stem.endswith(".tar")):
                            raise CLIValidationError(
                                f"{args.preprocess_dataset} preprocessing requires .tar.gz file, got: {input_path.name}"
                            )
                    # Add more format checks as needed for future dataset types

            # Validate output path if specified
            if hasattr(args, "preprocess_output") and args.preprocess_output:
                output_path = Path(args.preprocess_output)
                # Check output directory is writable
                output_dir = output_path.parent
                if not output_dir.exists():
                    try:
                        output_dir.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        raise CLIValidationError(
                            f"Cannot create output directory: {output_dir}. Error: {e}"
                        )

                # Check if output exists and force flag not set
                if output_path.exists() and not getattr(args, "preprocess_force", False):
                    self.logger.warning(
                        f"Output file already exists: {output_path}. "
                        f"Use --preprocess-force to overwrite."
                    )

            # Validate num_molecules if specified
            if hasattr(args, "preprocess_num_molecules") and args.preprocess_num_molecules:
                if args.preprocess_num_molecules < 1:
                    raise CLIValidationError(
                        f"preprocess_num_molecules must be >= 1, got: {args.preprocess_num_molecules}"
                    )

            # Validate feature_tier if specified
            if hasattr(args, "preprocess_feature_tier") and args.preprocess_feature_tier:
                valid_tiers = ["basic", "standard", "complete"]
                if args.preprocess_feature_tier not in valid_tiers:
                    raise CLIValidationError(
                        f"Invalid feature tier: {args.preprocess_feature_tier}. "
                        f"Must be one of: {valid_tiers}"
                    )

        # Validate preprocessing-only modes
        if hasattr(args, "validate_preprocessing_only") and args.validate_preprocessing_only:
            # Need either preprocess_config or preprocess_dataset
            if not (hasattr(args, "preprocess_config") and args.preprocess_config) and not (
                hasattr(args, "preprocess_dataset") and args.preprocess_dataset
            ):
                raise CLIValidationError(
                    "Preprocessing validation requires --preprocess-config or --preprocess-dataset"
                )

        # Note: test_preprocessor_only and list_preprocessors do NOT require --preprocess-dataset
        # They list ALL available preprocessors, not a specific one

        # =====================================================================
        # PHASE 5b: Validate prediction arguments (basic validation only)
        # =====================================================================
        # NOTE: Path existence validation is deferred to _apply_cli_overrides()
        # where config (and thus working_root_dir) is available for proper
        # path resolution. This matches the pattern used in main.py.
        # =====================================================================
        if getattr(args, "predict", False):
            # --model-path is required in predict mode
            if not getattr(args, "model_path", None):
                raise CLIValidationError(
                    "--model-path is required when using --predict mode.\n"
                    "Example: python main.py --predict --model-path ./model.pt --test-path ./data.csv"
                )

            # --test-path is required in predict mode
            if not getattr(args, "test_path", None):
                raise CLIValidationError(
                    "--test-path is required when using --predict mode.\n"
                    "Example: python main.py --predict --model-path ./model.pt --test-path ./data.csv"
                )

            # Validate predict_batch_size
            predict_batch_size = getattr(args, "predict_batch_size", 32)
            if predict_batch_size < 1:
                raise CLIValidationError(
                    f"Invalid predict-batch-size: {predict_batch_size}. Must be >= 1."
                )

            # Validate predict_num_samples if specified
            predict_num_samples = getattr(args, "predict_num_samples", None)
            if predict_num_samples is not None and predict_num_samples < 1:
                raise CLIValidationError(
                    f"Invalid predict-num-samples: {predict_num_samples}. Must be >= 1."
                )

            # Check for conflicting modes
            if getattr(args, "train", False):
                raise CLIValidationError(
                    "Cannot use --predict and --train together.\n"
                    "Use --predict for inference or --train for training."
                )

            self.logger.debug("Prediction arguments (basic) validated successfully")

    def load_and_merge_config(self, args: argparse.Namespace) -> dict[str, Any]:
        """
        Load configuration and merge with CLI arguments.

        CLI arguments take precedence over config file values.

        Args:
            args: Parsed CLI arguments

        Returns:
            Merged configuration dictionary

        Raises:
            CLIValidationError: If configuration loading fails
        """
        try:
            # Load base configuration
            self.config = load_config(args.config)

            # Apply CLI overrides
            self._apply_cli_overrides(args)

            return self.config

        except Exception as e:
            raise CLIValidationError(f"Failed to load configuration: {e}")

    def _apply_cli_overrides(self, args: argparse.Namespace) -> None:
        """
        Apply CLI argument overrides to configuration.

        Args:
            args: Parsed CLI arguments
        """
        # Override root directory
        if args.root_dir:
            self.config["dataset_root_dir"] = args.root_dir

        # Override filters
        if args.no_filters:
            self.config["filters"] = {}
        else:
            if args.max_atoms is not None:
                self.config.setdefault("filters", {})["max_atoms"] = args.max_atoms
            if args.min_atoms is not None:
                self.config.setdefault("filters", {})["min_atoms"] = args.min_atoms

        # Override test limit
        if args.test_limit is not None:
            self.config.setdefault("processing", {})["test_molecule_limit"] = args.test_limit

        # Override experimental setup
        if args.experimental_setup:
            if "transformations" not in self.config:
                self.config["transformations"] = {}
            self.config["transformations"]["default_setup"] = args.experimental_setup

        # Apply preprocessing CLI overrides
        if hasattr(args, "preprocess") and args.preprocess:
            # Initialize preprocessing config if not present
            if "preprocessing" not in self.config:
                self.config["preprocessing"] = {}

            # Override dataset type
            if hasattr(args, "preprocess_dataset") and args.preprocess_dataset:
                self.config["preprocessing"]["dataset_type"] = args.preprocess_dataset
                self.logger.debug(
                    f"CLI override: preprocessing.dataset_type = {args.preprocess_dataset}"
                )

            # Override input path
            if hasattr(args, "preprocess_input") and args.preprocess_input:
                self.config["preprocessing"]["input_path"] = args.preprocess_input
                self.logger.debug(
                    f"CLI override: preprocessing.input_path = {args.preprocess_input}"
                )

            # Override output path
            if hasattr(args, "preprocess_output") and args.preprocess_output:
                self.config["preprocessing"]["output_path"] = args.preprocess_output
                self.logger.debug(
                    f"CLI override: preprocessing.output_path = {args.preprocess_output}"
                )

            # Override num_molecules
            if hasattr(args, "preprocess_num_molecules") and args.preprocess_num_molecules:
                self.config["preprocessing"]["num_molecules"] = args.preprocess_num_molecules
                self.logger.debug(
                    f"CLI override: preprocessing.num_molecules = {args.preprocess_num_molecules}"
                )

            # Override feature_tier
            if hasattr(args, "preprocess_feature_tier") and args.preprocess_feature_tier:
                self.config["preprocessing"]["feature_tier"] = args.preprocess_feature_tier
                self.logger.debug(
                    f"CLI override: preprocessing.feature_tier = {args.preprocess_feature_tier}"
                )

            # Override operational flags
            if hasattr(args, "preprocess_force") and args.preprocess_force:
                self.config["preprocessing"]["force_overwrite"] = True
                self.logger.debug("CLI override: preprocessing.force_overwrite = True")

            if hasattr(args, "preprocess_cleanup"):
                self.config["preprocessing"]["cleanup_temp"] = args.preprocess_cleanup
                self.logger.debug(
                    f"CLI override: preprocessing.cleanup_temp = {args.preprocess_cleanup}"
                )

            if hasattr(args, "preprocess_progress") and args.preprocess_progress:
                self.config["preprocessing"]["show_progress"] = True
                self.logger.debug("CLI override: preprocessing.show_progress = True")

        # Training CLI Overrides
        if getattr(args, "train", False):
            if "models" not in self.config:
                self.config["models"] = {}

            # Warn if models not enabled in config
            if not self.config["models"].get("enabled", False):
                self.logger.warning(
                    "models.enabled is False in config.yaml, and the user is specified --train. "
                    "Enabling models for this run."
                )
                self.config["models"]["enabled"] = True

            # Override epochs
            if getattr(args, "epochs", None) is not None:
                self.config["models"].setdefault("training", {})["epochs"] = args.epochs
                self.logger.debug(f"CLI override: models.training.epochs = {args.epochs}")

            # Override batch size
            if getattr(args, "batch_size", None) is not None:
                self.config["models"].setdefault("training", {})["batch_size"] = args.batch_size
                self.logger.debug(f"CLI override: models.training.batch_size = {args.batch_size}")

            # Override learning rate
            if getattr(args, "learning_rate", None) is not None:
                self.config["models"].setdefault("training", {}).setdefault(
                    "optimizer", {}
                ).setdefault("params", {})["lr"] = args.learning_rate
                self.logger.debug(
                    f"CLI override: models.training.optimizer.params.lr = {args.learning_rate}"
                )

            # Override model name
            if getattr(args, "model_name", None) is not None:
                self.config["models"].setdefault("selection", {})["model_name"] = args.model_name
                self.logger.debug(f"CLI override: models.selection.model_name = {args.model_name}")

            # Override task type
            if getattr(args, "task_type", None) is not None:
                self.config["models"].setdefault("selection", {})["task_type"] = args.task_type
                self.logger.debug(f"CLI override: models.selection.task_type = {args.task_type}")

            # Override mode
            if getattr(args, "mode", None) is not None:
                self.config["models"].setdefault("selection", {})["mode"] = args.mode
                self.logger.debug(f"CLI override: models.selection.mode = {args.mode}")
        # -------

        # HPO CLI Overrides (--hpo / --no-hpo)
        # None = not passed (use config), True = --hpo, False = --no-hpo
        cli_hpo = getattr(args, "hpo", None)
        if cli_hpo is not None:
            if "models" not in self.config:
                self.config["models"] = {}
            self.config["models"].setdefault("hpo", {})["enabled"] = cli_hpo
            self.logger.debug(f"CLI override: models.hpo.enabled = {cli_hpo}")

            if getattr(args, "n_trials", None) is not None:
                self.config["models"]["hpo"]["n_trials"] = args.n_trials
                self.logger.debug(f"CLI override: models.hpo.n_trials = {args.n_trials}")

            if getattr(args, "hpo_timeout", None) is not None:
                self.config["models"]["hpo"]["timeout"] = args.hpo_timeout
                self.logger.debug(f"CLI override: models.hpo.timeout = {args.hpo_timeout}")

            if getattr(args, "cv_folds", None) is not None:
                self.config["models"]["hpo"]["cv_folds"] = args.cv_folds
                self.logger.debug(f"CLI override: models.hpo.cv_folds = {args.cv_folds}")

            if getattr(args, "sampler", None) is not None:
                self.config["models"]["hpo"].setdefault("sampler", {})["type"] = args.sampler
                self.logger.debug(f"CLI override: models.hpo.sampler.type = {args.sampler}")

            if getattr(args, "pruner", None) is not None:
                self.config["models"]["hpo"].setdefault("pruner", {})["type"] = args.pruner
                self.logger.debug(f"CLI override: models.hpo.pruner.type = {args.pruner}")

                # -------

        # =====================================================================
        # PHASE 5b: Validate prediction paths with working_root_dir resolution
        # =====================================================================
        # Path validation is done here (after config is loaded) so that
        # working_root_dir can be used for proper path resolution.
        # This matches the pattern used in main.py handle_predict_mode().
        # =====================================================================
        if getattr(args, "predict", False):
            # Get working_root_dir from config (same logic as main.py)
            working_root_dir = self._get_working_root_dir_for_validation()

            # Resolve and validate model path
            model_path = Path(args.model_path).expanduser()
            if not model_path.is_absolute():
                # Check in checkpoint directory first (same logic as main.py)
                checkpoint_dir = working_root_dir / "checkpoints"
                candidate = checkpoint_dir / model_path.name
                if candidate.exists():
                    model_path = candidate
                else:
                    model_path = working_root_dir / model_path

            if not model_path.exists():
                raise CLIValidationError(
                    f"Model checkpoint not found: {args.model_path}\n"
                    f"Searched locations:\n"
                    f"  1. {Path(args.model_path).expanduser()} (as provided)\n"
                    f"  2. {working_root_dir / 'checkpoints' / Path(args.model_path).name} (checkpoints dir)\n"
                    f"  3. {working_root_dir / args.model_path} (working_root_dir relative)\n"
                    f"Ensure the path to your trained model checkpoint is correct."
                )

            # Validate model path has .pt extension
            if model_path.suffix != ".pt":
                self.logger.warning(
                    f"Model path does not have .pt extension: {model_path}. "
                    "Continuing, but this may not be a valid checkpoint."
                )

            self.logger.debug(f"Resolved model path: {model_path}")

            # Resolve and validate test path
            test_path = Path(args.test_path).expanduser()
            if not test_path.is_absolute():
                test_path = working_root_dir / test_path

            if not test_path.exists():
                raise CLIValidationError(
                    f"Test path not found: {args.test_path}\n"
                    f"Searched locations:\n"
                    f"  1. {Path(args.test_path).expanduser()} (as provided)\n"
                    f"  2. {working_root_dir / args.test_path} (working_root_dir relative)\n"
                    f"Provide a valid path to: molecular file (CSV, XYZ, SDF), "
                    f"processed dataset (.pt), or miliaDataset directory."
                )

            self.logger.debug(f"Resolved test path: {test_path}")

            # Resolve and validate predictions output path
            preds_path_arg = getattr(args, "preds_path", None) or "./predictions.csv"
            preds_path = Path(preds_path_arg).expanduser()
            if not preds_path.is_absolute():
                preds_path = working_root_dir / preds_path

            preds_dir = preds_path.parent
            if not preds_dir.exists():
                try:
                    preds_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.debug(f"Created predictions output directory: {preds_dir}")
                except Exception as e:
                    raise CLIValidationError(
                        f"Cannot create predictions output directory: {preds_dir}. Error: {e}"
                    )

            self.logger.debug(f"Resolved predictions path: {preds_path}")
            self.logger.debug("Prediction path validation completed successfully")

    def _get_working_root_dir_for_validation(self) -> Path:
        """
        Get the working root directory from loaded config for path validation.

        This mirrors the logic in main.py _get_working_root_dir() to ensure
        consistent path resolution between CLI validation and main execution.

        Priority:
        1. config['global_paths']['working_root_dir']
        2. get_dataset_constants() root directory
        3. Current directory fallback

        Returns:
            Path to working root directory

        Note:
            Must be called after config is loaded (in _apply_cli_overrides or later).
        """
        working_root_dir = None

        # Priority 1: From loaded config
        if self.config is not None:
            global_paths = self.config.get("global_paths", {})
            working_root_dir_str = global_paths.get("working_root_dir")
            if working_root_dir_str:
                working_root_dir = Path(working_root_dir_str).expanduser()
                self.logger.debug(f"Using working_root_dir from config: {working_root_dir}")

        # Priority 2: From get_dataset_constants (if available)
        if working_root_dir is None:
            try:
                from milia_pipeline.config.config_accessors import get_dataset_constants

                _, _, dataset_root_dir = get_dataset_constants()
                working_root_dir = Path(dataset_root_dir).expanduser()
                self.logger.debug(
                    f"Using working_root_dir from dataset constants: {working_root_dir}"
                )
            except Exception:
                pass

        # Priority 3: Ultimate fallback to current directory
        if working_root_dir is None:
            working_root_dir = Path(".").resolve()
            self.logger.debug(f"Using current directory as working_root_dir: {working_root_dir}")

        return working_root_dir

    def validate_configuration(self, args: argparse.Namespace) -> bool:
        """
        Validate loaded configuration using YAML schema validator.

        Args:
            args: Parsed CLI arguments

        Returns:
            True if validation passes

        Raises:
            CLIValidationError: If validation fails
        """
        if args.skip_validation:
            self.logger.debug("Configuration validation skipped by user request")
            return True

        if not CONFIG_VALIDATION_AVAILABLE:
            # Schema validation not available - this is normal if config_schemas isn't fully set up
            # Basic validation still happens via config_loader and config_accessors
            self.logger.debug("YAML schema validation not available - using basic validation")
            return True

        try:
            # Both YAMLSchemaValidator and ValidationConfig available here
            validator = YAMLSchemaValidator()

            # Create lenient validation config for CLI usage
            validation_config = ValidationConfig(
                strict_mode=False,
                warn_on_unknown=True,
                require_descriptions=False,
                check_parameter_types=False,
                validate_research_context=False,
            )

            # Create lenient validation config for CLI usage
            validation_config = ValidationConfig(
                strict_mode=False,
                warn_on_unknown=True,
                require_descriptions=False,
                check_parameter_types=False,
                validate_research_context=False,
            )

            # Validate the configuration
            validation_result = validator.validate_config(self.config, validation_config)

            # Check validation result
            if not validation_result["valid"]:
                errors = validation_result.get("errors", [])
                if errors:
                    error_summary = "; ".join(errors)
                    raise CLIValidationError(f"Configuration validation failed: {error_summary}")

            # Log warnings if present (but don't fail)
            warnings = validation_result.get("warnings", [])
            if warnings:
                self.logger.debug(f"Configuration validation warnings: {len(warnings)}")
                for warning in warnings[:3]:  # Show first 3
                    self.logger.debug(f"  ℹ️  {warning}")

            # Log suggestions if present
            suggestions = validation_result.get("suggestions", [])
            if suggestions:
                for suggestion in suggestions[:2]:  # Show first 2
                    self.logger.debug(f"  💡 {suggestion}")

            # Success - only log at debug level to reduce noise
            self.logger.debug("✅ Configuration validation passed")
            return True

        except CLIValidationError:
            # Re-raise CLI validation errors
            raise

        except Exception as e:
            # Unexpected errors during validation
            self.logger.debug(f"Configuration validation encountered error: {e}")
            # Don't fail the entire process for validation issues
            # The system has multiple validation layers
            self.logger.debug("Continuing with basic configuration validation")
            return True

    def validate_descriptor_config(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate descriptor configuration.

        Phase 3 Integration: Validates descriptor configuration including
        selection mode, categories, plugin setup, and handler compatibility.

        Args:
            config: Loaded configuration dictionary

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check if descriptors section exists
        if "descriptors" not in config:
            return True, []  # Not enabled, no validation needed

        desc_config = config["descriptors"]

        # Validate enabled flag
        if not isinstance(desc_config.get("enabled", False), bool):
            issues.append("descriptors.enabled must be a boolean")

        if not desc_config.get("enabled", False):
            return True, []  # Disabled, no further validation needed

        # Validate selection mode
        selection_mode = desc_config.get("selection_mode", "explicit")
        if selection_mode not in ["explicit", "category", "all"]:
            issues.append(
                f"Invalid selection_mode '{selection_mode}'. "
                f"Must be 'explicit', 'category', or 'all'"
            )

        # Validate category mode
        if selection_mode == "category":
            categories = desc_config.get("selected_categories", [])
            if not categories:
                issues.append("selection_mode is 'category' but no categories specified")

            valid_categories = [
                "constitutional",
                "topological",
                "electronic",
                "geometric",
                "drug_likeness",
                "fragments",
            ]
            for cat in categories:
                if cat not in valid_categories:
                    issues.append(f"Invalid category '{cat}'")

        # Validate explicit mode
        elif selection_mode == "explicit":
            selected = desc_config.get("selected_descriptors", {})
            if not selected:
                issues.append("selection_mode is 'explicit' but no descriptors specified")

        # Validate plugin configuration
        if desc_config.get("plugins", {}).get("enabled", False):
            plugin_paths = desc_config.get("plugins", {}).get("plugin_paths", [])
            if not plugin_paths:
                issues.append("Descriptor plugins enabled but no plugin_paths specified")

        # Validate computation settings
        computation = desc_config.get("computation", {})
        batch_size = computation.get("batch_size", 100)
        if batch_size < 1 or batch_size > 10000:
            issues.append(f"Invalid batch_size {batch_size}. Must be between 1 and 10000")

        is_valid = len(issues) == 0
        return is_valid, issues

    def get_registry_integration_status(self) -> dict[str, Any]:
        """
        Get registry integration status for CLI diagnostics.

        PHASE 7: Provides information about registry availability and
        current dataset type support for CLI operations.

        Returns:
            Dict with registry status information including:
            - registry_available: Whether registry is accessible
            - available_dataset_types: List of registered dataset types
            - using_legacy_fallback: Whether fallback is active
            - phase_7_integration: Confirmation of Phase 7 completion
        """
        return get_cli_registry_status()

    def run_interactive_mode(self) -> argparse.Namespace:
        """
        Run interactive configuration wizard.

        Returns:
            Namespace with user-selected options
        """
        print("\n" + "=" * 60)
        print("milia Dataset Processing - Interactive Configuration")
        print("=" * 60 + "\n")

        # Create args namespace
        args = argparse.Namespace()

        # Basic options
        print("Basic Configuration:")
        args.root_dir = self._prompt_path("Dataset root directory", default=None, must_exist=False)

        args.force_reload = self._prompt_yes_no("Force reprocessing?", default=False)

        args.chunk_size = self._prompt_int(
            "Chunk size", default=5000, min_value=100, max_value=50000
        )

        # Processing mode
        print("\nProcessing Mode:")
        mode_options = ["Full processing", "Quick validation", "Statistics only"]
        mode_choice = self._prompt_choice("Select processing mode", mode_options, default=0)

        args.process = mode_choice == 0
        args.quick_validation = mode_choice == 1
        args.stats_only = mode_choice == 2

        # Transformation system
        if TRANSFORMS_AVAILABLE:
            print("\nTransformation System:")
            args.disable_transforms = not self._prompt_yes_no(
                "Enable transformation system?", default=True
            )

            if not args.disable_transforms:
                setups = self._get_available_setups()
                if setups:
                    setup_choice = self._prompt_choice(
                        "Select experimental setup", setups + ["Use default"], default=len(setups)
                    )
                    args.experimental_setup = (
                        setups[setup_choice] if setup_choice < len(setups) else None
                    )
                else:
                    args.experimental_setup = None
        else:
            args.disable_transforms = True
            args.experimental_setup = None

        # Plugin system
        if PLUGIN_SYSTEM_AVAILABLE:
            print("\nPlugin System:")
            use_plugins = self._prompt_yes_no("Enable custom plugin system?", default=True)

            if use_plugins:
                add_plugin_path = self._prompt_yes_no("Add custom plugin paths?", default=False)

                plugin_paths = []
                if add_plugin_path:
                    while True:
                        path = self._prompt_path(
                            "Plugin directory path (empty to finish)", default=None, must_exist=True
                        )
                        if not path:
                            break
                        plugin_paths.append(path)

                        more = self._prompt_yes_no("Add another path?", default=False)
                        if not more:
                            break

                args.plugin_path = plugin_paths if plugin_paths else None

                args.discover_plugins = self._prompt_yes_no("Discover plugins now?", default=False)

                args.auto_validate = (
                    self._prompt_yes_no("Auto-validate discovered plugins?", default=True)
                    if args.discover_plugins
                    else False
                )

                args.disable_plugin_system = False
            else:
                args.disable_plugin_system = True
                args.plugin_path = None
                args.discover_plugins = False
                args.auto_validate = False
        else:
            args.disable_plugin_system = True
            args.plugin_path = None
            args.discover_plugins = False
            args.auto_validate = False

        # Set other plugin defaults
        args.list_plugins = False
        args.plugin_info = None
        args.validate_plugin = None
        args.validate_plugin_comprehensive = None
        args.enable_plugin = None
        args.disable_plugin = None
        args.trust_plugin = None
        args.validated_only = False
        args.enabled_only = False
        args.run_performance_tests = False

        # Filters
        print("\nMolecule Filters:")
        use_filters = self._prompt_yes_no("Configure molecule filters?", default=True)

        if use_filters:
            args.max_atoms = (
                self._prompt_int(
                    "Maximum atoms per molecule (0 for no limit)", default=0, min_value=0
                )
                or None
            )

            args.min_atoms = (
                self._prompt_int(
                    "Minimum atoms per molecule (0 for no limit)", default=0, min_value=0
                )
                or None
            )
        else:
            args.no_filters = True
            args.max_atoms = None
            args.min_atoms = None

        # Logging
        print("\nLogging Configuration:")
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        log_choice = self._prompt_choice("Select log level", log_levels, default=1)
        args.log_level = log_levels[log_choice]

        save_log = self._prompt_yes_no("Save log to file?", default=False)

        if save_log:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_log = f"milia_{timestamp}.log"
            args.log_file = self._prompt_string("Log file path", default=default_log)
        else:
            args.log_file = None

        # Set remaining defaults
        # YAML Splitting Architecture: Use dynamic config path detection
        args.config = _get_default_config_path()
        args.interactive = False
        args.verbose = args.log_level == "DEBUG"
        args.quiet = False
        args.skip_validation = False

        print("\n" + "=" * 60)
        print("Configuration complete!")
        print("=" * 60 + "\n")

        return args

    def _prompt_string(self, prompt: str, default: str | None = None) -> str:
        """Prompt user for string input."""
        if default:
            prompt_text = f"{prompt} [{default}]: "
        else:
            prompt_text = f"{prompt}: "

        value = input(prompt_text).strip()
        return value if value else default

    def _prompt_int(
        self,
        prompt: str,
        default: int = 0,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> int:
        """Prompt user for integer input."""
        while True:
            try:
                value = input(f"{prompt} [{default}]: ").strip()
                result = int(value) if value else default

                if min_value is not None and result < min_value:
                    print(f"Value must be >= {min_value}")
                    continue
                if max_value is not None and result > max_value:
                    print(f"Value must be <= {max_value}")
                    continue

                return result
            except ValueError:
                print("Invalid integer value")

    def _prompt_yes_no(self, prompt: str, default: bool = True) -> bool:
        """Prompt user for yes/no input."""
        default_str = "Y/n" if default else "y/N"
        while True:
            value = input(f"{prompt} [{default_str}]: ").strip().lower()
            if not value:
                return default
            if value in ["y", "yes"]:
                return True
            if value in ["n", "no"]:
                return False
            print("Please enter 'y' or 'n'")

    def _prompt_choice(self, prompt: str, options: list[str], default: int = 0) -> int:
        """Prompt user to select from options."""
        print(f"\n{prompt}:")
        for i, option in enumerate(options):
            marker = "*" if i == default else " "
            print(f"  {marker} {i + 1}. {option}")

        while True:
            try:
                value = input(f"Select option [1-{len(options)}] [{default + 1}]: ").strip()
                choice = int(value) - 1 if value else default

                if 0 <= choice < len(options):
                    return choice
                print(f"Please enter a number between 1 and {len(options)}")
            except ValueError:
                print("Invalid selection")

    def _prompt_path(
        self, prompt: str, default: str | None = None, must_exist: bool = False
    ) -> str:
        """Prompt user for file/directory path."""
        while True:
            value = self._prompt_string(prompt, default)

            if not value:
                if default:
                    return default
                print("Path is required")
                continue

            path = Path(value)

            if must_exist and not path.exists():
                print(f"Path does not exist: {value}")
                retry = self._prompt_yes_no("Try again?", True)
                if not retry:
                    return value
                continue

            return value

    def _get_available_setups(self) -> list[str]:
        """Get list of available experimental setups."""
        try:
            return list_experimental_setups()
        except Exception:
            return []

    def print_configuration_summary(self, args: argparse.Namespace) -> None:
        """
        Print summary of active configuration.

        Args:
            args: Parsed arguments
        """
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)

        # Processing mode - use safe attribute access
        mode = "Special Mode"  # default
        if getattr(args, "process", False):
            mode = "Full Processing"
        elif getattr(args, "quick_validation", False):
            mode = "Quick Validation"
        elif getattr(args, "stats_only", False):
            mode = "Statistics Only"
        elif getattr(args, "interactive", False):
            mode = "Interactive Mode"

        print(f"Mode: {mode}")

        # Basic settings - safe access
        root_dir = getattr(args, "root_dir", None)
        if root_dir:
            print(f"Root Directory: {root_dir}")

        force_reload = getattr(args, "force_reload", False)
        print(f"Force Reload: {force_reload}")

        chunk_size = getattr(args, "chunk_size", 5000)
        print(f"Chunk Size: {chunk_size}")

        # Transformation system - safe access
        disable_transforms = getattr(args, "disable_transforms", False)
        experimental_setup = getattr(args, "experimental_setup", None)

        if not disable_transforms and TRANSFORMS_AVAILABLE:
            setup = experimental_setup or "default"
            print(f"Transformation System: ✅ Enabled (setup: {setup})")
        else:
            print("Transformation System: ❌ Disabled")

        # Handlers - safe access
        disable_handlers = getattr(args, "disable_handlers", False)
        if not disable_handlers:
            print("Handler Pattern: ✅ Enabled")
        else:
            print("Handler Pattern: ❌ Disabled")

        # Filters - safe access with hasattr
        filters_active = []
        if hasattr(args, "max_atoms") and args.max_atoms:
            filters_active.append(f"max_atoms={args.max_atoms}")
        if hasattr(args, "min_atoms") and args.min_atoms:
            filters_active.append(f"min_atoms={args.min_atoms}")
        if hasattr(args, "max_uncertainty") and args.max_uncertainty:
            filters_active.append(f"max_uncertainty={args.max_uncertainty}")

        no_filters = getattr(args, "no_filters", False)
        if filters_active:
            print(f"Filters: {', '.join(filters_active)}")
        elif no_filters:
            print("Filters: None (disabled)")
        else:
            print("Filters: From config.yaml")

        # Logging - safe access
        log_level = getattr(args, "log_level", "INFO")
        print(f"Log Level: {log_level}")

        log_file = getattr(args, "log_file", None)
        if log_file:
            print(f"Log File: {log_file}")

        # Plugin system - safe access
        disable_plugin_system = getattr(args, "disable_plugin_system", False)

        if PLUGIN_SYSTEM_AVAILABLE and not disable_plugin_system:
            print("Plugin System: ✓ Enabled")

            plugin_path = getattr(args, "plugin_path", None)
            if plugin_path:
                print(f"  Plugin Paths: {len(plugin_path)} configured")

            discover_plugins = getattr(args, "discover_plugins", False)
            if discover_plugins:
                print("  Auto-discovery: ✓ Enabled")

                auto_validate = getattr(args, "auto_validate", False)
                if auto_validate:
                    print("  Auto-validation: ✓ Enabled")
        else:
            print("Plugin System: ○ Disabled")

        run_experiment = getattr(args, "run_experiment", None)
        validate_experiment = getattr(args, "validate_experiment", None)
        list_experiments = getattr(args, "list_experiments", False)

        if run_experiment or validate_experiment or list_experiments:
            print("Research API: ✅ Active")
            if run_experiment:
                print(f"  Running: {run_experiment}")
            if validate_experiment:
                print(f"  Validating: {validate_experiment}")
            if list_experiments:
                print("  Listing experiments")

        print("=" * 60 + "\n")

        # Preprocessing Configuration
        if "preprocessing" in self.config:
            preprocessing = self.config["preprocessing"]
            self.logger.info("\nPreprocessing Configuration:")
            self.logger.info("=" * 60)

            if "dataset_type" in preprocessing:
                self.logger.info(f"  Dataset type: {preprocessing['dataset_type']}")

            if "input_path" in preprocessing:
                self.logger.info(f"  Input path: {preprocessing['input_path']}")

            if "output_path" in preprocessing:
                self.logger.info(f"  Output path: {preprocessing['output_path']}")

            if "num_molecules" in preprocessing:
                self.logger.info(f"  Molecules to process: {preprocessing['num_molecules']}")
            else:
                self.logger.info("  Molecules to process: ALL")

            if "feature_tier" in preprocessing:
                self.logger.info(f"  Feature tier: {preprocessing['feature_tier']}")

            if "force_overwrite" in preprocessing:
                self.logger.info(f"  Force overwrite: {preprocessing['force_overwrite']}")

            if "cleanup_temp" in preprocessing:
                self.logger.info(f"  Cleanup temp files: {preprocessing['cleanup_temp']}")

            if "show_progress" in preprocessing:
                self.logger.info(f"  Show progress: {preprocessing['show_progress']}")

    def handle_plugin_operations(self, args: argparse.Namespace) -> bool:
        """
        Handle plugin-specific CLI operations.

        Args:
            args: Parsed CLI arguments

        Returns:
            True if a plugin operation was executed (should exit), False otherwise

        Raises:
            CLIValidationError: If plugin operations fail
        """
        if not PLUGIN_SYSTEM_AVAILABLE:
            if any(
                [
                    args.list_plugins,
                    args.plugin_info,
                    args.validate_plugin,
                    args.validate_plugin_comprehensive,
                    args.discover_plugins,
                ]
            ):
                raise CLIValidationError(
                    "Plugin system not available. Install required dependencies."
                )
            return False

        # Discover plugins if requested
        if args.discover_plugins:
            self._discover_plugins_operation(args)
            return True

        # List plugins
        if args.list_plugins:
            self._list_plugins_operation(args)
            return True

        # Show plugin info
        if args.plugin_info:
            self._show_plugin_info_operation(args.plugin_info)
            return True

        # Validate plugin (basic)
        if args.validate_plugin:
            self._validate_plugin_operation(args.validate_plugin, args)
            return True

        # Validate plugin (comprehensive)
        if args.validate_plugin_comprehensive:
            self._validate_plugin_comprehensive_operation(args.validate_plugin_comprehensive, args)
            return True

        # Enable/disable plugins (non-exit operations)
        if args.enable_plugin:
            for plugin_name in args.enable_plugin:
                self._enable_plugin_operation(plugin_name)

        if args.disable_plugin:
            for plugin_name in args.disable_plugin:
                self._disable_plugin_operation(plugin_name)

        if args.trust_plugin:
            for plugin_name in args.trust_plugin:
                self._trust_plugin_operation(plugin_name)

        return False

    def handle_research_api_commands(self, args: argparse.Namespace) -> bool:
        """
        Handle research API commands.

        Research workflow integration for systematic experimentation.
        Supports ablation studies, parameter sweeps, and comparative analyses.

        Args:
            args: Parsed command-line arguments

        Returns:
            True if command was handled and program should exit
            False if no research command or should continue
        """
        # Import here to avoid circular dependencies and handle missing module gracefully
        try:
            from milia_pipeline.transformations.research_api import (
                get_experiment,
                list_available_experiments,
                load_experiments_from_config,
            )
        except ImportError as e:
            # Only error if user actually requested research API operations
            if any(
                [
                    getattr(args, "list_experiments", False),
                    getattr(args, "run_experiment", None),
                    getattr(args, "validate_experiment", None),
                ]
            ):
                self.logger.error(f"Research API not available: {e}")
                self.logger.error("Ensure research_api.py is properly installed")
                self.logger.info("Hint: Check milia_pipeline/transformations/research_api.py")
                return True
            return False

        # ========================================================================
        # COMMAND 1: List Available Experiments
        # ========================================================================
        if getattr(args, "list_experiments", False):
            print("\n" + "=" * 70)
            print("AVAILABLE EXPERIMENTS")
            print("=" * 70)

            try:
                # Determine config path
                config_path = None
                if hasattr(args, "experiment_config") and args.experiment_config:
                    config_path = Path(args.experiment_config)

                # Load experiments
                experiments = load_experiments_from_config(config_path)

                if not experiments:
                    print("\nNo experiments configured.")
                    print("\nTo add experiments:")
                    print("  1. Create research_experiments.yaml in project root, OR")
                    print("  2. Add 'experiments:' section to config.yaml")
                    print("\nSee Research API Operations documentation for format.")
                else:
                    # Display each experiment
                    for name, exp in experiments.items():
                        # Determine experiment type
                        exp_type = "Unknown"
                        num_variants = 0

                        if hasattr(exp, "ablations") and exp.ablations:
                            exp_type = "Ablation Study"
                            num_variants = len(exp.ablations)
                        elif hasattr(exp, "parameter_sweeps") and exp.parameter_sweeps:
                            exp_type = "Parameter Sweep"
                            num_variants = len(exp.parameter_sweeps)

                        # Display experiment info
                        print(f"\n📊 {name}")
                        print(f"   Type: {exp_type}")
                        print(f"   Description: {exp.description}")

                        # Show hypothesis if available
                        if hasattr(exp, "hypothesis") and exp.hypothesis:
                            hypothesis_preview = exp.hypothesis[:60]
                            if len(exp.hypothesis) > 60:
                                hypothesis_preview += "..."
                            print(f"   Hypothesis: {hypothesis_preview}")

                        # Show metrics
                        print(f"   Variants: {num_variants}")
                        print(f"   Runs per variant: {exp.num_runs}")

                        # Calculate total runs
                        if hasattr(exp, "get_total_runs"):
                            print(f"   Total runs: {exp.get_total_runs()}")

                    print(f"\n{'-' * 70}")
                    print(f"Total: {len(experiments)} experiment(s) configured")

            except Exception as e:
                print(f"\nError loading experiments: {e}")
                self.logger.error(f"Experiment loading failed: {e}", exc_info=True)
                return True

            print("=" * 70 + "\n")
            return True

        # ========================================================================
        # COMMAND 2: Validate Experiment Configuration
        # ========================================================================
        if hasattr(args, "validate_experiment") and args.validate_experiment:
            print("\n" + "=" * 70)
            print(f"VALIDATING EXPERIMENT: {args.validate_experiment}")
            print("=" * 70)

            try:
                # Determine config path
                config_path = None
                if hasattr(args, "experiment_config") and args.experiment_config:
                    config_path = Path(args.experiment_config)

                # Get experiment
                experiment = get_experiment(args.validate_experiment, config_path)

                # Display validation results
                print("\n✅ Experiment configuration is valid")
                print(f"\n   Name: {experiment.name}")
                print(f"   Description: {experiment.description}")

                # Determine experiment type and variant count
                if hasattr(experiment, "ablations") and experiment.ablations:
                    exp_type = "Ablation Study"
                    variants = experiment.ablations
                elif hasattr(experiment, "parameter_sweeps") and experiment.parameter_sweeps:
                    exp_type = "Parameter Sweep"
                    variants = experiment.parameter_sweeps
                else:
                    exp_type = "Unknown"
                    variants = []

                print(f"   Type: {exp_type}")
                print(f"   Variants: {len(variants)}")
                print(f"   Runs per variant: {experiment.num_runs}")

                # Calculate total runs
                if hasattr(experiment, "get_total_runs"):
                    print(f"   Total runs: {experiment.get_total_runs()}")

                # Show research metadata if available
                if hasattr(experiment, "hypothesis") and experiment.hypothesis:
                    print(f"\n   Hypothesis: {experiment.hypothesis}")

                if hasattr(experiment, "expected_outcome") and experiment.expected_outcome:
                    print(f"   Expected: {experiment.expected_outcome}")

                print("\n" + "=" * 70)

            except Exception as e:
                print(f"\n❌ Validation failed: {e}")
                self.logger.error(f"Validation error: {e}", exc_info=True)
                return True

            return True

        # ========================================================================
        # COMMAND 3: Run Experiment
        # ========================================================================
        if hasattr(args, "run_experiment") and args.run_experiment:
            print("\n" + "=" * 70)
            print(f"RUNNING EXPERIMENT: {args.run_experiment}")
            print("=" * 70)

            try:
                # Determine config path
                config_path = None
                if hasattr(args, "experiment_config") and args.experiment_config:
                    config_path = Path(args.experiment_config)

                # Load experiment configuration
                experiment = get_experiment(args.run_experiment, config_path)

                # Display experiment info
                print(f"\nExperiment: {experiment.name}")
                print(f"Description: {experiment.description}")

                if hasattr(experiment, "hypothesis") and experiment.hypothesis:
                    print(f"Hypothesis: {experiment.hypothesis}")

                # Setup output directory
                output_dir = Path(getattr(args, "experiment_output", "./experiments"))
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"\nOutput directory: {output_dir}")

                # NOTE: Full experiment execution requires dataset and model integration
                # This is a placeholder showing configuration validation
                print("\n⚠️  Full experiment execution requires dataset/model integration")
                print("Validating experiment configuration...")

                # Validate configuration
                if hasattr(experiment, "ablations") and experiment.ablations:
                    variants = experiment.ablations
                    exp_type = "ablation study"
                elif hasattr(experiment, "parameter_sweeps") and experiment.parameter_sweeps:
                    variants = experiment.parameter_sweeps
                    exp_type = "parameter sweep"
                else:
                    variants = []
                    exp_type = "experiment"

                print("\n✅ Experiment configuration valid")
                print(f"   Type: {exp_type}")
                print(f"   Variants to test: {len(variants)}")
                print(f"   Runs per variant: {experiment.num_runs}")
                print(f"   Total runs: {len(variants) * experiment.num_runs}")

                # Integration instructions
                print("\n" + "-" * 70)
                print("To integrate with your training pipeline:")
                print("1. Define dataset_loader function")
                print("2. Define model_trainer function")
                print("3. Define evaluator function")
                print("4. Call ExperimentRunner.run_experiment() with these functions")

                print("\nExample integration code:")
                print("""
from milia_pipeline.transformations.research_api import ExperimentRunner

runner = ExperimentRunner(experiment, output_dir)
results = runner.run_experiment(
    dataset_loader=your_dataset_loader,
    model_trainer=your_model_trainer,
    evaluator=your_evaluator,
    num_runs=args.num_runs or experiment.num_runs
)
""")

                print("\nSee research API documentation for complete examples")

            except Exception as e:
                print(f"\n❌ Error: {e}")
                self.logger.error(f"Experiment execution error: {e}", exc_info=True)
                return True

            print("\n" + "=" * 70)
            return True

        # No research API command executed
        return False

    def handle_descriptor_operations(self, args: argparse.Namespace) -> bool:
        """
        Handle descriptor-specific CLI operations.

        Phase 3 Integration: Handles descriptor listing, validation, and
        statistics display operations triggered by CLI flags.

        Args:
            args: Parsed command-line arguments

        Returns:
            True if operation was handled (should exit), False otherwise
        """
        # List descriptors
        if args.list_descriptors:
            self._list_available_descriptors()
            return True

        # Validate descriptor configuration
        if args.validate_descriptors:
            config = self.load_and_merge_config(args)
            is_valid, issues = self.validate_descriptor_config(config)

            if is_valid:
                self.logger.info("✓ Descriptor configuration is valid")
                print("\n✓ Descriptor configuration is VALID")
            else:
                self.logger.error("✗ Descriptor configuration has issues:")
                print("\n✗ Descriptor configuration has ISSUES:")
                for issue in issues:
                    self.logger.error(f"  - {issue}")
                    print(f"  - {issue}")

            return True

        return False

    def handle_training_operations(self, args: argparse.Namespace) -> bool:
        """
        Handle model training and HPO CLI operations.

        Phase 9 Integration: Handles training mode, HPO optimization,
        model evaluation, and checkpoint management.

        Args:
            args: Parsed command-line arguments

        Returns:
            True if operation was handled (should exit after main.py handles training),
            False if not a training operation
        """
        # Check if training mode is requested
        if not getattr(args, "train", False) and not getattr(args, "evaluate_only", False):
            return False

        # Determine effective mode from flags
        if getattr(args, "custom_architecture", False):
            args.mode = "custom"
        elif getattr(args, "ensemble", False):
            args.mode = "ensemble"

        # Log training configuration
        self.logger.info("Training mode detected - will be handled by main.py")
        self.logger.info(f"  Mode: {getattr(args, 'mode', 'single')}")
        self.logger.info(f"  HPO enabled: {getattr(args, 'hpo', False)}")

        if getattr(args, "hpo", False):
            self.logger.info(f"  N trials: {getattr(args, 'n_trials', 'from config')}")
            self.logger.info(f"  Backend: {getattr(args, 'hpo_backend', 'optuna')}")

        # Return True to signal training should be executed (by main.py)
        return True

    def _list_available_descriptors(self):
        """
        List all available descriptors organized by category.

        Displays descriptor registry contents for user reference.
        """
        try:
            from milia_pipeline.descriptors.descriptor_registry import DescriptorRegistry

            registry = DescriptorRegistry.get_instance()

            print("\n" + "=" * 70)
            print("AVAILABLE MOLECULAR DESCRIPTORS")
            print("=" * 70)

            categories = [
                "constitutional",
                "topological",
                "electronic",
                "geometric",
                "drug_likeness",
                "fragments",
            ]

            from milia_pipeline.descriptors.descriptor_categories import DescriptorCategory

            for category in categories:
                # Convert string to enum
                try:
                    category_enum = DescriptorCategory(category)
                    descriptors = registry.list_available_descriptors(category_enum)
                except ValueError:
                    descriptors = []
                print(f"\n{category.upper()} ({len(descriptors)} descriptors):")
                print("-" * 70)

                for i, desc in enumerate(sorted(descriptors), 1):
                    metadata = registry.get_metadata(desc)
                    requires = []
                    if metadata and metadata.requires_3d:
                        requires.append("3D")
                    if metadata and metadata.requires_charges:
                        requires.append("charges")

                    req_str = f" [requires: {', '.join(requires)}]" if requires else ""
                    print(f"  {i:3d}. {desc}{req_str}")

            # Plugin descriptors
            plugin_descriptors = registry.get_plugin_descriptors()
            if plugin_descriptors:
                print(f"\nPLUGIN DESCRIPTORS ({len(plugin_descriptors)} descriptors):")
                print("-" * 70)
                for i, (desc, plugin) in enumerate(sorted(plugin_descriptors.items()), 1):
                    print(f"  {i:3d}. {desc} (from plugin: {plugin})")

            print("\n" + "=" * 70)
            print(f"Total: {len(registry.list_all_descriptors())} descriptors available")
            print("=" * 70 + "\n")

        except ImportError as e:
            self.logger.error(f"Cannot list descriptors: {e}")
            print("\n✗ Error: Descriptor module not available")
            print("  Make sure Phase 1 implementation is complete")

    def _discover_plugins_operation(self, args: argparse.Namespace) -> None:
        """Execute plugin discovery operation."""
        print("\n" + "=" * 70)
        print("PLUGIN DISCOVERY")
        print("=" * 70 + "\n")

        # Show registered paths before discovery
        registered_paths = PluginRegistry.get_plugin_paths()  # Assumes this method exists
        if registered_paths:
            print(f"Searching in {len(registered_paths)} path(s):")
            for path in registered_paths:
                print(f"  • {path}")
            print()

        try:
            plugins = PluginRegistry.discover_plugins(auto_validate=args.auto_validate)

            print(f"Discovered {len(plugins)} plugin(s):\n")

            for plugin_name, metadata in plugins.items():
                status = "✓ Validated" if metadata.is_validated else "○ Not validated"
                print(f"  • {plugin_name} (v{metadata.version}) - {status}")
                print(f"    Transforms: {', '.join(metadata.transforms)}")
                print(f"    Author: {metadata.author}")
                if metadata.description:
                    print(f"    Description: {metadata.description}")
                print()

            if args.auto_validate:
                validated = sum(1 for m in plugins.values() if m.is_validated)
                print(f"Validation Summary: {validated}/{len(plugins)} passed")

        except Exception as e:
            # Handle plugin-specific exceptions if plugin system is available
            error_type = type(e).__name__

            if PLUGIN_SYSTEM_AVAILABLE and error_type == "PluginDiscoveryError":
                self.logger.error(f"Plugin discovery failed: {e}")
                raise CLIValidationError(f"Plugin discovery failed: {e}")
            else:
                self.logger.error(f"Unexpected error during discovery: {e}")
                raise CLIValidationError(f"Discovery error: {e}")

    def _list_plugins_operation(self, args: argparse.Namespace) -> None:
        """Execute list plugins operation."""
        print("\n" + "=" * 70)
        print("REGISTERED PLUGINS")
        print("=" * 70 + "\n")

        try:
            plugins = PluginRegistry.list_plugins(
                validated_only=args.validated_only, enabled_only=args.enabled_only
            )

            if not plugins:
                filter_desc = []
                if args.validated_only:
                    filter_desc.append("validated")
                if args.enabled_only:
                    filter_desc.append("enabled")
                filter_str = " and ".join(filter_desc) if filter_desc else ""
                print(f"No {filter_str} plugins found.".strip())
                return

            for plugin_name in plugins:
                info = PluginRegistry.get_plugin_info(plugin_name)
                enabled = PluginRegistry.is_plugin_enabled(plugin_name)

                status_parts = []
                if info.is_validated:
                    status_parts.append("✓ Validated")
                if enabled:
                    status_parts.append("✓ Enabled")
                else:
                    status_parts.append("○ Disabled")
                if info.trusted:
                    status_parts.append("🔒 Trusted")

                status = " | ".join(status_parts)

                print(f"📦 {plugin_name} (v{info.version})")
                print(f"   Status: {status}")
                print(f"   Transforms: {', '.join(info.transforms)}")
                print(f"   Author: {info.author}")
                if info.description:
                    print(f"   Description: {info.description}")
                print()

            print(f"Total: {len(plugins)} plugin(s)")

        except Exception as e:
            self.logger.error(f"Failed to list plugins: {e}")
            raise CLIValidationError(f"List operation failed: {e}")

    def _show_plugin_info_operation(self, plugin_name: str) -> None:
        """Execute show plugin info operation."""
        print("\n" + "=" * 70)
        print(f"PLUGIN INFO: {plugin_name}")
        print("=" * 70 + "\n")

        try:
            info = PluginRegistry.get_plugin_info(plugin_name)
            enabled = PluginRegistry.is_plugin_enabled(plugin_name)

            print(f"Name: {info.plugin_name}")
            print(f"Version: {info.version}")
            print(f"Author: {info.author}")
            if info.email:
                print(f"Email: {info.email}")
            if info.license:
                print(f"License: {info.license}")
            if info.homepage:
                print(f"Homepage: {info.homepage}")

            print("\nStatus:")
            print(f"  Validated: {'Yes' if info.is_validated else 'No'}")
            print(f"  Enabled: {'Yes' if enabled else 'No'}")
            print(f"  Trusted: {'Yes' if info.trusted else 'No'}")

            if info.validation_date:
                print(f"  Last Validated: {info.validation_date}")

            print(f"\nTransforms ({len(info.transforms)}):")
            for transform in info.transforms:
                print(f"  • {transform}")

            print("\nDependencies:")
            print(f"  milia: {info.milia_version}")
            print(f"  PyG: {info.pyg_version}")
            print(f"  Python: {info.python_version}")

            if info.dependencies:
                print("  Additional Packages:")
                for dep in info.dependencies:
                    print(f"    • {dep}")

            if info.description:
                print("\nDescription:")
                print(f"  {info.description}")

        except KeyError:
            raise CLIValidationError(f"Plugin '{plugin_name}' not found")
        except Exception as e:
            self.logger.error(f"Failed to get plugin info: {e}")
            raise CLIValidationError(f"Info operation failed: {e}")

    def _validate_plugin_operation(self, plugin_name: str, args: argparse.Namespace) -> None:
        """Execute basic plugin validation operation."""
        print("\n" + "=" * 70)
        print(f"VALIDATING PLUGIN: {plugin_name}")
        print("=" * 70 + "\n")

        try:
            results = PluginRegistry.validate_plugin(plugin_name)

            print(f"Validation Result: {'✓ PASSED' if results['passed'] else '✗ FAILED'}")
            print(f"Timestamp: {results['timestamp']}\n")

            print("Test Results:")
            for test_name, test_result in results["tests"].items():
                passed = test_result.get("passed", False)
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {status} {test_name}")

                if not passed:
                    details = test_result.get("details", "")
                    if details:
                        print(f"      Details: {details}")

                    failures = test_result.get("failures", [])
                    if failures:
                        for failure in failures[:3]:  # Show first 3
                            print(f"      - {failure}")

                    missing = test_result.get("missing", [])
                    if missing:
                        print(f"      Missing: {', '.join(missing)}")

            if results["passed"]:
                print("\n✓ Plugin validation successful!")
                print("  Plugin can be safely enabled.")
            else:
                print("\n✗ Plugin validation failed!")
                print("  Review errors above before enabling.")

        except KeyError:
            raise CLIValidationError(f"Plugin '{plugin_name}' not found")
        except Exception as e:
            # Handle plugin-specific exceptions if plugin system is available
            error_type = type(e).__name__

            if PLUGIN_SYSTEM_AVAILABLE and error_type == "PluginValidationError":
                self.logger.error(f"Plugin validation error: {e}")
                raise CLIValidationError(f"Validation failed: {e}")
            else:
                self.logger.error(f"Unexpected validation error: {e}")
                raise CLIValidationError(f"Validation error: {e}")

    def _validate_plugin_comprehensive_operation(
        self, plugin_name: str, args: argparse.Namespace
    ) -> None:
        """Execute comprehensive plugin validation operation."""
        print("\n" + "=" * 70)
        print(f"COMPREHENSIVE VALIDATION: {plugin_name}")
        print("=" * 70 + "\n")

        try:
            report = PluginValidator.validate_plugin_comprehensive(
                plugin_name, run_performance_tests=args.run_performance_tests
            )

            print(f"Overall Score: {report['overall_score']:.2f}")
            print(f"Recommendation: {report['recommendation']}")
            print(f"Timestamp: {report['timestamp']}\n")

            print("Section Scores:")
            for section_name, section_data in report["sections"].items():
                score = section_data["score"]
                weight = section_data["weight"]
                status = "✓" if score >= 0.7 else "○" if score >= 0.5 else "✗"
                print(
                    f"  {status} {section_name.replace('_', ' ').title()}: "
                    f"{score:.2f} (weight: {weight:.0%})"
                )

            # Show performance benchmarks if available
            if args.run_performance_tests and "performance" in report["sections"]:
                print("\nPerformance Benchmarks:")
                benchmarks = report["sections"]["performance"].get("benchmarks", {})
                for transform, metrics in benchmarks.items():
                    small_ms = metrics.get("small_molecule_ms", 0)
                    large_ms = metrics.get("large_molecule_ms", 0)
                    print(f"  {transform}:")
                    print(f"    Small molecules: {small_ms:.2f}ms")
                    print(f"    Large molecules: {large_ms:.2f}ms")

            # Show issues if any
            if report["issues"]:
                print("\nIssues Found:")
                for issue in report["issues"][:5]:  # Show first 5
                    print(f"  ⚠️  {issue}")

            # Show recommendations
            if report["recommendations"]:
                print("\nRecommendations:")
                for rec in report["recommendations"][:3]:  # Show first 3
                    print(f"  💡 {rec}")

            print(f"\n{'-' * 70}")
            print(f"Final Recommendation: {report['recommendation']}")

        except PluginValidationError as e:
            self.logger.error(f"Comprehensive validation error: {e}")
            raise CLIValidationError(f"Validation failed: {e}")
        except KeyError:
            raise CLIValidationError(f"Plugin '{plugin_name}' not found")
        except Exception as e:
            self.logger.error(f"Unexpected validation error: {e}")
            raise CLIValidationError(f"Validation error: {e}")

    def _enable_plugin_operation(self, plugin_name: str) -> None:
        """Execute enable plugin operation."""
        try:
            PluginRegistry.enable_plugin(plugin_name)
            self.logger.info(f"✓ Enabled plugin: {plugin_name}")
        except KeyError:
            raise CLIValidationError(f"Plugin '{plugin_name}' not found")
        except Exception as e:
            self.logger.error(f"Failed to enable plugin: {e}")
            raise CLIValidationError(f"Enable failed: {e}")

    def _disable_plugin_operation(self, plugin_name: str) -> None:
        """Execute disable plugin operation."""
        try:
            PluginRegistry.disable_plugin(plugin_name)
            self.logger.info(f"○ Disabled plugin: {plugin_name}")
        except KeyError:
            raise CLIValidationError(f"Plugin '{plugin_name}' not found")
        except Exception as e:
            self.logger.error(f"Failed to disable plugin: {e}")
            raise CLIValidationError(f"Disable failed: {e}")

    def _trust_plugin_operation(self, plugin_name: str) -> None:
        """Execute trust plugin operation."""
        try:
            info = PluginRegistry.get_plugin_info(plugin_name)
            info.trusted = True
            self.logger.info(f"🔒 Marked plugin as trusted: {plugin_name}")
            self.logger.warning(
                "Trusted plugins bypass security checks. Only trust plugins from verified sources."
            )
        except KeyError:
            raise CLIValidationError(f"Plugin '{plugin_name}' not found")
        except Exception as e:
            self.logger.error(f"Failed to trust plugin: {e}")
            raise CLIValidationError(f"Trust operation failed: {e}")


def create_cli_manager(logger: logging.Logger | None = None) -> CLIManager:
    """
    Factory function to create CLI manager instance.

    Args:
        logger: Optional logger instance

    Returns:
        Configured CLIManager instance
    """
    return CLIManager(logger)


# Convenience function for direct usage
def parse_cli_args(
    args: list[str] | None = None, logger: logging.Logger | None = None
) -> tuple[argparse.Namespace, CLIManager]:
    """
    Parse CLI arguments with validation.

    Args:
        args: Optional argument list
        logger: Optional logger instance

    Returns:
        Tuple of (parsed_args, cli_manager)

    Raises:
        CLIValidationError: If validation fails
    """
    cli_manager = create_cli_manager(logger)
    parsed_args = cli_manager.parse_args(args)

    return parsed_args, cli_manager
