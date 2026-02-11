# milia_pipeline/preprocessing/__init__.py

"""
Preprocessing Subsystem
=======================

One-time transformation of raw dataset files into .npz format for miliaDataset.

This subsystem provides a modular, extensible preprocessing pipeline for molecular
wavefunction data with registry-based preprocessor discovery, comprehensive utility
functions, and robust error handling.

Core Components
---------------
- **BasePreprocessor**: Abstract base class defining the preprocessor contract
- **PreprocessorRegistry**: Auto-discovery and management system for preprocessors
- **Utility Functions**: Archive extraction, format parsing, and NPZ building tools

Architecture
------------
The preprocessing subsystem follows a plugin-like architecture where:

1. Preprocessors inherit from BasePreprocessor and implement required methods
2. Preprocessors auto-register via the @PreprocessorRegistry.register() decorator
3. Import of preprocessor modules triggers automatic registration
4. Utilities provide reusable components for common preprocessing tasks

Workflow
--------
Typical preprocessing workflow:

    Raw Data (tar.gz, zip, etc.)
           ↓
    [Archive Extraction] → extract_from_targz()
           ↓
    [Format Parsing] → parse_molden_files()
           ↓
    [Data Processing] → Custom Preprocessor Logic
           ↓
    [NPZ Building] → build_npz()
           ↓
    [Validation] → validate_npz_structure()
           ↓
    Output .npz file ready for miliaDataset

Available Preprocessors
-----------------------
- **Wavefunction**: Processes quantum chemistry wavefunction data (MOLDEN, FCHK formats)

To see all registered preprocessors:
>>> from milia_pipeline.preprocessing import PreprocessorRegistry
>>> PreprocessorRegistry.list_preprocessors()
['Wavefunction']

Usage Examples
--------------

**Basic Preprocessing**:
    >>> from milia_pipeline.preprocessing import PreprocessorRegistry
    >>> import logging
    >>> 
    >>> # Get preprocessor class
    >>> PreprocessorClass = PreprocessorRegistry.get_preprocessor("Wavefunction")
    >>> 
    >>> # Configure preprocessing
    >>> config = {
    ...     'raw_tar_path': 'raw/wavefunctions.tar.gz',
    ...     'output_npz_path': 'processed/wavefunctions.npz',
    ...     'num_molecules': 100,
    ...     'feature_tier': 'standard'
    ... }
    >>> 
    >>> # Initialize and run
    >>> logger = logging.getLogger(__name__)
    >>> preprocessor = PreprocessorClass(config, logger)
    >>> output_path = preprocessor.run()
    >>> print(f"Generated: {output_path}")

**Check Preprocessing Support**:
    >>> from milia_pipeline.preprocessing import PreprocessorRegistry
    >>> 
    >>> if PreprocessorRegistry.supports_preprocessing("Wavefunction"):
    ...     print("Wavefunction preprocessing is available")

**Using Utility Functions Directly**:
    >>> from milia_pipeline.preprocessing import (
    ...     extract_from_targz,
    ...     parse_molden_files,
    ...     build_npz,
    ...     validate_npz_structure
    ... )
    >>> 
    >>> # Extract archive
    >>> extracted_dir = extract_from_targz('raw/data.tar.gz', 'temp/')
    >>> 
    >>> # Parse molecular data
    >>> molecules = parse_molden_files(extracted_dir)
    >>> 
    >>> # Build NPZ file
    >>> output_path = build_npz(molecules, 'processed/output.npz')
    >>> 
    >>> # Validate structure
    >>> validate_npz_structure(output_path)

Creating Custom Preprocessors
------------------------------
To create a custom preprocessor:

1. Inherit from BasePreprocessor
2. Implement required abstract methods (_validate_config, preprocess)
3. Register using the @PreprocessorRegistry.register() decorator
4. Place in the preprocessors/ subdirectory
5. Import the module to trigger registration

Example:
    >>> from milia_pipeline.preprocessing import BasePreprocessor, PreprocessorRegistry
    >>> 
    >>> @PreprocessorRegistry.register("CustomDataset")
    >>> class CustomPreprocessor(BasePreprocessor):
    ...     def _validate_config(self):
    ...         # Validate configuration
    ...         pass
    ...     
    ...     def preprocess(self):
    ...         # Implement preprocessing logic
    ...         return output_path

Module Structure
----------------
preprocessing/
├── __init__.py                  # This file - module initialization
├── base_preprocessor.py         # Abstract base class
├── registry.py                  # Preprocessor registry system
├── preprocessors/               # Preprocessor implementations
│   ├── __init__.py
│   └── wavefunction.py          # Wavefunction preprocessor
└── utils/                       # Utility functions
    ├── __init__.py
    ├── archive_handlers.py      # Archive extraction utilities
    ├── format_parsers.py        # Format parsing utilities
    └── npz_builders.py          # NPZ file construction utilities

Error Handling
--------------
The subsystem uses custom exceptions from milia_pipeline.exceptions:
- ConfigurationError: Raised for invalid configurations
- DataProcessingError: Raised for preprocessing failures

All preprocessors validate inputs and outputs, providing detailed error messages
for troubleshooting.

Performance Considerations
--------------------------
- Preprocessing is an OFFLINE operation (one-time transformation)
- Large datasets should be processed in chunks if memory-constrained
- NPZ format provides efficient binary storage and fast loading
- Archive extraction uses temporary directories (cleaned up after processing)

Version History
---------------
- 1.1 (November 2025): Production-ready release with enhanced documentation
- 1.0 (Initial): Base implementation with wavefunction preprocessing

Author: milia Pipeline Team
Date: November 2025
License: See project LICENSE file
"""

# ============================================================================
# Standard Library Imports
# ============================================================================
import logging
import warnings
from typing import List, Optional


# ============================================================================
# Core Component Imports
# ============================================================================
# These are the fundamental building blocks of the preprocessing subsystem

try:
    from milia_pipeline.preprocessing.base_preprocessor import BasePreprocessor
except ImportError as e:
    raise ImportError(
        "Failed to import BasePreprocessor. Ensure milia_pipeline.preprocessing.base_preprocessor "
        "is available and all dependencies are installed."
    ) from e

try:
    from milia_pipeline.preprocessing.registry import PreprocessorRegistry
except ImportError as e:
    raise ImportError(
        "Failed to import PreprocessorRegistry. Ensure milia_pipeline.preprocessing.registry "
        "is available and all dependencies are installed."
    ) from e


# ============================================================================
# Preprocessor Auto-Registration
# ============================================================================
# Import preprocessor modules to trigger automatic registration via decorators
# Each import statement registers the preprocessor with PreprocessorRegistry

_PREPROCESSOR_IMPORT_ERRORS = []

# Wavefunction Preprocessor
try:
    from milia_pipeline.preprocessing.preprocessors import wavefunction
except ImportError as e:
    _PREPROCESSOR_IMPORT_ERRORS.append(("wavefunction", str(e)))
    warnings.warn(
        f"Failed to import wavefunction preprocessor: {e}. "
        "Wavefunction preprocessing will not be available.",
        ImportWarning
    )

# QM9 Preprocessor
try:
    from milia_pipeline.preprocessing.preprocessors import qm9
except ImportError as e:
    _PREPROCESSOR_IMPORT_ERRORS.append(("qm9", str(e)))
    warnings.warn(
        f"Failed to import qm9 preprocessor: {e}. "
        "QM9 preprocessing will not be available.",
        ImportWarning
    )

# ANI1x Preprocessor
try:
    from milia_pipeline.preprocessing.preprocessors import ani1x
except ImportError as e:
    _PREPROCESSOR_IMPORT_ERRORS.append(("ani1x", str(e)))
    warnings.warn(
        f"Failed to import ani1x preprocessor: {e}. "
        "ANI1x preprocessing will not be available.",
        ImportWarning
    )


# ============================================================================
# Utility Function Imports
# ============================================================================
# These utilities can be used independently or as building blocks for custom
# preprocessors. They handle common tasks like archive extraction, format
# parsing, and NPZ file construction.

_UTILITY_IMPORT_ERRORS = []

# Archive Handling Utilities
try:
    from milia_pipeline.preprocessing.utils.archive_handlers import extract_from_targz
except ImportError as e:
    _UTILITY_IMPORT_ERRORS.append(("extract_from_targz", str(e)))
    warnings.warn(
        f"Failed to import archive_handlers utilities: {e}. "
        "Archive extraction functions will not be available.",
        ImportWarning
    )
    extract_from_targz = None

# Format Parsing Utilities
try:
    from milia_pipeline.preprocessing.utils.format_parsers import parse_molden_files
except ImportError as e:
    _UTILITY_IMPORT_ERRORS.append(("parse_molden_files", str(e)))
    warnings.warn(
        f"Failed to import format_parsers utilities: {e}. "
        "Format parsing functions will not be available.",
        ImportWarning
    )
    parse_molden_files = None

# NPZ Building Utilities
try:
    from milia_pipeline.preprocessing.utils.npz_builders import (
        build_npz,
        validate_npz_structure
    )
except ImportError as e:
    _UTILITY_IMPORT_ERRORS.append(("npz_builders", str(e)))
    warnings.warn(
        f"Failed to import npz_builders utilities: {e}. "
        "NPZ building and validation functions will not be available.",
        ImportWarning
    )
    build_npz = None
    validate_npz_structure = None


# ============================================================================
# Module Configuration
# ============================================================================

# Module version
__version__ = '1.1'

# Public API - explicitly define what gets exported with "from preprocessing import *"
__all__ = [
    # Core Classes
    'BasePreprocessor',         # Abstract base for custom preprocessors
    'PreprocessorRegistry',     # Registry for preprocessor management
    
    # Utility Functions - Archive Handling
    'extract_from_targz',       # Extract files from tar.gz archives
    
    # Utility Functions - Format Parsing
    'parse_molden_files',       # Parse MOLDEN format files
    
    # Utility Functions - NPZ Building
    'build_npz',                # Build NPZ files from molecular data
    'validate_npz_structure',   # Validate NPZ file structure
]


# ============================================================================
# Module Initialization
# ============================================================================

# Configure module-level logger
_logger = logging.getLogger(__name__)


def _log_initialization_status() -> None:
    """Log the initialization status of the preprocessing subsystem."""
    
    # Log registered preprocessors
    registered = PreprocessorRegistry.list_preprocessors()
    if registered:
        _logger.info(
            f"Preprocessing subsystem initialized with {len(registered)} "
            f"preprocessor(s): {', '.join(registered)}"
        )
    else:
        _logger.warning(
            "Preprocessing subsystem initialized but no preprocessors were registered. "
            "Check for import errors."
        )
    
    # Log preprocessor import errors if any
    if _PREPROCESSOR_IMPORT_ERRORS:
        _logger.warning(
            f"Encountered {len(_PREPROCESSOR_IMPORT_ERRORS)} preprocessor import error(s):"
        )
        for name, error in _PREPROCESSOR_IMPORT_ERRORS:
            _logger.warning(f"  - {name}: {error}")
    
    # Log utility import errors if any
    if _UTILITY_IMPORT_ERRORS:
        _logger.warning(
            f"Encountered {len(_UTILITY_IMPORT_ERRORS)} utility import error(s):"
        )
        for name, error in _UTILITY_IMPORT_ERRORS:
            _logger.warning(f"  - {name}: {error}")


def get_preprocessing_info() -> dict:
    """
    Get information about the preprocessing subsystem.
    
    Returns:
        Dictionary containing:
        - version: Module version
        - registered_preprocessors: List of registered preprocessor names
        - available_utilities: List of successfully imported utility functions
        - import_errors: List of import errors encountered
    
    Example:
        >>> from milia_pipeline.preprocessing import get_preprocessing_info
        >>> info = get_preprocessing_info()
        >>> print(f"Version: {info['version']}")
        >>> print(f"Preprocessors: {info['registered_preprocessors']}")
    """
    return {
        'version': __version__,
        'registered_preprocessors': PreprocessorRegistry.list_preprocessors(),
        'available_utilities': [
            name for name in [
                'extract_from_targz',
                'parse_molden_files',
                'build_npz',
                'validate_npz_structure'
            ] if globals().get(name) is not None
        ],
        'preprocessor_import_errors': _PREPROCESSOR_IMPORT_ERRORS,
        'utility_import_errors': _UTILITY_IMPORT_ERRORS,
    }


# Perform initialization logging
_log_initialization_status()


# ============================================================================
# Module-Level Validation
# ============================================================================

def _validate_critical_components() -> None:
    """
    Validate that critical components are available.
    
    Raises:
        RuntimeError: If critical components are missing
    """
    errors = []
    
    # Check core classes
    if 'BasePreprocessor' not in globals():
        errors.append("BasePreprocessor is not available")
    
    if 'PreprocessorRegistry' not in globals():
        errors.append("PreprocessorRegistry is not available")
    
    # Check if at least one preprocessor is registered
    if not PreprocessorRegistry.list_preprocessors():
        errors.append(
            "No preprocessors registered - check preprocessor imports"
        )
    
    if errors:
        raise RuntimeError(
            "Preprocessing subsystem initialization failed:\n  " +
            "\n  ".join(errors)
        )


# Validate critical components on module load
try:
    _validate_critical_components()
except RuntimeError as e:
    _logger.error(str(e))
    # Don't raise - allow module to import even if some components are missing
    # This enables graceful degradation and better error messages


# ============================================================================
# Convenience Functions
# ============================================================================

def list_available_preprocessors() -> List[str]:
    """
    List all available preprocessors.
    
    This is a convenience function that wraps PreprocessorRegistry.list_preprocessors()
    for easier access.
    
    Returns:
        List of registered preprocessor names
    
    Example:
        >>> from milia_pipeline.preprocessing import list_available_preprocessors
        >>> preprocessors = list_available_preprocessors()
        >>> print(f"Available: {preprocessors}")
    """
    return PreprocessorRegistry.list_preprocessors()


def supports_dataset(dataset_type: str) -> bool:
    """
    Check if preprocessing is supported for a given dataset type.
    
    This is a convenience function that wraps PreprocessorRegistry.supports_preprocessing()
    for easier access.
    
    Args:
        dataset_type: Name/type of dataset to check
    
    Returns:
        True if preprocessing is supported for this dataset type
    
    Example:
        >>> from milia_pipeline.preprocessing import supports_dataset
        >>> if supports_dataset("Wavefunction"):
        ...     print("Wavefunction preprocessing is supported")
    """
    return PreprocessorRegistry.supports_preprocessing(dataset_type)


# Add convenience functions to public API
__all__.extend([
    'get_preprocessing_info',
    'list_available_preprocessors',
    'supports_dataset',
])


# ============================================================================
# Module Cleanup
# ============================================================================

# Clean up internal variables that shouldn't be part of public API
del logging, warnings, Optional, List


# ============================================================================
# End of Module
# ============================================================================
