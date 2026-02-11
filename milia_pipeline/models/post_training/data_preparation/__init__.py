# models/post_training/data_preparation/__init__
"""
Data Preparation Subpackage

DYNAMIC, PRODUCTION-READY, FUTURE-PROOF data conversion for inference.
Converts various molecular formats to PyG Data objects.

Supported Formats:
    - pyg_data: PyG Data passthrough (always available)
    - dict: Dictionary with tensors (always available)
    - smiles: SMILES strings (requires RDKit)
    - inchi: InChI strings (requires RDKit)
    - sdf: SDF/MOL files (requires RDKit)
    - xyz: XYZ files (requires ASE)
    - ase_atoms: ASE Atoms objects (requires ASE)

DYNAMIC: Auto-detects input format via registry pattern.
         New formats can be added via @register_converter decorator.

PRODUCTION-READY: Graceful dependency handling - unavailable formats
                  don't break the import, just list as unavailable.

FUTURE-PROOF: Registry-based architecture allows adding new formats
              without modifying existing code.

Author: MILIA Team
Version: 1.0.0

Quick Start:
    >>> from milia_pipeline.models.post_training.data_preparation import (
    ...     convert_to_pyg,
    ...     convert_batch_to_pyg,
    ...     list_available_formats,
    ... )
    >>> 
    >>> # Convert SMILES (auto-detected)
    >>> data = convert_to_pyg("CCO")
    >>> 
    >>> # Convert XYZ file
    >>> data = convert_to_pyg("molecule.xyz")
    >>> 
    >>> # Convert batch
    >>> batch = convert_batch_to_pyg(["CCO", "CC(=O)O", "c1ccccc1"])
    >>> 
    >>> # Check available formats
    >>> print(list_available_formats())

Custom Converters:
    >>> from milia_pipeline.models.post_training.data_preparation import (
    ...     BaseDataConverter,
    ...     register_converter,
    ... )
    >>> 
    >>> @register_converter("my_format")
    >>> class MyConverter(BaseDataConverter):
    ...     @property
    ...     def format_name(self) -> str:
    ...         return "my_format"
    ...     
    ...     @property
    ...     def is_available(self) -> bool:
    ...         return True
    ...     
    ...     def can_convert(self, input_data) -> bool:
    ...         return isinstance(input_data, MyDataType)
    ...     
    ...     def convert(self, input_data, **kwargs):
    ...         # Convert to PyG Data
    ...         return Data(x=..., edge_index=...)
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


# =============================================================================
# VERSION INFORMATION
# =============================================================================

__version__ = "1.0.0"
__author__ = "MILIA Team"


# =============================================================================
# CORE EXPORTS
# =============================================================================

# Registry and base classes
from .data_converter import (
    # Protocol
    DataConverterProtocol,
    
    # Registry (singleton)
    DataConverterRegistry,
    get_registry,
    register_converter,
    
    # Base class for custom converters
    BaseDataConverter,
)

# Built-in converters (always available)
from .data_converter import (
    PyGDataConverter,
    DictConverter,
)

# Convenience functions (DYNAMIC API)
from .data_converter import (
    convert_to_pyg,
    convert_batch_to_pyg,
    list_available_formats,
    list_all_formats,
    smiles_to_data,  # Legacy alias
)


# =============================================================================
# OPTIONAL CONVERTERS (Dependency-gated imports)
# =============================================================================

# RDKit-dependent converters
_RDKIT_AVAILABLE = False
try:
    from .data_converter import (
        SMILESConverter,
        InChIConverter,
        SDFConverter,
    )
    _RDKIT_AVAILABLE = True
except ImportError:
    SMILESConverter = None
    InChIConverter = None
    SDFConverter = None

# ASE-dependent converters
_ASE_AVAILABLE = False
try:
    from .data_converter import (
        XYZConverter,
        ASEAtomsConverter,
    )
    _ASE_AVAILABLE = True
except ImportError:
    XYZConverter = None
    ASEAtomsConverter = None


# =============================================================================
# PUBLIC API
# =============================================================================

# Core exports (always available)
__all_core__ = [
    # Protocol
    'DataConverterProtocol',
    
    # Registry
    'DataConverterRegistry',
    'get_registry',
    'register_converter',
    
    # Base class
    'BaseDataConverter',
    
    # Built-in converters
    'PyGDataConverter',
    'DictConverter',
    
    # Convenience functions
    'convert_to_pyg',
    'convert_batch_to_pyg',
    'list_available_formats',
    'list_all_formats',
    'smiles_to_data',
]

# RDKit converters (conditional)
__all_rdkit__ = []
if _RDKIT_AVAILABLE:
    __all_rdkit__ = [
        'SMILESConverter',
        'InChIConverter',
        'SDFConverter',
    ]

# ASE converters (conditional)
__all_ase__ = []
if _ASE_AVAILABLE:
    __all_ase__ = [
        'XYZConverter',
        'ASEAtomsConverter',
    ]

# Complete public API
__all__ = (
    __all_core__ +
    __all_rdkit__ +
    __all_ase__
)


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

def get_available_components() -> Dict[str, List[str]]:
    """
    Get all available data preparation components.
    
    Returns:
        Dictionary mapping component types to available classes/functions
        
    Example:
        >>> from milia_pipeline.models.post_training.data_preparation import get_available_components
        >>> components = get_available_components()
        >>> print(f"Core: {components['core']}")
        >>> print(f"RDKit converters: {components['rdkit_converters']}")
        >>> print(f"ASE converters: {components['ase_converters']}")
    """
    return {
        'core': __all_core__,
        'rdkit_converters': __all_rdkit__,
        'ase_converters': __all_ase__,
        'available_formats': list_available_formats(),
        'all_formats': list_all_formats(),
    }


def print_available_components():
    """
    Print all available data preparation components to console.
    
    Useful for exploring available options during development.
    
    Example:
        >>> from milia_pipeline.models.post_training.data_preparation import print_available_components
        >>> print_available_components()
    """
    components = get_available_components()
    
    print("=" * 70)
    print("MILIA Pipeline - Data Preparation Module Components")
    print("=" * 70)
    
    print(f"\n📦 Core Components ({len(components['core'])} available):")
    for i, name in enumerate(components['core'], 1):
        print(f"  {i:2d}. {name}")
    
    if components['rdkit_converters']:
        print(f"\n🧪 RDKit Converters ({len(components['rdkit_converters'])} available):")
        for i, name in enumerate(components['rdkit_converters'], 1):
            print(f"  {i}. {name}")
    else:
        print("\n🧪 RDKit Converters: Not available (pip install rdkit)")
    
    if components['ase_converters']:
        print(f"\n⚛️  ASE Converters ({len(components['ase_converters'])} available):")
        for i, name in enumerate(components['ase_converters'], 1):
            print(f"  {i}. {name}")
    else:
        print("\n⚛️  ASE Converters: Not available (pip install ase)")
    
    print(f"\n📋 Available Formats: {components['available_formats']}")
    print(f"📋 All Formats: {components['all_formats']}")
    
    print("\n" + "=" * 70)
    print(f"Data Preparation Module v{__version__}")
    print("=" * 70)


def check_dependencies() -> Dict[str, bool]:
    """
    Check which optional dependencies are available.
    
    Returns:
        Dictionary of dependency name to availability status
        
    Example:
        >>> deps = check_dependencies()
        >>> if deps['rdkit']:
        ...     print("RDKit is available - SMILES/InChI/SDF conversion supported")
    """
    deps = {
        'rdkit': _RDKIT_AVAILABLE,
        'ase': _ASE_AVAILABLE,
    }
    
    # Check individual formats
    registry = get_registry()
    for fmt in registry.list_all():
        converter_cls = registry.get(fmt)
        try:
            instance = converter_cls()
            deps[f'format_{fmt}'] = instance.is_available
        except Exception:
            deps[f'format_{fmt}'] = False
    
    return deps


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info(
    f"data_preparation module loaded - "
    f"v{__version__} - "
    f"{len(__all__)} public components - "
    f"RDKit={'available' if _RDKIT_AVAILABLE else 'not available'} - "
    f"ASE={'available' if _ASE_AVAILABLE else 'not available'}"
)
