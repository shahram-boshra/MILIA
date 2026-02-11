# models/post_training/__init__
"""
MILIA Post-Training Module

Inference, transfer learning, and model deployment infrastructure.

DYNAMIC: Works with ANY model via registry lookup
PRODUCTION-READY: Returns complete model_info for downstream usage
FUTURE-PROOF: New models automatically supported

Dependency Injection Pattern (v2.0.0):
- All components require explicit `working_root_dir: Path` parameter
- No hidden config loading (Service Locator anti-pattern removed)
- Follows CallbackFactory pattern from models/training/callbacks.py
- path_utils.py has been REMOVED - callers compute working_root_dir

Implementation Status:
    - Phase 1: Checkpoint Management ✅
    - Phase 2: Model Loading & Inference ✅
    - Phase 3: Data Preparation ✅
    - Phase 4: Transfer Learning ✅
    - Phase 7: Dependency Injection Refactoring ✅ (v2.0.0)

Usage:
    # Caller computes working_root_dir from config (Dependency Injection)
    working_root_dir = Path(config['global_paths']['working_root_dir']).expanduser()
    
    # Load model for inference
    from milia_pipeline.models.post_training import load_model
    
    model, model_info = load_model(
        "final_model.pt",
        working_root_dir=working_root_dir
    )
    print(f"Task type: {model_info['task_type']}")
    print(f"Uses edge features: {model_info['uses_edge_features']}")
    
    # Make predictions
    from milia_pipeline.models.post_training import Predictor
    
    predictor = Predictor.from_checkpoint(
        "final_model.pt",
        working_root_dir=working_root_dir
    )
    predictions = predictor.predict(data)
    predictions = predictor.predict_batch(dataset)
    
    # Checkpoint management
    from milia_pipeline.models.post_training import CheckpointManager
    
    manager = CheckpointManager(working_root_dir=working_root_dir)
    checkpoint = manager.load("model.pt")
    if manager.is_v2_checkpoint(checkpoint):
        print("v2.0 checkpoint with model recreation metadata")
    
    # Data conversion (Phase 3)
    from milia_pipeline.models.post_training import convert_to_pyg
    
    data = convert_to_pyg("CCO")  # SMILES auto-detected
    data = convert_to_pyg("molecule.xyz")  # XYZ auto-detected
    
    # Transfer learning (Phase 4)
    from milia_pipeline.models.post_training import FineTuner, FreezeStrategy
    
    fine_tuner = FineTuner.from_checkpoint(
        "pretrained.pt",
        working_root_dir=working_root_dir
    )
    model = fine_tuner.prepare_for_finetuning(
        new_out_channels=5,
        freeze_strategy=FreezeStrategy.ENCODER
    )

Author: MILIA Team
Version: 2.0.0
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


# =============================================================================
# VERSION INFORMATION
# =============================================================================

__version__ = "2.0.0"
__author__ = "MILIA Team"


# =============================================================================
# PATH UTILITIES REMOVED (v2.0.0)
# =============================================================================
# Path resolution is now handled via Dependency Injection pattern.
# All components require explicit `working_root_dir: Path` parameter.
# Callers should compute working_root_dir from config and pass it explicitly.
#
# Migration guide:
#   OLD: from milia_pipeline.models.post_training import resolve_path
#        resolved = resolve_path("checkpoints/model.pt", config=config)
#
#   NEW: working_root_dir = Path(config['global_paths']['working_root_dir']).expanduser()
#        resolved = working_root_dir / "checkpoints/model.pt"
# =============================================================================


# =============================================================================
# CHECKPOINT MANAGEMENT (Phase 1)
# =============================================================================

from .checkpoint.checkpoint_manager import (
    CheckpointManager,
    CHECKPOINT_FORMAT_VERSION,
)


# =============================================================================
# MODEL LOADING & INFERENCE (Phase 2)
# =============================================================================

from .inference.model_loader import (
    ModelLoader,
    load_model,
    load_model_only,
)

from .inference.predictor import (
    Predictor,
    predict,
)


# =============================================================================
# DATA PREPARATION (Phase 3)
# =============================================================================

_DATA_PREPARATION_AVAILABLE = False

try:
    from .data_preparation.data_converter import (
        # Protocol
        DataConverterProtocol,
        
        # Registry
        DataConverterRegistry,
        get_registry,
        register_converter,
        
        # Base class
        BaseDataConverter,
        
        # Built-in converters
        PyGDataConverter,
        DictConverter,
        SMILESConverter,
        InChIConverter,
        XYZConverter,
        ASEAtomsConverter,
        SDFConverter,
        
        # Convenience functions
        convert_to_pyg,
        convert_batch_to_pyg,
        convert_sdf_to_pyg_list,  # FIX 24: Multi-molecule SDF support
        list_available_formats,
        list_all_formats,
        smiles_to_data,  # Legacy alias
    )
    _DATA_PREPARATION_AVAILABLE = True
except ImportError as e:
    # Dependencies not available (RDKit/ASE)
    logger.debug(f"Data preparation imports failed: {e}")
    pass


# =============================================================================
# TRANSFER LEARNING (Phase 4)
# =============================================================================

_TRANSFER_LEARNING_AVAILABLE = False

try:
    from .transfer_learning.fine_tuner import (
        FineTuner,
        FreezeStrategy,
    )
    _TRANSFER_LEARNING_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Transfer learning imports failed: {e}")
    pass


# =============================================================================
# PUBLIC API
# =============================================================================

# Path utilities removed in v2.0.0 - see migration guide above
__all_path_utils__ = []

# Core exports (always available)
__all_checkpoint__ = [
    'CheckpointManager',
    'CHECKPOINT_FORMAT_VERSION',
]

__all_inference__ = [
    'ModelLoader',
    'load_model',
    'load_model_only',
    'Predictor',
    'predict',
]

# Conditional exports - Data Preparation (Phase 3)
__all_data_preparation__ = []
if _DATA_PREPARATION_AVAILABLE:
    __all_data_preparation__ = [
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
        'SMILESConverter',
        'InChIConverter',
        'XYZConverter',
        'ASEAtomsConverter',
        'SDFConverter',
        
        # Convenience functions
        'convert_to_pyg',
        'convert_batch_to_pyg',
        'convert_sdf_to_pyg_list',  # FIX 24: Multi-molecule SDF support
        'list_available_formats',
        'list_all_formats',
        'smiles_to_data',
    ]

# Conditional exports - Transfer Learning (Phase 4)
__all_transfer_learning__ = []
if _TRANSFER_LEARNING_AVAILABLE:
    __all_transfer_learning__ = [
        'FineTuner',
        'FreezeStrategy',
    ]

# Complete public API
__all__ = (
    __all_path_utils__ +
    __all_checkpoint__ +
    __all_inference__ +
    __all_data_preparation__ +
    __all_transfer_learning__
)


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

def get_available_components() -> Dict[str, List[str]]:
    """
    Get all available post-training components.
    
    Returns:
        Dictionary mapping component types to available classes/functions
        
    Example:
        >>> from milia_pipeline.models.post_training import get_available_components
        >>> components = get_available_components()
        >>> print(f"Checkpoint: {components['checkpoint']}")
        >>> print(f"Inference: {components['inference']}")
    """
    return {
        'checkpoint': __all_checkpoint__,
        'inference': __all_inference__,
        'data_preparation': __all_data_preparation__,
        'transfer_learning': __all_transfer_learning__,
    }


def print_available_components():
    """
    Print all available post-training components to console.
    
    Useful for exploring available options during development.
    
    Example:
        >>> from milia_pipeline.models.post_training import print_available_components
        >>> print_available_components()
    """
    components = get_available_components()
    
    print("=" * 70)
    print("MILIA Pipeline - Post-Training Module Components")
    print("=" * 70)
    
    print(f"\n📦 Checkpoint Management ({len(components['checkpoint'])} available):")
    for i, name in enumerate(components['checkpoint'], 1):
        print(f"  {i}. {name}")
    
    print(f"\n🔮 Inference ({len(components['inference'])} available):")
    for i, name in enumerate(components['inference'], 1):
        print(f"  {i}. {name}")
    
    if components['data_preparation']:
        print(f"\n🧪 Data Preparation ({len(components['data_preparation'])} available):")
        for i, name in enumerate(components['data_preparation'], 1):
            print(f"  {i}. {name}")
    else:
        print("\n🧪 Data Preparation: Not available (install rdkit/ase)")
    
    if components['transfer_learning']:
        print(f"\n🔄 Transfer Learning ({len(components['transfer_learning'])} available):")
        for i, name in enumerate(components['transfer_learning'], 1):
            print(f"  {i}. {name}")
    else:
        print("\n🔄 Transfer Learning: Not available")
    
    print("\n" + "=" * 70)
    print(f"Post-Training Module v{__version__} (Dependency Injection pattern)")
    print("=" * 70)


def get_implementation_status() -> Dict[str, bool]:
    """
    Get implementation status of all phases.
    
    Returns:
        Dictionary mapping phase names to implementation status
        
    Example:
        >>> status = get_implementation_status()
        >>> for phase, implemented in status.items():
        ...     print(f"{phase}: {'✅' if implemented else '🔲'}")
    """
    return {
        'Phase 1 - Checkpoint Management': True,
        'Phase 2 - Model Loading & Inference': True,
        'Phase 3 - Data Preparation': _DATA_PREPARATION_AVAILABLE,
        'Phase 4 - Transfer Learning': _TRANSFER_LEARNING_AVAILABLE,
        'Phase 7 - Dependency Injection Refactoring': True,
    }


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info(
    f"post_training module loaded - "
    f"v{__version__} - "
    f"{len(__all__)} public components - "
    f"data_prep={'available' if _DATA_PREPARATION_AVAILABLE else 'unavailable'} - "
    f"transfer={'available' if _TRANSFER_LEARNING_AVAILABLE else 'unavailable'} - "
    f"pattern=dependency_injection"
)
