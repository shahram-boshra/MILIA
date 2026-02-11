# models/post_training/transfer_learning/__init__
"""
Transfer Learning Subpackage

Transfer learning and fine-tuning infrastructure for pre-trained models.

DYNAMIC: Works with ANY model loaded from checkpoint via ModelLoader.
         Freeze strategies are configurable, not hard-coded to specific architectures.

PRODUCTION-READY: Comprehensive logging of parameter status.
                  Handles output head replacement dynamically.

FUTURE-PROOF: FreezeStrategy enum allows easy extension.
              Works with any model that has standard PyTorch modules.

Author: MILIA Team
Version: 1.0.0

Quick Start:
    >>> from milia_pipeline.models.post_training.transfer_learning import (
    ...     FineTuner,
    ...     FreezeStrategy,
    ... )
    >>> 
    >>> # Load pre-trained model and prepare for fine-tuning
    >>> fine_tuner = FineTuner.from_checkpoint("pretrained_model.pt")
    >>> 
    >>> # Prepare model: freeze encoder, replace output head
    >>> model = fine_tuner.prepare_for_finetuning(
    ...     new_out_channels=5,  # New task has 5 targets
    ...     freeze_strategy=FreezeStrategy.ENCODER
    ... )
    >>> 
    >>> # Then train with Trainer as normal
    >>> trainer = Trainer(model=model, ...)
    >>> trainer.fit()

Freeze Strategies:
    - FreezeStrategy.NONE: Train all parameters (full fine-tuning)
    - FreezeStrategy.ENCODER: Freeze GNN encoder, train output head only
    - FreezeStrategy.ENCODER_PARTIAL: Freeze first N layers of encoder
    - FreezeStrategy.ALL_BUT_LAST: Freeze all but the last linear layer

Transfer Learning Workflow:
    1. Train a model on source task (e.g., large dataset)
    2. Save checkpoint with full metadata (Phase 1)
    3. Load checkpoint with FineTuner.from_checkpoint()
    4. Call prepare_for_finetuning() with new output size
    5. Train on target task with standard Trainer
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

from .fine_tuner import (
    # Enum for freeze strategies
    FreezeStrategy,
    
    # Main fine-tuning class
    FineTuner,
)


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Freeze strategy enum
    'FreezeStrategy',
    
    # Main class
    'FineTuner',
]


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

def get_available_components() -> Dict[str, List[str]]:
    """
    Get all available transfer learning components.
    
    Returns:
        Dictionary mapping component types to available classes/functions
        
    Example:
        >>> from milia_pipeline.models.post_training.transfer_learning import get_available_components
        >>> components = get_available_components()
        >>> print(f"Classes: {components['classes']}")
        >>> print(f"Freeze strategies: {components['freeze_strategies']}")
    """
    return {
        'classes': ['FineTuner'],
        'enums': ['FreezeStrategy'],
        'freeze_strategies': [s.value for s in FreezeStrategy],
    }


def print_available_components():
    """
    Print all available transfer learning components to console.
    
    Useful for exploring available options during development.
    
    Example:
        >>> from milia_pipeline.models.post_training.transfer_learning import print_available_components
        >>> print_available_components()
    """
    components = get_available_components()
    
    print("=" * 70)
    print("MILIA Pipeline - Transfer Learning Module Components")
    print("=" * 70)
    
    print(f"\n📦 Classes ({len(components['classes'])} available):")
    for i, name in enumerate(components['classes'], 1):
        print(f"  {i}. {name}")
    
    print(f"\n🔧 Enums ({len(components['enums'])} available):")
    for i, name in enumerate(components['enums'], 1):
        print(f"  {i}. {name}")
    
    print(f"\n❄️  Freeze Strategies ({len(components['freeze_strategies'])} available):")
    for i, strategy in enumerate(components['freeze_strategies'], 1):
        desc = {
            'none': 'Train all parameters (full fine-tuning)',
            'encoder': 'Freeze GNN encoder, train head only',
            'encoder_partial': 'Freeze first N layers of encoder',
            'all_but_last': 'Freeze all but last linear layer',
        }.get(strategy, '')
        print(f"  {i}. {strategy}: {desc}")
    
    print("\n" + "=" * 70)
    print(f"Transfer Learning Module v{__version__}")
    print("=" * 70)


def list_freeze_strategies() -> List[str]:
    """
    List all available freeze strategies.
    
    Returns:
        List of freeze strategy names
        
    Example:
        >>> strategies = list_freeze_strategies()
        >>> print(strategies)
        ['none', 'encoder', 'encoder_partial', 'all_but_last']
    """
    return [s.value for s in FreezeStrategy]


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info(
    f"transfer_learning module loaded - "
    f"v{__version__} - "
    f"{len(__all__)} public components - "
    f"{len(list_freeze_strategies())} freeze strategies"
)
