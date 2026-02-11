# models/post_training/inference/__init__
"""
Inference Module

Model loading and prediction infrastructure for post-training inference.
Handles checkpoint loading, model recreation, and PyG-compatible predictions.

Key Features:
- ModelLoader: Load models from v2.0 checkpoints with DYNAMIC recreation
- Predictor: PyG-compatible inference with batching and device management
- Convenience functions: load_model(), predict() for quick usage

DYNAMIC: Uses COMPLETE hyperparameters dict from checkpoints.
         No hard-coded parameter handling - ANY model works automatically.

PRODUCTION-READY: Recreates models with ALL original settings including
                  wrappers (GraphLevelModelWrapper, EdgeLevelModelWrapper).
                  Full error handling and clear messages.

FUTURE-PROOF: Works with ANY model in registry without code changes.
              New models automatically supported via ModelRegistry lookup.

Author: MILIA Team
Version: 1.0.0

Quick Start:
    >>> from milia_pipeline.models.post_training.inference import (
    ...     ModelLoader,
    ...     Predictor,
    ...     load_model,
    ...     load_model_only,
    ...     predict,
    ... )
    >>> 
    >>> # Load model with full info
    >>> model, model_info = load_model("final_model.pt")
    >>> print(f"Task: {model_info['task_type']}")
    >>> print(f"Uses edge features: {model_info['uses_edge_features']}")
    >>> 
    >>> # Quick prediction
    >>> predictions = predict("final_model.pt", my_data)
    >>> 
    >>> # Batch prediction with Predictor
    >>> predictor = Predictor.from_checkpoint("final_model.pt")
    >>> results = predictor.predict_batch(test_dataset, batch_size=32)
    >>> 
    >>> # Check checkpoint info without loading model
    >>> info = ModelLoader.get_checkpoint_info("final_model.pt")
    >>> print(f"Format: v{info['format_version']}, Model: {info['model_name']}")

Module Structure:
    - model_loader: ModelLoader class and load_model() convenience functions
    - predictor: Predictor class and predict() convenience function
"""

import logging
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch


logger = logging.getLogger(__name__)


# =============================================================================
# VERSION INFORMATION
# =============================================================================

__version__ = "1.0.0"
__author__ = "MILIA Team"


# =============================================================================
# MODEL LOADING COMPONENTS
# =============================================================================

# ModelLoader class with static methods:
#   - load_from_checkpoint(checkpoint_path, ...) -> Tuple[model, model_info]
#   - get_checkpoint_info(checkpoint_path) -> Dict with checkpoint metadata
from .model_loader import (
    # Main class
    ModelLoader,
    
    # Convenience functions
    load_model,
    load_model_only,
)


# =============================================================================
# PREDICTION COMPONENTS
# =============================================================================

# Predictor class with methods:
#   - from_checkpoint(checkpoint_path, ...) -> Predictor [class method]
#   - predict(data, ...) -> Tensor/ndarray
#   - predict_batch(dataset, ...) -> Tensor/ndarray
from .predictor import (
    # Main class
    Predictor,
    
    # Convenience function
    predict,
)


# =============================================================================
# PUBLIC API - ORGANIZED BY CATEGORY
# =============================================================================

# Model loading components
__all_model_loading__ = [
    'ModelLoader',
    'load_model',
    'load_model_only',
]

# Prediction components
__all_prediction__ = [
    'Predictor',
    'predict',
]

# Complete public API
__all__ = (
    __all_model_loading__ +
    __all_prediction__
)


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

def get_available_components() -> Dict[str, List[str]]:
    """
    Get all available inference components.
    
    Returns:
        Dictionary mapping component types to available classes/functions
        
    Example:
        >>> from milia_pipeline.models.post_training.inference import get_available_components
        >>> components = get_available_components()
        >>> print(f"Model loading: {components['model_loading']}")
        >>> print(f"Prediction: {components['prediction']}")
    """
    return {
        'model_loading': __all_model_loading__,
        'prediction': __all_prediction__,
    }


def print_available_components():
    """
    Print all available inference components to console.
    
    Useful for exploring available options during development.
    
    Example:
        >>> from milia_pipeline.models.post_training.inference import print_available_components
        >>> print_available_components()
    """
    components = get_available_components()
    
    print("=" * 70)
    print("MILIA Pipeline - Inference Module Components")
    print("=" * 70)
    
    print(f"\n📦 Model Loading ({len(components['model_loading'])} available):")
    for i, name in enumerate(components['model_loading'], 1):
        print(f"  {i}. {name}")
    
    print(f"\n🔮 Prediction ({len(components['prediction'])} available):")
    for i, name in enumerate(components['prediction'], 1):
        print(f"  {i}. {name}")
    
    print("\n" + "=" * 70)
    print(f"Inference Module v{__version__}")
    print("=" * 70)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info(
    f"inference module loaded - "
    f"v{__version__} - "
    f"{len(__all__)} public components"
)
