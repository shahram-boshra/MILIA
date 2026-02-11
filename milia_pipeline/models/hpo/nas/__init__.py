"""
Neural Architecture Search (NAS) Module

Provides neural architecture search capabilities for Graph Neural Networks (GNNs)
using the HPO infrastructure.

This module enables automated discovery of optimal GNN architectures by searching
over configurable architecture components including layer types, dimensions,
pooling strategies, and connectivity patterns.

Location: milia_pipeline/models/hpo/nas/__init__.py
Blueprint Reference: MILIA_HPO_Implementation_Blueprint.md (lines 4138-4144)

Components
----------
**Search Space Definition** (from search_space.py):
    LayerType : Enum
        Available GNN layer types (GCN, GAT, SAGE, GIN, etc.)
    PoolingType : Enum
        Graph pooling strategies (mean, max, attention, etc.)
    AggregationType : Enum
        Message aggregation types (mean, max, sum, etc.)
    ActivationType : Enum
        Activation functions (relu, gelu, elu, etc.)
    LayerConfig : dataclass
        Configuration for a single GNN layer
    GNNArchitectureSpace : dataclass
        Defines the complete GNN architecture search space

**NAS Manager** (from nas_manager.py):
    NASConfig : dataclass
        Configuration for neural architecture search
    NASManager : class
        Main orchestrator for architecture search
    HeterogeneousGNN : nn.Module
        GNN model with mixed layer types

**Factory Functions**:
    create_gnn_search_space(model_type, **kwargs)
        Create preset or custom GNN search spaces
    get_default_gnn_search_space()
        Get default GNN architecture search space
    create_nas_manager(arch_space, n_trials, **kwargs)
        Convenience function to create NASManager

Usage Examples
--------------
Basic architecture search:

    >>> from milia_pipeline.models.hpo.nas import (
    ...     NASManager,
    ...     GNNArchitectureSpace,
    ... )
    >>> 
    >>> # Create search space
    >>> arch_space = GNNArchitectureSpace()
    >>> 
    >>> # Create NAS manager and run search
    >>> nas = NASManager(arch_space)
    >>> best_arch = nas.search(dataset=train_dataset)
    >>> 
    >>> # Build model from best architecture
    >>> model = nas.build_model(best_arch, in_channels=10, out_channels=1)

Custom search space for attention models:

    >>> from milia_pipeline.models.hpo.nas import (
    ...     GNNArchitectureSpace,
    ...     LayerType,
    ...     PoolingType,
    ...     NASManager,
    ...     NASConfig,
    ... )
    >>> 
    >>> # Create attention-focused search space
    >>> arch_space = GNNArchitectureSpace(
    ...     min_layers=2,
    ...     max_layers=6,
    ...     layer_types=[LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER],
    ...     heads=[2, 4, 8, 16],
    ...     pooling_types=[PoolingType.ATTENTION, PoolingType.MEAN],
    ...     allow_mixed_layers=True,
    ... )
    >>> 
    >>> # Configure NAS
    >>> nas_config = NASConfig(n_trials=100, cv_folds=5)
    >>> nas = NASManager(arch_space, nas_config=nas_config)
    >>> 
    >>> # Run search
    >>> best_arch = nas.search(dataset=dataset)

Using factory functions:

    >>> from milia_pipeline.models.hpo.nas import (
    ...     create_gnn_search_space,
    ...     create_nas_manager,
    ... )
    >>> 
    >>> # Create GAT-specific search space
    >>> arch_space = create_gnn_search_space('gat')
    >>> 
    >>> # Quick NAS setup
    >>> nas = create_nas_manager(arch_space, n_trials=50)
    >>> best_arch = nas.search(dataset=dataset)

Dependencies
------------
- search_space.py : GNN architecture search space definitions
- nas_manager.py : NAS orchestration and model building
- hpo_manager.py : HPO infrastructure for optimization
- hpo_config.py : HPO configuration classes

See Also
--------
- milia_pipeline.models.hpo.hpo_manager : HPO orchestration
- milia_pipeline.models.hpo.hpo_config : HPO configuration
- milia_pipeline.models.hpo.backends : HPO backends (Optuna, Ray Tune)

Author: Milia Team
Version: 1.0.0
"""

# =============================================================================
# SEARCH SPACE IMPORTS
# =============================================================================

from .search_space import (
    # Enums
    LayerType,
    PoolingType,
    AggregationType,
    ActivationType,
    # Dataclasses
    LayerConfig,
    GNNArchitectureSpace,
    # Factory functions
    create_gnn_search_space,
    get_default_gnn_search_space,
)

# =============================================================================
# NAS MANAGER IMPORTS
# =============================================================================

from .nas_manager import (
    # Configuration
    NASConfig,
    # Main classes
    NASManager,
    HeterogeneousGNN,
    # Convenience functions
    create_nas_manager,
)

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums (search_space.py)
    'LayerType',
    'PoolingType',
    'AggregationType',
    'ActivationType',
    # Dataclasses (search_space.py)
    'LayerConfig',
    'GNNArchitectureSpace',
    # Factory functions (search_space.py)
    'create_gnn_search_space',
    'get_default_gnn_search_space',
    # Configuration (nas_manager.py)
    'NASConfig',
    # Main classes (nas_manager.py)
    'NASManager',
    'HeterogeneousGNN',
    # Convenience functions (nas_manager.py)
    'create_nas_manager',
]

# =============================================================================
# MODULE METADATA
# =============================================================================

__version__ = '1.0.0'
__author__ = 'Milia Team'


# =============================================================================
# MODULE INFORMATION FUNCTIONS
# =============================================================================

def get_nas_module_info() -> dict:
    """
    Get information about the NAS module.
    
    Returns a dictionary containing version, available components,
    and capability information.
    
    Returns
    -------
    dict
        Dictionary containing:
        - version: Module version string
        - layer_types: Available GNN layer types
        - pooling_types: Available pooling strategies
        - aggregation_types: Available aggregation types
        - activation_types: Available activation functions
        - components: List of main exported components
        
    Examples
    --------
    >>> from milia_pipeline.models.hpo.nas import get_nas_module_info
    >>> info = get_nas_module_info()
    >>> print(f"NAS Module v{info['version']}")
    >>> print(f"Layer types: {info['layer_types']}")
    """
    return {
        'version': __version__,
        'author': __author__,
        'layer_types': [lt.value for lt in LayerType],
        'pooling_types': [pt.value for pt in PoolingType],
        'aggregation_types': [at.value for at in AggregationType],
        'activation_types': [at.value for at in ActivationType],
        'components': __all__,
    }


def check_nas_dependencies() -> dict:
    """
    Check availability of NAS module dependencies.
    
    Verifies that required dependencies (PyTorch, PyTorch Geometric, Optuna)
    are available for NAS functionality.
    
    Returns
    -------
    dict
        Dictionary with dependency availability:
        - torch: PyTorch availability
        - torch_geometric: PyTorch Geometric availability
        - optuna: Optuna availability
        - all_available: True if all dependencies available
        
    Examples
    --------
    >>> from milia_pipeline.models.hpo.nas import check_nas_dependencies
    >>> deps = check_nas_dependencies()
    >>> if deps['all_available']:
    ...     print("All NAS dependencies available")
    >>> else:
    ...     print(f"Missing: {[k for k, v in deps.items() if not v]}")
    """
    deps = {}
    
    try:
        import torch
        deps['torch'] = True
        deps['torch_version'] = torch.__version__
    except ImportError:
        deps['torch'] = False
        deps['torch_version'] = None
    
    try:
        import torch_geometric
        deps['torch_geometric'] = True
        deps['torch_geometric_version'] = torch_geometric.__version__
    except ImportError:
        deps['torch_geometric'] = False
        deps['torch_geometric_version'] = None
    
    try:
        import optuna
        deps['optuna'] = True
        deps['optuna_version'] = optuna.__version__
    except ImportError:
        deps['optuna'] = False
        deps['optuna_version'] = None
    
    deps['all_available'] = all([
        deps['torch'],
        deps['torch_geometric'],
        deps['optuna'],
    ])
    
    return deps
