#/models/builders/__init__.py
"""
Models Builder Module

Dynamic architecture building, layer composition, and model ensembles.

Public API:
    - LayerRegistry: Catalog of GNN layers
    - LayerCategory: Layer category enum
    - LayerMetadata: Layer metadata structure
    - FunctionalLayerWrapper: Wrapper for functional operations
    - ArchitectureBuilder: Dynamic layer composition
    - LayerConfig: Layer configuration data class
    - ArchitectureConfig: Architecture configuration data class
    - ModelComposer: Multi-model ensemble composer
    - ModelSpec: Model specification for ensembles
    - EnsembleConfig: Ensemble configuration data class
    - ParallelEnsemble: Parallel ensemble module
    - SequentialStack: Sequential stacking module
    - HierarchicalComposition: Hierarchical composition module
    - ArchitectureTemplates: Pre-built architecture templates
    - ArchitectureConfigParser: Configuration parser for YAML/JSON
    - ArchitectureValidator: Architecture validation
    - get_layer: Get layer from registry
    - list_layers: List available layers
    - parse_custom_architecture: Parse custom architecture from config
    - parse_ensemble: Parse ensemble from config
    - load_config: Load configuration file
    - validate_config: Validate configuration structure
    - validate_architecture: Validate architecture
    - validate_data_compatibility: Validate data compatibility

Version: 1.0.0
"""

from typing import Dict, List, Optional, Any

# Import layer registry components
from .layer_registry import (
    LayerRegistry,
    LayerCategory,
    LayerMetadata,
    FunctionalLayerWrapper,
    LayerNotFoundError,
    get_layer,
    list_layers,
    get_layer_metadata,
    registry as layer_registry,
)

# Import architecture builder components
from .architecture_builder import (
    ArchitectureBuilder,
    LayerConfig,
    ArchitectureConfig,
    ResidualConnection,
    ArchitectureError,
    ChannelMismatchError,
    CustomArchitecture,
)

# Import model composer components
from .model_composer import (
    ModelSpec,
    EnsembleConfig,
    CompositionError,
    ModelComposer,
    ParallelEnsemble,
    SequentialStack,
    HierarchicalComposition,
)

# Import templates
from .templates import (
    ArchitectureTemplates,
)

# Import config parser components
from .config_parser import (
    ArchitectureConfigParser,
    parse_custom_architecture,
    parse_ensemble,
    load_config,
    validate_config,
)

# Import validation components
from .validation import (
    ArchitectureValidator,
    validate_architecture,
    validate_data_compatibility,
)

__version__ = "1.0.0"
__author__ = "Milia Team"

__all__ = [
    # Layer Registry
    "LayerRegistry",
    "LayerCategory",
    "LayerMetadata",
    "FunctionalLayerWrapper",
    "LayerNotFoundError",
    "get_layer",
    "list_layers",
    "get_layer_metadata",
    "layer_registry",
    
    # Architecture Builder
    "ArchitectureBuilder",
    "LayerConfig",
    "ArchitectureConfig",
    "ResidualConnection",
    "ArchitectureError",
    "ChannelMismatchError",
    "CustomArchitecture",
    
    # Model Composer
    "ModelSpec",
    "EnsembleConfig",
    "CompositionError",
    "ModelComposer",
    "ParallelEnsemble",
    "SequentialStack",
    "HierarchicalComposition",
    
    # Templates
    "ArchitectureTemplates",
    
    # Config Parser
    "ArchitectureConfigParser",
    "parse_custom_architecture",
    "parse_ensemble",
    "load_config",
    "validate_config",
    
    # Validation
    "ArchitectureValidator",
    "validate_architecture",
    "validate_data_compatibility",
    
    # Metadata
    "__version__",
]
