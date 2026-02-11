# search_space.py - GNN Architecture Search Space for Neural Architecture Search

"""
GNN Architecture Search Space Module

Defines searchable architecture components for graph neural networks,
enabling automated discovery of optimal GNN architectures through
Neural Architecture Search (NAS).

Location: milia_pipeline/models/hpo/nas/search_space.py
Pattern: Follows frozen dataclass pattern from config_bridge.py (lines 167-649)
         and hpo_config.py dataclass patterns
Dependencies: 
    - milia_pipeline.exceptions (SearchSpaceError, ConfigurationError)
    - hpo_config.py (configuration patterns)

Classes:
    LayerType: Enum of available GNN layer types
    PoolingType: Enum of graph pooling strategies
    AggregationType: Enum of message aggregation types
    ActivationType: Enum of activation functions
    LayerConfig: Frozen dataclass for single GNN layer configuration
    GNNArchitectureSpace: Main dataclass defining the GNN search space

Integration Points:
    - Used by NASManager (nas_manager.py) for architecture search
    - Exported via hpo/nas/__init__.py
    - Returns Optuna-compatible search space format via to_optuna_search_space()

Example:
    >>> from milia_pipeline.models.hpo.nas.search_space import (
    ...     GNNArchitectureSpace,
    ...     LayerType,
    ...     PoolingType,
    ... )
    >>> 
    >>> # Create default search space
    >>> space = GNNArchitectureSpace()
    >>> 
    >>> # Custom search space for attention models
    >>> attention_space = GNNArchitectureSpace(
    ...     layer_types=[LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER],
    ...     heads=[1, 2, 4, 8, 16],
    ...     allow_mixed_layers=True,
    ... )
    >>> 
    >>> # Convert to Optuna search space format
    >>> optuna_space = space.to_optuna_search_space()

Blueprint Reference: MILIA_HPO_Implementation_Blueprint.md (lines 4146-4285)

Author: Milia Team
Version: 1.1.0

Pydantic V2 Migration (Phase 14):
    - Migrated LayerConfig from @dataclass(frozen=True) to BaseModel with frozen=True
    - Migrated GNNArchitectureSpace from @dataclass to mutable BaseModel
    - Uses @field_validator for individual field validation
    - Uses @model_validator(mode='after') for cross-field validation
    - NON-BREAKING: Same constructor API, attribute access, to_dict(), from_dict() preserved
    - Follows established patterns from warm_start.py (Phase 13) and hpo_config.py (Phase 6b)
"""

from pydantic import BaseModel, field_validator, model_validator, Field
from typing_extensions import Self
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


# =============================================================================
# GNN LAYER TYPE ENUM
# =============================================================================

class LayerType(Enum):
    """
    Available GNN layer types for architecture search.
    
    Defines the message-passing layer types that can be searched
    during neural architecture search. Each type has different
    computational characteristics and expressiveness.
    
    Attributes:
        GCN: Graph Convolutional Network layer (Kipf & Welling, 2017)
        GAT: Graph Attention Network layer (Veličković et al., 2018)
        SAGE: GraphSAGE layer with neighborhood sampling (Hamilton et al., 2017)
        GIN: Graph Isomorphism Network layer (Xu et al., 2019)
        GATV2: Improved Graph Attention layer (Brody et al., 2021)
        TRANSFORMER: Graph Transformer layer with self-attention
        PNA: Principal Neighbourhood Aggregation layer (Corso et al., 2020)
    
    Example:
        >>> layer = LayerType.GAT
        >>> print(layer.value)
        'gat'
        >>> 
        >>> # Check if layer supports attention heads
        >>> attention_layers = [LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER]
        >>> if layer in attention_layers:
        ...     print("Layer supports attention heads")
    """
    GCN = "gcn"
    GAT = "gat"
    SAGE = "sage"
    GIN = "gin"
    GATV2 = "gatv2"
    TRANSFORMER = "transformer"
    PNA = "pna"


# =============================================================================
# GRAPH POOLING TYPE ENUM
# =============================================================================

class PoolingType(Enum):
    """
    Available graph pooling types for readout operations.
    
    Defines strategies for aggregating node representations into
    graph-level representations for graph classification/regression.
    
    Attributes:
        MEAN: Mean pooling over all nodes
        MAX: Max pooling over all nodes
        SUM: Sum pooling over all nodes
        ATTENTION: Attention-weighted pooling (Set2Set variant)
        SET2SET: Set2Set pooling with LSTM (Vinyals et al., 2016)
        TOPK: Top-K pooling with learnable scoring (Gao & Ji, 2019)
    
    Example:
        >>> pooling = PoolingType.ATTENTION
        >>> print(pooling.value)
        'attention'
    """
    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    ATTENTION = "attention"
    SET2SET = "set2set"
    TOPK = "topk"


# =============================================================================
# AGGREGATION TYPE ENUM
# =============================================================================

class AggregationType(Enum):
    """
    Available message aggregation types for GNN layers.
    
    Defines how messages from neighboring nodes are aggregated
    in message-passing neural networks.
    
    Attributes:
        MEAN: Mean aggregation of neighbor messages
        MAX: Max aggregation of neighbor messages
        SUM: Sum aggregation of neighbor messages
        LSTM: LSTM-based sequential aggregation
        MULTI: Multi-aggregator (PNA-style) combining multiple methods
    
    Example:
        >>> agg = AggregationType.MULTI
        >>> print(agg.value)
        'multi'
    """
    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    LSTM = "lstm"
    MULTI = "multi"


# =============================================================================
# ACTIVATION TYPE ENUM
# =============================================================================

class ActivationType(Enum):
    """
    Available activation functions for GNN layers.
    
    Defines non-linear activation functions used between layers
    in graph neural network architectures.
    
    Attributes:
        RELU: Rectified Linear Unit
        GELU: Gaussian Error Linear Unit
        ELU: Exponential Linear Unit
        LEAKY_RELU: Leaky Rectified Linear Unit
        SILU: Sigmoid Linear Unit (Swish)
        TANH: Hyperbolic tangent
        PRELU: Parametric ReLU
    
    Example:
        >>> activation = ActivationType.GELU
        >>> print(activation.value)
        'gelu'
    """
    RELU = "relu"
    GELU = "gelu"
    ELU = "elu"
    LEAKY_RELU = "leaky_relu"
    SILU = "silu"
    TANH = "tanh"
    PRELU = "prelu"


# =============================================================================
# LAYER CONFIGURATION (Pydantic V2 BaseModel)
# =============================================================================

class LayerConfig(BaseModel, frozen=True):
    """
    Configuration for a single GNN layer.
    
    Pattern: Follows frozen BaseModel pattern from warm_start.py (Pydantic V2)
    
    This frozen BaseModel defines the configuration for an individual
    GNN layer within an architecture. It captures layer type, dimensions,
    regularization, and architectural choices.
    
    Attributes:
        type: GNN layer type (GCN, GAT, SAGE, etc.)
        hidden_channels: Number of hidden channels/features
        heads: Number of attention heads (for attention-based layers)
        dropout: Dropout probability for regularization
        activation: Activation function name
        batch_norm: Whether to apply batch normalization
        residual: Whether to use residual/skip connection
    
    Raises:
        ValidationError: If validation fails (e.g., negative values)
    
    Example:
        >>> # Basic GCN layer
        >>> layer = LayerConfig(
        ...     type=LayerType.GCN,
        ...     hidden_channels=64,
        ... )
        >>> 
        >>> # GAT layer with multi-head attention
        >>> gat_layer = LayerConfig(
        ...     type=LayerType.GAT,
        ...     hidden_channels=64,
        ...     heads=4,
        ...     dropout=0.2,
        ...     batch_norm=True,
        ...     residual=True,
        ... )
    """
    type: LayerType
    hidden_channels: int
    heads: int = 1
    dropout: float = 0.0
    activation: str = "relu"
    batch_norm: bool = True
    residual: bool = False
    
    @field_validator('hidden_channels')
    @classmethod
    def validate_hidden_channels(cls, v: int) -> int:
        """Validate hidden_channels is positive."""
        if v <= 0:
            raise ValueError("hidden_channels must be positive")
        return v
    
    @field_validator('heads')
    @classmethod
    def validate_heads(cls, v: int) -> int:
        """Validate heads is at least 1."""
        if v < 1:
            raise ValueError("heads must be at least 1")
        return v
    
    @field_validator('dropout')
    @classmethod
    def validate_dropout(cls, v: float) -> float:
        """Validate dropout is between 0 and 1."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("dropout must be between 0 and 1")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert layer configuration to dictionary.
        
        Returns:
            Dictionary representation of the layer configuration.
        
        Example:
            >>> layer = LayerConfig(type=LayerType.GAT, hidden_channels=64, heads=4)
            >>> config_dict = layer.to_dict()
            >>> print(config_dict['type'])
            'gat'
        """
        data = self.model_dump()
        # Convert enum to string value for backward compatibility
        data['type'] = self.type.value
        return data
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LayerConfig':
        """
        Create LayerConfig from dictionary.
        
        Args:
            config_dict: Dictionary containing layer configuration.
        
        Returns:
            LayerConfig instance.
        
        Example:
            >>> config = {'type': 'gat', 'hidden_channels': 64, 'heads': 4}
            >>> layer = LayerConfig.from_dict(config)
        """
        config_copy = config_dict.copy()
        if isinstance(config_copy.get('type'), str):
            config_copy['type'] = LayerType(config_copy['type'])
        return cls.model_validate(config_copy)


# =============================================================================
# GNN ARCHITECTURE SEARCH SPACE (Pydantic V2 BaseModel)
# =============================================================================

class GNNArchitectureSpace(BaseModel):
    """
    Defines the search space for GNN architectures.
    
    Pattern: Follows mutable BaseModel pattern from warm_start.py (Pydantic V2)
    Blueprint Reference: MILIA_HPO_Implementation_Blueprint.md (lines 4203-4285)
    
    This BaseModel defines the complete search space for neural architecture
    search over graph neural networks. It includes searchable components for:
    - Network depth (number of layers)
    - Layer types (can vary per layer if allow_mixed_layers=True)
    - Hidden dimensions
    - Attention heads (for attention-based layers)
    - Skip/residual connections
    - Pooling strategy for graph-level tasks
    - Message aggregation functions
    
    Note: This BaseModel is NOT frozen to allow modification of search
    space parameters before optimization. Uses Field(default_factory=...)
    for mutable default values.
    
    Attributes:
        min_layers: Minimum number of GNN layers
        max_layers: Maximum number of GNN layers
        layer_types: List of layer types to search over
        hidden_channels: List of hidden channel options
        heads: List of attention head options
        dropout_range: Tuple of (min, max) dropout values
        allow_skip_connections: Whether to search over skip connections
        allow_dense_connections: Whether to allow DenseGCN-style connections
        allow_mixed_layers: Whether different layers can have different types
        pooling_types: List of pooling types for graph-level readout
        aggregation_types: List of aggregation types for message passing
        activation_types: List of activation functions to search
        batch_norm_options: List of batch normalization options [True, False]
    
    Example:
        >>> # Default search space
        >>> space = GNNArchitectureSpace()
        >>> 
        >>> # Narrow search space for attention models
        >>> attention_space = GNNArchitectureSpace(
        ...     min_layers=2,
        ...     max_layers=4,
        ...     layer_types=[LayerType.GAT, LayerType.GATV2],
        ...     heads=[2, 4, 8],
        ...     hidden_channels=[64, 128],
        ... )
        >>> 
        >>> # Convert to Optuna format
        >>> optuna_space = space.to_optuna_search_space()
    """
    
    # =========================================================================
    # LAYER SEARCH SPACE
    # =========================================================================
    min_layers: int = 2
    max_layers: int = 8
    layer_types: List[LayerType] = Field(
        default_factory=lambda: [LayerType.GCN, LayerType.GAT, LayerType.SAGE]
    )
    
    # =========================================================================
    # DIMENSION SEARCH SPACE
    # =========================================================================
    hidden_channels: List[int] = Field(
        default_factory=lambda: [32, 64, 128, 256]
    )
    
    # =========================================================================
    # ATTENTION SEARCH SPACE (for GAT, Transformer)
    # =========================================================================
    heads: List[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    
    # =========================================================================
    # REGULARIZATION SEARCH SPACE
    # =========================================================================
    dropout_range: Tuple[float, float] = (0.0, 0.6)
    
    # =========================================================================
    # ARCHITECTURE OPTIONS
    # =========================================================================
    allow_skip_connections: bool = True
    allow_dense_connections: bool = False
    allow_mixed_layers: bool = True
    
    # =========================================================================
    # POOLING SEARCH SPACE
    # =========================================================================
    pooling_types: List[PoolingType] = Field(
        default_factory=lambda: [PoolingType.MEAN, PoolingType.ATTENTION]
    )
    
    # =========================================================================
    # AGGREGATION SEARCH SPACE
    # =========================================================================
    aggregation_types: List[AggregationType] = Field(
        default_factory=lambda: [AggregationType.MEAN, AggregationType.SUM]
    )
    
    # =========================================================================
    # ACTIVATION SEARCH SPACE
    # =========================================================================
    activation_types: List[ActivationType] = Field(
        default_factory=lambda: [ActivationType.RELU, ActivationType.GELU, ActivationType.ELU]
    )
    
    # =========================================================================
    # BATCH NORMALIZATION OPTIONS
    # =========================================================================
    batch_norm_options: List[bool] = Field(
        default_factory=lambda: [True, False]
    )
    
    @field_validator('min_layers')
    @classmethod
    def validate_min_layers(cls, v: int) -> int:
        """Validate min_layers is at least 1."""
        if v < 1:
            raise ValueError("min_layers must be at least 1")
        return v
    
    @field_validator('hidden_channels')
    @classmethod
    def validate_hidden_channels_list(cls, v: List[int]) -> List[int]:
        """Validate hidden_channels list is not empty and all values are positive."""
        if not v:
            raise ValueError("hidden_channels cannot be empty")
        for hc in v:
            if hc <= 0:
                raise ValueError(f"hidden_channels values must be positive, got {hc}")
        return v
    
    @field_validator('heads')
    @classmethod
    def validate_heads_list(cls, v: List[int]) -> List[int]:
        """Validate heads list is not empty and all values are >= 1."""
        if not v:
            raise ValueError("heads cannot be empty")
        for h in v:
            if h < 1:
                raise ValueError(f"heads values must be >= 1, got {h}")
        return v
    
    @field_validator('dropout_range')
    @classmethod
    def validate_dropout_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        """Validate dropout_range is valid tuple with 0 <= min <= max <= 1."""
        if len(v) != 2:
            raise ValueError("dropout_range must be a tuple of (min, max)")
        if not (0.0 <= v[0] <= v[1] <= 1.0):
            raise ValueError("dropout_range must satisfy 0 <= min <= max <= 1")
        return v
    
    @field_validator('layer_types')
    @classmethod
    def validate_layer_types(cls, v: List[LayerType]) -> List[LayerType]:
        """Validate layer_types list is not empty."""
        if not v:
            raise ValueError("layer_types cannot be empty")
        return v
    
    @field_validator('pooling_types')
    @classmethod
    def validate_pooling_types(cls, v: List[PoolingType]) -> List[PoolingType]:
        """Validate pooling_types list is not empty."""
        if not v:
            raise ValueError("pooling_types cannot be empty")
        return v
    
    @field_validator('aggregation_types')
    @classmethod
    def validate_aggregation_types(cls, v: List[AggregationType]) -> List[AggregationType]:
        """Validate aggregation_types list is not empty."""
        if not v:
            raise ValueError("aggregation_types cannot be empty")
        return v
    
    @model_validator(mode='after')
    def validate_layer_range(self) -> Self:
        """Validate max_layers >= min_layers (cross-field validation)."""
        if self.max_layers < self.min_layers:
            raise ValueError("max_layers must be >= min_layers")
        return self
    
    def to_optuna_search_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert architecture space to Optuna search space format.
        
        Generates a dictionary structure compatible with Optuna's search
        space configuration. The output can be used directly with
        HPO search space builders.
        
        Returns:
            Dictionary with 'architecture' key containing parameter definitions
            in Optuna-compatible format (type, low/high or choices).
        
        Example:
            >>> space = GNNArchitectureSpace()
            >>> optuna_space = space.to_optuna_search_space()
            >>> print(optuna_space['architecture']['num_layers'])
            {'type': 'int', 'low': 2, 'high': 8}
            >>> 
            >>> # Per-layer type choices (if allow_mixed_layers=True)
            >>> print(optuna_space['architecture']['layer_0_type'])
            {'type': 'categorical', 'choices': ['gcn', 'gat', 'sage']}
        """
        space: Dict[str, Dict[str, Any]] = {
            'architecture': {
                'num_layers': {
                    'type': 'int',
                    'low': self.min_layers,
                    'high': self.max_layers,
                },
                'hidden_channels': {
                    'type': 'categorical',
                    'choices': self.hidden_channels,
                },
                'pooling': {
                    'type': 'categorical',
                    'choices': [p.value for p in self.pooling_types],
                },
                'dropout': {
                    'type': 'float',
                    'low': self.dropout_range[0],
                    'high': self.dropout_range[1],
                },
                'aggregation': {
                    'type': 'categorical',
                    'choices': [a.value for a in self.aggregation_types],
                },
                'activation': {
                    'type': 'categorical',
                    'choices': [a.value for a in self.activation_types],
                },
                'batch_norm': {
                    'type': 'categorical',
                    'choices': self.batch_norm_options,
                },
            }
        }
        
        if self.allow_skip_connections:
            space['architecture']['use_skip_connections'] = {
                'type': 'categorical',
                'choices': [True, False],
            }
        
        if self.allow_dense_connections:
            space['architecture']['use_dense_connections'] = {
                'type': 'categorical',
                'choices': [True, False],
            }
        
        if self.allow_mixed_layers:
            for i in range(self.max_layers):
                space['architecture'][f'layer_{i}_type'] = {
                    'type': 'categorical',
                    'choices': [lt.value for lt in self.layer_types],
                }
                attention_layer_types = [
                    LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER
                ]
                if any(lt in self.layer_types for lt in attention_layer_types):
                    space['architecture'][f'layer_{i}_heads'] = {
                        'type': 'categorical',
                        'choices': self.heads,
                    }
        else:
            space['architecture']['layer_type'] = {
                'type': 'categorical',
                'choices': [lt.value for lt in self.layer_types],
            }
            attention_layer_types = [
                LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER
            ]
            if any(lt in self.layer_types for lt in attention_layer_types):
                space['architecture']['heads'] = {
                    'type': 'categorical',
                    'choices': self.heads,
                }
        
        return space
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert architecture space to dictionary representation.
        
        Returns:
            Dictionary containing all search space configuration.
        
        Example:
            >>> space = GNNArchitectureSpace()
            >>> config = space.to_dict()
            >>> print(config['min_layers'])
            2
        """
        return {
            'min_layers': self.min_layers,
            'max_layers': self.max_layers,
            'layer_types': [lt.value for lt in self.layer_types],
            'hidden_channels': self.hidden_channels,
            'heads': self.heads,
            'dropout_range': self.dropout_range,
            'allow_skip_connections': self.allow_skip_connections,
            'allow_dense_connections': self.allow_dense_connections,
            'allow_mixed_layers': self.allow_mixed_layers,
            'pooling_types': [pt.value for pt in self.pooling_types],
            'aggregation_types': [at.value for at in self.aggregation_types],
            'activation_types': [at.value for at in self.activation_types],
            'batch_norm_options': self.batch_norm_options,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GNNArchitectureSpace':
        """
        Create GNNArchitectureSpace from dictionary.
        
        Pattern: Follows HPOConfig.from_dict() (hpo_config.py:589-658)
        
        Args:
            config_dict: Dictionary containing architecture space configuration.
        
        Returns:
            GNNArchitectureSpace instance with validated configuration.
        
        Raises:
            SearchSpaceError: If configuration is invalid.
        
        Example:
            >>> config = {
            ...     'min_layers': 2,
            ...     'max_layers': 6,
            ...     'layer_types': ['gcn', 'gat'],
            ...     'hidden_channels': [64, 128, 256],
            ... }
            >>> space = GNNArchitectureSpace.from_dict(config)
        """
        config_copy = config_dict.copy()
        
        if 'layer_types' in config_copy:
            config_copy['layer_types'] = [
                LayerType(lt) if isinstance(lt, str) else lt
                for lt in config_copy['layer_types']
            ]
        
        if 'pooling_types' in config_copy:
            config_copy['pooling_types'] = [
                PoolingType(pt) if isinstance(pt, str) else pt
                for pt in config_copy['pooling_types']
            ]
        
        if 'aggregation_types' in config_copy:
            config_copy['aggregation_types'] = [
                AggregationType(at) if isinstance(at, str) else at
                for at in config_copy['aggregation_types']
            ]
        
        if 'activation_types' in config_copy:
            config_copy['activation_types'] = [
                ActivationType(at) if isinstance(at, str) else at
                for at in config_copy['activation_types']
            ]
        
        if 'dropout_range' in config_copy and isinstance(config_copy['dropout_range'], list):
            config_copy['dropout_range'] = tuple(config_copy['dropout_range'])
        
        return cls.model_validate(config_copy)
    
    def get_attention_layer_types(self) -> List[LayerType]:
        """
        Get layer types that support attention heads.
        
        Returns:
            List of layer types that use multi-head attention.
        
        Example:
            >>> space = GNNArchitectureSpace()
            >>> attention_types = space.get_attention_layer_types()
            >>> print([lt.value for lt in attention_types])
            ['gat']
        """
        attention_types = [LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER]
        return [lt for lt in self.layer_types if lt in attention_types]
    
    def has_attention_layers(self) -> bool:
        """
        Check if search space includes attention-based layers.
        
        Returns:
            True if any layer type supports attention heads.
        
        Example:
            >>> space = GNNArchitectureSpace(layer_types=[LayerType.GCN])
            >>> print(space.has_attention_layers())
            False
        """
        return len(self.get_attention_layer_types()) > 0
    
    def get_search_dimensions(self) -> int:
        """
        Calculate the approximate number of search dimensions.
        
        Returns:
            Integer representing the number of hyperparameters being searched.
        
        Example:
            >>> space = GNNArchitectureSpace()
            >>> dims = space.get_search_dimensions()
            >>> print(f"Searching over {dims} dimensions")
        """
        dims = 7
        
        if self.allow_skip_connections:
            dims += 1
        
        if self.allow_dense_connections:
            dims += 1
        
        if self.allow_mixed_layers:
            dims += self.max_layers
            if self.has_attention_layers():
                dims += self.max_layers
        else:
            dims += 1
            if self.has_attention_layers():
                dims += 1
        
        return dims
    
    def estimate_search_space_size(self) -> int:
        """
        Estimate the total size of the search space.
        
        Calculates the approximate number of unique architectures
        that can be explored within this search space.
        
        Returns:
            Estimated number of unique architecture configurations.
        
        Note:
            This is an approximation; actual search space may be smaller
            due to constraints and dependencies between parameters.
        
        Example:
            >>> space = GNNArchitectureSpace()
            >>> size = space.estimate_search_space_size()
            >>> print(f"Search space contains ~{size:,} architectures")
        """
        num_layer_options = self.max_layers - self.min_layers + 1
        num_hidden_options = len(self.hidden_channels)
        num_pooling_options = len(self.pooling_types)
        num_activation_options = len(self.activation_types)
        num_aggregation_options = len(self.aggregation_types)
        num_batchnorm_options = len(self.batch_norm_options)
        
        base_size = (
            num_layer_options *
            num_hidden_options *
            num_pooling_options *
            num_activation_options *
            num_aggregation_options *
            num_batchnorm_options *
            10
        )
        
        if self.allow_skip_connections:
            base_size *= 2
        
        if self.allow_dense_connections:
            base_size *= 2
        
        if self.allow_mixed_layers:
            num_layer_type_options = len(self.layer_types)
            base_size *= num_layer_type_options ** self.max_layers
            if self.has_attention_layers():
                num_head_options = len(self.heads)
                base_size *= num_head_options ** self.max_layers
        else:
            base_size *= len(self.layer_types)
            if self.has_attention_layers():
                base_size *= len(self.heads)
        
        return base_size
    
    def create_default_layer_config(
        self,
        layer_type: Optional[LayerType] = None,
        hidden_channels: Optional[int] = None,
    ) -> LayerConfig:
        """
        Create a default LayerConfig from this search space.
        
        Creates a LayerConfig using the first (default) options from
        each search space parameter.
        
        Args:
            layer_type: Override layer type (uses first in list if None)
            hidden_channels: Override hidden channels (uses first if None)
        
        Returns:
            LayerConfig with default parameters from search space.
        
        Example:
            >>> space = GNNArchitectureSpace()
            >>> default_layer = space.create_default_layer_config()
            >>> print(default_layer.type.value)
            'gcn'
        """
        lt = layer_type if layer_type is not None else self.layer_types[0]
        hc = hidden_channels if hidden_channels is not None else self.hidden_channels[0]
        
        heads = 1
        if lt in [LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER]:
            heads = self.heads[0] if self.heads else 1
        
        return LayerConfig(
            type=lt,
            hidden_channels=hc,
            heads=heads,
            dropout=self.dropout_range[0],
            activation=self.activation_types[0].value if self.activation_types else "relu",
            batch_norm=self.batch_norm_options[0] if self.batch_norm_options else True,
            residual=self.allow_skip_connections,
        )


# =============================================================================
# CONVENIENCE FACTORY FUNCTIONS
# =============================================================================

def create_gnn_search_space(
    model_type: Optional[str] = None,
    **kwargs,
) -> GNNArchitectureSpace:
    """
    Factory function to create a GNNArchitectureSpace with preset configurations.
    
    Creates a search space optionally tailored to a specific model type
    or with custom parameters provided via kwargs.
    
    Args:
        model_type: Optional model type for preset configuration.
                   Supported: 'gcn', 'gat', 'sage', 'gin', 'transformer', 'pna'
        **kwargs: Additional parameters to override defaults.
    
    Returns:
        Configured GNNArchitectureSpace instance.
    
    Raises:
        ValueError: If model_type is not recognized.
    
    Example:
        >>> # Default search space
        >>> space = create_gnn_search_space()
        >>> 
        >>> # GAT-focused search space
        >>> gat_space = create_gnn_search_space('gat')
        >>> 
        >>> # Custom search space
        >>> custom_space = create_gnn_search_space(
        ...     min_layers=3,
        ...     max_layers=10,
        ...     hidden_channels=[128, 256, 512],
        ... )
    """
    presets: Dict[str, Dict[str, Any]] = {
        'gcn': {
            'layer_types': [LayerType.GCN],
            'heads': [1],
            'allow_mixed_layers': False,
        },
        'gat': {
            'layer_types': [LayerType.GAT, LayerType.GATV2],
            'heads': [1, 2, 4, 8],
            'allow_mixed_layers': False,
        },
        'sage': {
            'layer_types': [LayerType.SAGE],
            'aggregation_types': [AggregationType.MEAN, AggregationType.MAX, AggregationType.LSTM],
            'heads': [1],
            'allow_mixed_layers': False,
        },
        'gin': {
            'layer_types': [LayerType.GIN],
            'heads': [1],
            'allow_mixed_layers': False,
        },
        'transformer': {
            'layer_types': [LayerType.TRANSFORMER],
            'heads': [2, 4, 8, 16],
            'allow_mixed_layers': False,
        },
        'pna': {
            'layer_types': [LayerType.PNA],
            'aggregation_types': [AggregationType.MULTI],
            'heads': [1],
            'allow_mixed_layers': False,
        },
        'mixed': {
            'layer_types': [LayerType.GCN, LayerType.GAT, LayerType.SAGE, LayerType.GIN],
            'allow_mixed_layers': True,
        },
    }
    
    if model_type is not None:
        model_type_lower = model_type.lower()
        if model_type_lower not in presets:
            valid_types = list(presets.keys())
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Valid options: {valid_types}"
            )
        config = presets[model_type_lower].copy()
        config.update(kwargs)
        return GNNArchitectureSpace(**config)
    
    return GNNArchitectureSpace(**kwargs)


def get_default_gnn_search_space() -> GNNArchitectureSpace:
    """
    Get the default GNN architecture search space.
    
    Returns a search space suitable for general-purpose GNN
    architecture search with reasonable defaults.
    
    Returns:
        Default GNNArchitectureSpace instance.
    
    Example:
        >>> space = get_default_gnn_search_space()
        >>> print(space.min_layers, space.max_layers)
        2 8
    """
    return GNNArchitectureSpace()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'LayerType',
    'PoolingType',
    'AggregationType',
    'ActivationType',
    'LayerConfig',
    'GNNArchitectureSpace',
    'create_gnn_search_space',
    'get_default_gnn_search_space',
]


# =============================================================================
# MODULE METADATA
# =============================================================================

__version__ = '1.1.0'
__author__ = 'Milia Team'
