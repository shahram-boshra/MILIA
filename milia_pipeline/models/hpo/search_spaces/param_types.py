# param_types.py - Parameter Type Definitions for HPO Search Spaces

"""
Parameter type definitions for HPO search space configuration.

This module provides the core type definitions for hyperparameter search spaces,
including the ParamType enum for parameter type classification and the
SearchSpaceParamConfig frozen BaseModel for individual parameter configuration.

Location: milia_pipeline/models/hpo/search_spaces/param_types.py
Pattern: Follows frozen BaseModel pattern from config_containers.py (Pydantic V2)
Dependencies: None (standalone module within HPO)

Pydantic V2 Migration (Phase 6a):
    - Migrated from @dataclass(frozen=True) to BaseModel with frozen=True
    - Uses @field_validator for type-specific validation
    - Uses @model_validator(mode='before') for cross-field initialization
    - NON-BREAKING: Same constructor API and attribute access

Classes:
    ParamType: Enum defining supported parameter types for search spaces
    SearchSpaceParamConfig: Frozen BaseModel for single hyperparameter configuration

Example:
    >>> from milia_pipeline.models.hpo.search_spaces.param_types import (
    ...     ParamType,
    ...     SearchSpaceParamConfig,
    ... )
    >>> 
    >>> # Integer parameter with step
    >>> hidden_channels = SearchSpaceParamConfig(
    ...     type=ParamType.INT,
    ...     low=32,
    ...     high=256,
    ...     step=32,
    ... )
    >>> 
    >>> # Log-uniform parameter for learning rate
    >>> lr = SearchSpaceParamConfig(
    ...     type=ParamType.LOGUNIFORM,
    ...     low=1e-5,
    ...     high=1e-2,
    ... )
    >>> 
    >>> # Categorical parameter
    >>> activation = SearchSpaceParamConfig(
    ...     type=ParamType.CATEGORICAL,
    ...     choices=['relu', 'elu', 'leaky_relu'],
    ... )
"""

from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, field_validator, model_validator


class ParamType(Enum):
    """
    Parameter types for search space definition.
    
    Defines the supported hyperparameter types that can be used in
    search space configurations for HPO optimization.
    
    Attributes:
        INT: Integer parameter with optional step size
        FLOAT: Floating-point parameter (linear scale)
        CATEGORICAL: Categorical parameter with discrete choices
        LOGUNIFORM: Log-uniform distributed parameter
        UNIFORM: Uniformly distributed floating-point parameter
        INT_UNIFORM: Uniformly distributed integer parameter
        DISCRETE_UNIFORM: Discrete uniform distribution with step
    
    Example:
        >>> param_type = ParamType.LOGUNIFORM
        >>> print(param_type.value)
        'loguniform'
    """
    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    LOGUNIFORM = "loguniform"
    UNIFORM = "uniform"
    INT_UNIFORM = "int_uniform"
    DISCRETE_UNIFORM = "discrete_uniform"


class SearchSpaceParamConfig(BaseModel, frozen=True):
    """
    Configuration for a single hyperparameter in search space.
    
    Pattern: Follows frozen BaseModel pattern from config_containers.py (Pydantic V2)
    
    This frozen BaseModel defines the configuration for a single hyperparameter
    that will be optimized during HPO. It supports various parameter types
    including numeric ranges and categorical choices.
    
    Attributes:
        type: Parameter type (int, float, categorical, loguniform, etc.)
        low: Lower bound for numeric types (required for numeric types)
        high: Upper bound for numeric types (required for numeric types)
        step: Step size for int types (optional, default=1)
        choices: List of choices for categorical type (required for categorical)
        log: Whether to use log scale (for float type, default=False)
    
    Raises:
        ValueError: If validation fails based on parameter type
    
    Example:
        >>> # Valid integer parameter
        >>> config = SearchSpaceParamConfig(
        ...     type=ParamType.INT,
        ...     low=2,
        ...     high=10,
        ...     step=2,
        ... )
        >>> 
        >>> # Valid categorical parameter
        >>> config = SearchSpaceParamConfig(
        ...     type=ParamType.CATEGORICAL,
        ...     choices=['adam', 'sgd', 'adamw'],
        ... )
        >>> 
        >>> # Invalid: missing bounds for numeric type
        >>> config = SearchSpaceParamConfig(
        ...     type=ParamType.FLOAT,
        ...     low=0.0,
        ...     # high is missing - raises ValueError
        ... )
    """
    type: ParamType
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[int] = None
    choices: Optional[List[Any]] = None
    log: bool = False
    
    @model_validator(mode='before')
    @classmethod
    def validate_type_requirements(cls, data: Any) -> Any:
        """
        Validate configuration requirements based on parameter type.
        
        Uses mode='before' to validate before field assignment, following
        the established pattern from config_containers.py for frozen models.
        
        Raises:
            ValueError: If required fields are missing or invalid for the type
        """
        if isinstance(data, dict):
            param_type = data.get('type')
            
            # Convert string to enum if needed
            if isinstance(param_type, str):
                try:
                    param_type = ParamType(param_type)
                except ValueError:
                    # Let Pydantic handle the enum validation error
                    return data
            
            # Numeric types require low and high
            numeric_types = (
                ParamType.INT, ParamType.FLOAT, ParamType.LOGUNIFORM,
                ParamType.UNIFORM, ParamType.INT_UNIFORM, ParamType.DISCRETE_UNIFORM
            )
            
            if param_type in numeric_types:
                low = data.get('low')
                high = data.get('high')
                
                if low is None or high is None:
                    raise ValueError(
                        f"Parameter type '{param_type.value}' requires 'low' and 'high'. "
                        f"Got low={low}, high={high}"
                    )
                
                if low >= high:
                    raise ValueError(
                        f"'low' must be less than 'high'. "
                        f"Got low={low}, high={high}"
                    )
            
            # Categorical requires non-empty choices
            if param_type == ParamType.CATEGORICAL:
                choices = data.get('choices')
                if not choices or len(choices) == 0:
                    raise ValueError(
                        "Categorical parameter requires non-empty 'choices' list"
                    )
        
        return data
    
    def to_dict(self) -> dict:
        """Backward compatible dict conversion."""
        return self.model_dump()
