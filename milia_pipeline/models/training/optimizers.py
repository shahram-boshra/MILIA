"""
Optimizers Registry

Centralized registry for optimizers with:
- Common PyTorch optimizers (Adam, SGD, etc.)
- Easy instantiation via string names
- Parameter group support

Author: milia Team
Version: 1.0.0
"""

import logging
from typing import Dict, Any, List, Optional, Iterator

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


# =============================================================================
# OPTIMIZER REGISTRY
# =============================================================================

class OptimizerRegistry:
    """
    Centralized registry for optimizers.
    
    Provides easy access to common PyTorch optimizers via string names.
    
    Usage:
        >>> from milia_pipeline.models import OptimizerRegistry
        >>> optimizer = OptimizerRegistry.get_optimizer(
        ...     "adam",
        ...     model.parameters(),
        ...     {"lr": 0.001, "weight_decay": 1e-5}
        ... )
    """
    
    _optimizers = {
        # =====================================================================
        # ADAPTIVE LEARNING RATE OPTIMIZERS
        # =====================================================================
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "adamax": torch.optim.Adamax,
        "adadelta": torch.optim.Adadelta,
        "adagrad": torch.optim.Adagrad,
        "rmsprop": torch.optim.RMSprop,
        
        # =====================================================================
        # STOCHASTIC GRADIENT DESCENT VARIANTS
        # =====================================================================
        "sgd": torch.optim.SGD,
        "asgd": torch.optim.ASGD,
        
        # =====================================================================
        # SECOND-ORDER METHODS
        # =====================================================================
        "lbfgs": torch.optim.LBFGS,
        
        # =====================================================================
        # OTHER OPTIMIZERS
        # =====================================================================
        "rprop": torch.optim.Rprop,
        "nadam": torch.optim.NAdam,
        "radam": torch.optim.RAdam,
    }
    
    # Default parameters for each optimizer
    _defaults = {
        "adam": {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0},
        "adamw": {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01},
        "sgd": {"lr": 0.01, "momentum": 0, "dampening": 0, "weight_decay": 0, "nesterov": False},
        "rmsprop": {"lr": 0.01, "alpha": 0.99, "eps": 1e-8, "weight_decay": 0, "momentum": 0},
        "adagrad": {"lr": 0.01, "lr_decay": 0, "weight_decay": 0, "eps": 1e-10},
    }
    
    @classmethod
    def get_optimizer(
        cls,
        name: str,
        model_parameters: Iterator[nn.Parameter],
        params: Optional[Dict[str, Any]] = None
    ) -> torch.optim.Optimizer:
        """
        Get optimizer by name.
        
        Merges provided parameters with registry defaults and dynamically filters
        to only those accepted by the optimizer class. This prevents errors when
        unsupported parameters are passed.
        
        Args:
            name: Optimizer name (e.g., "adam", "sgd", "adamw")
            model_parameters: Model parameters (from model.parameters())
            params: Optional dictionary of optimizer parameters.
                   These override registry defaults. Invalid parameters
                   are automatically filtered out.
            
        Returns:
            Instantiated optimizer
            
        Raises:
            ValueError: If optimizer name not found in registry
            
        Example:
            >>> # Simple usage with registry defaults
            >>> optimizer = OptimizerRegistry.get_optimizer(
            ...     "adam",
            ...     model.parameters()
            ... )
            >>> 
            >>> # With custom parameters (merged with defaults)
            >>> optimizer = OptimizerRegistry.get_optimizer(
            ...     "adam",
            ...     model.parameters(),
            ...     {"lr": 0.001, "weight_decay": 1e-5}
            ... )
            >>> 
            >>> # Safe usage - invalid params are filtered
            >>> optimizer = OptimizerRegistry.get_optimizer(
            ...     "adam",
            ...     model.parameters(),
            ...     {"lr": 0.001, "invalid_param": 123}
            ... )
            >>> # Works! 'invalid_param' is filtered out
        """
        if name not in cls._optimizers:
            available = ', '.join(sorted(cls._optimizers.keys()))
            raise ValueError(
                f"Unknown optimizer: '{name}'. "
                f"Available optimizers: {available}"
            )
        
        opt_cls = cls._optimizers[name]
        
        # Merge defaults with provided params (provided params take precedence)
        merged_params = {**cls._defaults.get(name, {}), **(params or {})}
        
        # Filter parameters to only those accepted by the optimizer class
        filtered_params = cls._filter_params(opt_cls, merged_params)
        
        try:
            optimizer = opt_cls(model_parameters, **filtered_params)
            
            if filtered_params:
                logger.debug(f"Initialized {name} optimizer with params: {filtered_params}")
            else:
                logger.debug(f"Initialized {name} optimizer with default params")
            
            # Log filtered out params at debug level
            ignored = set(merged_params.keys()) - set(filtered_params.keys())
            if ignored:
                logger.debug(f"Optimizer '{name}': ignored unsupported params {ignored}")
            
            return optimizer
            
        except TypeError as e:
            raise ValueError(
                f"Invalid parameters for optimizer '{name}': {filtered_params}. Error: {e}"
            ) from e
    
    @classmethod
    def _filter_params(cls, target_cls: type, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter parameters to only those accepted by the target optimizer constructor.
        
        Uses inspect.signature() for dynamic introspection, ensuring only valid
        parameters are passed to the constructor. This is DYNAMIC (works with any
        optimizer), PRODUCTION-READY (handles edge cases), and FUTURE-PROOF (no 
        hardcoded parameter lists).
        
        Args:
            target_cls: The optimizer class whose constructor parameters to check
            params: Dictionary of parameters to filter
            
        Returns:
            Filtered dictionary containing only valid parameters
        """
        import inspect
        
        if not params:
            return {}
        
        try:
            sig = inspect.signature(target_cls.__init__)
            # Exclude 'self' and 'params' (the model parameters argument)
            valid_param_names = set(sig.parameters.keys()) - {'self', 'params'}
            filtered = {k: v for k, v in params.items() if k in valid_param_names}
            return filtered
        except (ValueError, TypeError):
            # Fallback: return original params if introspection fails
            return params
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available optimizer names.
        
        Returns:
            List of optimizer names
            
        Example:
            >>> optimizers = OptimizerRegistry.list_available()
            >>> print(f"Available optimizers: {', '.join(optimizers)}")
        """
        return sorted(cls._optimizers.keys())
    
    @classmethod
    def get_optimizer_info(cls, name: str) -> Dict[str, Any]:
        """
        Get information about an optimizer.
        
        Args:
            name: Optimizer name
            
        Returns:
            Dictionary with optimizer information including valid parameters
            
        Example:
            >>> info = OptimizerRegistry.get_optimizer_info("adam")
            >>> print(info['valid_params'])
        """
        if name not in cls._optimizers:
            raise ValueError(f"Unknown optimizer: '{name}'")
        
        opt_cls = cls._optimizers[name]
        
        return {
            'name': name,
            'class': opt_cls.__name__,
            'module': opt_cls.__module__,
            'default_params': cls._defaults.get(name, {}),
            'valid_params': cls.get_valid_params(name),
            'doc': opt_cls.__doc__
        }
    
    @classmethod
    def get_default_params(cls, name: str) -> Dict[str, Any]:
        """
        Get default parameters for an optimizer from registry defaults.
        
        Note: These are registry-defined defaults, not introspected from the class.
        Use get_valid_params() to see all valid parameters with their class defaults.
        
        Args:
            name: Optimizer name
            
        Returns:
            Dictionary of registry default parameters
            
        Example:
            >>> defaults = OptimizerRegistry.get_default_params("adam")
            >>> print(defaults)
            {'lr': 0.001, 'betas': (0.9, 0.999), ...}
        """
        if name not in cls._optimizers:
            raise ValueError(f"Unknown optimizer: '{name}'")
        
        return cls._defaults.get(name, {}).copy()
    
    @classmethod
    def get_valid_params(cls, name: str) -> Dict[str, Any]:
        """
        Get valid parameters for an optimizer using introspection.
        
        Uses inspect.signature() to dynamically discover what parameters
        the optimizer's constructor accepts, along with their default values.
        
        Args:
            name: Optimizer name
            
        Returns:
            Dictionary mapping parameter names to their default values
            (None if no default / required parameter)
            
        Example:
            >>> params = OptimizerRegistry.get_valid_params("adam")
            >>> print(params)
            {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, ...}
        """
        import inspect
        
        if name not in cls._optimizers:
            raise ValueError(f"Unknown optimizer: '{name}'")
        
        opt_cls = cls._optimizers[name]
        
        try:
            sig = inspect.signature(opt_cls.__init__)
            params = {}
            for param_name, param in sig.parameters.items():
                if param_name in ('self', 'params'):
                    continue
                if param.default is not inspect.Parameter.empty:
                    params[param_name] = param.default
                else:
                    params[param_name] = None  # Required param, no default
            return params
        except (ValueError, TypeError):
            # Cannot introspect (e.g., built-in C extension)
            return {}
    
    @classmethod
    def register_custom_optimizer(
        cls,
        name: str,
        optimizer_class: type,
        default_params: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ):
        """
        Register a custom optimizer.
        
        Args:
            name: Name to register optimizer under
            optimizer_class: Optimizer class (must be torch.optim.Optimizer subclass)
            default_params: Optional default parameters
            overwrite: Whether to overwrite existing optimizer with same name
            
        Raises:
            ValueError: If name exists and overwrite=False
            TypeError: If optimizer_class is not Optimizer subclass
            
        Example:
            >>> class MyOptimizer(torch.optim.Optimizer):
            ...     def __init__(self, params, lr=0.01):
            ...         defaults = dict(lr=lr)
            ...         super().__init__(params, defaults)
            ...     
            ...     def step(self, closure=None):
            ...         # Custom optimization step
            ...         pass
            >>> 
            >>> OptimizerRegistry.register_custom_optimizer(
            ...     "my_opt",
            ...     MyOptimizer,
            ...     {"lr": 0.01}
            ... )
        """
        if not issubclass(optimizer_class, torch.optim.Optimizer):
            raise TypeError(
                f"optimizer_class must be a subclass of torch.optim.Optimizer, "
                f"got {type(optimizer_class)}"
            )
        
        if name in cls._optimizers and not overwrite:
            raise ValueError(
                f"Optimizer '{name}' already registered. "
                f"Use overwrite=True to replace."
            )
        
        cls._optimizers[name] = optimizer_class
        if default_params:
            cls._defaults[name] = default_params
        
        logger.info(f"Registered custom optimizer: '{name}'")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_optimizer(
    name: str,
    model_parameters: Iterator[nn.Parameter],
    params: Optional[Dict[str, Any]] = None
) -> torch.optim.Optimizer:
    """
    Convenience function to get optimizer.
    
    Example:
        >>> from milia_pipeline.models import get_optimizer
        >>> optimizer = get_optimizer("adam", model.parameters(), {"lr": 0.001})
    """
    return OptimizerRegistry.get_optimizer(name, model_parameters, params)


def list_optimizers() -> List[str]:
    """
    Convenience function to list available optimizers.
    
    Example:
        >>> from milia_pipeline.models import list_optimizers
        >>> print(list_optimizers())
    """
    return OptimizerRegistry.list_available()


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info(
    f"optimizers module loaded - {len(OptimizerRegistry._optimizers)} optimizers available"
)
