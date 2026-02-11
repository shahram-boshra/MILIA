"""
Model Registry System

Central registry for model discovery, registration, and management.
Thread-safe singleton pattern following descriptors module architecture.

**Version 1.1.0 (Phase 3 Migration)**:
- auto_discover_pyg_models() now uses dynamic introspection via get_introspector()
- get_metadata() falls back to dynamic introspection for unregistered models
- Added _auto_discovered flag to prevent redundant discovery
- Uses PyGModelIntrospector singleton for runtime model discovery

**Version 1.2.0 (Pydantic V2 Migration - Phase 24)**:
- Migrated ModelRegistration from @dataclass to Pydantic BaseModel (mutable)
- Uses ConfigDict(arbitrary_types_allowed=True) for Type[torch.nn.Module] field
- Added to_dict() method wrapping model_dump() for backward compatibility
- NON-BREAKING: Same constructor API and attribute access preserved

Author: Milia Team
Version: 1.2.0
"""

import threading
import logging
import importlib
from typing import Dict, List, Set, Optional, Type, Any, Tuple
from pydantic import BaseModel, ConfigDict
from collections import defaultdict
from datetime import datetime

import torch

# Dynamic introspection replaces static model_categories
from .pyg_introspector import (
    ModelCategory,
    ModelMetadata,  # Alias for DynamicModelMetadata
    get_model_metadata,
    get_all_model_names,
    get_introspector,
)

# Import exceptions
try:
    from milia_pipeline.exceptions import ModelError, ModelValidationError
except ImportError:
    # Fallback exceptions
    class ModelError(Exception):
        """Base exception for model-related errors."""
        pass
    
    class ModelValidationError(ModelError):
        """Exception raised when model validation fails."""
        pass


logger = logging.getLogger(__name__)


# =============================================================================
# MODEL REGISTRATION PYDANTIC MODEL
# =============================================================================

class ModelRegistration(BaseModel):
    """
    Container for model registration information.
    
    Pydantic V2 Migration (Phase 24):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses ConfigDict(arbitrary_types_allowed=True) for Type[torch.nn.Module]
        - Added to_dict() method wrapping model_dump() for backward compatibility
        - NON-BREAKING: Same constructor API and attribute access preserved
    
    Attributes:
        name: Model name (e.g., "GCN", "GAT")
        model_class: PyG model class
        metadata: ModelMetadata object with specifications
        is_builtin: True if PyG model, False if custom/plugin
        plugin_name: Name of plugin (if from plugin)
        registered_at: Timestamp of registration
    """
    # Allow arbitrary types for Type[torch.nn.Module] field
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str
    model_class: Type[torch.nn.Module]
    metadata: ModelMetadata
    is_builtin: bool = True
    plugin_name: Optional[str] = None
    registered_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Backward compatible method wrapping Pydantic V2's model_dump().
        
        Note:
            - model_class is serialized as string representation
            - metadata is serialized via its own to_dict() (Pydantic model_dump)
        
        Returns:
            Dictionary with all registration fields
        """
        return self.model_dump()


# =============================================================================
# MODEL REGISTRY (SINGLETON)
# =============================================================================

class ModelRegistry:
    """
    Thread-safe singleton registry for ML/DL models.
    
    Manages:
    - PyG built-in models (120+ models)
    - Custom plugin models
    - Model discovery and validation
    - Category-based organization
    
    Usage:
        >>> registry = ModelRegistry()
        >>> registry.auto_discover_pyg_models()
        >>> model_class = registry.get_model("GCN")
        >>> all_names = registry.list_available_models()
    
    Thread Safety:
        All operations are thread-safe using RLock for reentrant locking.
    
    Singleton Pattern:
        Only one instance exists per class. Multiple calls to ModelRegistry()
        return the same instance.
    """
    
    _instances: Dict[type, 'ModelRegistry'] = {}
    _class_lock: threading.Lock = threading.Lock()
    
    def __new__(cls):
        """Implement singleton pattern with thread safety"""
        if cls not in cls._instances:
            with cls._class_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]
    
    @classmethod
    def get_instance(cls) -> 'ModelRegistry':
        """
        Get singleton instance of the registry.
        
        Thread-safe factory method that returns the singleton instance.
        This is the recommended way to access the registry.
        
        Returns:
            ModelRegistry singleton instance
            
        Example:
            >>> registry = ModelRegistry.get_instance()
            >>> registry.auto_discover_pyg_models()
        """
        return cls()
    
    def __init__(self):
        """Initialize registry (only once due to singleton)"""
        if not hasattr(self, '_initialized'):
            self._models: Dict[str, ModelRegistration] = {}
            self._by_category: Dict[ModelCategory, Set[str]] = defaultdict(set)
            self._plugin_models: Dict[str, str] = {}  # name -> plugin_name
            self._lock = threading.RLock()  # Use RLock for reentrant locking
            
            # Initialize tracking for discovery process
            self._failed_models = []
            self._auto_discovered = False  # Phase 3: Track if discovery already ran
            self._discovery_stats = {
                'total_attempted': 0,
                'successful': 0,
                'failed': 0,
                'last_discovery': None
            }
            
            self._initialized = True
            logger.info("ModelRegistry initialized")
            
            # Auto-discover PyG models
            self.auto_discover_pyg_models()
            
            # Log comprehensive availability summary
            self.log_availability_summary()
    
    # =========================================================================
    # AUTO-DISCOVERY
    # =========================================================================
    
    def auto_discover_pyg_models(self) -> int:
        """
        Auto-discover and register all PyG models dynamically.
        
        Uses runtime introspection instead of static model_categories.
        Gracefully handles:
        - Missing models in current PyG version
        - Import errors
        - Version incompatibilities
        
        Returns:
            Number of models successfully registered
            
        Example:
            >>> registry = ModelRegistry()
            >>> count = registry.auto_discover_pyg_models()
            >>> print(f"Discovered {count} models")
        """
        with self._lock:
            # Prevent redundant discovery
            if self._auto_discovered:
                return self._discovery_stats.get('successful', 0)
            
            logger.info("Starting dynamic auto-discovery of PyG models...")
            
            discovered_count = 0
            failed_models = []
            
            # Get all model names from DYNAMIC INTROSPECTION
            introspector = get_introspector()
            discovered_names = introspector.get_all_model_names()  # DYNAMIC
            self._discovery_stats['total_attempted'] = len(discovered_names)
            
            # Discover from PyG
            for name in discovered_names:
                metadata = introspector.get_model_metadata(name)  # DYNAMIC
                if metadata is None:
                    logger.debug(f"Could not introspect model '{name}', skipping")
                    failed_models.append((name, "introspection_failed"))
                    continue
                
                try:
                    # Try to import model class
                    model_class = self._import_pyg_model(name, metadata)
                    
                    if model_class is None:
                        failed_models.append((name, "not_found"))
                        continue
                    
                    # Register the model
                    self._register_internal(
                        name=name,
                        model_class=model_class,
                        metadata=metadata,
                        is_builtin=True
                    )
                    discovered_count += 1
                    logger.debug(f"✓ Registered: {name}")
                    
                except Exception as e:
                    logger.debug(f"Failed to discover model {name}: {e}")
                    failed_models.append((name, f"error: {str(e)}"))
            
            self._auto_discovered = True
            self._failed_models = failed_models
            self._discovery_stats['successful'] = discovered_count
            self._discovery_stats['failed'] = len(failed_models)
            self._discovery_stats['last_discovery'] = datetime.now().isoformat()
            
            logger.info(
                f"Dynamic auto-discovery complete: {discovered_count}/{len(discovered_names)} "
                f"models registered"
            )
            
            if failed_models:
                logger.debug(f"Failed to register {len(failed_models)} models")
                if len(failed_models) <= 10:
                    for name, reason in failed_models:
                        logger.debug(f"  - {name}: {reason}")
            
            return discovered_count
    
    def _import_pyg_model(
        self, 
        name: str, 
        metadata: ModelMetadata
    ) -> Optional[Type[torch.nn.Module]]:
        """
        Import PyG model class dynamically.
        
        Args:
            name: Model name
            metadata: Model metadata with import path
            
        Returns:
            Model class or None if import fails
            
        Example:
            >>> metadata = get_model_metadata("GCN")
            >>> model_class = registry._import_pyg_model("GCN", metadata)
            >>> print(model_class)
            <class 'torch_geometric.nn.models.GCN'>
        """
        try:
            # Metadata contains import path (e.g., "torch_geometric.nn.models.GCN")
            import_path = metadata.import_path
            
            # Split into module and class
            parts = import_path.rsplit('.', 1)
            if len(parts) != 2:
                logger.debug(f"Invalid import path for {name}: {import_path}")
                return None
            
            module_name, class_name = parts
            
            # Import module
            module = importlib.import_module(module_name)
            
            # Get class
            model_class = getattr(module, class_name, None)
            
            if model_class is None:
                logger.debug(f"Class {class_name} not found in {module_name}")
                return None
            
            # Verify it's a PyTorch module
            if not issubclass(model_class, torch.nn.Module):
                logger.debug(f"{name} is not a torch.nn.Module subclass")
                return None
            
            return model_class
            
        except ImportError as e:
            logger.debug(f"Import error for {name}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Unexpected error importing {name}: {e}")
            return None
    
    # =========================================================================
    # REGISTRATION
    # =========================================================================
    
    def _register_internal(
        self,
        name: str,
        model_class: Type[torch.nn.Module],
        metadata: ModelMetadata,
        is_builtin: bool = True,
        plugin_name: Optional[str] = None
    ) -> None:
        """
        Internal registration method.
        
        Args:
            name: Model name
            model_class: Model class
            metadata: Model metadata
            is_builtin: True if PyG model, False if plugin
            plugin_name: Name of plugin (if from plugin)
        """
        with self._lock:
            registration = ModelRegistration(
                name=name,
                model_class=model_class,
                metadata=metadata,
                is_builtin=is_builtin,
                plugin_name=plugin_name,
                registered_at=datetime.now().isoformat()
            )
            
            self._models[name] = registration
            self._by_category[metadata.category].add(name)
            
            if plugin_name:
                self._plugin_models[name] = plugin_name
    
    def register_model(
        self,
        name: str,
        model_class: Type[torch.nn.Module],
        metadata: ModelMetadata,
        plugin_name: Optional[str] = None,
        force: bool = False
    ) -> bool:
        """
        Register a custom model.
        
        Args:
            name: Model name (must be unique)
            model_class: Model class (must be torch.nn.Module subclass)
            metadata: Model metadata
            plugin_name: Plugin name (if from plugin)
            force: Force registration even if already exists
            
        Returns:
            True if registration successful, False otherwise
            
        Raises:
            ModelError: If model_class is not torch.nn.Module subclass
            
        Example:
            >>> class MyModel(torch.nn.Module):
            ...     pass
            >>> metadata = ModelMetadata(
            ...     name="MyModel",
            ...     category=ModelCategory.BASIC_GNN,
            ...     import_path="custom.MyModel",
            ...     description="Custom model",
            ...     supported_tasks=["graph_classification"]
            ... )
            >>> registry.register_model("MyModel", MyModel, metadata)
            True
        """
        # Validate model_class
        if not issubclass(model_class, torch.nn.Module):
            raise ModelError(
                f"model_class must be torch.nn.Module subclass, got {type(model_class)}"
            )
        
        with self._lock:
            # Check if already exists
            if name in self._models and not force:
                logger.warning(
                    f"Model '{name}' already registered. Use force=True to override."
                )
                return False
            
            # Register
            self._register_internal(
                name=name,
                model_class=model_class,
                metadata=metadata,
                is_builtin=False,
                plugin_name=plugin_name
            )
            
            logger.info(f"Registered custom model: {name}")
            return True
    
    def unregister_model(self, name: str) -> bool:
        """
        Unregister a model.
        
        Args:
            name: Model name
            
        Returns:
            True if unregistration successful, False if model not found
            
        Example:
            >>> registry.unregister_model("MyModel")
            True
        """
        with self._lock:
            if name not in self._models:
                logger.warning(f"Model '{name}' not found in registry")
                return False
            
            registration = self._models[name]
            
            # Remove from main dict
            del self._models[name]
            
            # Remove from category
            self._by_category[registration.metadata.category].discard(name)
            
            # Remove from plugin models
            if name in self._plugin_models:
                del self._plugin_models[name]
            
            logger.info(f"Unregistered model: {name}")
            return True
    
    # =========================================================================
    # QUERY METHODS
    # =========================================================================
    
    def get_model(self, name: str) -> Optional[Type[torch.nn.Module]]:
        """
        Get model class by name.
        
        Args:
            name: Model name
            
        Returns:
            Model class or None if not found
            
        Example:
            >>> model_class = registry.get_model("GCN")
            >>> model = model_class(in_channels=10, out_channels=5)
        """
        with self._lock:
            registration = self._models.get(name)
            return registration.model_class if registration else None
    
    def has_model(self, name: str) -> bool:
        """
        Check if model exists in registry.
        
        Args:
            name: Model name
            
        Returns:
            True if model exists, False otherwise
            
        Example:
            >>> registry.has_model("GCN")
            True
            >>> registry.has_model("NonExistentModel")
            False
        """
        with self._lock:
            return name in self._models
    
    def list_available_models(
        self,
        category: Optional[ModelCategory] = None,
        task_type: Optional[str] = None,
        supports_heterogeneous: Optional[bool] = None,
        tags: Optional[List[str]] = None
    ) -> List[str]:
        """
        List available models with optional filtering.
        
        Args:
            category: Filter by category
            task_type: Filter by supported task type
            supports_heterogeneous: Filter by heterogeneous graph support
            tags: Filter by tags (model must have ALL specified tags)
            
        Returns:
            List of model names (sorted)
            
        Example:
            >>> # All models
            >>> all_models = registry.list_available_models()
            
            >>> # Basic GNN models
            >>> basic = registry.list_available_models(
            ...     category=ModelCategory.BASIC_GNN
            ... )
            
            >>> # Models for graph regression
            >>> regression = registry.list_available_models(
            ...     task_type="graph_regression"
            ... )
            
            >>> # Attention-based models
            >>> attention = registry.list_available_models(
            ...     tags=["attention"]
            ... )
        """
        with self._lock:
            models = list(self._models.keys())
            
            # Filter by category
            if category:
                models = [m for m in models if m in self._by_category[category]]
            
            # Filter by task type
            if task_type:
                models = [
                    m for m in models
                    if task_type in self._models[m].metadata.supported_tasks
                ]
            
            # Filter by heterogeneous support
            if supports_heterogeneous is not None:
                models = [
                    m for m in models
                    if self._models[m].metadata.supports_heterogeneous == supports_heterogeneous
                ]
            
            # Filter by tags
            if tags:
                models = [
                    m for m in models
                    if all(tag in self._models[m].metadata.tags for tag in tags)
                ]
            
            return sorted(models)
    
    def get_registration(self, name: str) -> Optional[ModelRegistration]:
        """
        Get full registration object for a model.
        
        Args:
            name: Model name
            
        Returns:
            ModelRegistration object or None if not found
            
        Example:
            >>> reg = registry.get_registration("GCN")
            >>> print(reg.metadata.description)
            'Graph Convolutional Network - Semi-supervised node classification'
        """
        with self._lock:
            return self._models.get(name)
    
    def get_metadata(self, name: str) -> Optional[ModelMetadata]:
        """
        Get metadata for a model.
        
        First checks registered models, then falls back to dynamic introspection
        for models that may not be registered yet.
        
        Args:
            name: Model name
            
        Returns:
            ModelMetadata object or None if not found
            
        Example:
            >>> metadata = registry.get_metadata("GCN")
            >>> print(metadata.supported_tasks)
            ['node_classification', 'node_regression', ...]
        """
        with self._lock:
            registration = self._models.get(name)
            if registration:
                return registration.metadata
        
        # Fallback to dynamic introspection for unregistered models
        return get_introspector().get_model_metadata(name)
    
    def list_by_category(self) -> Dict[str, List[str]]:
        """
        List models organized by category.
        
        Returns:
            Dictionary mapping category name to list of model names
            
        Example:
            >>> by_cat = registry.list_by_category()
            >>> print(by_cat['basic_gnn'])
            ['GCN', 'GraphSAGE', 'GIN', 'GAT', 'EdgeCNN', 'PNA']
        """
        with self._lock:
            return {
                cat.value: sorted(list(models))
                for cat, models in self._by_category.items()
                if len(models) > 0
            }
    
    def search_models(
        self,
        query: str,
        search_in: Optional[List[str]] = None
    ) -> List[str]:
        """
        Search models by keyword.
        
        Args:
            query: Search query (case-insensitive)
            search_in: Fields to search in ['name', 'description', 'tags']
                      Default: all fields
            
        Returns:
            List of matching model names
            
        Example:
            >>> # Search by name
            >>> results = registry.search_models("attention")
            >>> print(results)
            ['GAT', 'GATv2', 'SuperGATConv', 'TransformerConv', ...]
            
            >>> # Search in description only
            >>> results = registry.search_models(
            ...     "temporal",
            ...     search_in=["description"]
            ... )
        """
        if search_in is None:
            search_in = ["name", "description", "tags"]
        
        query_lower = query.lower()
        results = []
        
        with self._lock:
            for name, registration in self._models.items():
                metadata = registration.metadata
                
                # Search in name
                if "name" in search_in and query_lower in name.lower():
                    results.append(name)
                    continue
                
                # Search in description
                if "description" in search_in and query_lower in metadata.description.lower():
                    results.append(name)
                    continue
                
                # Search in tags
                if "tags" in search_in:
                    if any(query_lower in tag.lower() for tag in metadata.tags):
                        results.append(name)
                        continue
        
        return sorted(results)
    
    # =========================================================================
    # PLUGIN MANAGEMENT
    # =========================================================================
    
    def list_plugin_models(self) -> Dict[str, List[str]]:
        """
        List plugin models organized by plugin name.
        
        Returns:
            Dictionary mapping plugin name to list of model names
            
        Example:
            >>> plugin_models = registry.list_plugin_models()
            >>> print(plugin_models)
            {'custom_molecular_gnn': ['CustomMolGNN', 'CustomGAT']}
        """
        with self._lock:
            result = defaultdict(list)
            for model_name, plugin_name in self._plugin_models.items():
                result[plugin_name].append(model_name)
            return dict(result)
    
    def get_builtin_models(self) -> List[str]:
        """
        Get list of built-in PyG models.
        
        Returns:
            List of built-in model names
        """
        with self._lock:
            return sorted([
                name for name, reg in self._models.items()
                if reg.is_builtin
            ])
    
    def get_custom_models(self) -> List[str]:
        """
        Get list of custom/plugin models.
        
        Returns:
            List of custom model names
        """
        with self._lock:
            return sorted([
                name for name, reg in self._models.items()
                if not reg.is_builtin
            ])
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with statistics about registered models
            
        Example:
            >>> stats = registry.get_statistics()
            >>> print(f"Total models: {stats['total_models']}")
            >>> print(f"By category: {stats['by_category']}")
        """
        with self._lock:
            total = len(self._models)
            builtins = sum(1 for reg in self._models.values() if reg.is_builtin)
            plugins = total - builtins
            
            by_category = {
                cat.value: len(names)
                for cat, names in self._by_category.items()
                if len(names) > 0
            }
            
            return {
                "total_models": total,
                "builtin_models": builtins,
                "plugin_models": plugins,
                "by_category": by_category,
                "plugins": len(set(self._plugin_models.values())),
                "discovery_stats": self._discovery_stats.copy(),
                "failed_models_count": len(self._failed_models)
            }
    
    def get_availability_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report of model availability.
        
        Returns:
            Dictionary containing:
            - total_registered: Number of successfully registered models
            - failed_models: List of models that couldn't be registered
            - by_category: Breakdown by category
            - discovery_stats: Discovery process statistics
        """
        with self._lock:
            # Count by category
            by_category = {}
            for reg in self._models.values():
                cat = reg.metadata.category.value
                by_category[cat] = by_category.get(cat, 0) + 1
            
            report = {
                'total_registered': len(self._models),
                'total_attempted': self._discovery_stats['total_attempted'],
                'failed_models': self._failed_models.copy(),
                'by_category': by_category,
                'discovery_stats': self._discovery_stats.copy(),
                'success_rate': (
                    len(self._models) / self._discovery_stats['total_attempted'] * 100
                    if self._discovery_stats['total_attempted'] > 0 else 0
                )
            }
            
            return report
    
    def log_availability_summary(self):
        """Log a human-readable summary of model availability."""
        report = self.get_availability_report()
        
        logger.info("=" * 70)
        logger.info("MODEL REGISTRY SUMMARY")
        logger.info("=" * 70)
        logger.info(
            f"Total Registered: {report['total_registered']} / "
            f"{report['total_attempted']} ({report['success_rate']:.1f}%)"
        )
        
        if report['by_category']:
            logger.info("\nBy Category:")
            for category, count in sorted(report['by_category'].items()):
                logger.info(f"  {category:20s}: {count:3d} models")
        
        if report['failed_models']:
            logger.info(
                f"\nUnavailable in Current PyG Version: "
                f"{len(report['failed_models'])}"
            )
            if len(report['failed_models']) <= 10:
                for name, reason in report['failed_models']:
                    logger.debug(f"  - {name}: {reason}")
            else:
                logger.debug(
                    f"  (First 10 shown, {len(report['failed_models']) - 10} more)"
                )
                for name, reason in report['failed_models'][:10]:
                    logger.debug(f"  - {name}: {reason}")
        
        logger.info("=" * 70)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def reset(self) -> None:
        """
        Reset the registry (useful for testing).
        
        Clears all registered models and statistics.
        
        Warning:
            This will clear all models including plugins.
            Use with caution in production.
        """
        with self._lock:
            self._models.clear()
            self._by_category.clear()
            self._plugin_models.clear()
            self._failed_models.clear()
            self._auto_discovered = False  # Allow re-discovery after reset
            self._discovery_stats = {
                'total_attempted': 0,
                'successful': 0,
                'failed': 0,
                'last_discovery': None
            }
            logger.info("Registry reset")
    
    def __len__(self) -> int:
        """Return number of registered models."""
        with self._lock:
            return len(self._models)
    
    def __contains__(self, name: str) -> bool:
        """Check if model exists in registry."""
        return self.has_model(name)
    
    def __repr__(self) -> str:
        """String representation of registry."""
        with self._lock:
            return (
                f"ModelRegistry(total={len(self._models)}, "
                f"builtin={sum(1 for r in self._models.values() if r.is_builtin)}, "
                f"plugin={sum(1 for r in self._models.values() if not r.is_builtin)})"
            )


# =============================================================================
# GLOBAL REGISTRY INSTANCE
# =============================================================================

# Global registry instance (like descriptors module)
registry = ModelRegistry()

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_model(name: str) -> Optional[Type[torch.nn.Module]]:
    """
    Get model class from global registry.
    
    Args:
        name: Model name
        
    Returns:
        Model class or None if not found
        
    Example:
        >>> from milia_pipeline.models import get_model
        >>> GCN = get_model("GCN")
        >>> model = GCN(in_channels=10, out_channels=5)
    """
    return registry.get_model(name)


def has_model(name: str) -> bool:
    """
    Check if model exists in global registry.
    
    Args:
        name: Model name
        
    Returns:
        True if model exists
        
    Example:
        >>> from milia_pipeline.models import has_model
        >>> has_model("GCN")
        True
    """
    return registry.has_model(name)


def list_models(
    category: Optional[ModelCategory] = None,
    task_type: Optional[str] = None
) -> List[str]:
    """
    List available models from global registry.
    
    Args:
        category: Filter by category
        task_type: Filter by task type
        
    Returns:
        List of model names
        
    Example:
        >>> from milia_pipeline.models import list_models, ModelCategory
        >>> all_models = list_models()
        >>> basic_gnn = list_models(category=ModelCategory.BASIC_GNN)
        >>> regression = list_models(task_type="graph_regression")
    """
    return registry.list_available_models(category=category, task_type=task_type)


def get_model_info(name: str) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive information about a model.
    
    Args:
        name: Model name
        
    Returns:
        Dictionary with model information or None if not found
        
    Example:
        >>> from milia_pipeline.models import get_model_info
        >>> info = get_model_info("GCN")
        >>> print(info['description'])
        >>> print(info['supported_tasks'])
    """
    registration = registry.get_registration(name)
    if not registration:
        return None
    
    return {
        'name': registration.name,
        'class': registration.model_class.__name__,
        'description': registration.metadata.description,
        'category': registration.metadata.category.value,
        'supported_tasks': registration.metadata.supported_tasks,
        'is_builtin': registration.is_builtin,
        'plugin_name': registration.plugin_name,
        'paper_url': registration.metadata.paper_url,
        'tags': registration.metadata.tags,
        'requires_edge_features': registration.metadata.requires_edge_features,
        'requires_edge_weights': registration.metadata.requires_edge_weights,
        'supports_heterogeneous': registration.metadata.supports_heterogeneous,
        'registered_at': registration.registered_at
    }
