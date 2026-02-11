"""
Dataset registry for managing registered dataset types.

Design decisions:
- NOT a singleton: Can create isolated instances for testing
- Explicit registration: No auto-discovery magic
- Type-safe: Only accepts BaseDataset subclasses
- Thread-safe: Protected by RLock for concurrent access
- Cache invalidation: Callbacks notify consumers of registry changes
"""

from typing import Dict, List, Type, Optional, Iterator, Callable
from threading import RLock
import logging

from milia_pipeline.datasets.base import BaseDataset
from milia_pipeline.exceptions import DatasetNotFoundError, DatasetRegistrationError

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """
    Thread-safe registry for dataset types.
    """
    
    def __init__(self):
        """Initialize empty registry with thread lock."""
        self._datasets: Dict[str, Type[BaseDataset]] = {}
        self._lock = RLock()
        self._on_change_callbacks: List[Callable[[], None]] = []
    
    def add_on_change_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when registry changes."""
        with self._lock:
            self._on_change_callbacks.append(callback)
    
    def remove_on_change_callback(self, callback: Callable[[], None]) -> bool:
        """Remove a previously registered callback."""
        with self._lock:
            try:
                self._on_change_callbacks.remove(callback)
                return True
            except ValueError:
                return False
    
    def _notify_change(self) -> None:
        """Notify all registered callbacks of registry change."""
        for callback in self._on_change_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Registry change callback failed: {e}")
    
    def register(self, dataset_class: Type[BaseDataset]) -> None:
        """Register a dataset class."""
        if not isinstance(dataset_class, type):
            raise TypeError(f"Expected class, got {type(dataset_class).__name__}")
        
        if not issubclass(dataset_class, BaseDataset):
            raise TypeError(
                f"Dataset class must be subclass of BaseDataset, got {dataset_class.__name__}"
            )
        
        if hasattr(dataset_class, '__abstractmethods__') and dataset_class.__abstractmethods__:
            raise DatasetRegistrationError(
                message=f"Cannot register abstract class '{dataset_class.__name__}'",
                dataset_name=dataset_class.__name__,
                details=f"Missing implementations: {dataset_class.__abstractmethods__}"
            )
        
        name = dataset_class.metadata.name
        
        with self._lock:
            if name in self._datasets:
                existing = self._datasets[name]
                if existing is not dataset_class:
                    raise DatasetRegistrationError(
                        message=f"Dataset '{name}' already registered",
                        dataset_name=name,
                        conflicting_class=existing.__name__,
                        details=f"Cannot register {dataset_class.__name__} with same name"
                    )
                else:
                    logger.debug(f"Dataset '{name}' re-registered (same class)")
                    return
            
            self._datasets[name] = dataset_class
            logger.info(f"Registered dataset: {name} ({dataset_class.__name__})")
            self._notify_change()
    
    def unregister(self, name: str) -> bool:
        """Unregister a dataset by name. Returns True if found and removed."""
        with self._lock:
            if name in self._datasets:
                del self._datasets[name]
                logger.info(f"Unregistered dataset: {name}")
                self._notify_change()
                return True
            return False
    
    def get(self, name: str) -> Type[BaseDataset]:
        """Get dataset class by name. Raises DatasetNotFoundError if not found."""
        with self._lock:
            if name not in self._datasets:
                available = list(self._datasets.keys())
                raise DatasetNotFoundError(
                    message=f"Dataset '{name}' not registered",
                    dataset_name=name,
                    available_datasets=available
                )
            return self._datasets[name]
    
    def get_or_none(self, name: str) -> Optional[Type[BaseDataset]]:
        """Get dataset class by name, or None if not found."""
        with self._lock:
            return self._datasets.get(name)
    
    def list_all(self) -> List[str]:
        """List all registered dataset names."""
        with self._lock:
            return list(self._datasets.keys())
    
    def list_all_classes(self) -> List[Type[BaseDataset]]:
        """List all registered dataset classes."""
        with self._lock:
            return list(self._datasets.values())
    
    def is_registered(self, name: str) -> bool:
        """Check if dataset is registered."""
        with self._lock:
            return name in self._datasets
    
    def __contains__(self, name: str) -> bool:
        """Support 'in' operator."""
        return self.is_registered(name)
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over registered dataset names."""
        with self._lock:
            return iter(list(self._datasets.keys()))
    
    def __len__(self) -> int:
        """Return number of registered datasets."""
        with self._lock:
            return len(self._datasets)
    
    def clear(self) -> None:
        """Clear all registrations (mainly for testing)."""
        with self._lock:
            self._datasets.clear()
            logger.warning("Dataset registry cleared")
            self._notify_change()


_default_registry = DatasetRegistry()


def get_default_registry() -> DatasetRegistry:
    """Get the default global registry."""
    return _default_registry


def register(dataset_class: Type[BaseDataset]) -> Type[BaseDataset]:
    """
    Register dataset class with default registry. Can be used as decorator.
    
    Usage:
        @register
        class QM9Dataset(BaseDataset):
            ...
    """
    _default_registry.register(dataset_class)
    return dataset_class


def get(name: str) -> Type[BaseDataset]:
    """Get dataset from default registry."""
    return _default_registry.get(name)


def list_all() -> List[str]:
    """List all datasets in default registry."""
    return _default_registry.list_all()


def is_registered(name: str) -> bool:
    """Check if dataset is registered in default registry."""
    return _default_registry.is_registered(name)
