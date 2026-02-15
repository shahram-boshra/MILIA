# milia_pipeline/handlers/handler_registry.py

"""
Handler Registry
================

Thread-safe registry for managing dataset handler types with automatic
registration via decorator.

Design decisions (following datasets/registry.py pattern):
- NOT a singleton: Can create isolated instances for testing
- Explicit registration: @register_handler decorator triggers registration
- Type-safe: Only accepts DatasetHandler subclasses
- Thread-safe: Protected by RLock for concurrent access
- Cache invalidation: Callbacks notify consumers of registry changes

Usage:
    from milia_pipeline.handlers.handler_registry import register_handler

    @register_handler
    class MyDatasetHandler(DatasetHandler):
        def get_dataset_type(self) -> str:
            return "MyDataset"
        ...

Phase 6 Integration:
- Works alongside datasets/registry.py for handler discovery
- Enables zero-modification extensibility for new handlers
- Auto-registration via decorator when implementation files are imported
"""

import logging
from collections.abc import Callable, Iterator
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


class HandlerRegistrationError(Exception):
    """Raised when handler registration fails."""

    def __init__(
        self,
        message: str,
        handler_name: str,
        conflicting_class: str | None = None,
        details: str | None = None,
    ):
        self.handler_name = handler_name
        self.conflicting_class = conflicting_class
        self.details = details
        super().__init__(message)


class HandlerNotFoundError(Exception):
    """Raised when requested handler is not found in registry."""

    def __init__(
        self, message: str, handler_name: str, available_handlers: list[str] | None = None
    ):
        self.handler_name = handler_name
        self.available_handlers = available_handlers or []
        super().__init__(message)


class HandlerRegistry:
    """
    Thread-safe registry for dataset handler types.

    This registry follows the same pattern as datasets/registry.py to maintain
    consistency across the codebase. Handler classes are registered by their
    dataset_type (from get_dataset_type() method).
    """

    def __init__(self):
        """Initialize empty registry with thread lock."""
        self._handlers: dict[str, type] = {}  # Type hint without forward reference
        self._lock = RLock()
        self._on_change_callbacks: list[Callable[[], None]] = []

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
                logger.warning(f"Handler registry change callback failed: {e}")

    def register(self, handler_class: type) -> None:
        """
        Register a handler class.

        The handler class must:
        1. Be a class (not an instance)
        2. Have a get_dataset_type() method that returns the handler name
        3. Not be abstract (must have all abstract methods implemented)

        Args:
            handler_class: The handler class to register

        Raises:
            TypeError: If handler_class is not a class or doesn't have required methods
            HandlerRegistrationError: If registration fails (e.g., duplicate name)
        """
        if not isinstance(handler_class, type):
            raise TypeError(f"Expected class, got {type(handler_class).__name__}")

        # Verify the class has get_dataset_type method
        if not hasattr(handler_class, "get_dataset_type"):
            raise TypeError(
                f"Handler class must have get_dataset_type() method, got {handler_class.__name__}"
            )

        # Check for abstract methods
        if hasattr(handler_class, "__abstractmethods__") and handler_class.__abstractmethods__:
            raise HandlerRegistrationError(
                message=f"Cannot register abstract class '{handler_class.__name__}'",
                handler_name=handler_class.__name__,
                details=f"Missing implementations: {handler_class.__abstractmethods__}",
            )

        # Get the handler name from get_dataset_type
        # For class methods or instances, we need to handle both cases
        try:
            # Try calling as class method first
            if isinstance(handler_class.get_dataset_type, classmethod):
                name = handler_class.get_dataset_type()
            else:
                # For regular methods, we need an instance or check if it's defined at class level
                # Try to get from a temporary instance check
                # This is a design consideration - we'll use the class name pattern
                name = handler_class.__name__.replace("DatasetHandler", "").replace("Handler", "")
                if not name:
                    name = handler_class.__name__
        except Exception:
            # Fallback: derive name from class name
            name = handler_class.__name__.replace("DatasetHandler", "").replace("Handler", "")
            if not name:
                name = handler_class.__name__

        with self._lock:
            if name in self._handlers:
                existing = self._handlers[name]
                # Check if it's the same class - use qualname and module comparison
                # to handle re-imports through different import paths
                # (which create different class objects in memory)
                same_class = existing is handler_class
                if not same_class:
                    # Check by qualified name and module to handle re-import scenarios
                    # This occurs when the same handler module is imported via different paths:
                    # e.g., 'milia_pipeline.handlers.implementations.qdpi' vs relative import
                    existing_qualname = getattr(existing, "__qualname__", existing.__name__)
                    handler_qualname = getattr(
                        handler_class, "__qualname__", handler_class.__name__
                    )
                    existing_module = getattr(existing, "__module__", "")
                    handler_module = getattr(handler_class, "__module__", "")

                    # Same class if qualnames match and both are from handlers.implementations
                    same_class = (
                        existing_qualname == handler_qualname
                        and "handlers.implementations" in existing_module
                        and "handlers.implementations" in handler_module
                    )

                if not same_class:
                    raise HandlerRegistrationError(
                        message=f"Handler '{name}' already registered",
                        handler_name=name,
                        conflicting_class=existing.__name__,
                        details=f"Cannot register {handler_class.__name__} with same name",
                    )
                else:
                    logger.debug(
                        f"Handler '{name}' re-registered (same class, possibly different import path)"
                    )
                    return

            self._handlers[name] = handler_class
            logger.info(f"Registered handler: {name} ({handler_class.__name__})")
            self._notify_change()

    def unregister(self, name: str) -> bool:
        """Unregister a handler by name. Returns True if found and removed."""
        with self._lock:
            if name in self._handlers:
                del self._handlers[name]
                logger.info(f"Unregistered handler: {name}")
                self._notify_change()
                return True
            return False

    def get(self, name: str) -> type:
        """Get handler class by name. Raises HandlerNotFoundError if not found."""
        with self._lock:
            if name not in self._handlers:
                available = list(self._handlers.keys())
                raise HandlerNotFoundError(
                    message=f"Handler '{name}' not registered",
                    handler_name=name,
                    available_handlers=available,
                )
            return self._handlers[name]

    def get_or_none(self, name: str) -> type | None:
        """Get handler class by name, or None if not found."""
        with self._lock:
            return self._handlers.get(name)

    def list_all(self) -> list[str]:
        """List all registered handler names."""
        with self._lock:
            return list(self._handlers.keys())

    def list_all_classes(self) -> list[type]:
        """List all registered handler classes."""
        with self._lock:
            return list(self._handlers.values())

    def is_registered(self, name: str) -> bool:
        """Check if handler is registered."""
        with self._lock:
            return name in self._handlers

    def __contains__(self, name: str) -> bool:
        """Support 'in' operator."""
        return self.is_registered(name)

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered handler names."""
        with self._lock:
            return iter(list(self._handlers.keys()))

    def __len__(self) -> int:
        """Return number of registered handlers."""
        with self._lock:
            return len(self._handlers)

    def clear(self) -> None:
        """Clear all registrations (mainly for testing)."""
        with self._lock:
            self._handlers.clear()
            logger.warning("Handler registry cleared")
            self._notify_change()

    def get_registry_info(self) -> dict[str, Any]:
        """
        Get comprehensive registry information for diagnostics.

        Returns:
            Dict with registry status, registered handlers, and their details
        """
        with self._lock:
            return {
                "total_handlers": len(self._handlers),
                "registered_handlers": list(self._handlers.keys()),
                "handler_classes": {name: cls.__name__ for name, cls in self._handlers.items()},
                "callback_count": len(self._on_change_callbacks),
            }


# ============================================================================
# Default Global Registry
# ============================================================================

_default_registry = HandlerRegistry()


def get_default_registry() -> HandlerRegistry:
    """Get the default global registry."""
    return _default_registry


def register_handler(handler_class: type) -> type:
    """
    Register handler class with default registry. Can be used as decorator.

    Usage:
        @register_handler
        class QM9DatasetHandler(DatasetHandler):
            def get_dataset_type(self) -> str:
                return "QM9"
            ...

    Args:
        handler_class: The handler class to register

    Returns:
        The handler class (unchanged, for decorator pattern)
    """
    _default_registry.register(handler_class)
    return handler_class


def get(name: str) -> type:
    """Get handler from default registry."""
    return _default_registry.get(name)


def list_all() -> list[str]:
    """List all handlers in default registry."""
    return _default_registry.list_all()


def is_registered(name: str) -> bool:
    """Check if handler is registered in default registry."""
    return _default_registry.is_registered(name)


def get_registry_info() -> dict[str, Any]:
    """Get registry information from default registry."""
    return _default_registry.get_registry_info()


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Main class
    "HandlerRegistry",
    # Exceptions
    "HandlerRegistrationError",
    "HandlerNotFoundError",
    # Default registry functions
    "get_default_registry",
    "register_handler",
    "get",
    "list_all",
    "is_registered",
    "get_registry_info",
]
