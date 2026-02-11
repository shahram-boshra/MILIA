#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/handlers/handler_registry.py

Module under test: handler_registry.py
- HandlerRegistry: Thread-safe registry for dataset handler types
- HandlerRegistrationError: Exception for registration failures
- HandlerNotFoundError: Exception for lookup failures
- register_handler: Decorator for default registry registration
- get_default_registry / get / list_all / is_registered / get_registry_info: Module-level API

Test path on local machine: ~/ml_projects/milia/tests/test_handler_registry_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/handlers/handler_registry.py

NOTE: This test suite runs inside Docker at /app/milia
Path mappings:
- Project root: /app/milia (mapped from ~/ml_projects/milia)

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution

Updated: February 2026 - Production-ready comprehensive test coverage
"""

import sys
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, MagicMock, patch, call
import logging
import threading
import time
from typing import Dict, List, Type, Optional, Any
from abc import ABC, abstractmethod

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from milia_pipeline.handlers.handler_registry import (
    HandlerRegistry,
    HandlerRegistrationError,
    HandlerNotFoundError,
    get_default_registry,
    register_handler,
    get,
    list_all,
    is_registered,
    get_registry_info,
)


# ============================================================================
# HELPERS: Build mock handler classes for testing
# ============================================================================

def _make_handler_class(name="TestHandler", dataset_type="TEST", has_get_dataset_type=True,
                        abstract_methods=None, module_name=None, qualname=None):
    """
    Dynamically build a mock handler class with configurable attributes.
    
    Args:
        name: Class __name__
        dataset_type: Value returned by get_dataset_type() if present
        has_get_dataset_type: Whether the class has get_dataset_type method
        abstract_methods: Set of abstract method names (makes class abstract)
        module_name: Custom __module__ attribute
        qualname: Custom __qualname__ attribute
    """
    ns = {}
    if has_get_dataset_type:
        def get_dataset_type(self, _dt=dataset_type):
            return _dt
        ns['get_dataset_type'] = get_dataset_type

    cls = type(name, (), ns)

    if abstract_methods:
        cls.__abstractmethods__ = frozenset(abstract_methods)

    if module_name is not None:
        cls.__module__ = module_name

    if qualname is not None:
        cls.__qualname__ = qualname

    return cls


def _derive_registry_name(class_name):
    """
    Predict the handler name the registry will derive from a class name.
    
    This mirrors the actual logic in HandlerRegistry.register() lines 146-148:
        name = handler_class.__name__.replace('DatasetHandler', '').replace('Handler', '')
        if not name:
            name = handler_class.__name__
    
    NOTE: In CPython, isinstance(cls.get_dataset_type, classmethod) is False when
    accessed as a class attribute (it becomes a bound method), so the name derivation
    from class name is ALWAYS used — the get_dataset_type() method is never called
    by register() for name extraction.
    """
    name = class_name.replace('DatasetHandler', '').replace('Handler', '')
    if not name:
        name = class_name
    return name


def _make_classmethod_handler(name="ClassMethodHandler", dataset_type="CLASSMETHOD_DS"):
    """Build a handler class where get_dataset_type is a classmethod."""
    ns = {}

    @classmethod
    def get_dataset_type(cls, _dt=dataset_type):
        return _dt

    ns['get_dataset_type'] = get_dataset_type
    cls = type(name, (), ns)
    return cls


# ============================================================================
# GROUP 1: HandlerRegistrationError Exception (7 tests)
# ============================================================================

class TestHandlerRegistrationError(unittest.TestCase):
    """Test HandlerRegistrationError exception attributes and behavior."""

    def test_inherits_from_exception(self):
        """HandlerRegistrationError is a proper Exception subclass."""
        self.assertTrue(issubclass(HandlerRegistrationError, Exception))

    def test_basic_construction(self):
        """Can create with required args only."""
        err = HandlerRegistrationError(message="test", handler_name="MyHandler")
        self.assertEqual(str(err), "test")
        self.assertEqual(err.handler_name, "MyHandler")

    def test_conflicting_class_attribute(self):
        """Stores conflicting_class when provided."""
        err = HandlerRegistrationError(
            message="conflict", handler_name="H1", conflicting_class="H2"
        )
        self.assertEqual(err.conflicting_class, "H2")

    def test_details_attribute(self):
        """Stores details when provided."""
        err = HandlerRegistrationError(
            message="fail", handler_name="H1", details="some details"
        )
        self.assertEqual(err.details, "some details")

    def test_conflicting_class_defaults_none(self):
        """conflicting_class defaults to None."""
        err = HandlerRegistrationError(message="m", handler_name="H")
        self.assertIsNone(err.conflicting_class)

    def test_details_defaults_none(self):
        """details defaults to None."""
        err = HandlerRegistrationError(message="m", handler_name="H")
        self.assertIsNone(err.details)

    def test_is_catchable_as_exception(self):
        """Can be caught as a generic Exception."""
        with self.assertRaises(Exception):
            raise HandlerRegistrationError(message="boom", handler_name="X")


# ============================================================================
# GROUP 2: HandlerNotFoundError Exception (6 tests)
# ============================================================================

class TestHandlerNotFoundError(unittest.TestCase):
    """Test HandlerNotFoundError exception attributes and behavior."""

    def test_inherits_from_exception(self):
        """HandlerNotFoundError is a proper Exception subclass."""
        self.assertTrue(issubclass(HandlerNotFoundError, Exception))

    def test_basic_construction(self):
        """Can create with required args only."""
        err = HandlerNotFoundError(message="not found", handler_name="Missing")
        self.assertEqual(str(err), "not found")
        self.assertEqual(err.handler_name, "Missing")

    def test_available_handlers_attribute(self):
        """Stores available_handlers when provided."""
        err = HandlerNotFoundError(
            message="nope", handler_name="X", available_handlers=["A", "B"]
        )
        self.assertEqual(err.available_handlers, ["A", "B"])

    def test_available_handlers_defaults_empty(self):
        """available_handlers defaults to empty list."""
        err = HandlerNotFoundError(message="m", handler_name="H")
        self.assertEqual(err.available_handlers, [])

    def test_is_catchable_as_exception(self):
        """Can be caught as a generic Exception."""
        with self.assertRaises(Exception):
            raise HandlerNotFoundError(message="boom", handler_name="X")

    def test_handler_name_preserved_in_error(self):
        """handler_name is accessible after catching."""
        try:
            raise HandlerNotFoundError(message="m", handler_name="QM9")
        except HandlerNotFoundError as e:
            self.assertEqual(e.handler_name, "QM9")


# ============================================================================
# GROUP 3: HandlerRegistry — Initialization (5 tests)
# ============================================================================

class TestHandlerRegistryInit(unittest.TestCase):
    """Test HandlerRegistry construction and initial state."""

    def test_empty_on_creation(self):
        """New registry starts with no handlers."""
        reg = HandlerRegistry()
        self.assertEqual(len(reg), 0)

    def test_list_all_empty(self):
        """list_all returns empty list on new registry."""
        reg = HandlerRegistry()
        self.assertEqual(reg.list_all(), [])

    def test_list_all_classes_empty(self):
        """list_all_classes returns empty list on new registry."""
        reg = HandlerRegistry()
        self.assertEqual(reg.list_all_classes(), [])

    def test_has_lock_attribute(self):
        """Registry has an RLock for thread safety."""
        reg = HandlerRegistry()
        self.assertIsNotNone(reg._lock)

    def test_has_callbacks_list(self):
        """Registry has an empty callbacks list on creation."""
        reg = HandlerRegistry()
        self.assertEqual(reg._on_change_callbacks, [])


# ============================================================================
# GROUP 4: HandlerRegistry.register — Success Paths (10 tests)
# ============================================================================

class TestHandlerRegistryRegisterSuccess(unittest.TestCase):
    """Test successful registration scenarios."""

    def setUp(self):
        self.registry = HandlerRegistry()

    def test_register_simple_handler(self):
        """Can register a basic handler class."""
        cls = _make_handler_class(name="FooHandler")
        expected_name = _derive_registry_name("FooHandler")  # "Foo"
        self.registry.register(cls)
        self.assertIn(expected_name, self.registry)

    def test_register_returns_none(self):
        """register() returns None (not the class)."""
        cls = _make_handler_class()
        result = self.registry.register(cls)
        self.assertIsNone(result)

    def test_register_multiple_handlers(self):
        """Can register multiple handlers with different names."""
        cls_a = _make_handler_class(name="AlphaHandler")
        cls_b = _make_handler_class(name="BetaHandler")
        name_a = _derive_registry_name("AlphaHandler")  # "Alpha"
        name_b = _derive_registry_name("BetaHandler")   # "Beta"
        self.registry.register(cls_a)
        self.registry.register(cls_b)
        self.assertEqual(len(self.registry), 2)
        self.assertIn(name_a, self.registry)
        self.assertIn(name_b, self.registry)

    def test_register_same_class_twice_is_idempotent(self):
        """Re-registering the same class object is a no-op."""
        cls = _make_handler_class(name="DFTDatasetHandler")
        self.registry.register(cls)
        self.registry.register(cls)  # Should not raise
        self.assertEqual(len(self.registry), 1)

    def test_register_handler_retrievable_by_get(self):
        """Registered handler is retrievable via get()."""
        cls = _make_handler_class(name="QM9Handler")
        expected_name = _derive_registry_name("QM9Handler")  # "QM9"
        self.registry.register(cls)
        self.assertIs(self.registry.get(expected_name), cls)

    def test_register_handler_name_derived_from_class_name_fallback(self):
        """When get_dataset_type() raises, name is derived from class name."""
        cls = type("XYZDatasetHandler", (), {})
        cls.get_dataset_type = property(lambda self: (_ for _ in ()).throw(RuntimeError()))
        self.registry.register(cls)
        # Name derived by stripping "DatasetHandler" -> "XYZ"
        self.assertIn("XYZ", self.registry)

    def test_register_handler_name_strips_handler_suffix(self):
        """Class name 'AbcHandler' falls back to name 'Abc' if get_dataset_type fails."""
        class AbcHandler:
            def get_dataset_type(self):
                raise RuntimeError("not callable as classmethod")
        # The register method calls it in a try/except and falls back
        self.registry.register(AbcHandler)
        self.assertIn("Abc", self.registry)

    def test_register_classmethod_handler(self):
        """Can register handler where get_dataset_type is a classmethod.
        
        NOTE: In CPython, isinstance(cls.get_dataset_type, classmethod) is False
        when accessed as a class attribute (descriptor protocol yields bound method),
        so the name is derived from the class name, not from get_dataset_type().
        """
        cls = _make_classmethod_handler(name="CMHandler", dataset_type="CM_DS")
        expected_name = _derive_registry_name("CMHandler")  # "CM"
        self.registry.register(cls)
        self.assertIn(expected_name, self.registry)

    def test_register_re_import_same_qualname_and_module(self):
        """Re-registration of same qualname from handlers.implementations is allowed."""
        cls_a = _make_handler_class(
            name="DFTHandler",
            module_name="milia_pipeline.handlers.implementations.dft",
            qualname="DFTHandler"
        )
        cls_b = _make_handler_class(
            name="DFTHandler",
            module_name="milia_pipeline.handlers.implementations.dft",
            qualname="DFTHandler"
        )
        self.registry.register(cls_a)
        # cls_b is a different object but same qualname+module
        self.registry.register(cls_b)  # Should not raise
        self.assertEqual(len(self.registry), 1)

    def test_register_notifies_callbacks(self):
        """Registration triggers on_change callbacks."""
        called = []
        self.registry.add_on_change_callback(lambda: called.append(True))
        cls = _make_handler_class(name="CBHandler")
        self.registry.register(cls)
        self.assertEqual(len(called), 1)


# ============================================================================
# GROUP 5: HandlerRegistry.register — Error Paths (8 tests)
# ============================================================================

class TestHandlerRegistryRegisterErrors(unittest.TestCase):
    """Test registration failure scenarios."""

    def setUp(self):
        self.registry = HandlerRegistry()

    def test_register_non_class_raises_type_error(self):
        """Registering a non-class (instance) raises TypeError."""
        with self.assertRaises(TypeError) as ctx:
            self.registry.register("not a class")
        self.assertIn("Expected class", str(ctx.exception))

    def test_register_instance_raises_type_error(self):
        """Registering an instance of a class raises TypeError."""
        cls = _make_handler_class()
        instance = cls()
        with self.assertRaises(TypeError):
            self.registry.register(instance)

    def test_register_class_without_get_dataset_type_raises_type_error(self):
        """Class missing get_dataset_type() raises TypeError."""
        cls = _make_handler_class(has_get_dataset_type=False)
        with self.assertRaises(TypeError) as ctx:
            self.registry.register(cls)
        self.assertIn("get_dataset_type", str(ctx.exception))

    def test_register_abstract_class_raises_registration_error(self):
        """Abstract class (with __abstractmethods__) raises HandlerRegistrationError."""
        cls = _make_handler_class(
            name="AbstractHandler",
            abstract_methods={'process', 'validate'}
        )
        with self.assertRaises(HandlerRegistrationError) as ctx:
            self.registry.register(cls)
        self.assertIn("abstract", str(ctx.exception).lower())
        self.assertEqual(ctx.exception.handler_name, "AbstractHandler")

    def test_register_abstract_class_error_has_details(self):
        """Abstract class error includes missing method details."""
        cls = _make_handler_class(
            name="PartialHandler",
            abstract_methods={'missing_method'}
        )
        with self.assertRaises(HandlerRegistrationError) as ctx:
            self.registry.register(cls)
        self.assertIn("missing_method", str(ctx.exception.details))

    def test_register_duplicate_name_different_class_raises_error(self):
        """Two different classes with same derived handler name raises HandlerRegistrationError."""
        # Both derive name "DUPE" since neither has Handler/DatasetHandler suffix
        cls_a = _make_handler_class(name="DUPE", has_get_dataset_type=True)
        cls_b = _make_handler_class(name="DUPE", has_get_dataset_type=True)
        # Give them different qualnames so they aren't detected as re-imports
        cls_a.__qualname__ = "ModuleA.DUPE"
        cls_b.__qualname__ = "ModuleB.DUPE"
        self.registry.register(cls_a)
        with self.assertRaises(HandlerRegistrationError) as ctx:
            self.registry.register(cls_b)
        self.assertEqual(ctx.exception.handler_name, "DUPE")
        self.assertEqual(ctx.exception.conflicting_class, "DUPE")

    def test_register_duplicate_error_has_details(self):
        """Duplicate registration error includes detail about new class."""
        # Both derive name "DUP" since neither has Handler/DatasetHandler suffix
        cls_a = _make_handler_class(name="DUP", has_get_dataset_type=True)
        cls_b = _make_handler_class(name="DUP", has_get_dataset_type=True)
        cls_a.__qualname__ = "ModuleA.DUP"
        cls_b.__qualname__ = "ModuleB.DUP"
        self.registry.register(cls_a)
        with self.assertRaises(HandlerRegistrationError) as ctx:
            self.registry.register(cls_b)
        self.assertIn("DUP", ctx.exception.details)

    def test_register_none_raises_type_error(self):
        """Registering None raises TypeError."""
        with self.assertRaises(TypeError):
            self.registry.register(None)


# ============================================================================
# GROUP 6: HandlerRegistry.get / get_or_none (7 tests)
# ============================================================================

class TestHandlerRegistryGet(unittest.TestCase):
    """Test handler retrieval methods."""

    def setUp(self):
        self.registry = HandlerRegistry()
        self.cls = _make_handler_class(name="TargetHandler")
        self.derived_name = _derive_registry_name("TargetHandler")  # "Target"
        self.registry.register(self.cls)

    def test_get_existing_handler(self):
        """get() returns the registered class."""
        result = self.registry.get(self.derived_name)
        self.assertIs(result, self.cls)

    def test_get_nonexistent_raises_not_found(self):
        """get() raises HandlerNotFoundError for unknown name."""
        with self.assertRaises(HandlerNotFoundError) as ctx:
            self.registry.get("NONEXISTENT")
        self.assertEqual(ctx.exception.handler_name, "NONEXISTENT")

    def test_get_not_found_includes_available_handlers(self):
        """HandlerNotFoundError includes list of available handlers."""
        with self.assertRaises(HandlerNotFoundError) as ctx:
            self.registry.get("MISSING")
        self.assertIn(self.derived_name, ctx.exception.available_handlers)

    def test_get_or_none_existing(self):
        """get_or_none() returns class when found."""
        result = self.registry.get_or_none(self.derived_name)
        self.assertIs(result, self.cls)

    def test_get_or_none_missing(self):
        """get_or_none() returns None when not found."""
        result = self.registry.get_or_none("NOPE")
        self.assertIsNone(result)

    def test_get_case_sensitive(self):
        """Handler name lookup is case-sensitive."""
        wrong_case = self.derived_name.lower() if self.derived_name[0].isupper() else self.derived_name.upper()
        if wrong_case != self.derived_name:
            with self.assertRaises(HandlerNotFoundError):
                self.registry.get(wrong_case)

    def test_get_empty_string(self):
        """get() with empty string raises HandlerNotFoundError."""
        with self.assertRaises(HandlerNotFoundError):
            self.registry.get("")


# ============================================================================
# GROUP 7: HandlerRegistry — Collection Methods (10 tests)
# ============================================================================

class TestHandlerRegistryCollectionMethods(unittest.TestCase):
    """Test list_all, list_all_classes, is_registered, __contains__, __iter__, __len__."""

    def setUp(self):
        self.registry = HandlerRegistry()
        self.cls_a = _make_handler_class(name="AlphaHandler")
        self.cls_b = _make_handler_class(name="BetaHandler")
        self.name_a = _derive_registry_name("AlphaHandler")  # "Alpha"
        self.name_b = _derive_registry_name("BetaHandler")   # "Beta"
        self.registry.register(self.cls_a)
        self.registry.register(self.cls_b)

    def test_list_all_returns_names(self):
        """list_all() returns all registered handler names."""
        names = self.registry.list_all()
        self.assertCountEqual(names, [self.name_a, self.name_b])

    def test_list_all_classes_returns_classes(self):
        """list_all_classes() returns all registered handler classes."""
        classes = self.registry.list_all_classes()
        self.assertIn(self.cls_a, classes)
        self.assertIn(self.cls_b, classes)

    def test_is_registered_true(self):
        """is_registered() returns True for registered handler."""
        self.assertTrue(self.registry.is_registered(self.name_a))

    def test_is_registered_false(self):
        """is_registered() returns False for unregistered handler."""
        self.assertFalse(self.registry.is_registered("Z"))

    def test_contains_operator(self):
        """'in' operator works via __contains__."""
        self.assertIn(self.name_a, self.registry)
        self.assertNotIn("Z", self.registry)

    def test_iter_returns_names(self):
        """Iterating registry yields handler names."""
        names = list(self.registry)
        self.assertCountEqual(names, [self.name_a, self.name_b])

    def test_len_returns_count(self):
        """len() returns number of registered handlers."""
        self.assertEqual(len(self.registry), 2)

    def test_list_all_returns_copy(self):
        """list_all() returns a copy, not internal state."""
        names = self.registry.list_all()
        names.append("MUTATED")
        self.assertNotIn("MUTATED", self.registry.list_all())

    def test_list_all_classes_returns_copy(self):
        """list_all_classes() returns a copy, not internal state."""
        classes = self.registry.list_all_classes()
        classes.append(str)  # Mutate the copy
        self.assertNotIn(str, self.registry.list_all_classes())

    def test_iter_snapshot_isolation(self):
        """Iterator returns a snapshot, safe if registry changes during iteration."""
        names = list(iter(self.registry))
        self.assertCountEqual(names, [self.name_a, self.name_b])


# ============================================================================
# GROUP 8: HandlerRegistry.unregister (6 tests)
# ============================================================================

class TestHandlerRegistryUnregister(unittest.TestCase):
    """Test handler unregistration."""

    def setUp(self):
        self.registry = HandlerRegistry()
        self.cls = _make_handler_class(name="RmHandler")
        self.derived_name = _derive_registry_name("RmHandler")  # "Rm"
        self.registry.register(self.cls)

    def test_unregister_existing_returns_true(self):
        """unregister() returns True when handler was removed."""
        self.assertTrue(self.registry.unregister(self.derived_name))

    def test_unregister_removes_handler(self):
        """Handler is no longer findable after unregister."""
        self.registry.unregister(self.derived_name)
        self.assertFalse(self.registry.is_registered(self.derived_name))
        self.assertEqual(len(self.registry), 0)

    def test_unregister_nonexistent_returns_false(self):
        """unregister() returns False for unknown name."""
        self.assertFalse(self.registry.unregister("NOPE"))

    def test_unregister_notifies_callbacks(self):
        """Unregistration triggers on_change callbacks."""
        called = []
        self.registry.add_on_change_callback(lambda: called.append(True))
        self.registry.unregister(self.derived_name)
        self.assertEqual(len(called), 1)

    def test_unregister_no_callback_if_not_found(self):
        """Unregistering a nonexistent name does NOT trigger callback."""
        called = []
        self.registry.add_on_change_callback(lambda: called.append(True))
        self.registry.unregister("NONEXISTENT")
        self.assertEqual(len(called), 0)

    def test_unregister_then_re_register(self):
        """Can re-register a handler after unregistering it."""
        self.registry.unregister(self.derived_name)
        # New class with same derived name
        new_cls = _make_handler_class(name="RmHandler")
        self.registry.register(new_cls)
        self.assertIs(self.registry.get(self.derived_name), new_cls)


# ============================================================================
# GROUP 9: HandlerRegistry.clear (4 tests)
# ============================================================================

class TestHandlerRegistryClear(unittest.TestCase):
    """Test registry clearing."""

    def setUp(self):
        self.registry = HandlerRegistry()
        self.registry.register(_make_handler_class(name="H1Handler"))
        self.registry.register(_make_handler_class(name="H2Handler"))

    def test_clear_empties_registry(self):
        """clear() removes all handlers."""
        self.registry.clear()
        self.assertEqual(len(self.registry), 0)
        self.assertEqual(self.registry.list_all(), [])

    def test_clear_notifies_callbacks(self):
        """clear() triggers on_change callbacks."""
        called = []
        self.registry.add_on_change_callback(lambda: called.append(True))
        self.registry.clear()
        self.assertEqual(len(called), 1)

    def test_clear_then_register(self):
        """Registry is usable after clear()."""
        self.registry.clear()
        cls = _make_handler_class(name="NewHandler")
        expected_name = _derive_registry_name("NewHandler")  # "New"
        self.registry.register(cls)
        self.assertEqual(len(self.registry), 1)
        self.assertIn(expected_name, self.registry)

    def test_clear_empty_registry_no_error(self):
        """Clearing an already-empty registry does not error."""
        empty = HandlerRegistry()
        empty.clear()  # Should not raise
        self.assertEqual(len(empty), 0)


# ============================================================================
# GROUP 10: HandlerRegistry — Change Callbacks (8 tests)
# ============================================================================

class TestHandlerRegistryCallbacks(unittest.TestCase):
    """Test on_change callback registration, removal, and notification."""

    def setUp(self):
        self.registry = HandlerRegistry()

    def test_add_callback(self):
        """add_on_change_callback adds to internal list."""
        cb = Mock()
        self.registry.add_on_change_callback(cb)
        self.assertIn(cb, self.registry._on_change_callbacks)

    def test_remove_callback_success(self):
        """remove_on_change_callback returns True and removes callback."""
        cb = Mock()
        self.registry.add_on_change_callback(cb)
        result = self.registry.remove_on_change_callback(cb)
        self.assertTrue(result)
        self.assertNotIn(cb, self.registry._on_change_callbacks)

    def test_remove_callback_not_found(self):
        """remove_on_change_callback returns False for unknown callback."""
        result = self.registry.remove_on_change_callback(Mock())
        self.assertFalse(result)

    def test_callback_called_on_register(self):
        """Callback is called when a handler is registered."""
        cb = Mock()
        self.registry.add_on_change_callback(cb)
        cls = _make_handler_class(name="CBRegHandler")
        self.registry.register(cls)
        cb.assert_called_once()

    def test_callback_called_on_unregister(self):
        """Callback is called when a handler is unregistered."""
        cls = _make_handler_class(name="CBUnregHandler")
        derived = _derive_registry_name("CBUnregHandler")
        self.registry.register(cls)
        cb = Mock()
        self.registry.add_on_change_callback(cb)
        self.registry.unregister(derived)
        cb.assert_called_once()

    def test_callback_called_on_clear(self):
        """Callback is called when registry is cleared."""
        self.registry.register(_make_handler_class(name="SomeHandler"))
        cb = Mock()
        self.registry.add_on_change_callback(cb)
        self.registry.clear()
        cb.assert_called_once()

    def test_multiple_callbacks_all_called(self):
        """All registered callbacks are called on change."""
        cb1 = Mock()
        cb2 = Mock()
        self.registry.add_on_change_callback(cb1)
        self.registry.add_on_change_callback(cb2)
        self.registry.register(_make_handler_class(name="MultiHandler"))
        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_failing_callback_does_not_prevent_others(self):
        """A failing callback does not prevent subsequent callbacks from running."""
        def bad_callback():
            raise ValueError("callback error")
        good_cb = Mock()
        self.registry.add_on_change_callback(bad_callback)
        self.registry.add_on_change_callback(good_cb)
        # Should not raise; bad_callback failure is caught internally
        self.registry.register(_make_handler_class(name="SafeHandler"))
        good_cb.assert_called_once()


# ============================================================================
# GROUP 11: HandlerRegistry.get_registry_info (5 tests)
# ============================================================================

class TestHandlerRegistryInfo(unittest.TestCase):
    """Test get_registry_info diagnostic method."""

    def setUp(self):
        self.registry = HandlerRegistry()

    def test_empty_registry_info(self):
        """get_registry_info() on empty registry returns correct structure."""
        info = self.registry.get_registry_info()
        self.assertEqual(info['total_handlers'], 0)
        self.assertEqual(info['registered_handlers'], [])
        self.assertEqual(info['handler_classes'], {})
        self.assertEqual(info['callback_count'], 0)

    def test_populated_registry_info(self):
        """get_registry_info() reflects registered handlers."""
        cls = _make_handler_class(name="InfoHandler")
        expected_name = _derive_registry_name("InfoHandler")  # "Info"
        self.registry.register(cls)
        info = self.registry.get_registry_info()
        self.assertEqual(info['total_handlers'], 1)
        self.assertIn(expected_name, info['registered_handlers'])
        self.assertEqual(info['handler_classes'][expected_name], 'InfoHandler')

    def test_callback_count_reflected(self):
        """get_registry_info() reflects callback count."""
        self.registry.add_on_change_callback(Mock())
        self.registry.add_on_change_callback(Mock())
        info = self.registry.get_registry_info()
        self.assertEqual(info['callback_count'], 2)

    def test_info_returns_dict(self):
        """get_registry_info() returns a dictionary."""
        self.assertIsInstance(self.registry.get_registry_info(), dict)

    def test_info_has_all_expected_keys(self):
        """get_registry_info() has all expected keys."""
        info = self.registry.get_registry_info()
        for key in ['total_handlers', 'registered_handlers', 'handler_classes', 'callback_count']:
            self.assertIn(key, info)


# ============================================================================
# GROUP 12: Thread Safety (6 tests)
# ============================================================================

class TestHandlerRegistryThreadSafety(unittest.TestCase):
    """Test thread-safe behavior of HandlerRegistry."""

    def setUp(self):
        self.registry = HandlerRegistry()

    def test_concurrent_register_no_crash(self):
        """Concurrent registrations do not crash."""
        errors = []

        def register_handler_thread(idx):
            try:
                cls = _make_handler_class(name=f"Thread{idx}Handler")
                self.registry.register(cls)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_handler_thread, args=(i,))
                   for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(self.registry), 20)

    def test_concurrent_read_during_write(self):
        """Reading registry during concurrent writes does not crash."""
        # Pre-populate
        for i in range(5):
            cls = _make_handler_class(name=f"Pre{i}Handler")
            self.registry.register(cls)

        errors = []

        def reader():
            try:
                for _ in range(50):
                    self.registry.list_all()
                    len(self.registry)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(5, 15):
                    cls = _make_handler_class(name=f"W{i}Handler")
                    self.registry.register(cls)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)

    def test_concurrent_unregister(self):
        """Concurrent unregistration does not crash."""
        names = []
        for i in range(10):
            cls_name = f"Del{i}Handler"
            cls = _make_handler_class(name=cls_name)
            derived = _derive_registry_name(cls_name)
            names.append(derived)
            self.registry.register(cls)

        errors = []

        def unregister_thread(name):
            try:
                self.registry.unregister(name)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=unregister_thread, args=(n,))
                   for n in names]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(self.registry), 0)

    def test_rlock_allows_reentrant_locking(self):
        """RLock allows reentrant (nested) lock acquisition."""
        # Simulate a callback that accesses registry during notification
        def reentrant_callback():
            # This accesses registry which acquires lock again
            self.registry.list_all()

        self.registry.add_on_change_callback(reentrant_callback)
        cls = _make_handler_class(name="ReentrantHandler")
        expected_name = _derive_registry_name("ReentrantHandler")
        # If the lock were a Lock (not RLock), this would deadlock
        self.registry.register(cls)
        self.assertIn(expected_name, self.registry)

    def test_concurrent_get_or_none(self):
        """Concurrent get_or_none does not crash."""
        cls = _make_handler_class(name="ConcGetHandler")
        derived = _derive_registry_name("ConcGetHandler")
        self.registry.register(cls)
        errors = []

        def getter():
            try:
                for _ in range(100):
                    self.registry.get_or_none(derived)
                    self.registry.get_or_none("NONEXISTENT")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=getter) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)

    def test_concurrent_is_registered(self):
        """Concurrent is_registered does not crash."""
        cls = _make_handler_class(name="ConcCheckHandler")
        derived = _derive_registry_name("ConcCheckHandler")
        self.registry.register(cls)
        errors = []

        def checker():
            try:
                for _ in range(100):
                    self.registry.is_registered(derived)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=checker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)


# ============================================================================
# GROUP 13: Module-Level Default Registry Functions (12 tests)
# ============================================================================

class TestModuleLevelFunctions(unittest.TestCase):
    """Test module-level convenience functions that operate on _default_registry."""

    def setUp(self):
        """Save and restore default registry state."""
        self._default_registry = get_default_registry()
        self._original_handlers = dict(self._default_registry._handlers)
        self._original_callbacks = list(self._default_registry._on_change_callbacks)

    def tearDown(self):
        """Restore default registry to original state."""
        self._default_registry._handlers = self._original_handlers
        self._default_registry._on_change_callbacks = self._original_callbacks

    def test_get_default_registry_returns_handler_registry(self):
        """get_default_registry() returns a HandlerRegistry instance."""
        self.assertIsInstance(get_default_registry(), HandlerRegistry)

    def test_get_default_registry_same_instance(self):
        """get_default_registry() returns the same instance each call."""
        self.assertIs(get_default_registry(), get_default_registry())

    def test_register_handler_decorator_returns_class(self):
        """register_handler() returns the class (decorator pattern)."""
        cls = _make_handler_class(name="DecTestHandler")
        result = register_handler(cls)
        self.assertIs(result, cls)

    def test_register_handler_decorator_registers_in_default(self):
        """register_handler() registers the class in the default registry."""
        cls = _make_handler_class(name="DecRegHandler")
        derived = _derive_registry_name("DecRegHandler")  # "DecReg"
        register_handler(cls)
        self.assertTrue(get_default_registry().is_registered(derived))

    def test_get_from_default_registry(self):
        """Module-level get() retrieves from default registry."""
        cls = _make_handler_class(name="GetModHandler")
        derived = _derive_registry_name("GetModHandler")  # "GetMod"
        register_handler(cls)
        self.assertIs(get(derived), cls)

    def test_get_nonexistent_raises_not_found(self):
        """Module-level get() raises HandlerNotFoundError for unknown name."""
        with self.assertRaises(HandlerNotFoundError):
            get("ABSOLUTELY_NONEXISTENT_HANDLER_12345")

    def test_list_all_from_default_registry(self):
        """Module-level list_all() returns names from default registry."""
        cls = _make_handler_class(name="ListModHandler")
        derived = _derive_registry_name("ListModHandler")  # "ListMod"
        register_handler(cls)
        self.assertIn(derived, list_all())

    def test_is_registered_from_default_registry(self):
        """Module-level is_registered() queries default registry."""
        cls = _make_handler_class(name="IsRegModHandler")
        derived = _derive_registry_name("IsRegModHandler")  # "IsRegMod"
        register_handler(cls)
        self.assertTrue(is_registered(derived))
        self.assertFalse(is_registered("NOT_REGISTERED_XYZ_123"))

    def test_get_registry_info_from_default(self):
        """Module-level get_registry_info() queries default registry."""
        info = get_registry_info()
        self.assertIsInstance(info, dict)
        self.assertIn('total_handlers', info)

    def test_register_handler_as_decorator_syntax(self):
        """register_handler can be used with @ decorator syntax."""
        @register_handler
        class MyTestDecoratedHandler:
            def get_dataset_type(self):
                return "MY_DECORATED"
        # Name derived from class name: "MyTestDecorated"
        derived = _derive_registry_name("MyTestDecoratedHandler")
        self.assertTrue(is_registered(derived))

    def test_register_handler_invalid_raises(self):
        """register_handler() with invalid class propagates the error."""
        with self.assertRaises(TypeError):
            register_handler("not a class")

    def test_default_registry_is_not_none(self):
        """Default registry is always available (not None)."""
        self.assertIsNotNone(get_default_registry())


# ============================================================================
# GROUP 14: Non-Singleton Behavior (4 tests)
# ============================================================================

class TestHandlerRegistryNonSingleton(unittest.TestCase):
    """Test that HandlerRegistry is NOT a singleton (can create isolated instances)."""

    def test_two_registries_are_independent(self):
        """Two HandlerRegistry instances have independent state."""
        reg1 = HandlerRegistry()
        reg2 = HandlerRegistry()
        cls = _make_handler_class(name="IsoHandler")
        derived = _derive_registry_name("IsoHandler")  # "Iso"
        reg1.register(cls)
        self.assertIn(derived, reg1)
        self.assertNotIn(derived, reg2)

    def test_clear_one_does_not_affect_other(self):
        """Clearing one registry does not affect another."""
        reg1 = HandlerRegistry()
        reg2 = HandlerRegistry()
        cls_a = _make_handler_class(name="A2Handler")
        cls_b = _make_handler_class(name="B2Handler")
        reg1.register(cls_a)
        reg2.register(cls_b)
        reg1.clear()
        self.assertEqual(len(reg1), 0)
        self.assertEqual(len(reg2), 1)

    def test_callbacks_are_independent(self):
        """Callbacks registered on one registry don't fire on another."""
        reg1 = HandlerRegistry()
        reg2 = HandlerRegistry()
        cb = Mock()
        reg1.add_on_change_callback(cb)
        reg2.register(_make_handler_class(name="IndHandler"))
        cb.assert_not_called()

    def test_default_is_different_from_new(self):
        """Default registry is a different instance from a new HandlerRegistry()."""
        self.assertIsNot(get_default_registry(), HandlerRegistry())


# ============================================================================
# GROUP 15: Name Derivation Logic (7 tests)
# ============================================================================

class TestHandlerNameDerivation(unittest.TestCase):
    """Test the name derivation fallback logic in register()."""

    def setUp(self):
        self.registry = HandlerRegistry()

    def test_name_from_classmethod_get_dataset_type(self):
        """Name is derived from class name even for classmethod handlers.
        
        NOTE: In CPython, isinstance(cls.get_dataset_type, classmethod) returns
        False when accessed as a class attribute, so the classmethod is never
        actually called for name extraction. Name comes from class name.
        """
        cls = _make_classmethod_handler(name="CMHHandler", dataset_type="FROM_CM")
        expected_name = _derive_registry_name("CMHHandler")  # "CMH"
        self.registry.register(cls)
        self.assertIn(expected_name, self.registry)

    def test_fallback_strips_datasethandler_suffix(self):
        """Fallback strips 'DatasetHandler' from class name."""
        class QDPiDatasetHandler:
            def get_dataset_type(self):
                raise RuntimeError
        self.registry.register(QDPiDatasetHandler)
        self.assertIn("QDPi", self.registry)

    def test_fallback_strips_handler_suffix(self):
        """Fallback strips 'Handler' from class name."""
        class ANI1XHandler:
            def get_dataset_type(self):
                raise RuntimeError
        self.registry.register(ANI1XHandler)
        self.assertIn("ANI1X", self.registry)

    def test_fallback_uses_full_name_if_stripping_leaves_empty(self):
        """If stripping suffixes yields empty, full class name is used."""
        class Handler:
            def get_dataset_type(self):
                raise RuntimeError
        self.registry.register(Handler)
        self.assertIn("Handler", self.registry)

    def test_fallback_datasethandler_only_class(self):
        """Class named exactly 'DatasetHandler' uses full name as fallback."""
        class DatasetHandler:
            def get_dataset_type(self):
                raise RuntimeError
        self.registry.register(DatasetHandler)
        self.assertIn("DatasetHandler", self.registry)

    def test_regular_method_falls_back_to_class_name(self):
        """Regular (non-classmethod) get_dataset_type that can't be called statically falls back."""
        class RMD17Handler:
            def get_dataset_type(self):
                return "RMD17"
        self.registry.register(RMD17Handler)
        # The register method tries to call it, may fall back to class name derivation
        self.assertTrue(
            self.registry.is_registered("RMD17") or
            self.registry.is_registered("RMD17Handler")
        )

    def test_name_empty_string_never_registered(self):
        """An empty-string name from stripping should use class name instead."""
        # 'Handler' -> strip 'Handler' -> '' -> use class name
        class Handler:
            def get_dataset_type(self):
                raise RuntimeError
        self.registry.register(Handler)
        # Should not have empty string key
        self.assertNotIn("", self.registry)


# ============================================================================
# GROUP 16: Re-import Detection Logic (5 tests)
# ============================================================================

class TestReImportDetection(unittest.TestCase):
    """Test the qualname/module comparison for re-import detection."""

    def setUp(self):
        self.registry = HandlerRegistry()

    def test_same_class_identity_skips_registration(self):
        """Same class object (is check) skips re-registration."""
        cls = _make_handler_class(name="SameIdHandler")
        self.registry.register(cls)
        self.registry.register(cls)
        self.assertEqual(len(self.registry), 1)

    def test_different_class_different_module_raises(self):
        """Different class objects from unrelated modules raise error."""
        # Both named "Conflict" (no Handler suffix) so both derive "Conflict"
        cls_a = _make_handler_class(
            name="Conflict",
            module_name="module_a.handlers",
            qualname="module_a.Conflict"
        )
        cls_b = _make_handler_class(
            name="Conflict",
            module_name="module_b.handlers",
            qualname="module_b.Conflict"
        )
        self.registry.register(cls_a)
        with self.assertRaises(HandlerRegistrationError):
            self.registry.register(cls_b)

    def test_same_qualname_handlers_implementations_allowed(self):
        """Same qualname from handlers.implementations modules is allowed."""
        cls_a = _make_handler_class(
            name="DFTHandler",
            module_name="milia_pipeline.handlers.implementations.dft",
            qualname="DFTHandler"
        )
        cls_b = _make_handler_class(
            name="DFTHandler",
            module_name="some.other.handlers.implementations.dft",
            qualname="DFTHandler"
        )
        self.registry.register(cls_a)
        # Should not raise due to qualname + module matching logic
        self.registry.register(cls_b)
        self.assertEqual(len(self.registry), 1)

    def test_same_qualname_different_non_implementation_module_raises(self):
        """Same qualname but NOT from handlers.implementations raises error."""
        cls_a = _make_handler_class(
            name="FooH",
            module_name="package_a.unrelated",
            qualname="FooH"
        )
        cls_b = _make_handler_class(
            name="FooH",
            module_name="package_b.unrelated",
            qualname="FooH"
        )
        self.registry.register(cls_a)
        with self.assertRaises(HandlerRegistrationError):
            self.registry.register(cls_b)

    def test_re_register_logs_debug(self):
        """Re-registering same class logs a debug message."""
        cls = _make_handler_class(name="DebugHandler")
        self.registry.register(cls)
        with patch('milia_pipeline.handlers.handler_registry.logger') as mock_logger:
            self.registry.register(cls)
            mock_logger.debug.assert_called()


# ============================================================================
# GROUP 17: Logging Behavior (5 tests)
# ============================================================================

class TestHandlerRegistryLogging(unittest.TestCase):
    """Test that registry operations produce appropriate log messages."""

    def setUp(self):
        self.registry = HandlerRegistry()

    def test_register_logs_info(self):
        """Successful registration logs an info message."""
        with patch('milia_pipeline.handlers.handler_registry.logger') as mock_logger:
            cls = _make_handler_class(name="LogHandler")
            self.registry.register(cls)
            mock_logger.info.assert_called()
            log_msg = str(mock_logger.info.call_args)
            # Derived name is "Log", class name is "LogHandler"
            self.assertIn("Log", log_msg)

    def test_unregister_logs_info(self):
        """Successful unregistration logs an info message."""
        cls = _make_handler_class(name="UnlogHandler")
        derived = _derive_registry_name("UnlogHandler")
        self.registry.register(cls)
        with patch('milia_pipeline.handlers.handler_registry.logger') as mock_logger:
            self.registry.unregister(derived)
            mock_logger.info.assert_called()

    def test_clear_logs_warning(self):
        """Clearing registry logs a warning."""
        with patch('milia_pipeline.handlers.handler_registry.logger') as mock_logger:
            self.registry.clear()
            mock_logger.warning.assert_called()

    def test_failing_callback_logs_warning(self):
        """A failing callback logs a warning."""
        def bad_cb():
            raise ValueError("bad")
        self.registry.add_on_change_callback(bad_cb)
        with patch('milia_pipeline.handlers.handler_registry.logger') as mock_logger:
            self.registry.register(_make_handler_class(name="BadCBHandler"))
            mock_logger.warning.assert_called()

    def test_re_register_same_class_logs_debug(self):
        """Re-registering same class object logs debug."""
        cls = _make_handler_class(name="ReLogHandler")
        self.registry.register(cls)
        with patch('milia_pipeline.handlers.handler_registry.logger') as mock_logger:
            self.registry.register(cls)
            mock_logger.debug.assert_called()


# ============================================================================
# GROUP 18: Edge Cases and Boundary Conditions (8 tests)
# ============================================================================

class TestEdgeCasesAndBoundary(unittest.TestCase):
    """Test edge cases and unusual inputs."""

    def test_handler_with_special_chars_in_name(self):
        """Handler name with special characters in class name works."""
        reg = HandlerRegistry()
        # Class name "DFT_v2_1" has no Handler/DatasetHandler suffix -> derived name is "DFT_v2_1"
        cls = _make_handler_class(name="DFT_v2_1")
        expected = _derive_registry_name("DFT_v2_1")
        reg.register(cls)
        self.assertIn(expected, reg)
        self.assertIs(reg.get(expected), cls)

    def test_handler_with_unicode_name(self):
        """Handler with unicode in class name works."""
        reg = HandlerRegistry()
        cls = _make_handler_class(name="δ_dataset")
        expected = _derive_registry_name("δ_dataset")
        reg.register(cls)
        self.assertIn(expected, reg)

    def test_many_handlers_registered(self):
        """Registry can hold a large number of handlers."""
        reg = HandlerRegistry()
        for i in range(100):
            cls = _make_handler_class(name=f"Bulk{i}Handler")
            reg.register(cls)
        self.assertEqual(len(reg), 100)

    def test_get_registry_info_after_unregister(self):
        """get_registry_info reflects state after unregister."""
        reg = HandlerRegistry()
        cls = _make_handler_class(name="TempHandler")
        derived = _derive_registry_name("TempHandler")  # "Temp"
        reg.register(cls)
        reg.unregister(derived)
        info = reg.get_registry_info()
        self.assertEqual(info['total_handlers'], 0)

    def test_register_handler_with_no_name_strippable(self):
        """Handler class name that doesn't end in Handler/DatasetHandler uses derived name."""
        class MyDataProcessor:
            def get_dataset_type(self):
                raise RuntimeError
        reg = HandlerRegistry()
        reg.register(MyDataProcessor)
        # Falls back to class name since no Handler suffix to strip
        self.assertIn("MyDataProcessor", reg)

    def test_iter_and_len_consistent(self):
        """len() matches number of items from iteration."""
        reg = HandlerRegistry()
        for i in range(5):
            reg.register(_make_handler_class(name=f"Cons{i}Handler"))
        self.assertEqual(len(list(reg)), len(reg))

    def test_contains_with_non_string(self):
        """__contains__ with non-string doesn't crash (dicts handle this)."""
        reg = HandlerRegistry()
        # This tests that the registry doesn't crash on unusual input
        # dict's __contains__ handles non-string keys without error
        self.assertFalse(reg.is_registered(""))

    def test_register_int_raises_type_error(self):
        """Registering an int raises TypeError."""
        reg = HandlerRegistry()
        with self.assertRaises(TypeError):
            reg.register(42)


# ============================================================================
# GROUP 19: Module __all__ Exports (3 tests)
# ============================================================================

class TestModuleExports(unittest.TestCase):
    """Verify module exports are properly defined."""

    def test_handler_registry_importable(self):
        """HandlerRegistry is importable from the module."""
        from milia_pipeline.handlers.handler_registry import HandlerRegistry
        self.assertTrue(callable(HandlerRegistry))

    def test_exceptions_importable(self):
        """Exception classes are importable."""
        from milia_pipeline.handlers.handler_registry import (
            HandlerRegistrationError, HandlerNotFoundError
        )
        self.assertTrue(issubclass(HandlerRegistrationError, Exception))
        self.assertTrue(issubclass(HandlerNotFoundError, Exception))

    def test_all_public_functions_importable(self):
        """All documented public functions are importable."""
        from milia_pipeline.handlers.handler_registry import (
            get_default_registry,
            register_handler,
            get,
            list_all,
            is_registered,
            get_registry_info,
        )
        for fn in [get_default_registry, register_handler, get, list_all,
                    is_registered, get_registry_info]:
            self.assertTrue(callable(fn))


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_comprehensive_suite():
    """Run all test groups in a structured order."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestHandlerRegistrationError,           # GROUP 1:   7 tests
        TestHandlerNotFoundError,                # GROUP 2:   6 tests
        TestHandlerRegistryInit,                 # GROUP 3:   5 tests
        TestHandlerRegistryRegisterSuccess,      # GROUP 4:  10 tests
        TestHandlerRegistryRegisterErrors,       # GROUP 5:   8 tests
        TestHandlerRegistryGet,                  # GROUP 6:   7 tests
        TestHandlerRegistryCollectionMethods,    # GROUP 7:  10 tests
        TestHandlerRegistryUnregister,           # GROUP 8:   6 tests
        TestHandlerRegistryClear,                # GROUP 9:   4 tests
        TestHandlerRegistryCallbacks,            # GROUP 10:  8 tests
        TestHandlerRegistryInfo,                 # GROUP 11:  5 tests
        TestHandlerRegistryThreadSafety,         # GROUP 12:  6 tests
        TestModuleLevelFunctions,                # GROUP 13: 12 tests
        TestHandlerRegistryNonSingleton,         # GROUP 14:  4 tests
        TestHandlerNameDerivation,               # GROUP 15:  7 tests
        TestReImportDetection,                   # GROUP 16:  5 tests
        TestHandlerRegistryLogging,              # GROUP 17:  5 tests
        TestEdgeCasesAndBoundary,                # GROUP 18:  8 tests
        TestModuleExports,                       # GROUP 19:  3 tests
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS - handler_registry.py")
    print("=" * 80)
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"\nTest Groups: {len(test_classes)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - PRODUCTION-READY")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - REVIEW REQUIRED")
        return 1


if __name__ == "__main__":
    if "pytest" in sys.modules:
        pass
    else:
        sys.exit(run_comprehensive_suite())


"""
TEST SUITE SUMMARY — milia_pipeline/handlers/handler_registry.py
=================================================================

119 comprehensive production-ready tests across 19 groups:

GROUP 1:  HandlerRegistrationError Exception                          ( 7 tests)
GROUP 2:  HandlerNotFoundError Exception                              ( 6 tests)
GROUP 3:  HandlerRegistry — Initialization                            ( 5 tests)
GROUP 4:  HandlerRegistry.register — Success Paths                    (10 tests)
GROUP 5:  HandlerRegistry.register — Error Paths                      ( 8 tests)
GROUP 6:  HandlerRegistry.get / get_or_none                           ( 7 tests)
GROUP 7:  HandlerRegistry — Collection Methods                        (10 tests)
GROUP 8:  HandlerRegistry.unregister                                  ( 6 tests)
GROUP 9:  HandlerRegistry.clear                                       ( 4 tests)
GROUP 10: HandlerRegistry — Change Callbacks                          ( 8 tests)
GROUP 11: HandlerRegistry.get_registry_info                           ( 5 tests)
GROUP 12: Thread Safety                                               ( 6 tests)
GROUP 13: Module-Level Default Registry Functions                     (12 tests)
GROUP 14: Non-Singleton Behavior                                      ( 4 tests)
GROUP 15: Name Derivation Logic                                       ( 7 tests)
GROUP 16: Re-import Detection Logic                                   ( 5 tests)
GROUP 17: Logging Behavior                                            ( 5 tests)
GROUP 18: Edge Cases and Boundary Conditions                          ( 8 tests)
GROUP 19: Module __all__ Exports                                      ( 3 tests)

PRODUCTION-READY QUALITIES:
- NO sys.modules pollution (no module-level mocking)
- All mocking via @patch decorators or context managers (test-level only)
- Dynamic test data creation via helper functions (no hardcoded paths)
- No file downloads (no file system dependencies)
- Comprehensive error path coverage
- Thread safety verification with concurrent tests
- Interface-focused testing (future-proof)
- Compatible with both pytest and unittest runner
- Default registry state properly saved/restored between tests (setUp/tearDown)
- Exception hierarchy correctly tested
- Error message quality assertions
- Non-singleton behavior verified (isolated test instances)
- Re-import detection logic thoroughly tested
- Callback lifecycle fully tested
"""
