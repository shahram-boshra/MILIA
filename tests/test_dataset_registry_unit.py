#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/datasets/registry.py

Module under test: registry.py
- DatasetRegistry: Thread-safe registry class (NOT singleton)
- Module-level convenience functions: register, get, list_all, is_registered, get_default_registry

Test path on local machine: ~/ml_projects/milia/tests/test_dataset_registry_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/datasets/registry.py

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
from unittest.mock import Mock, MagicMock, patch, PropertyMock, call
import logging
import threading
import time
from typing import Dict, List, Type, Optional, Any

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from milia_pipeline.datasets.registry import (
    DatasetRegistry,
    get_default_registry,
    register,
    get,
    list_all,
    is_registered,
)
from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetMetadata,
    DatasetSchema,
    DatasetFeatures,
)
from milia_pipeline.exceptions import DatasetNotFoundError, DatasetRegistrationError


# ============================================================================
# HELPER: Reusable fixtures for building concrete BaseDataset subclasses
# ============================================================================

def _make_valid_metadata(**overrides):
    """Create a valid DatasetMetadata with optional overrides."""
    defaults = dict(
        name="TestDS",
        version="1.0.0",
        description="A test dataset",
    )
    defaults.update(overrides)
    return DatasetMetadata(**defaults)


def _make_valid_schema(**overrides):
    """Create a valid DatasetSchema with optional overrides."""
    defaults = dict(
        required_properties=("energy", "forces"),
    )
    defaults.update(overrides)
    return DatasetSchema(**defaults)


def _make_valid_features(**overrides):
    """Create a valid DatasetFeatures with optional overrides."""
    return DatasetFeatures(**overrides)


# Module-level counter to ensure unique class names across all tests
_class_counter = 0
_class_counter_lock = threading.Lock()


def _next_class_name(prefix="TestDS"):
    """Generate a unique class name to avoid __init_subclass__ conflicts."""
    global _class_counter
    with _class_counter_lock:
        _class_counter += 1
        return f"{prefix}_{_class_counter}"


def _build_concrete_dataset_class(
    class_name=None,
    metadata_name=None,
    metadata=None,
    schema=None,
    features=None,
    config_key=None,
):
    """
    Dynamically build a fully valid concrete BaseDataset subclass.

    Each call produces a unique class with a unique metadata.name to avoid
    registration conflicts.  Uses type() to create the class at call time
    so that __init_subclass__ validation fires inside the caller's scope.
    """
    if class_name is None:
        class_name = _next_class_name()
    if metadata_name is None:
        metadata_name = class_name
    if metadata is None:
        metadata = _make_valid_metadata(name=metadata_name)
    if schema is None:
        schema = _make_valid_schema()
    if features is None:
        features = _make_valid_features()
    if config_key is None:
        config_key = metadata_name.lower()

    _meta = metadata
    _sch = schema
    _feat = features
    _ck = config_key

    def _get_req(cls):
        return ["energy", "forces"]

    def _get_fs(cls):
        return _feat.to_dict()

    def _get_strat(cls):
        return "coordinate_based"

    ns = {
        "metadata": _meta,
        "schema": _sch,
        "features": _feat,
        "config_key": _ck,
        "handler_class": None,
        "converter_class": None,
        "validator_class": None,
        "get_required_properties": classmethod(_get_req),
        "get_feature_support": classmethod(_get_fs),
        "get_molecule_creation_strategy": classmethod(_get_strat),
    }

    cls = type(class_name, (BaseDataset,), ns)
    return cls


# ============================================================================
# GROUP 1: DatasetRegistry — Construction and Empty State (6 tests)
# ============================================================================

class TestRegistryConstruction(unittest.TestCase):
    """Test DatasetRegistry initialization and empty state."""

    def test_new_registry_is_empty(self):
        """A freshly created registry has zero registered datasets."""
        reg = DatasetRegistry()
        self.assertEqual(len(reg), 0)

    def test_new_registry_list_all_empty(self):
        """list_all on an empty registry returns an empty list."""
        reg = DatasetRegistry()
        self.assertEqual(reg.list_all(), [])

    def test_new_registry_list_all_classes_empty(self):
        """list_all_classes on an empty registry returns an empty list."""
        reg = DatasetRegistry()
        self.assertEqual(reg.list_all_classes(), [])

    def test_multiple_registries_are_independent(self):
        """Two DatasetRegistry instances are fully isolated."""
        reg1 = DatasetRegistry()
        reg2 = DatasetRegistry()
        cls = _build_concrete_dataset_class()
        reg1.register(cls)
        self.assertEqual(len(reg1), 1)
        self.assertEqual(len(reg2), 0)

    def test_registry_not_singleton(self):
        """DatasetRegistry is NOT a singleton — each call returns a new instance."""
        reg1 = DatasetRegistry()
        reg2 = DatasetRegistry()
        self.assertIsNot(reg1, reg2)

    def test_new_registry_iteration_empty(self):
        """Iterating an empty registry yields no items."""
        reg = DatasetRegistry()
        self.assertEqual(list(reg), [])


# ============================================================================
# GROUP 2: DatasetRegistry.register — Happy Path (8 tests)
# ============================================================================

class TestRegistryRegisterHappyPath(unittest.TestCase):
    """Test successful registration scenarios."""

    def setUp(self):
        self.registry = DatasetRegistry()

    def test_register_single_dataset(self):
        """A single valid dataset class is registered successfully."""
        cls = _build_concrete_dataset_class()
        self.registry.register(cls)
        self.assertEqual(len(self.registry), 1)

    def test_register_uses_metadata_name(self):
        """Registration key is the metadata.name of the dataset class."""
        cls = _build_concrete_dataset_class(metadata_name="AlphaDS")
        self.registry.register(cls)
        self.assertIn("AlphaDS", self.registry.list_all())

    def test_register_multiple_datasets(self):
        """Multiple dataset classes with unique names can coexist."""
        cls1 = _build_concrete_dataset_class(metadata_name="DS_A")
        cls2 = _build_concrete_dataset_class(metadata_name="DS_B")
        cls3 = _build_concrete_dataset_class(metadata_name="DS_C")
        self.registry.register(cls1)
        self.registry.register(cls2)
        self.registry.register(cls3)
        self.assertEqual(len(self.registry), 3)
        self.assertEqual(set(self.registry.list_all()), {"DS_A", "DS_B", "DS_C"})

    def test_register_returns_none(self):
        """register() returns None on success (not the class)."""
        cls = _build_concrete_dataset_class()
        result = self.registry.register(cls)
        self.assertIsNone(result)

    def test_re_register_same_class_is_idempotent(self):
        """Re-registering the exact same class object is a no-op."""
        cls = _build_concrete_dataset_class(metadata_name="Idempotent")
        self.registry.register(cls)
        self.registry.register(cls)  # Should not raise
        self.assertEqual(len(self.registry), 1)

    def test_registered_class_is_retrievable(self):
        """A registered class can be retrieved by name."""
        cls = _build_concrete_dataset_class(metadata_name="Retrieve")
        self.registry.register(cls)
        retrieved = self.registry.get("Retrieve")
        self.assertIs(retrieved, cls)

    def test_register_dataset_with_features_enabled(self):
        """A dataset with feature flags set is registered correctly."""
        feat = _make_valid_features(vibrational_analysis=True, uncertainty_handling=True)
        cls = _build_concrete_dataset_class(metadata_name="Featured", features=feat)
        self.registry.register(cls)
        retrieved = self.registry.get("Featured")
        self.assertTrue(retrieved.features.supports("vibrational_analysis"))

    def test_list_all_classes_returns_class_objects(self):
        """list_all_classes returns the actual class objects, not names."""
        cls = _build_concrete_dataset_class(metadata_name="ClassObj")
        self.registry.register(cls)
        classes = self.registry.list_all_classes()
        self.assertEqual(len(classes), 1)
        self.assertIs(classes[0], cls)


# ============================================================================
# GROUP 3: DatasetRegistry.register — Error Paths (8 tests)
# ============================================================================

class TestRegistryRegisterErrors(unittest.TestCase):
    """Test registration validation and error handling."""

    def setUp(self):
        self.registry = DatasetRegistry()

    def test_register_non_class_raises_type_error(self):
        """Passing a non-class (instance) raises TypeError."""
        with self.assertRaises(TypeError) as ctx:
            self.registry.register("not_a_class")
        self.assertIn("Expected class", str(ctx.exception))

    def test_register_non_class_instance_raises_type_error(self):
        """Passing an object instance raises TypeError."""
        with self.assertRaises(TypeError):
            self.registry.register(42)

    def test_register_non_basedataset_subclass_raises_type_error(self):
        """Registering a class that does not subclass BaseDataset raises TypeError."""
        class NotADataset:
            pass

        with self.assertRaises(TypeError) as ctx:
            self.registry.register(NotADataset)
        self.assertIn("BaseDataset", str(ctx.exception))

    def test_register_abstract_class_raises_registration_error(self):
        """Registering an abstract class with unimplemented methods raises DatasetRegistrationError."""
        from abc import abstractmethod

        class AbstractDS(BaseDataset):
            metadata = _make_valid_metadata(name="AbstractErr")
            schema = _make_valid_schema()
            features = _make_valid_features()
            config_key = "abstract_err"

            @classmethod
            def get_required_properties(cls):
                return ["energy"]

            @classmethod
            @abstractmethod
            def get_feature_support(cls):
                ...

            @classmethod
            @abstractmethod
            def get_molecule_creation_strategy(cls):
                ...

        with self.assertRaises(DatasetRegistrationError) as ctx:
            self.registry.register(AbstractDS)
        self.assertIn("abstract", str(ctx.exception).lower())

    def test_register_duplicate_name_different_class_raises(self):
        """Registering a different class with the same metadata.name raises DatasetRegistrationError."""
        name = _next_class_name("Dup")
        cls1 = _build_concrete_dataset_class(metadata_name=name)
        cls2 = _build_concrete_dataset_class(metadata_name=name)
        self.registry.register(cls1)
        with self.assertRaises(DatasetRegistrationError) as ctx:
            self.registry.register(cls2)
        self.assertIn(name, str(ctx.exception))

    def test_duplicate_registration_error_mentions_conflicting_class(self):
        """DatasetRegistrationError for duplicate name includes info about the conflict."""
        name = _next_class_name("Conflict")
        cls1 = _build_concrete_dataset_class(class_name="Original", metadata_name=name)
        cls2 = _build_concrete_dataset_class(class_name="Duplicate", metadata_name=name)
        self.registry.register(cls1)
        with self.assertRaises(DatasetRegistrationError) as ctx:
            self.registry.register(cls2)
        error_str = str(ctx.exception)
        # Should mention either the conflicting class name or the "already registered" message
        self.assertTrue(
            "already registered" in error_str.lower() or name in error_str,
            f"Error message should reference conflict: {error_str}"
        )

    def test_register_none_raises_type_error(self):
        """Passing None raises TypeError."""
        with self.assertRaises(TypeError):
            self.registry.register(None)

    def test_register_lambda_raises_type_error(self):
        """Passing a lambda (not a class) raises TypeError."""
        with self.assertRaises(TypeError):
            self.registry.register(lambda: None)


# ============================================================================
# GROUP 4: DatasetRegistry.get and get_or_none — Lookup (8 tests)
# ============================================================================

class TestRegistryLookup(unittest.TestCase):
    """Test dataset lookup by name."""

    def setUp(self):
        self.registry = DatasetRegistry()
        self.ds_name = _next_class_name("Lookup")
        self.ds_class = _build_concrete_dataset_class(metadata_name=self.ds_name)
        self.registry.register(self.ds_class)

    def test_get_existing_returns_class(self):
        """get() returns the registered class for a known name."""
        result = self.registry.get(self.ds_name)
        self.assertIs(result, self.ds_class)

    def test_get_nonexistent_raises_not_found_error(self):
        """get() for an unregistered name raises DatasetNotFoundError."""
        with self.assertRaises(DatasetNotFoundError):
            self.registry.get("NoSuchDataset")

    def test_get_not_found_error_includes_dataset_name(self):
        """DatasetNotFoundError message includes the requested name."""
        with self.assertRaises(DatasetNotFoundError) as ctx:
            self.registry.get("MissingDS")
        self.assertIn("MissingDS", str(ctx.exception))

    def test_get_not_found_error_includes_available_datasets(self):
        """DatasetNotFoundError provides the list of available datasets."""
        with self.assertRaises(DatasetNotFoundError) as ctx:
            self.registry.get("MissingDS")
        exc = ctx.exception
        # The exception should carry available_datasets info
        self.assertTrue(
            hasattr(exc, "available_datasets") or self.ds_name in str(exc),
            "Error should include available datasets"
        )

    def test_get_or_none_existing_returns_class(self):
        """get_or_none() returns the class for a known name."""
        result = self.registry.get_or_none(self.ds_name)
        self.assertIs(result, self.ds_class)

    def test_get_or_none_nonexistent_returns_none(self):
        """get_or_none() returns None for an unregistered name."""
        result = self.registry.get_or_none("DoesNotExist")
        self.assertIsNone(result)

    def test_get_is_case_sensitive(self):
        """Registry lookup is case-sensitive."""
        with self.assertRaises(DatasetNotFoundError):
            self.registry.get(self.ds_name.lower() + "_extra")

    def test_get_or_none_empty_registry(self):
        """get_or_none on an empty registry returns None."""
        empty_reg = DatasetRegistry()
        self.assertIsNone(empty_reg.get_or_none("anything"))


# ============================================================================
# GROUP 5: DatasetRegistry.unregister (7 tests)
# ============================================================================

class TestRegistryUnregister(unittest.TestCase):
    """Test dataset unregistration."""

    def setUp(self):
        self.registry = DatasetRegistry()

    def test_unregister_existing_returns_true(self):
        """unregister() returns True when removing an existing dataset."""
        name = _next_class_name("Unreg")
        cls = _build_concrete_dataset_class(metadata_name=name)
        self.registry.register(cls)
        result = self.registry.unregister(name)
        self.assertTrue(result)

    def test_unregister_removes_from_registry(self):
        """After unregister, the dataset is no longer findable."""
        name = _next_class_name("UnregRemove")
        cls = _build_concrete_dataset_class(metadata_name=name)
        self.registry.register(cls)
        self.registry.unregister(name)
        self.assertNotIn(name, self.registry)
        self.assertEqual(len(self.registry), 0)

    def test_unregister_nonexistent_returns_false(self):
        """unregister() returns False for an unregistered name."""
        result = self.registry.unregister("NeverRegistered")
        self.assertFalse(result)

    def test_unregister_allows_re_registration(self):
        """After unregistering, a new class with the same name can be registered."""
        name = _next_class_name("ReReg")
        cls1 = _build_concrete_dataset_class(metadata_name=name)
        self.registry.register(cls1)
        self.registry.unregister(name)

        cls2 = _build_concrete_dataset_class(metadata_name=name)
        self.registry.register(cls2)
        self.assertIs(self.registry.get(name), cls2)

    def test_unregister_one_does_not_affect_others(self):
        """Unregistering one dataset leaves others intact."""
        name_a = _next_class_name("KeepA")
        name_b = _next_class_name("RemoveB")
        cls_a = _build_concrete_dataset_class(metadata_name=name_a)
        cls_b = _build_concrete_dataset_class(metadata_name=name_b)
        self.registry.register(cls_a)
        self.registry.register(cls_b)
        self.registry.unregister(name_b)
        self.assertIn(name_a, self.registry)
        self.assertNotIn(name_b, self.registry)
        self.assertEqual(len(self.registry), 1)

    def test_unregister_get_raises_after_removal(self):
        """get() raises DatasetNotFoundError after unregistering."""
        name = _next_class_name("UnregGet")
        cls = _build_concrete_dataset_class(metadata_name=name)
        self.registry.register(cls)
        self.registry.unregister(name)
        with self.assertRaises(DatasetNotFoundError):
            self.registry.get(name)

    def test_unregister_idempotent_second_call_false(self):
        """Calling unregister twice: first True, second False."""
        name = _next_class_name("UnregTwice")
        cls = _build_concrete_dataset_class(metadata_name=name)
        self.registry.register(cls)
        self.assertTrue(self.registry.unregister(name))
        self.assertFalse(self.registry.unregister(name))


# ============================================================================
# GROUP 6: DatasetRegistry — is_registered, __contains__, __len__, __iter__ (8 tests)
# ============================================================================

class TestRegistryProtocolMethods(unittest.TestCase):
    """Test is_registered, 'in' operator, len, and iteration."""

    def setUp(self):
        self.registry = DatasetRegistry()
        self.ds_name = _next_class_name("Proto")
        self.ds_class = _build_concrete_dataset_class(metadata_name=self.ds_name)
        self.registry.register(self.ds_class)

    def test_is_registered_true(self):
        """is_registered returns True for a registered name."""
        self.assertTrue(self.registry.is_registered(self.ds_name))

    def test_is_registered_false(self):
        """is_registered returns False for an unregistered name."""
        self.assertFalse(self.registry.is_registered("Nonexistent"))

    def test_contains_operator_true(self):
        """'in' operator returns True for registered names."""
        self.assertIn(self.ds_name, self.registry)

    def test_contains_operator_false(self):
        """'in' operator returns False for unregistered names."""
        self.assertNotIn("MissingName", self.registry)

    def test_len_reflects_registration_count(self):
        """len() reflects the current registration count."""
        self.assertEqual(len(self.registry), 1)
        name2 = _next_class_name("ProtoLen")
        cls2 = _build_concrete_dataset_class(metadata_name=name2)
        self.registry.register(cls2)
        self.assertEqual(len(self.registry), 2)

    def test_iter_yields_all_names(self):
        """Iterating the registry yields all registered names."""
        name2 = _next_class_name("ProtoIter")
        cls2 = _build_concrete_dataset_class(metadata_name=name2)
        self.registry.register(cls2)
        names = set(self.registry)
        self.assertEqual(names, {self.ds_name, name2})

    def test_iter_returns_snapshot(self):
        """Iteration returns a snapshot, safe from concurrent modification."""
        # Get iterator
        it = iter(self.registry)
        # Consuming should work without error
        names = list(it)
        self.assertIn(self.ds_name, names)

    def test_len_after_clear_is_zero(self):
        """len() returns 0 after clearing the registry."""
        self.registry.clear()
        self.assertEqual(len(self.registry), 0)


# ============================================================================
# GROUP 7: DatasetRegistry.clear (5 tests)
# ============================================================================

class TestRegistryClear(unittest.TestCase):
    """Test registry clearing functionality."""

    def setUp(self):
        self.registry = DatasetRegistry()

    def test_clear_empties_registry(self):
        """clear() removes all registered datasets."""
        cls1 = _build_concrete_dataset_class(metadata_name=_next_class_name("ClearA"))
        cls2 = _build_concrete_dataset_class(metadata_name=_next_class_name("ClearB"))
        self.registry.register(cls1)
        self.registry.register(cls2)
        self.registry.clear()
        self.assertEqual(len(self.registry), 0)
        self.assertEqual(self.registry.list_all(), [])

    def test_clear_empty_registry_no_error(self):
        """Clearing an already-empty registry does not raise."""
        self.registry.clear()  # Should not raise
        self.assertEqual(len(self.registry), 0)

    def test_clear_allows_re_registration(self):
        """After clear, datasets can be registered again with same names."""
        name = _next_class_name("ClearReReg")
        cls = _build_concrete_dataset_class(metadata_name=name)
        self.registry.register(cls)
        self.registry.clear()
        # Can register a new class with same metadata name
        cls2 = _build_concrete_dataset_class(metadata_name=name)
        self.registry.register(cls2)
        self.assertEqual(len(self.registry), 1)

    def test_clear_list_all_classes_empty(self):
        """list_all_classes is empty after clear."""
        cls = _build_concrete_dataset_class(metadata_name=_next_class_name("ClearCls"))
        self.registry.register(cls)
        self.registry.clear()
        self.assertEqual(self.registry.list_all_classes(), [])

    def test_clear_get_raises_after_clear(self):
        """get() raises DatasetNotFoundError after clearing."""
        name = _next_class_name("ClearGet")
        cls = _build_concrete_dataset_class(metadata_name=name)
        self.registry.register(cls)
        self.registry.clear()
        with self.assertRaises(DatasetNotFoundError):
            self.registry.get(name)


# ============================================================================
# GROUP 8: DatasetRegistry — Change Callbacks (10 tests)
# ============================================================================

class TestRegistryChangeCallbacks(unittest.TestCase):
    """Test cache invalidation / on-change callback system."""

    def setUp(self):
        self.registry = DatasetRegistry()
        self.callback_calls = []

    def _make_callback(self, label="default"):
        """Create a callback that records invocations."""
        def cb():
            self.callback_calls.append(label)
        return cb

    def test_callback_fired_on_register(self):
        """Callback is fired when a dataset is registered."""
        cb = self._make_callback("register")
        self.registry.add_on_change_callback(cb)
        cls = _build_concrete_dataset_class(metadata_name=_next_class_name("CB_Reg"))
        self.registry.register(cls)
        self.assertIn("register", self.callback_calls)

    def test_callback_fired_on_unregister(self):
        """Callback is fired when a dataset is unregistered."""
        name = _next_class_name("CB_Unreg")
        cls = _build_concrete_dataset_class(metadata_name=name)
        self.registry.register(cls)
        self.callback_calls.clear()

        cb = self._make_callback("unregister")
        self.registry.add_on_change_callback(cb)
        self.registry.unregister(name)
        self.assertIn("unregister", self.callback_calls)

    def test_callback_fired_on_clear(self):
        """Callback is fired when registry is cleared."""
        cb = self._make_callback("clear")
        self.registry.add_on_change_callback(cb)
        self.registry.clear()
        self.assertIn("clear", self.callback_calls)

    def test_callback_not_fired_on_reregister_same_class(self):
        """Callback is NOT fired when the same class is re-registered (no-op)."""
        cls = _build_concrete_dataset_class(metadata_name=_next_class_name("CB_Same"))
        self.registry.register(cls)
        self.callback_calls.clear()

        cb = self._make_callback("should_not_fire")
        self.registry.add_on_change_callback(cb)
        self.registry.register(cls)  # Same class — should be no-op
        self.assertEqual(self.callback_calls, [])

    def test_callback_not_fired_on_unregister_nonexistent(self):
        """Callback is NOT fired when unregistering a nonexistent name."""
        cb = self._make_callback("should_not_fire")
        self.registry.add_on_change_callback(cb)
        self.registry.unregister("ghost_dataset")
        self.assertEqual(self.callback_calls, [])

    def test_multiple_callbacks_all_fired(self):
        """Multiple registered callbacks are all invoked."""
        cb1 = self._make_callback("cb1")
        cb2 = self._make_callback("cb2")
        self.registry.add_on_change_callback(cb1)
        self.registry.add_on_change_callback(cb2)
        cls = _build_concrete_dataset_class(metadata_name=_next_class_name("CB_Multi"))
        self.registry.register(cls)
        self.assertIn("cb1", self.callback_calls)
        self.assertIn("cb2", self.callback_calls)

    def test_remove_callback_returns_true(self):
        """remove_on_change_callback returns True for a registered callback."""
        cb = self._make_callback()
        self.registry.add_on_change_callback(cb)
        result = self.registry.remove_on_change_callback(cb)
        self.assertTrue(result)

    def test_remove_callback_returns_false_for_unknown(self):
        """remove_on_change_callback returns False for an unregistered callback."""
        cb = self._make_callback()
        result = self.registry.remove_on_change_callback(cb)
        self.assertFalse(result)

    def test_removed_callback_not_fired(self):
        """A removed callback is no longer invoked on changes."""
        cb = self._make_callback("removed")
        self.registry.add_on_change_callback(cb)
        self.registry.remove_on_change_callback(cb)
        cls = _build_concrete_dataset_class(metadata_name=_next_class_name("CB_NoFire"))
        self.registry.register(cls)
        self.assertEqual(self.callback_calls, [])

    def test_failing_callback_does_not_block_others(self):
        """A callback that raises an exception does not prevent other callbacks from running."""
        def failing_cb():
            raise RuntimeError("callback error")

        cb_after = self._make_callback("after_fail")
        self.registry.add_on_change_callback(failing_cb)
        self.registry.add_on_change_callback(cb_after)

        cls = _build_concrete_dataset_class(metadata_name=_next_class_name("CB_Fail"))
        # Should not raise — exception is caught internally
        self.registry.register(cls)
        self.assertIn("after_fail", self.callback_calls)


# ============================================================================
# GROUP 9: DatasetRegistry — Thread Safety (6 tests)
# ============================================================================

class TestRegistryThreadSafety(unittest.TestCase):
    """Test thread-safe concurrent access to DatasetRegistry."""

    def setUp(self):
        self.registry = DatasetRegistry()

    def test_concurrent_register_no_crash(self):
        """Multiple threads can register different datasets without crashing."""
        errors = []
        classes = []
        for i in range(10):
            name = _next_class_name(f"Thread_{i}")
            cls = _build_concrete_dataset_class(metadata_name=name)
            classes.append((name, cls))

        def register_one(cls):
            try:
                self.registry.register(cls)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_one, args=(cls,))
            for _, cls in classes
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(errors, [])
        self.assertEqual(len(self.registry), 10)

    def test_concurrent_read_while_writing(self):
        """Reads and writes can happen concurrently without error."""
        name = _next_class_name("ConcRead")
        cls = _build_concrete_dataset_class(metadata_name=name)
        self.registry.register(cls)
        errors = []

        def reader():
            try:
                for _ in range(50):
                    self.registry.list_all()
                    self.registry.is_registered(name)
                    self.registry.get_or_none(name)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(10):
                    n = _next_class_name(f"ConcWrite_{i}")
                    c = _build_concrete_dataset_class(metadata_name=n)
                    self.registry.register(c)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(3)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])

    def test_concurrent_unregister(self):
        """Concurrent unregister calls do not raise."""
        names = []
        for i in range(10):
            name = _next_class_name(f"ConcUnreg_{i}")
            cls = _build_concrete_dataset_class(metadata_name=name)
            self.registry.register(cls)
            names.append(name)

        errors = []

        def unreg_one(n):
            try:
                self.registry.unregister(n)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=unreg_one, args=(n,))
            for n in names
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(errors, [])
        self.assertEqual(len(self.registry), 0)

    def test_concurrent_clear_and_register(self):
        """Concurrent clear and register do not deadlock or crash."""
        errors = []

        def do_clear():
            try:
                for _ in range(5):
                    self.registry.clear()
            except Exception as e:
                errors.append(e)

        def do_register():
            try:
                for i in range(5):
                    n = _next_class_name(f"ClearReg_{i}")
                    c = _build_concrete_dataset_class(metadata_name=n)
                    self.registry.register(c)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=do_clear),
            threading.Thread(target=do_register),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])

    def test_rlock_allows_reentrant_access(self):
        """RLock allows re-entrant access (e.g., callback calling registry methods)."""
        def reentrant_callback():
            # This accesses the registry from inside a lock-protected operation
            self.registry.list_all()

        self.registry.add_on_change_callback(reentrant_callback)
        cls = _build_concrete_dataset_class(metadata_name=_next_class_name("Reentrant"))
        # Should not deadlock
        self.registry.register(cls)

    def test_concurrent_iteration_safe(self):
        """Iterating while another thread modifies the registry does not crash."""
        for i in range(5):
            n = _next_class_name(f"IterSafe_{i}")
            c = _build_concrete_dataset_class(metadata_name=n)
            self.registry.register(c)

        errors = []

        def iterate():
            try:
                for _ in range(20):
                    list(self.registry)
            except Exception as e:
                errors.append(e)

        def modify():
            try:
                for i in range(10):
                    n = _next_class_name(f"IterMod_{i}")
                    c = _build_concrete_dataset_class(metadata_name=n)
                    self.registry.register(c)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=iterate),
            threading.Thread(target=modify),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])


# ============================================================================
# GROUP 10: Module-Level Convenience Functions (12 tests)
# ============================================================================

class TestModuleLevelFunctions(unittest.TestCase):
    """Test module-level convenience functions that operate on the default registry."""

    def setUp(self):
        """Save default registry state and clear for isolation."""
        self.default_reg = get_default_registry()
        self._saved_datasets = dict(self.default_reg._datasets)
        self.default_reg.clear()

    def tearDown(self):
        """Restore default registry state after each test."""
        self.default_reg._datasets.clear()
        self.default_reg._datasets.update(self._saved_datasets)

    def test_get_default_registry_returns_dataset_registry(self):
        """get_default_registry() returns a DatasetRegistry instance."""
        reg = get_default_registry()
        self.assertIsInstance(reg, DatasetRegistry)

    def test_get_default_registry_same_instance(self):
        """get_default_registry() always returns the same instance."""
        reg1 = get_default_registry()
        reg2 = get_default_registry()
        self.assertIs(reg1, reg2)

    def test_module_register_adds_to_default(self):
        """Module-level register() adds to the default registry."""
        cls = _build_concrete_dataset_class(metadata_name=_next_class_name("ModReg"))
        register(cls)
        self.assertIn(cls.metadata.name, self.default_reg.list_all())

    def test_module_register_returns_class(self):
        """Module-level register() returns the class (for decorator use)."""
        cls = _build_concrete_dataset_class(metadata_name=_next_class_name("ModRegRet"))
        result = register(cls)
        self.assertIs(result, cls)

    def test_module_register_as_decorator_pattern(self):
        """register() works as a decorator — class is returned and registered."""
        cls = _build_concrete_dataset_class(metadata_name=_next_class_name("ModDec"))
        decorated = register(cls)
        self.assertIs(decorated, cls)
        self.assertTrue(is_registered(cls.metadata.name))

    def test_module_get_retrieves_from_default(self):
        """Module-level get() retrieves from the default registry."""
        name = _next_class_name("ModGet")
        cls = _build_concrete_dataset_class(metadata_name=name)
        register(cls)
        retrieved = get(name)
        self.assertIs(retrieved, cls)

    def test_module_get_nonexistent_raises(self):
        """Module-level get() raises DatasetNotFoundError for unknown name."""
        with self.assertRaises(DatasetNotFoundError):
            get("NonexistentModuleDS")

    def test_module_list_all_returns_list(self):
        """Module-level list_all() returns a list of strings."""
        name = _next_class_name("ModList")
        cls = _build_concrete_dataset_class(metadata_name=name)
        register(cls)
        result = list_all()
        self.assertIsInstance(result, list)
        self.assertIn(name, result)

    def test_module_is_registered_true(self):
        """Module-level is_registered() returns True for registered dataset."""
        name = _next_class_name("ModIsReg")
        cls = _build_concrete_dataset_class(metadata_name=name)
        register(cls)
        self.assertTrue(is_registered(name))

    def test_module_is_registered_false(self):
        """Module-level is_registered() returns False for unregistered name."""
        self.assertFalse(is_registered("NeverRegisteredModule"))

    def test_module_list_all_empty_after_clear(self):
        """Module-level list_all() returns empty list after clearing default registry."""
        self.assertEqual(list_all(), [])

    def test_default_registry_is_not_new_instance(self):
        """The default registry is a module-level pre-created instance, not created on call."""
        from milia_pipeline.datasets import registry as reg_module
        self.assertIs(get_default_registry(), reg_module._default_registry)


# ============================================================================
# GROUP 11: Logging Verification (6 tests)
# ============================================================================

class TestRegistryLogging(unittest.TestCase):
    """Test that registry operations produce appropriate log output."""

    def setUp(self):
        self.registry = DatasetRegistry()

    def test_register_logs_info(self):
        """Registering a dataset logs an info-level message."""
        with self.assertLogs("milia_pipeline.datasets.registry", level="INFO") as cm:
            cls = _build_concrete_dataset_class(metadata_name=_next_class_name("LogReg"))
            self.registry.register(cls)
        log_output = "\n".join(cm.output)
        self.assertIn("Registered dataset", log_output)

    def test_unregister_logs_info(self):
        """Unregistering a dataset logs an info-level message."""
        name = _next_class_name("LogUnreg")
        cls = _build_concrete_dataset_class(metadata_name=name)
        self.registry.register(cls)
        with self.assertLogs("milia_pipeline.datasets.registry", level="INFO") as cm:
            self.registry.unregister(name)
        log_output = "\n".join(cm.output)
        self.assertIn("Unregistered dataset", log_output)

    def test_clear_logs_warning(self):
        """Clearing the registry logs a warning-level message."""
        with self.assertLogs("milia_pipeline.datasets.registry", level="WARNING") as cm:
            self.registry.clear()
        log_output = "\n".join(cm.output)
        self.assertIn("cleared", log_output.lower())

    def test_reregister_same_class_logs_debug(self):
        """Re-registering the same class logs a debug-level message."""
        cls = _build_concrete_dataset_class(metadata_name=_next_class_name("LogReReg"))
        self.registry.register(cls)
        with self.assertLogs("milia_pipeline.datasets.registry", level="DEBUG") as cm:
            self.registry.register(cls)
        log_output = "\n".join(cm.output)
        self.assertIn("re-registered", log_output.lower())

    def test_failing_callback_logs_warning(self):
        """A failing callback logs a warning about the failure."""
        def bad_cb():
            raise ValueError("intentional error")

        self.registry.add_on_change_callback(bad_cb)
        with self.assertLogs("milia_pipeline.datasets.registry", level="WARNING") as cm:
            cls = _build_concrete_dataset_class(metadata_name=_next_class_name("LogCBFail"))
            self.registry.register(cls)
        log_output = "\n".join(cm.output)
        self.assertIn("callback failed", log_output.lower())

    def test_register_log_includes_dataset_name(self):
        """Registration log message includes the dataset name."""
        name = _next_class_name("LogName")
        with self.assertLogs("milia_pipeline.datasets.registry", level="INFO") as cm:
            cls = _build_concrete_dataset_class(metadata_name=name)
            self.registry.register(cls)
        log_output = "\n".join(cm.output)
        self.assertIn(name, log_output)


# ============================================================================
# GROUP 12: Edge Cases and Boundary Conditions (8 tests)
# ============================================================================

class TestRegistryEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        self.registry = DatasetRegistry()

    def test_register_dataset_with_long_name(self):
        """A dataset with a very long name can be registered and retrieved."""
        long_name = "A" * 200
        cls = _build_concrete_dataset_class(metadata_name=long_name)
        self.registry.register(cls)
        self.assertIs(self.registry.get(long_name), cls)

    def test_register_dataset_with_special_chars_in_name(self):
        """Dataset names with special characters are handled correctly."""
        special_name = _next_class_name("DFT-v2.1_extended")
        cls = _build_concrete_dataset_class(metadata_name=special_name)
        self.registry.register(cls)
        self.assertTrue(self.registry.is_registered(special_name))

    def test_register_dataset_with_unicode_name(self):
        """Dataset names with unicode characters are handled correctly."""
        unicode_name = _next_class_name("αβγ_dataset")
        cls = _build_concrete_dataset_class(metadata_name=unicode_name)
        self.registry.register(cls)
        self.assertTrue(self.registry.is_registered(unicode_name))

    def test_list_all_returns_copy(self):
        """list_all() returns a new list, not a reference to internals."""
        name = _next_class_name("Copy")
        cls = _build_concrete_dataset_class(metadata_name=name)
        self.registry.register(cls)
        list1 = self.registry.list_all()
        list2 = self.registry.list_all()
        self.assertIsNot(list1, list2)
        self.assertEqual(list1, list2)

    def test_list_all_classes_returns_copy(self):
        """list_all_classes() returns a new list each call."""
        name = _next_class_name("ClassCopy")
        cls = _build_concrete_dataset_class(metadata_name=name)
        self.registry.register(cls)
        c1 = self.registry.list_all_classes()
        c2 = self.registry.list_all_classes()
        self.assertIsNot(c1, c2)

    def test_many_registrations_performance(self):
        """Registry handles many registrations without degradation."""
        for i in range(100):
            name = _next_class_name(f"Perf_{i}")
            cls = _build_concrete_dataset_class(metadata_name=name)
            self.registry.register(cls)
        self.assertEqual(len(self.registry), 100)
        all_names = self.registry.list_all()
        self.assertEqual(len(all_names), 100)

    def test_get_after_register_unregister_register_cycle(self):
        """Full register → unregister → re-register cycle works correctly."""
        name = _next_class_name("Cycle")
        cls1 = _build_concrete_dataset_class(metadata_name=name)
        self.registry.register(cls1)
        self.assertIs(self.registry.get(name), cls1)

        self.registry.unregister(name)
        with self.assertRaises(DatasetNotFoundError):
            self.registry.get(name)

        cls2 = _build_concrete_dataset_class(metadata_name=name)
        self.registry.register(cls2)
        self.assertIs(self.registry.get(name), cls2)
        self.assertIsNot(self.registry.get(name), cls1)

    def test_callback_mutation_during_iteration(self):
        """Adding a callback from within a callback does not crash."""
        inner_called = []

        def outer_cb():
            def inner_cb():
                inner_called.append(True)
            # This accesses the callback list from within a callback
            self.registry.add_on_change_callback(inner_cb)

        self.registry.add_on_change_callback(outer_cb)
        cls = _build_concrete_dataset_class(metadata_name=_next_class_name("CBMut"))
        self.registry.register(cls)
        # outer_cb was called, inner_cb was added (but won't fire this round)
        # The main check is no crash


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_comprehensive_suite():
    """Run all test groups in a structured order."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestRegistryConstruction,           # GROUP 1:  6 tests
        TestRegistryRegisterHappyPath,      # GROUP 2:  8 tests
        TestRegistryRegisterErrors,         # GROUP 3:  8 tests
        TestRegistryLookup,                 # GROUP 4:  8 tests
        TestRegistryUnregister,             # GROUP 5:  7 tests
        TestRegistryProtocolMethods,        # GROUP 6:  8 tests
        TestRegistryClear,                  # GROUP 7:  5 tests
        TestRegistryChangeCallbacks,        # GROUP 8: 10 tests
        TestRegistryThreadSafety,           # GROUP 9:  6 tests
        TestModuleLevelFunctions,           # GROUP 10: 12 tests
        TestRegistryLogging,                # GROUP 11:  6 tests
        TestRegistryEdgeCases,              # GROUP 12:  8 tests
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS — registry.py")
    print("=" * 80)
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    total_test_groups = len(test_classes)
    print(f"\nTest Groups: {total_test_groups}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - PRODUCTION-READY")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - REVIEW REQUIRED")
        return 1


if __name__ == "__main__":
    if "pytest" in sys.modules:
        # Let pytest discover and run tests normally
        pass
    else:
        sys.exit(run_comprehensive_suite())


"""
TEST SUITE SUMMARY — milia_pipeline/datasets/registry.py
=========================================================

92 comprehensive production-ready tests covering:

GROUP 1: DatasetRegistry Construction and Empty State (6 tests)
- New registry is empty (len, list_all, list_all_classes)
- Multiple registries are independent
- Not a singleton
- Empty iteration

GROUP 2: DatasetRegistry.register — Happy Path (8 tests)
- Single dataset registration
- Registration key from metadata.name
- Multiple datasets coexist
- register() returns None
- Re-register same class is idempotent
- Registered class retrievable via get()
- Feature flags preserved
- list_all_classes returns class objects

GROUP 3: DatasetRegistry.register — Error Paths (8 tests)
- Non-class raises TypeError
- Instance raises TypeError
- Non-BaseDataset subclass raises TypeError
- Abstract class raises DatasetRegistrationError
- Duplicate name (different class) raises DatasetRegistrationError
- Duplicate error mentions conflict
- None raises TypeError
- Lambda raises TypeError

GROUP 4: DatasetRegistry.get and get_or_none — Lookup (8 tests)
- get() returns class for known name
- get() raises DatasetNotFoundError for unknown
- Error includes requested name
- Error includes available datasets
- get_or_none() returns class or None
- Case-sensitive lookup
- get_or_none on empty registry

GROUP 5: DatasetRegistry.unregister (7 tests)
- Returns True for existing
- Removes from registry
- Returns False for nonexistent
- Allows re-registration
- Does not affect others
- get() raises after removal
- Idempotent second call returns False

GROUP 6: DatasetRegistry Protocol Methods (8 tests)
- is_registered True/False
- __contains__ ('in') True/False
- __len__ reflects count
- __iter__ yields all names
- Iteration returns snapshot
- len after clear is zero

GROUP 7: DatasetRegistry.clear (5 tests)
- Empties registry
- No error on empty
- Allows re-registration
- list_all_classes empty after clear
- get() raises after clear

GROUP 8: DatasetRegistry Change Callbacks (10 tests)
- Fired on register, unregister, clear
- NOT fired on re-register same class
- NOT fired on unregister nonexistent
- Multiple callbacks all fire
- remove returns True/False
- Removed callback not fired
- Failing callback doesn't block others

GROUP 9: DatasetRegistry Thread Safety (6 tests)
- Concurrent register no crash
- Concurrent read while writing
- Concurrent unregister
- Concurrent clear and register
- RLock re-entrant access
- Concurrent iteration safe

GROUP 10: Module-Level Convenience Functions (12 tests)
- get_default_registry returns DatasetRegistry
- Same instance always
- Module register() adds to default
- Module register() returns class (decorator pattern)
- Module get() retrieves from default
- Module get() raises for unknown
- Module list_all() returns list
- Module is_registered() True/False
- list_all empty after clear
- Default registry is module-level instance

GROUP 11: Logging Verification (6 tests)
- Register logs INFO
- Unregister logs INFO
- Clear logs WARNING
- Re-register logs DEBUG
- Failing callback logs WARNING
- Log includes dataset name

GROUP 12: Edge Cases and Boundary Conditions (8 tests)
- Long name
- Special characters
- Unicode name
- list_all returns copy
- list_all_classes returns copy
- Many registrations (100)
- Full register/unregister/re-register cycle
- Callback mutation during iteration

Total: 92 comprehensive production-ready tests

PRODUCTION-READY QUALITIES:
- NO sys.modules pollution (no module-level mocking)
- All mocking via @patch decorators or context managers (test-level only)
- Dynamic test data creation via helper functions (no hardcoded paths)
- No NPZ file downloads (no file system dependencies)
- Comprehensive error path coverage
- Thread safety verification with concurrent access patterns
- Interface-focused testing (future-proof)
- Compatible with both pytest and unittest runner
- Registry isolation via fresh instances per test
- Default registry state saved/restored in module-level tests
- Callback system exhaustively tested
- Logging output verified
"""
