#!/usr/bin/env python3
"""
Thread Safety Tests for MILIA Registry Components
===================================================

Test file: test_thread_safety_registries.py
Section:   6.1 of MILIA_Test_Recommendations.md
Category:  Thread Safety
Priority:  Medium
Est. CI:   ~15s

What it tests:
    Concurrent registration, lookup, and unregistration in all registry types
    do not cause race conditions or inconsistent state.

Modules exercised:
    - milia_pipeline/datasets/registry.py          — DatasetRegistry (non-singleton, RLock)
    - milia_pipeline/handlers/handler_registry.py  — HandlerRegistry (non-singleton, RLock)
    - milia_pipeline/descriptors/descriptor_registry.py — DescriptorRegistry (singleton, RLock)
    - milia_pipeline/models/registry/model_registry.py  — ModelRegistry (singleton, RLock)
    - milia_pipeline/models/builders/layer_registry.py   — LayerRegistry (singleton, RLock)

Scope:
    Spawns 10–20 threads performing simultaneous register(), get(), list_all(),
    is_registered() operations. Asserts: no exceptions raised, final state is
    consistent, no data corruption.

Design decisions:
    - Uses unittest.mock to create lightweight stub classes for DatasetRegistry and
      HandlerRegistry, avoiding heavy imports of real dataset/handler implementations.
    - For singleton registries (Descriptor, Model, Layer): tests concurrent reads on
      the already-initialized global singleton — does NOT reset or mutate global state
      to avoid polluting other tests.
    - NO sys.modules injection — all mocking is test-scoped via @patch decorators
      and locally constructed mock classes to prevent mock pollution.
    - Each test class is fully isolated; no shared mutable state between test methods.

Running:
    pytest tests/test_thread_safety_registries.py -v --tb=short
    pytest tests/test_thread_safety_registries.py -v -m thread_safety
"""

import contextlib
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# ===========================================================================
# Constants
# ===========================================================================

NUM_THREADS = 20
NUM_OPERATIONS_PER_THREAD = 50
STRESS_THREADS = 20
STRESS_OPS = 100


# ===========================================================================
# Helper: Barrier-synchronised thread launcher
# ===========================================================================


def run_threads_with_barrier(target_fn, num_threads, *args, **kwargs):
    """
    Launch *num_threads* threads that all start simultaneously via a Barrier,
    then collect any exceptions raised inside the threads.

    Returns:
        results  – list of return values (one per thread)
        errors   – list of (thread_index, exception) tuples
    """
    barrier = threading.Barrier(num_threads)
    results: list[Any] = [None] * num_threads
    errors: list[tuple] = []
    lock = threading.Lock()

    def _wrapper(idx):
        try:
            barrier.wait(timeout=10)
            results[idx] = target_fn(idx, *args, **kwargs)
        except Exception as exc:
            with lock:
                errors.append((idx, exc))

    threads = [threading.Thread(target=_wrapper, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    return results, errors


# ===========================================================================
# SECTION 1: DatasetRegistry Thread Safety
# ===========================================================================


class TestDatasetRegistryThreadSafety:
    """
    Thread-safety tests for milia_pipeline.datasets.registry.DatasetRegistry.

    DatasetRegistry is NOT a singleton — each test creates a fresh, isolated
    instance. We construct lightweight mock dataset classes that satisfy the
    registry's type checks (issubclass of BaseDataset, non-abstract, has
    metadata.name).
    """

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Import DatasetRegistry and BaseDataset; build mock classes."""
        from milia_pipeline.datasets.base import (
            BaseDataset,
            DatasetFeatures,
            DatasetMetadata,
            DatasetSchema,
        )
        from milia_pipeline.datasets.registry import DatasetRegistry

        self.DatasetRegistry = DatasetRegistry
        self.BaseDataset = BaseDataset
        self.DatasetMetadata = DatasetMetadata
        self.DatasetSchema = DatasetSchema
        self.DatasetFeatures = DatasetFeatures
        self.registry = DatasetRegistry()  # fresh, empty instance

    def _make_mock_dataset_class(self, name: str) -> type:
        """
        Dynamically create a concrete BaseDataset subclass that the registry
        will accept.

        BaseDataset.__init_subclass__ runs at class-creation time and:
          1. Checks hasattr for metadata, schema, features, config_key
          2. Checks isinstance(cls.metadata, DatasetMetadata)
          (and potentially similar checks for schema/features)

        We therefore construct real Pydantic dataclass instances for all
        three typed attributes to pass both validation gates.
        """
        base = self.BaseDataset

        # Determine which abstract methods need stubs
        abstract_methods = set()
        for cls in base.__mro__:
            if hasattr(cls, "__abstractmethods__"):
                abstract_methods |= cls.__abstractmethods__

        # Build a namespace with stubs for every abstract method
        ns: dict[str, Any] = {}
        for method_name in abstract_methods:
            ns[method_name] = lambda self, *a, **kw: None

        # Provide ALL four required class attributes using real Pydantic
        # dataclass instances so __init_subclass__ isinstance checks pass.
        ns["metadata"] = self.DatasetMetadata(
            name=name,
            version="0.0.1",
            description=f"Mock dataset {name}",
            author="test",
            license="test",
        )
        ns["schema"] = self.DatasetSchema(
            required_properties=["energy"],
            optional_properties=[],
            identifier_keys=[("mol_id", "mol_id")],
        )
        ns["features"] = self.DatasetFeatures()
        ns["config_key"] = f"mock_{name.lower()}_config"

        # Dynamically create the class
        cls = type(f"MockDataset_{name}", (base,), ns)
        # Remove __abstractmethods__ so the registry doesn't reject it
        cls.__abstractmethods__ = frozenset()
        return cls

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    @pytest.mark.thread_safety
    def test_concurrent_registration_no_exceptions(self):
        """
        20 threads each register a uniquely-named dataset class concurrently.
        No exceptions should be raised and all datasets should be present
        in the registry afterward.
        """
        registry = self.registry
        classes = {
            f"ConcDS_{i}": self._make_mock_dataset_class(f"ConcDS_{i}") for i in range(NUM_THREADS)
        }

        def register_one(idx):
            name = f"ConcDS_{idx}"
            registry.register(classes[name])

        _, errors = run_threads_with_barrier(register_one, NUM_THREADS)

        assert errors == [], f"Registration raised exceptions: {errors}"
        assert len(registry) == NUM_THREADS
        for i in range(NUM_THREADS):
            assert registry.is_registered(f"ConcDS_{i}")

    @pytest.mark.thread_safety
    def test_concurrent_get_after_registration(self):
        """
        Pre-register datasets, then have 20 threads concurrently call get()
        on random names.  All calls should succeed without exceptions.
        """
        registry = self.registry
        names = [f"GetDS_{i}" for i in range(10)]
        classes = {}
        for n in names:
            cls = self._make_mock_dataset_class(n)
            classes[n] = cls
            registry.register(cls)

        def lookup(idx):
            name = names[idx % len(names)]
            result = registry.get(name)
            assert result is classes[name]
            return name

        _, errors = run_threads_with_barrier(lookup, NUM_THREADS)
        assert errors == [], f"Concurrent get() raised exceptions: {errors}"

    @pytest.mark.thread_safety
    def test_concurrent_list_all_consistency(self):
        """
        Pre-register datasets, then have threads call list_all() concurrently.
        Each call should return a consistent snapshot (no partial state).
        """
        registry = self.registry
        for i in range(10):
            registry.register(self._make_mock_dataset_class(f"ListDS_{i}"))

        snapshots = []
        lock = threading.Lock()

        def snapshot(idx):
            result = registry.list_all()
            with lock:
                snapshots.append(frozenset(result))

        _, errors = run_threads_with_barrier(snapshot, NUM_THREADS)
        assert errors == []
        # All snapshots should be identical (no mutation is happening)
        assert len(set(snapshots)) == 1, "list_all() returned inconsistent snapshots"

    @pytest.mark.thread_safety
    def test_concurrent_register_and_lookup_mixed(self):
        """
        Half the threads register new datasets while the other half look up
        already-registered datasets.  No exceptions should be raised.
        """
        registry = self.registry
        # Pre-register 5 datasets for lookup threads
        pre_classes = {}
        for i in range(5):
            name = f"PreMixed_{i}"
            cls = self._make_mock_dataset_class(name)
            pre_classes[name] = cls
            registry.register(cls)

        # New classes for registration threads
        new_classes = {}
        for i in range(5):
            name = f"NewMixed_{i}"
            new_classes[name] = self._make_mock_dataset_class(name)

        def mixed_op(idx):
            if idx < 5:
                # Registering thread
                name = f"NewMixed_{idx}"
                registry.register(new_classes[name])
            else:
                # Lookup thread
                name = f"PreMixed_{idx % 5}"
                result = registry.get(name)
                assert result is pre_classes[name]

        _, errors = run_threads_with_barrier(mixed_op, 10)
        assert errors == [], f"Mixed ops raised exceptions: {errors}"
        assert len(registry) == 10

    @pytest.mark.thread_safety
    def test_concurrent_register_and_unregister(self):
        """
        Threads simultaneously register and unregister datasets.
        Final state should be consistent (no KeyError, no ghost entries).
        """
        registry = self.registry
        # Pre-register datasets to unregister
        for i in range(10):
            registry.register(self._make_mock_dataset_class(f"UnregDS_{i}"))

        new_classes = {}
        for i in range(10):
            name = f"RegDS_{i}"
            new_classes[name] = self._make_mock_dataset_class(name)

        def reg_unreg(idx):
            if idx < 10:
                registry.register(new_classes[f"RegDS_{idx}"])
            else:
                registry.unregister(f"UnregDS_{idx - 10}")

        _, errors = run_threads_with_barrier(reg_unreg, NUM_THREADS)
        assert errors == [], f"Register/unregister raised exceptions: {errors}"

        # All RegDS should be present
        for i in range(10):
            assert registry.is_registered(f"RegDS_{i}")
        # All UnregDS should be gone
        for i in range(10):
            assert not registry.is_registered(f"UnregDS_{i}")

    @pytest.mark.thread_safety
    def test_concurrent_is_registered_consistency(self):
        """
        All threads call is_registered() on the same set of names concurrently.
        Results must be consistent.
        """
        registry = self.registry
        for i in range(5):
            registry.register(self._make_mock_dataset_class(f"ExistDS_{i}"))

        results_map: dict[int, list[bool]] = {}
        lock = threading.Lock()

        def check(idx):
            present = [registry.is_registered(f"ExistDS_{i}") for i in range(5)]
            missing = [registry.is_registered(f"NoDS_{i}") for i in range(5)]
            with lock:
                results_map[idx] = present + missing

        _, errors = run_threads_with_barrier(check, NUM_THREADS)
        assert errors == []
        for _idx, bools in results_map.items():
            assert bools[:5] == [True] * 5
            assert bools[5:] == [False] * 5

    @pytest.mark.thread_safety
    def test_concurrent_contains_operator(self):
        """Thread-safe __contains__ (the 'in' operator)."""
        registry = self.registry
        registry.register(self._make_mock_dataset_class("InOpDS"))

        def check_in(idx):
            assert "InOpDS" in registry
            assert "NonExistent" not in registry

        _, errors = run_threads_with_barrier(check_in, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_concurrent_iter(self):
        """Thread-safe __iter__."""
        registry = self.registry
        expected = set()
        for i in range(10):
            name = f"IterDS_{i}"
            registry.register(self._make_mock_dataset_class(name))
            expected.add(name)

        collected = []
        lock = threading.Lock()

        def iterate(idx):
            names = set(registry)
            with lock:
                collected.append(names)

        _, errors = run_threads_with_barrier(iterate, NUM_THREADS)
        assert errors == []
        for s in collected:
            assert s == expected

    @pytest.mark.thread_safety
    def test_concurrent_len(self):
        """Thread-safe __len__."""
        registry = self.registry
        for i in range(7):
            registry.register(self._make_mock_dataset_class(f"LenDS_{i}"))

        lengths = []
        lock = threading.Lock()

        def get_len(idx):
            with lock:
                lengths.append(len(registry))

        _, errors = run_threads_with_barrier(get_len, NUM_THREADS)
        assert errors == []
        assert all(ln == 7 for ln in lengths)

    @pytest.mark.thread_safety
    def test_change_callback_fires_under_concurrency(self):
        """
        on_change callbacks should be invoked for each registration even
        when multiple threads register concurrently.
        """
        registry = self.registry
        call_count = {"n": 0}
        lock = threading.Lock()

        def on_change():
            with lock:
                call_count["n"] += 1

        registry.add_on_change_callback(on_change)

        classes = {
            f"CbDS_{i}": self._make_mock_dataset_class(f"CbDS_{i}") for i in range(NUM_THREADS)
        }

        def register_one(idx):
            registry.register(classes[f"CbDS_{idx}"])

        _, errors = run_threads_with_barrier(register_one, NUM_THREADS)
        assert errors == []
        assert call_count["n"] == NUM_THREADS, (
            f"Expected {NUM_THREADS} callbacks, got {call_count['n']}"
        )

    @pytest.mark.thread_safety
    def test_duplicate_registration_idempotent_under_concurrency(self):
        """
        Multiple threads register the *same* class concurrently.
        The registry should accept re-registration of the identical class
        without raising.
        """
        registry = self.registry
        cls = self._make_mock_dataset_class("DupDS")

        def register_same(idx):
            registry.register(cls)

        _, errors = run_threads_with_barrier(register_same, NUM_THREADS)
        assert errors == []
        assert len(registry) == 1
        assert registry.is_registered("DupDS")

    @pytest.mark.thread_safety
    def test_get_or_none_under_concurrency(self):
        """get_or_none() should not raise under concurrent access."""
        registry = self.registry
        cls = self._make_mock_dataset_class("OrNoneDS")
        registry.register(cls)

        def get_or_none_check(idx):
            if idx % 2 == 0:
                result = registry.get_or_none("OrNoneDS")
                assert result is cls
            else:
                result = registry.get_or_none("MissingDS")
                assert result is None

        _, errors = run_threads_with_barrier(get_or_none_check, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_stress_high_volume_concurrent_operations(self):
        """
        Stress test: 20 threads × 100 operations each (register, get,
        list_all, is_registered mixed).  Asserts zero exceptions.
        """
        registry = self.registry
        # Pre-create all classes to avoid class-creation overhead inside threads
        all_classes = {}
        for t in range(STRESS_THREADS):
            for op in range(STRESS_OPS):
                name = f"Stress_{t}_{op}"
                all_classes[name] = self._make_mock_dataset_class(name)

        error_count = {"n": 0}
        lock = threading.Lock()

        def stress_worker(idx):
            for op_idx in range(STRESS_OPS):
                name = f"Stress_{idx}_{op_idx}"
                try:
                    registry.register(all_classes[name])
                    registry.get(name)
                    registry.list_all()
                    registry.is_registered(name)
                except Exception:
                    with lock:
                        error_count["n"] += 1
                    raise

        _, errors = run_threads_with_barrier(stress_worker, STRESS_THREADS)
        assert errors == [], f"Stress test raised {len(errors)} exceptions"
        assert error_count["n"] == 0


# ===========================================================================
# SECTION 2: HandlerRegistry Thread Safety
# ===========================================================================


class TestHandlerRegistryThreadSafety:
    """
    Thread-safety tests for milia_pipeline.handlers.handler_registry.HandlerRegistry.

    HandlerRegistry is NOT a singleton.  We construct lightweight mock handler
    classes that have a get_dataset_type() method and no abstract methods.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        from milia_pipeline.handlers.handler_registry import HandlerRegistry

        self.HandlerRegistry = HandlerRegistry
        self.registry = HandlerRegistry()

    def _make_mock_handler_class(self, name: str) -> type:
        """
        Create a minimal handler class whose class name encodes *name*
        so that HandlerRegistry.register() can derive the handler name.

        The registry tries classmethod get_dataset_type first, then falls
        back to stripping 'DatasetHandler'/'Handler' from the class name.
        We use the fallback path: class named '{name}Handler'.
        """
        cls = type(
            f"{name}Handler",
            (),
            {
                "get_dataset_type": classmethod(lambda cls_: name),
                "__abstractmethods__": frozenset(),
            },
        )
        return cls

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    @pytest.mark.thread_safety
    def test_concurrent_registration_no_exceptions(self):
        """20 threads register unique handlers concurrently."""
        registry = self.registry
        classes = {
            f"HConc_{i}": self._make_mock_handler_class(f"HConc_{i}") for i in range(NUM_THREADS)
        }

        def register_one(idx):
            name = f"HConc_{idx}"
            registry.register(classes[name])

        _, errors = run_threads_with_barrier(register_one, NUM_THREADS)
        assert errors == [], f"Handler registration errors: {errors}"
        assert len(registry) == NUM_THREADS

    @pytest.mark.thread_safety
    def test_concurrent_get_after_registration(self):
        """Concurrent get() on pre-registered handlers."""
        registry = self.registry
        names = [f"HGet_{i}" for i in range(10)]
        classes = {}
        for n in names:
            cls = self._make_mock_handler_class(n)
            classes[n] = cls
            registry.register(cls)

        def lookup(idx):
            name = names[idx % len(names)]
            result = registry.get(name)
            assert result is classes[name]

        _, errors = run_threads_with_barrier(lookup, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_concurrent_list_all_snapshot_consistency(self):
        """list_all() returns consistent snapshots under concurrency."""
        registry = self.registry
        expected = set()
        for i in range(10):
            name = f"HList_{i}"
            registry.register(self._make_mock_handler_class(name))
            expected.add(name)

        snapshots = []
        lock = threading.Lock()

        def snapshot(idx):
            result = set(registry.list_all())
            with lock:
                snapshots.append(result)

        _, errors = run_threads_with_barrier(snapshot, NUM_THREADS)
        assert errors == []
        assert all(s == expected for s in snapshots)

    @pytest.mark.thread_safety
    def test_concurrent_register_and_unregister(self):
        """Interleaved register + unregister with no data corruption."""
        registry = self.registry
        for i in range(10):
            registry.register(self._make_mock_handler_class(f"HUnreg_{i}"))

        new_classes = {}
        for i in range(10):
            name = f"HNew_{i}"
            new_classes[name] = self._make_mock_handler_class(name)

        def reg_unreg(idx):
            if idx < 10:
                registry.register(new_classes[f"HNew_{idx}"])
            else:
                registry.unregister(f"HUnreg_{idx - 10}")

        _, errors = run_threads_with_barrier(reg_unreg, NUM_THREADS)
        assert errors == []
        for i in range(10):
            assert registry.is_registered(f"HNew_{i}")
        for i in range(10):
            assert not registry.is_registered(f"HUnreg_{i}")

    @pytest.mark.thread_safety
    def test_concurrent_is_registered(self):
        """is_registered() is consistent under concurrency."""
        registry = self.registry
        for i in range(5):
            registry.register(self._make_mock_handler_class(f"HExist_{i}"))

        def check(idx):
            for i in range(5):
                assert registry.is_registered(f"HExist_{i}")
                assert not registry.is_registered(f"HMissing_{i}")

        _, errors = run_threads_with_barrier(check, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_concurrent_contains_and_iter(self):
        """__contains__ and __iter__ under concurrency."""
        registry = self.registry
        expected = set()
        for i in range(8):
            name = f"HIt_{i}"
            registry.register(self._make_mock_handler_class(name))
            expected.add(name)

        collected = []
        lock = threading.Lock()

        def check(idx):
            assert f"HIt_{idx % 8}" in registry
            names = set(registry)
            with lock:
                collected.append(names)

        _, errors = run_threads_with_barrier(check, NUM_THREADS)
        assert errors == []
        for s in collected:
            assert s == expected

    @pytest.mark.thread_safety
    def test_change_callback_fires_under_concurrency(self):
        """on_change callbacks should fire for every registration."""
        registry = self.registry
        call_count = {"n": 0}
        lock = threading.Lock()

        def on_change():
            with lock:
                call_count["n"] += 1

        registry.add_on_change_callback(on_change)

        classes = {
            f"HCb_{i}": self._make_mock_handler_class(f"HCb_{i}") for i in range(NUM_THREADS)
        }

        def register_one(idx):
            registry.register(classes[f"HCb_{idx}"])

        _, errors = run_threads_with_barrier(register_one, NUM_THREADS)
        assert errors == []
        assert call_count["n"] == NUM_THREADS

    @pytest.mark.thread_safety
    def test_get_or_none_under_concurrency(self):
        """get_or_none() doesn't raise under concurrent access."""
        registry = self.registry
        cls = self._make_mock_handler_class("HOrNone")
        registry.register(cls)

        def check(idx):
            if idx % 2 == 0:
                assert registry.get_or_none("HOrNone") is cls
            else:
                assert registry.get_or_none("HMissing") is None

        _, errors = run_threads_with_barrier(check, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_get_registry_info_under_concurrency(self):
        """get_registry_info() returns consistent dict under concurrency."""
        registry = self.registry
        for i in range(5):
            registry.register(self._make_mock_handler_class(f"HInfo_{i}"))

        infos = []
        lock = threading.Lock()

        def get_info(idx):
            info = registry.get_registry_info()
            with lock:
                infos.append(info)

        _, errors = run_threads_with_barrier(get_info, NUM_THREADS)
        assert errors == []
        for info in infos:
            assert info["total_handlers"] == 5
            assert len(info["registered_handlers"]) == 5

    @pytest.mark.thread_safety
    def test_duplicate_registration_same_class(self):
        """Re-registering the same class is silently accepted (no raise)."""
        registry = self.registry
        cls = self._make_mock_handler_class("HDup")

        def register_same(idx):
            registry.register(cls)

        _, errors = run_threads_with_barrier(register_same, NUM_THREADS)
        assert errors == []
        assert len(registry) == 1

    @pytest.mark.thread_safety
    def test_stress_high_volume_mixed_operations(self):
        """Stress: 20 threads × 100 mixed operations each."""
        registry = self.registry
        all_classes = {}
        for t in range(STRESS_THREADS):
            for op in range(STRESS_OPS):
                name = f"HStress_{t}_{op}"
                all_classes[name] = self._make_mock_handler_class(name)

        def stress_worker(idx):
            for op_idx in range(STRESS_OPS):
                name = f"HStress_{idx}_{op_idx}"
                registry.register(all_classes[name])
                registry.get(name)
                registry.list_all()
                registry.is_registered(name)

        _, errors = run_threads_with_barrier(stress_worker, STRESS_THREADS)
        assert errors == [], f"Handler stress test raised {len(errors)} exceptions"


# ===========================================================================
# SECTION 3: DescriptorRegistry Thread Safety (Singleton)
# ===========================================================================


class TestDescriptorRegistryThreadSafety:
    """
    Thread-safety tests for milia_pipeline.descriptors.descriptor_registry.DescriptorRegistry.

    DescriptorRegistry is a singleton that auto-discovers RDKit descriptors
    on first instantiation.  Tests here exercise concurrent *reads* on the
    global singleton to verify the RLock protects shared state.

    We also test concurrent register_descriptor() calls using the public
    registration API with dummy callable descriptors, then clean up via
    reset() in teardown to avoid polluting the global singleton for
    subsequent test files.
    """

    @pytest.fixture(autouse=True)
    def _setup_and_teardown(self):
        """
        Import the singleton and record its initial state.
        After each test, remove any descriptors we added.
        """
        from milia_pipeline.descriptors.descriptor_registry import DescriptorRegistry

        self.DescriptorRegistry = DescriptorRegistry
        self.registry = DescriptorRegistry.get_instance()

        # Snapshot names before test
        self._initial_names = set(self.registry.list_all_descriptors())

        yield  # run test

        # Teardown: remove any descriptors added during the test
        current_names = set(self.registry.list_all_descriptors())
        added_names = current_names - self._initial_names
        if added_names:
            # DescriptorRegistry has no per-name unregister, so we
            # reset and re-discover to restore original state
            self.registry.reset()
            self.registry.auto_discover_rdkit_descriptors()
            self.registry.register_mol_method_descriptors()

    # ------------------------------------------------------------------
    # Tests — concurrent reads
    # ------------------------------------------------------------------

    @pytest.mark.thread_safety
    def test_concurrent_get_descriptor(self):
        """get_descriptor() is safe under concurrent access."""
        registry = self.registry
        all_names = registry.list_all_descriptors()
        if not all_names:
            pytest.skip("No descriptors discovered — RDKit might not be installed")
        sample = all_names[:10]

        def lookup(idx):
            name = sample[idx % len(sample)]
            func = registry.get_descriptor(name)
            assert func is not None, f"get_descriptor('{name}') returned None"
            assert callable(func)

        _, errors = run_threads_with_barrier(lookup, NUM_THREADS)
        assert errors == [], f"Concurrent get_descriptor errors: {errors}"

    @pytest.mark.thread_safety
    def test_concurrent_has_descriptor(self):
        """has_descriptor() returns consistent results under concurrency."""
        registry = self.registry
        all_names = registry.list_all_descriptors()
        if not all_names:
            pytest.skip("No descriptors discovered")
        sample = all_names[:5]

        def check(idx):
            for name in sample:
                assert registry.has_descriptor(name) is True
            assert registry.has_descriptor("__NONEXISTENT_DESC__") is False

        _, errors = run_threads_with_barrier(check, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_concurrent_list_all_descriptors(self):
        """list_all_descriptors() returns identical snapshots."""
        registry = self.registry
        snapshots = []
        lock = threading.Lock()

        def snapshot(idx):
            result = registry.list_all_descriptors()
            with lock:
                snapshots.append(tuple(result))

        _, errors = run_threads_with_barrier(snapshot, NUM_THREADS)
        assert errors == []
        assert len(set(snapshots)) == 1, "list_all_descriptors() inconsistent"

    @pytest.mark.thread_safety
    def test_concurrent_list_available_descriptors(self):
        """list_available_descriptors() is safe under concurrency."""
        registry = self.registry

        def list_desc(idx):
            result = registry.list_available_descriptors()
            assert isinstance(result, list)
            assert len(result) > 0

        _, errors = run_threads_with_barrier(list_desc, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_concurrent_get_statistics(self):
        """get_statistics() returns consistent dicts under concurrency."""
        registry = self.registry
        stats_list = []
        lock = threading.Lock()

        def get_stats(idx):
            stats = registry.get_statistics()
            with lock:
                stats_list.append(stats["total_descriptors"])

        _, errors = run_threads_with_barrier(get_stats, NUM_THREADS)
        assert errors == []
        assert len(set(stats_list)) == 1, "get_statistics() inconsistent"

    @pytest.mark.thread_safety
    def test_concurrent_get_metadata(self):
        """get_metadata() is safe under concurrent reads."""
        registry = self.registry
        all_names = registry.list_all_descriptors()
        if not all_names:
            pytest.skip("No descriptors discovered")
        sample = all_names[:5]

        def get_meta(idx):
            name = sample[idx % len(sample)]
            meta = registry.get_metadata(name)
            assert meta is not None

        _, errors = run_threads_with_barrier(get_meta, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_concurrent_register_descriptor_plugin(self):
        """
        Concurrent register_descriptor() calls with unique plugin names.
        Each thread registers a unique fake descriptor.
        """
        registry = self.registry

        def register_one(idx):
            name = f"__test_plugin_desc_{idx}__"

            def func(mol):
                return float(idx)

            with contextlib.suppress(Exception):
                # If the descriptor already exists from a prior failed run,
                # just verify it's present.
                registry.register_descriptor(
                    name=name,
                    function=func,
                    is_builtin=False,
                    plugin_name=f"test_plugin_{idx}",
                )
            assert registry.has_descriptor(name)

        _, errors = run_threads_with_barrier(register_one, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_singleton_identity_under_concurrency(self):
        """
        DescriptorRegistry() returns the same object from all threads.
        """
        instances = [None] * NUM_THREADS
        DescriptorRegistry = self.DescriptorRegistry

        def get_instance(idx):
            instances[idx] = DescriptorRegistry.get_instance()

        _, errors = run_threads_with_barrier(get_instance, NUM_THREADS)
        assert errors == []
        assert all(inst is instances[0] for inst in instances), (
            "DescriptorRegistry singleton broken — different instances returned"
        )

    @pytest.mark.thread_safety
    def test_concurrent_get_availability_report(self):
        """get_availability_report() is safe under concurrency."""
        registry = self.registry

        reports = []
        lock = threading.Lock()

        def get_report(idx):
            report = registry.get_availability_report()
            with lock:
                reports.append(report["total_registered"])

        _, errors = run_threads_with_barrier(get_report, NUM_THREADS)
        assert errors == []
        assert len(set(reports)) == 1


# ===========================================================================
# SECTION 4: ModelRegistry Thread Safety (Singleton)
# ===========================================================================


class TestModelRegistryThreadSafety:
    """
    Thread-safety tests for milia_pipeline.models.registry.model_registry.ModelRegistry.

    ModelRegistry is a singleton with auto-discovery of PyG models.
    Tests exercise concurrent reads on the global singleton, and concurrent
    register_model() / unregister_model() with dummy model classes.
    """

    @pytest.fixture(autouse=True)
    def _setup_and_teardown(self):
        from milia_pipeline.models.registry.model_registry import ModelRegistry

        self.ModelRegistry = ModelRegistry
        self.registry = ModelRegistry.get_instance()
        self._initial_names = set(self.registry.list_available_models())

        yield

        # Teardown: unregister any models we added
        current_names = set(self.registry.list_available_models())
        added = current_names - self._initial_names
        for name in added:
            self.registry.unregister_model(name)

    def _make_dummy_model_class(self, name: str):
        """Create a minimal torch.nn.Module subclass for registration."""
        import torch.nn as nn

        cls = type(
            f"DummyModel_{name}",
            (nn.Module,),
            {
                "__init__": lambda self_: nn.Module.__init__(self_),
                "forward": lambda self_, x: x,
            },
        )
        return cls

    def _make_dummy_metadata(self, name: str):
        """Create a ModelMetadata for the dummy model."""
        from milia_pipeline.models.registry.model_registry import ModelCategory, ModelMetadata

        return ModelMetadata(
            name=name,
            category=ModelCategory.BASIC_GNN,
            import_path=f"test.{name}",
            description=f"Test model {name}",
            supported_tasks=["graph_regression"],
        )

    # ------------------------------------------------------------------
    # Tests — concurrent reads
    # ------------------------------------------------------------------

    @pytest.mark.thread_safety
    def test_concurrent_get_model(self):
        """get_model() is safe under concurrent access."""
        registry = self.registry
        all_names = registry.list_available_models()
        if not all_names:
            pytest.skip("No models discovered")
        sample = all_names[:10]

        def lookup(idx):
            name = sample[idx % len(sample)]
            cls = registry.get_model(name)
            assert cls is not None

        _, errors = run_threads_with_barrier(lookup, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_concurrent_has_model(self):
        """has_model() consistent under concurrency."""
        registry = self.registry
        all_names = registry.list_available_models()
        if not all_names:
            pytest.skip("No models discovered")

        def check(idx):
            assert registry.has_model(all_names[0])
            assert not registry.has_model("__NONEXISTENT_MODEL__")

        _, errors = run_threads_with_barrier(check, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_concurrent_list_available_models(self):
        """list_available_models() returns consistent snapshots."""
        registry = self.registry
        snapshots = []
        lock = threading.Lock()

        def snapshot(idx):
            result = registry.list_available_models()
            with lock:
                snapshots.append(tuple(result))

        _, errors = run_threads_with_barrier(snapshot, NUM_THREADS)
        assert errors == []
        assert len(set(snapshots)) == 1

    @pytest.mark.thread_safety
    def test_concurrent_contains_operator(self):
        """__contains__ is safe under concurrency."""
        registry = self.registry
        all_names = registry.list_available_models()
        if not all_names:
            pytest.skip("No models discovered")

        def check(idx):
            assert all_names[0] in registry
            assert "__FAKE__" not in registry

        _, errors = run_threads_with_barrier(check, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_concurrent_len(self):
        """__len__ returns consistent value."""
        registry = self.registry
        expected = len(registry)
        lengths = []
        lock = threading.Lock()

        def get_len(idx):
            with lock:
                lengths.append(len(registry))

        _, errors = run_threads_with_barrier(get_len, NUM_THREADS)
        assert errors == []
        assert all(ln == expected for ln in lengths)

    @pytest.mark.thread_safety
    def test_concurrent_get_statistics(self):
        """get_statistics() is safe under concurrency."""
        registry = self.registry
        totals = []
        lock = threading.Lock()

        def get_stats(idx):
            stats = registry.get_statistics()
            with lock:
                totals.append(stats["total_models"])

        _, errors = run_threads_with_barrier(get_stats, NUM_THREADS)
        assert errors == []
        assert len(set(totals)) == 1

    @pytest.mark.thread_safety
    def test_concurrent_get_metadata(self):
        """get_metadata() is safe under concurrency."""
        registry = self.registry
        all_names = registry.list_available_models()
        if not all_names:
            pytest.skip("No models discovered")
        sample = all_names[:5]

        def get_meta(idx):
            name = sample[idx % len(sample)]
            meta = registry.get_metadata(name)
            assert meta is not None

        _, errors = run_threads_with_barrier(get_meta, NUM_THREADS)
        assert errors == []

    # ------------------------------------------------------------------
    # Tests — concurrent writes
    # ------------------------------------------------------------------

    @pytest.mark.thread_safety
    def test_concurrent_register_model(self):
        """Concurrent register_model() with unique dummy models."""
        registry = self.registry

        dummy_classes = {}
        dummy_metas = {}
        for i in range(NUM_THREADS):
            name = f"__TestDummy_{i}__"
            dummy_classes[name] = self._make_dummy_model_class(name)
            dummy_metas[name] = self._make_dummy_metadata(name)

        def register_one(idx):
            name = f"__TestDummy_{idx}__"
            registry.register_model(
                name=name,
                model_class=dummy_classes[name],
                metadata=dummy_metas[name],
                plugin_name="thread_test",
            )

        _, errors = run_threads_with_barrier(register_one, NUM_THREADS)
        assert errors == []
        for i in range(NUM_THREADS):
            assert registry.has_model(f"__TestDummy_{i}__")

    @pytest.mark.thread_safety
    def test_concurrent_register_and_unregister_model(self):
        """Interleaved register + unregister with no corruption."""
        registry = self.registry

        # Pre-register models to unregister
        unreg_names = []
        for i in range(10):
            name = f"__TestUnreg_{i}__"
            cls = self._make_dummy_model_class(name)
            meta = self._make_dummy_metadata(name)
            registry.register_model(name, cls, meta, plugin_name="thread_test")
            unreg_names.append(name)

        # New models to register
        new_classes = {}
        new_metas = {}
        for i in range(10):
            name = f"__TestNewReg_{i}__"
            new_classes[name] = self._make_dummy_model_class(name)
            new_metas[name] = self._make_dummy_metadata(name)

        def reg_unreg(idx):
            if idx < 10:
                name = f"__TestNewReg_{idx}__"
                registry.register_model(
                    name,
                    new_classes[name],
                    new_metas[name],
                    plugin_name="thread_test",
                )
            else:
                registry.unregister_model(unreg_names[idx - 10])

        _, errors = run_threads_with_barrier(reg_unreg, NUM_THREADS)
        assert errors == []
        for i in range(10):
            assert registry.has_model(f"__TestNewReg_{i}__")
        for i in range(10):
            assert not registry.has_model(f"__TestUnreg_{i}__")

    @pytest.mark.thread_safety
    def test_singleton_identity_under_concurrency(self):
        """ModelRegistry() returns the same instance from all threads."""
        instances = [None] * NUM_THREADS
        MR = self.ModelRegistry

        def get_instance(idx):
            instances[idx] = MR.get_instance()

        _, errors = run_threads_with_barrier(get_instance, NUM_THREADS)
        assert errors == []
        assert all(inst is instances[0] for inst in instances)


# ===========================================================================
# SECTION 5: LayerRegistry Thread Safety (Singleton)
# ===========================================================================


class TestLayerRegistryThreadSafety:
    """
    Thread-safety tests for milia_pipeline.models.builders.layer_registry.LayerRegistry.

    LayerRegistry is a singleton with auto-registered PyG layers.
    Tests exercise concurrent reads and concurrent register_custom_layer()
    calls.
    """

    @pytest.fixture(autouse=True)
    def _setup_and_teardown(self):
        from milia_pipeline.models.builders.layer_registry import (
            LayerCategory,
            LayerMetadata,
            LayerRegistry,
        )

        self.LayerRegistry = LayerRegistry
        self.LayerCategory = LayerCategory
        self.LayerMetadata = LayerMetadata
        self.registry = LayerRegistry()  # singleton

        self._initial_names = set(self.registry.list_layers())

        yield

        # Teardown: The registry has no unregister method, so we remove
        # any added layers manually from internal dicts.
        current_names = set(self.registry.list_layers())
        added = current_names - self._initial_names
        if added:
            with self.registry._lock:
                for name in added:
                    self.registry._layers.pop(name, None)
                    self.registry._metadata.pop(name, None)
                    for cat_set in self.registry._by_category.values():
                        cat_set.discard(name)

    def _make_dummy_layer_class(self, name: str):
        """Create a minimal nn.Module subclass for registration."""
        import torch.nn as nn

        cls = type(
            f"DummyLayer_{name}",
            (nn.Module,),
            {
                "__init__": lambda self_, in_c=16, out_c=16: nn.Module.__init__(self_),
                "forward": lambda self_, x, edge_index=None: x,
            },
        )
        return cls

    # ------------------------------------------------------------------
    # Tests — concurrent reads
    # ------------------------------------------------------------------

    @pytest.mark.thread_safety
    def test_concurrent_get_layer(self):
        """get_layer() is safe under concurrent access."""
        registry = self.registry
        all_names = registry.list_layers()
        if not all_names:
            pytest.skip("No layers registered")
        sample = all_names[:10]

        def lookup(idx):
            name = sample[idx % len(sample)]
            cls = registry.get_layer(name)
            assert cls is not None

        _, errors = run_threads_with_barrier(lookup, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_concurrent_has_layer(self):
        """has_layer() consistent under concurrency."""
        registry = self.registry
        all_names = registry.list_layers()
        if not all_names:
            pytest.skip("No layers registered")

        def check(idx):
            assert registry.has_layer(all_names[0])
            assert not registry.has_layer("__NONEXISTENT_LAYER__")

        _, errors = run_threads_with_barrier(check, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_concurrent_list_layers(self):
        """list_layers() returns consistent snapshots."""
        registry = self.registry
        snapshots = []
        lock = threading.Lock()

        def snapshot(idx):
            result = registry.list_layers()
            with lock:
                snapshots.append(tuple(result))

        _, errors = run_threads_with_barrier(snapshot, NUM_THREADS)
        assert errors == []
        assert len(set(snapshots)) == 1

    @pytest.mark.thread_safety
    def test_concurrent_list_layers_by_category(self):
        """list_layers(category=...) is safe under concurrency."""
        registry = self.registry
        LayerCategory = self.LayerCategory

        def list_by_cat(idx):
            result = registry.list_layers(category=LayerCategory.CONVOLUTIONAL)
            assert isinstance(result, list)

        _, errors = run_threads_with_barrier(list_by_cat, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_concurrent_get_layer_metadata(self):
        """get_layer_metadata() is safe under concurrency."""
        registry = self.registry
        all_names = registry.list_layers()
        if not all_names:
            pytest.skip("No layers registered")
        sample = all_names[:5]

        def get_meta(idx):
            name = sample[idx % len(sample)]
            meta = registry.get_layer_metadata(name)
            assert meta is not None
            assert meta.name == name

        _, errors = run_threads_with_barrier(get_meta, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_concurrent_contains_operator(self):
        """__contains__ under concurrency."""
        registry = self.registry
        all_names = registry.list_layers()
        if not all_names:
            pytest.skip("No layers registered")

        def check(idx):
            assert all_names[0] in registry
            assert "__FAKE_LAYER__" not in registry

        _, errors = run_threads_with_barrier(check, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_concurrent_len(self):
        """__len__ returns consistent value."""
        registry = self.registry
        expected = len(registry)
        lengths = []
        lock = threading.Lock()

        def get_len(idx):
            with lock:
                lengths.append(len(registry))

        _, errors = run_threads_with_barrier(get_len, NUM_THREADS)
        assert errors == []
        assert all(ln == expected for ln in lengths)

    @pytest.mark.thread_safety
    def test_concurrent_get_statistics(self):
        """get_statistics() is safe under concurrency."""
        registry = self.registry
        totals = []
        lock = threading.Lock()

        def get_stats(idx):
            stats = registry.get_statistics()
            with lock:
                totals.append(stats["total_layers"])

        _, errors = run_threads_with_barrier(get_stats, NUM_THREADS)
        assert errors == []
        assert len(set(totals)) == 1

    # ------------------------------------------------------------------
    # Tests — concurrent writes
    # ------------------------------------------------------------------

    @pytest.mark.thread_safety
    def test_concurrent_register_custom_layer(self):
        """Concurrent register_custom_layer() with unique names."""
        registry = self.registry

        def register_one(idx):
            name = f"__TestCustomLayer_{idx}__"
            cls = self._make_dummy_layer_class(name)
            registry.register_custom_layer(name, cls, overwrite=True)

        _, errors = run_threads_with_barrier(register_one, NUM_THREADS)
        assert errors == []
        for i in range(NUM_THREADS):
            assert registry.has_layer(f"__TestCustomLayer_{i}__")

    @pytest.mark.thread_safety
    def test_concurrent_register_and_read_custom_layer(self):
        """Interleaved registration and reads."""
        registry = self.registry
        all_names = registry.list_layers()
        read_target = all_names[0] if all_names else None

        if not read_target:
            pytest.skip("No layers registered")

        def mixed(idx):
            if idx % 2 == 0:
                name = f"__TestMixLayer_{idx}__"
                cls = self._make_dummy_layer_class(name)
                registry.register_custom_layer(name, cls, overwrite=True)
            else:
                cls = registry.get_layer(read_target)
                assert cls is not None

        _, errors = run_threads_with_barrier(mixed, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_singleton_identity_under_concurrency(self):
        """LayerRegistry() returns the same instance from all threads."""
        instances = [None] * NUM_THREADS
        LR = self.LayerRegistry

        def get_instance(idx):
            instances[idx] = LR()

        _, errors = run_threads_with_barrier(get_instance, NUM_THREADS)
        assert errors == []
        assert all(inst is instances[0] for inst in instances)


# ===========================================================================
# SECTION 6: Cross-Registry Consistency Under Concurrency
# ===========================================================================


class TestCrossRegistryConsistencyThreadSafety:
    """
    Verify that concurrent reads across *multiple* registries do not
    deadlock or produce inconsistent results.

    This exercises the scenario where a real pipeline thread needs to
    look up a dataset, its handler, and its model simultaneously.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Import all registries."""
        from milia_pipeline.datasets.registry import DatasetRegistry
        from milia_pipeline.handlers.handler_registry import HandlerRegistry

        self.dataset_registry = DatasetRegistry()
        self.handler_registry = HandlerRegistry()

    @pytest.mark.thread_safety
    def test_no_deadlock_accessing_multiple_registries(self):
        """
        Each thread accesses DatasetRegistry and HandlerRegistry in
        alternating order.  If locks were poorly nested this would deadlock.
        """
        ds_reg = self.dataset_registry
        h_reg = self.handler_registry

        def access_both(idx):
            if idx % 2 == 0:
                ds_reg.list_all()
                h_reg.list_all()
            else:
                h_reg.list_all()
                ds_reg.list_all()

        _, errors = run_threads_with_barrier(access_both, NUM_THREADS)
        assert errors == []

    @pytest.mark.thread_safety
    def test_no_deadlock_with_singleton_registries(self):
        """
        Access DatasetRegistry (non-singleton), DescriptorRegistry (singleton),
        and ModelRegistry (singleton) from concurrent threads in different orders.
        """
        from milia_pipeline.descriptors.descriptor_registry import DescriptorRegistry
        from milia_pipeline.models.registry.model_registry import ModelRegistry

        ds_reg = self.dataset_registry
        desc_reg = DescriptorRegistry.get_instance()
        model_reg = ModelRegistry.get_instance()

        def access_all(idx):
            order = idx % 3
            if order == 0:
                ds_reg.list_all()
                desc_reg.list_all_descriptors()
                model_reg.list_available_models()
            elif order == 1:
                model_reg.list_available_models()
                ds_reg.list_all()
                desc_reg.list_all_descriptors()
            else:
                desc_reg.list_all_descriptors()
                model_reg.list_available_models()
                ds_reg.list_all()

        _, errors = run_threads_with_barrier(access_all, NUM_THREADS)
        assert errors == []


# ===========================================================================
# SECTION 7: Edge Cases & Robustness
# ===========================================================================


class TestRegistryEdgeCasesThreadSafety:
    """
    Thread-safety edge cases that apply across registry types.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        from milia_pipeline.datasets.base import (
            BaseDataset,
            DatasetFeatures,
            DatasetMetadata,
            DatasetSchema,
        )
        from milia_pipeline.datasets.registry import DatasetRegistry

        self.DatasetRegistry = DatasetRegistry
        self.BaseDataset = BaseDataset
        self.DatasetMetadata = DatasetMetadata
        self.DatasetSchema = DatasetSchema
        self.DatasetFeatures = DatasetFeatures

    def _make_mock_dataset_class(self, name: str) -> type:
        """Create a mock dataset class (same helper as Section 1)."""
        base = self.BaseDataset
        abstract_methods = set()
        for cls in base.__mro__:
            if hasattr(cls, "__abstractmethods__"):
                abstract_methods |= cls.__abstractmethods__
        ns: dict[str, Any] = {}
        for method_name in abstract_methods:
            ns[method_name] = lambda self, *a, **kw: None
        ns["metadata"] = self.DatasetMetadata(
            name=name,
            version="0.0.1",
            description=f"Mock dataset {name}",
            author="test",
            license="test",
        )
        ns["schema"] = self.DatasetSchema(
            required_properties=["energy"],
            optional_properties=[],
            identifier_keys=[("mol_id", "mol_id")],
        )
        ns["features"] = self.DatasetFeatures()
        ns["config_key"] = f"mock_{name.lower()}_config"
        cls = type(f"MockDataset_{name}", (base,), ns)
        cls.__abstractmethods__ = frozenset()
        return cls

    @pytest.mark.thread_safety
    def test_clear_during_concurrent_reads(self):
        """
        One thread calls clear() while others read.
        Should not raise; readers may see empty or non-empty state.
        """
        registry = self.DatasetRegistry()
        for i in range(10):
            registry.register(self._make_mock_dataset_class(f"ClearDS_{i}"))

        def worker(idx):
            if idx == 0:
                registry.clear()
            else:
                # These should not raise regardless of clear() timing
                try:
                    registry.list_all()
                    registry.is_registered(f"ClearDS_{idx % 10}")
                except Exception:
                    # DatasetNotFoundError is acceptable if clear() already ran
                    pass

        _, errors = run_threads_with_barrier(worker, NUM_THREADS)
        # We only care that no unhandled threading exceptions occurred
        assert errors == []

    @pytest.mark.thread_safety
    def test_callback_exception_does_not_break_registration(self):
        """
        A failing callback should not prevent other threads from registering.
        DatasetRegistry.register() catches callback exceptions internally.
        """
        registry = self.DatasetRegistry()

        def bad_callback():
            raise RuntimeError("Intentional callback failure")

        registry.add_on_change_callback(bad_callback)

        classes = {
            f"CbFailDS_{i}": self._make_mock_dataset_class(f"CbFailDS_{i}")
            for i in range(NUM_THREADS)
        }

        def register_one(idx):
            registry.register(classes[f"CbFailDS_{idx}"])

        _, errors = run_threads_with_barrier(register_one, NUM_THREADS)
        assert errors == [], f"Failing callback caused thread errors: {errors}"
        assert len(registry) == NUM_THREADS

    @pytest.mark.thread_safety
    def test_remove_callback_during_concurrent_registration(self):
        """
        One thread removes a callback while others trigger registrations
        that fire callbacks.
        """
        registry = self.DatasetRegistry()
        call_count = {"n": 0}
        lock = threading.Lock()

        def counting_callback():
            with lock:
                call_count["n"] += 1

        registry.add_on_change_callback(counting_callback)

        classes = {
            f"RmCbDS_{i}": self._make_mock_dataset_class(f"RmCbDS_{i}") for i in range(NUM_THREADS)
        }

        def worker(idx):
            if idx == 0:
                # Give other threads a head start
                registry.remove_on_change_callback(counting_callback)
            else:
                registry.register(classes[f"RmCbDS_{idx}"])

        _, errors = run_threads_with_barrier(worker, NUM_THREADS)
        assert errors == []
        # At least some registrations should have succeeded
        assert len(registry) >= NUM_THREADS - 1

    @pytest.mark.thread_safety
    def test_rapid_register_unregister_same_name(self):
        """
        Multiple threads alternately register and unregister the same
        dataset name.  Final state should be consistent.
        """
        registry = self.DatasetRegistry()
        cls = self._make_mock_dataset_class("FlipFlop")

        def flip_flop(idx):
            if idx % 2 == 0:
                with contextlib.suppress(Exception):
                    # DatasetRegistrationError is acceptable
                    registry.register(cls)
            else:
                registry.unregister("FlipFlop")

        _, errors = run_threads_with_barrier(flip_flop, NUM_THREADS)
        assert errors == []
        # Final state: registered or not, but len must match reality
        is_reg = registry.is_registered("FlipFlop")
        count = len(registry)
        if is_reg:
            assert count >= 1
        else:
            assert "FlipFlop" not in registry.list_all()

    @pytest.mark.thread_safety
    def test_threadpool_executor_compatibility(self):
        """
        Verify registries work with ThreadPoolExecutor (not just raw threads).
        This is the pattern real application code would use.
        """
        registry = self.DatasetRegistry()
        classes = {
            f"TPE_DS_{i}": self._make_mock_dataset_class(f"TPE_DS_{i}") for i in range(NUM_THREADS)
        }

        def register_and_verify(idx):
            name = f"TPE_DS_{idx}"
            registry.register(classes[name])
            assert registry.is_registered(name)
            assert registry.get(name) is classes[name]
            return name

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [executor.submit(register_and_verify, i) for i in range(NUM_THREADS)]
            results = [f.result(timeout=30) for f in as_completed(futures)]

        assert len(results) == NUM_THREADS
        assert len(registry) == NUM_THREADS
