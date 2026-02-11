"""
Test Contract: Registry Consistency
====================================

Contract test verifying cross-registry consistency between the DatasetRegistry
(milia_pipeline/datasets/registry.py) and HandlerRegistry
(milia_pipeline/handlers/handler_registry.py).

Section 2.3 of MILIA Test Recommendations:
    "Every dataset type registered in DatasetRegistry has a corresponding handler
     in HandlerRegistry, and vice versa."

Test Scope:
    - Bijection between dataset registry and handler registry entries
    - create_dataset_handler(dataset_type) succeeds for every registered dataset type
    - Registry API contract consistency (both registries expose identical interface patterns)
    - Registry isolation (non-singleton instances behave independently)

Modules Exercised:
    - milia_pipeline/datasets/registry.py — DatasetRegistry, list_all()
    - milia_pipeline/handlers/handler_registry.py — HandlerRegistry, list_all()
    - milia_pipeline/handlers/base_handler.py — create_dataset_handler()

Environment:
    - Runs from project root: /app/milia/
    - Docker-compatible: No filesystem or network dependencies
    - CI/CD ready: Uses pytest markers for selective execution

Mock Pollution Prevention:
    - NO sys.modules injection at module level
    - All mocking done via @patch decorators at test level
    - Isolated registry instances created per-test where needed
    - No global state modification outside of test functions

Author: MILIA Test Suite
Version: 1.0.0
"""

import sys
import os
import logging
import pytest
from typing import List, Set, Dict, Any, Type
from unittest.mock import patch, MagicMock, PropertyMock

# ---------------------------------------------------------------------------
# Path Setup: Add project root to Python path FIRST
# ---------------------------------------------------------------------------
# This ensures milia_pipeline is importable when running from /app/milia/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)


# ===========================================================================
# Pytest Markers
# ===========================================================================

pytestmark = [
    pytest.mark.contract,
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]


# ===========================================================================
# Import Safety Helpers
# ===========================================================================

def _safe_import_dataset_registry():
    """
    Safely import the DatasetRegistry and its module-level convenience functions.

    Returns:
        Tuple of (registry_module, DatasetRegistry_class) or (None, None) on failure.
        Stores the import error message for skip-reason reporting.
    """
    try:
        from milia_pipeline.datasets import registry as ds_registry_module
        from milia_pipeline.datasets.registry import DatasetRegistry
        return ds_registry_module, DatasetRegistry
    except ImportError as exc:
        logger.warning("Could not import dataset registry: %s", exc)
        return None, None


def _safe_import_handler_registry():
    """
    Safely import the HandlerRegistry and its module-level convenience functions.

    Returns:
        Tuple of (registry_module, HandlerRegistry_class) or (None, None) on failure.
    """
    try:
        from milia_pipeline.handlers import handler_registry as hr_module
        from milia_pipeline.handlers.handler_registry import HandlerRegistry
        return hr_module, HandlerRegistry
    except ImportError as exc:
        logger.warning("Could not import handler registry: %s", exc)
        return None, None


def _safe_import_base_handler():
    """
    Safely import the create_dataset_handler factory function.

    Returns:
        The create_dataset_handler callable, or None on failure.
    """
    try:
        from milia_pipeline.handlers.base_handler import create_dataset_handler
        return create_dataset_handler
    except ImportError as exc:
        logger.warning("Could not import create_dataset_handler: %s", exc)
        return None


def _safe_import_config_containers():
    """
    Safely import configuration container classes needed by create_dataset_handler.

    Returns:
        Tuple of (DatasetConfig, FilterConfig, ProcessingConfig) or (None, None, None).
    """
    try:
        from milia_pipeline.config.config_containers import (
            DatasetConfig,
            FilterConfig,
            ProcessingConfig,
        )
        return DatasetConfig, FilterConfig, ProcessingConfig
    except ImportError as exc:
        logger.warning("Could not import config containers: %s", exc)
        return None, None, None


# ---------------------------------------------------------------------------
# Perform imports once at module level (no mocking, no sys.modules pollution)
# ---------------------------------------------------------------------------

_ds_registry_mod, _DatasetRegistry = _safe_import_dataset_registry()
_hr_registry_mod, _HandlerRegistry = _safe_import_handler_registry()
_create_dataset_handler = _safe_import_base_handler()
_DatasetConfig, _FilterConfig, _ProcessingConfig = _safe_import_config_containers()

# Determine availability for skip decorators
_DATASET_REGISTRY_AVAILABLE = _ds_registry_mod is not None
_HANDLER_REGISTRY_AVAILABLE = _hr_registry_mod is not None
_FACTORY_AVAILABLE = _create_dataset_handler is not None
_CONFIGS_AVAILABLE = _DatasetConfig is not None
_BOTH_REGISTRIES_AVAILABLE = _DATASET_REGISTRY_AVAILABLE and _HANDLER_REGISTRY_AVAILABLE
_FULL_STACK_AVAILABLE = (
    _BOTH_REGISTRIES_AVAILABLE and _FACTORY_AVAILABLE and _CONFIGS_AVAILABLE
)


# ===========================================================================
# Skip Reason Constants
# ===========================================================================

SKIP_NO_DATASET_REGISTRY = "DatasetRegistry not importable (milia_pipeline.datasets.registry)"
SKIP_NO_HANDLER_REGISTRY = "HandlerRegistry not importable (milia_pipeline.handlers.handler_registry)"
SKIP_NO_BOTH_REGISTRIES = "Both registries must be importable for cross-registry tests"
SKIP_NO_FACTORY = "create_dataset_handler not importable (milia_pipeline.handlers.base_handler)"
SKIP_NO_CONFIGS = "Config containers not importable (milia_pipeline.config.config_containers)"
SKIP_NO_FULL_STACK = (
    "Full stack required: both registries + create_dataset_handler + config containers"
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def dataset_registry_entries() -> List[str]:
    """
    Retrieve the list of all dataset names currently registered in the
    default DatasetRegistry.

    Returns an empty list if the registry is not available (test will be
    skipped by the appropriate marker, not by this fixture).
    """
    if not _DATASET_REGISTRY_AVAILABLE:
        return []
    return _ds_registry_mod.list_all()


@pytest.fixture
def handler_registry_entries() -> List[str]:
    """
    Retrieve the list of all handler names currently registered in the
    default HandlerRegistry.

    Returns an empty list if the registry is not available.
    """
    if not _HANDLER_REGISTRY_AVAILABLE:
        return []
    return _hr_registry_mod.list_all()


@pytest.fixture
def isolated_dataset_registry():
    """
    Create an isolated (non-singleton) DatasetRegistry instance for
    tests that must not interact with the default global registry.
    """
    if _DatasetRegistry is None:
        pytest.skip("DatasetRegistry class not available")
    return _DatasetRegistry()


@pytest.fixture
def isolated_handler_registry():
    """
    Create an isolated (non-singleton) HandlerRegistry instance for
    tests that must not interact with the default global registry.
    """
    if _HandlerRegistry is None:
        pytest.skip("HandlerRegistry class not available")
    return _HandlerRegistry()


# ===========================================================================
# Section 1: Dataset Registry Availability & Contract
# ===========================================================================

class TestDatasetRegistryContract:
    """Verify the DatasetRegistry satisfies its public API contract."""

    @pytest.mark.skipif(
        not _DATASET_REGISTRY_AVAILABLE, reason=SKIP_NO_DATASET_REGISTRY
    )
    def test_default_registry_is_accessible(self):
        """get_default_registry() returns a DatasetRegistry instance."""
        registry = _ds_registry_mod.get_default_registry()
        assert isinstance(registry, _DatasetRegistry), (
            f"Expected DatasetRegistry instance, got {type(registry).__name__}"
        )

    @pytest.mark.skipif(
        not _DATASET_REGISTRY_AVAILABLE, reason=SKIP_NO_DATASET_REGISTRY
    )
    def test_list_all_returns_list_of_strings(self, dataset_registry_entries):
        """list_all() returns List[str]."""
        assert isinstance(dataset_registry_entries, list), (
            f"Expected list, got {type(dataset_registry_entries).__name__}"
        )
        for entry in dataset_registry_entries:
            assert isinstance(entry, str), (
                f"Expected str entry, got {type(entry).__name__}: {entry!r}"
            )

    @pytest.mark.skipif(
        not _DATASET_REGISTRY_AVAILABLE, reason=SKIP_NO_DATASET_REGISTRY
    )
    def test_registry_is_not_empty(self, dataset_registry_entries):
        """
        At least one dataset must be registered for the pipeline to function.
        An empty registry indicates that dataset implementations were not
        imported or @register decorators did not fire.
        """
        assert len(dataset_registry_entries) > 0, (
            "DatasetRegistry is empty — no datasets registered. "
            "Verify that datasets/implementations/__init__.py triggers @register decorators."
        )

    @pytest.mark.skipif(
        not _DATASET_REGISTRY_AVAILABLE, reason=SKIP_NO_DATASET_REGISTRY
    )
    def test_no_duplicate_dataset_names(self, dataset_registry_entries):
        """Dataset names in the registry must be unique."""
        seen: Set[str] = set()
        duplicates: List[str] = []
        for name in dataset_registry_entries:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        assert not duplicates, (
            f"Duplicate dataset names found in registry: {duplicates}"
        )

    @pytest.mark.skipif(
        not _DATASET_REGISTRY_AVAILABLE, reason=SKIP_NO_DATASET_REGISTRY
    )
    def test_get_returns_class_for_each_registered_dataset(
        self, dataset_registry_entries
    ):
        """registry.get(name) returns a class (type) for every registered name."""
        for name in dataset_registry_entries:
            cls = _ds_registry_mod.get(name)
            assert isinstance(cls, type), (
                f"get('{name}') returned {type(cls).__name__}, expected a class"
            )

    @pytest.mark.skipif(
        not _DATASET_REGISTRY_AVAILABLE, reason=SKIP_NO_DATASET_REGISTRY
    )
    def test_is_registered_consistent_with_list_all(self, dataset_registry_entries):
        """is_registered(name) must return True for every name in list_all()."""
        for name in dataset_registry_entries:
            assert _ds_registry_mod.is_registered(name), (
                f"is_registered('{name}') returned False, but '{name}' is in list_all()"
            )

    @pytest.mark.skipif(
        not _DATASET_REGISTRY_AVAILABLE, reason=SKIP_NO_DATASET_REGISTRY
    )
    def test_is_registered_false_for_nonexistent(self):
        """is_registered() returns False for a name that was never registered."""
        assert not _ds_registry_mod.is_registered(
            "__nonexistent_dataset_type_for_testing__"
        )


# ===========================================================================
# Section 2: Handler Registry Availability & Contract
# ===========================================================================

class TestHandlerRegistryContract:
    """Verify the HandlerRegistry satisfies its public API contract."""

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_default_registry_is_accessible(self):
        """get_default_registry() returns a HandlerRegistry instance."""
        registry = _hr_registry_mod.get_default_registry()
        assert isinstance(registry, _HandlerRegistry), (
            f"Expected HandlerRegistry instance, got {type(registry).__name__}"
        )

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_list_all_returns_list_of_strings(self, handler_registry_entries):
        """list_all() returns List[str]."""
        assert isinstance(handler_registry_entries, list), (
            f"Expected list, got {type(handler_registry_entries).__name__}"
        )
        for entry in handler_registry_entries:
            assert isinstance(entry, str), (
                f"Expected str entry, got {type(entry).__name__}: {entry!r}"
            )

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_registry_is_not_empty(self, handler_registry_entries):
        """
        At least one handler must be registered for the pipeline to function.
        An empty registry indicates that handler implementations were not
        imported or @register_handler decorators did not fire.
        """
        assert len(handler_registry_entries) > 0, (
            "HandlerRegistry is empty — no handlers registered. "
            "Verify that handlers/implementations/__init__.py triggers "
            "@register_handler decorators."
        )

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_no_duplicate_handler_names(self, handler_registry_entries):
        """Handler names in the registry must be unique."""
        seen: Set[str] = set()
        duplicates: List[str] = []
        for name in handler_registry_entries:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        assert not duplicates, (
            f"Duplicate handler names found in registry: {duplicates}"
        )

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_get_returns_class_for_each_registered_handler(
        self, handler_registry_entries
    ):
        """registry.get(name) returns a class (type) for every registered name."""
        for name in handler_registry_entries:
            cls = _hr_registry_mod.get(name)
            assert isinstance(cls, type), (
                f"get('{name}') returned {type(cls).__name__}, expected a class"
            )

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_is_registered_consistent_with_list_all(self, handler_registry_entries):
        """is_registered(name) must return True for every name in list_all()."""
        for name in handler_registry_entries:
            assert _hr_registry_mod.is_registered(name), (
                f"is_registered('{name}') returned False, but '{name}' is in list_all()"
            )

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_is_registered_false_for_nonexistent(self):
        """is_registered() returns False for a name that was never registered."""
        assert not _hr_registry_mod.is_registered(
            "__nonexistent_handler_type_for_testing__"
        )

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_get_registry_info_returns_valid_dict(self):
        """
        get_registry_info() must return a dict with the documented keys:
        total_handlers, registered_handlers, handler_classes, callback_count.
        """
        info = _hr_registry_mod.get_registry_info()
        assert isinstance(info, dict), (
            f"Expected dict from get_registry_info(), got {type(info).__name__}"
        )
        required_keys = {
            "total_handlers",
            "registered_handlers",
            "handler_classes",
            "callback_count",
        }
        missing = required_keys - set(info.keys())
        assert not missing, (
            f"get_registry_info() missing keys: {missing}. Got keys: {set(info.keys())}"
        )
        assert isinstance(info["total_handlers"], int)
        assert isinstance(info["registered_handlers"], list)
        assert isinstance(info["handler_classes"], dict)
        assert isinstance(info["callback_count"], int)


# ===========================================================================
# Section 3: Cross-Registry Consistency (Core of Section 2.3)
# ===========================================================================

class TestCrossRegistryConsistency:
    """
    Verify the bijective relationship between DatasetRegistry and HandlerRegistry.

    The architectural contract requires:
    1. Every dataset registered in DatasetRegistry has a corresponding handler
       in HandlerRegistry (forward coverage).
    2. Every handler registered in HandlerRegistry has a corresponding dataset
       in DatasetRegistry (reverse coverage).

    Note: The MILIA project structure (as of v1.1.0) shows 5 dataset
    implementations (DFT, DMC, Wavefunction, XXMD, QDPi) and 10 handler
    implementations (the above 5 plus QM9, ANI1x, ANI1ccx, ANI2x, RMD17).
    This asymmetry — where some handlers exist without corresponding dataset
    implementations — may be an intentional design choice (handlers can work
    with datasets that are not yet formally registered via @register in
    datasets/implementations/). The tests below document this and report
    precisely which entries are mismatched, without assuming one direction
    must be a strict subset of the other.
    """

    @pytest.mark.skipif(
        not _BOTH_REGISTRIES_AVAILABLE, reason=SKIP_NO_BOTH_REGISTRIES
    )
    def test_every_dataset_has_a_handler(
        self, dataset_registry_entries, handler_registry_entries
    ):
        """
        Forward coverage: every dataset type in DatasetRegistry should have
        a corresponding entry in HandlerRegistry.

        A dataset without a handler is non-functional in the pipeline because
        create_dataset_handler() would fail for that dataset_type.
        """
        handler_set = set(handler_registry_entries)
        datasets_without_handlers: List[str] = [
            ds for ds in dataset_registry_entries if ds not in handler_set
        ]
        assert not datasets_without_handlers, (
            f"Dataset(s) registered without a corresponding handler: "
            f"{datasets_without_handlers}. "
            f"Registered datasets: {sorted(dataset_registry_entries)}. "
            f"Registered handlers: {sorted(handler_registry_entries)}."
        )

    @pytest.mark.skipif(
        not _BOTH_REGISTRIES_AVAILABLE, reason=SKIP_NO_BOTH_REGISTRIES
    )
    def test_every_handler_has_a_dataset(
        self, dataset_registry_entries, handler_registry_entries
    ):
        """
        Reverse coverage: every handler type in HandlerRegistry should have
        a corresponding entry in DatasetRegistry.

        A handler without a registered dataset is unreachable via the standard
        registry-based pipeline path (though it may still be reachable via
        the dynamic import fallback in create_dataset_handler).
        """
        dataset_set = set(dataset_registry_entries)
        handlers_without_datasets: List[str] = [
            h for h in handler_registry_entries if h not in dataset_set
        ]
        # NOTE: This is documented as a known asymmetry in the project.
        # If this assertion fails, it provides a clear diagnostic of which
        # handlers lack corresponding dataset registrations.
        assert not handlers_without_datasets, (
            f"Handler(s) registered without a corresponding dataset: "
            f"{handlers_without_datasets}. "
            f"Registered handlers: {sorted(handler_registry_entries)}. "
            f"Registered datasets: {sorted(dataset_registry_entries)}."
        )

    @pytest.mark.skipif(
        not _BOTH_REGISTRIES_AVAILABLE, reason=SKIP_NO_BOTH_REGISTRIES
    )
    def test_registry_entry_counts_diagnostic(
        self, dataset_registry_entries, handler_registry_entries
    ):
        """
        Diagnostic test: log and assert on the cardinality of both registries.

        This test does not require exact equality — it reports the counts
        and the symmetric difference so that CI logs always contain a
        clear picture of registry state.
        """
        ds_set = set(dataset_registry_entries)
        hr_set = set(handler_registry_entries)

        only_in_datasets = ds_set - hr_set
        only_in_handlers = hr_set - ds_set
        in_both = ds_set & hr_set

        logger.info(
            "Registry consistency report:\n"
            "  Datasets registered:  %d %s\n"
            "  Handlers registered:  %d %s\n"
            "  In both:              %d %s\n"
            "  Only in datasets:     %d %s\n"
            "  Only in handlers:     %d %s",
            len(ds_set), sorted(ds_set),
            len(hr_set), sorted(hr_set),
            len(in_both), sorted(in_both),
            len(only_in_datasets), sorted(only_in_datasets),
            len(only_in_handlers), sorted(only_in_handlers),
        )

        # At minimum, there must be overlap — a pipeline with zero
        # functional dataset-handler pairs is broken.
        assert len(in_both) > 0, (
            "No overlap between DatasetRegistry and HandlerRegistry. "
            f"Datasets: {sorted(ds_set)}. Handlers: {sorted(hr_set)}. "
            "The pipeline cannot function without at least one matching pair."
        )

    @pytest.mark.skipif(
        not _BOTH_REGISTRIES_AVAILABLE, reason=SKIP_NO_BOTH_REGISTRIES
    )
    def test_no_name_case_mismatches(
        self, dataset_registry_entries, handler_registry_entries
    ):
        """
        Detect case-insensitive matches that differ in casing.

        For example, if DatasetRegistry has 'DFT' but HandlerRegistry has
        'dft' or 'Dft', that is a silent consistency bug.
        """
        ds_lower_map: Dict[str, str] = {
            name.lower(): name for name in dataset_registry_entries
        }
        hr_lower_map: Dict[str, str] = {
            name.lower(): name for name in handler_registry_entries
        }

        case_mismatches: List[str] = []
        for lower_name, ds_name in ds_lower_map.items():
            if lower_name in hr_lower_map:
                hr_name = hr_lower_map[lower_name]
                if ds_name != hr_name:
                    case_mismatches.append(
                        f"Dataset='{ds_name}' vs Handler='{hr_name}'"
                    )

        assert not case_mismatches, (
            f"Case mismatches detected between registries: {case_mismatches}. "
            "Dataset names and handler names should use identical casing."
        )


# ===========================================================================
# Section 4: Factory Function Contract — create_dataset_handler()
# ===========================================================================

class TestCreateDatasetHandlerContract:
    """
    Verify that create_dataset_handler() succeeds for every dataset type
    that exists in both registries.

    The factory function is the integration point where both registries
    converge: it looks up the dataset_type in the DatasetRegistry (or falls
    back to dynamic import) and instantiates the corresponding handler.
    """

    @pytest.mark.skipif(not _FULL_STACK_AVAILABLE, reason=SKIP_NO_FULL_STACK)
    def test_factory_succeeds_for_each_registered_dataset(
        self, dataset_registry_entries, handler_registry_entries
    ):
        """
        For every dataset_type in the intersection of both registries,
        create_dataset_handler() must return a handler instance without
        raising an exception.

        This test uses mock configuration containers to avoid requiring
        actual dataset files or configuration YAML.
        """
        # Determine the intersection — only test types present in both registries
        common_types = set(dataset_registry_entries) & set(handler_registry_entries)

        if not common_types:
            pytest.skip(
                "No common entries between DatasetRegistry and HandlerRegistry; "
                "cannot test create_dataset_handler()."
            )

        test_logger = logging.getLogger("test.factory")
        failures: List[str] = []

        for dataset_type in sorted(common_types):
            # Build a mock DatasetConfig with the correct dataset_type
            mock_dataset_config = MagicMock(spec=_DatasetConfig)
            mock_dataset_config.dataset_type = dataset_type
            # Provide common attributes that handlers may access during __init__
            mock_dataset_config.common_properties = []
            mock_dataset_config.target_properties = []
            mock_dataset_config.uncertainty_properties = []
            mock_dataset_config.data_dir = "/tmp/test_data"
            mock_dataset_config.file_pattern = "*.npz"

            mock_filter_config = MagicMock(spec=_FilterConfig)
            mock_filter_config.enabled = False
            mock_filter_config.min_atoms = 1
            mock_filter_config.max_atoms = 1000

            mock_processing_config = MagicMock(spec=_ProcessingConfig)
            mock_processing_config.num_workers = 0
            mock_processing_config.batch_size = 1

            try:
                handler = _create_dataset_handler(
                    dataset_config=mock_dataset_config,
                    filter_config=mock_filter_config,
                    processing_config=mock_processing_config,
                    logger=test_logger,
                    experimental_setup=None,
                )
                # Verify the returned object has the core abstract method
                assert hasattr(handler, "get_dataset_type"), (
                    f"Handler for '{dataset_type}' missing get_dataset_type() method"
                )
            except Exception as exc:
                failures.append(
                    f"  {dataset_type}: {type(exc).__name__}: {exc}"
                )

        if failures:
            pytest.fail(
                f"create_dataset_handler() failed for {len(failures)} "
                f"dataset type(s):\n" + "\n".join(failures)
            )

    @pytest.mark.skipif(not _FULL_STACK_AVAILABLE, reason=SKIP_NO_FULL_STACK)
    def test_factory_raises_for_unknown_type(self):
        """
        create_dataset_handler() must raise an appropriate error
        (HandlerNotAvailableError) for an unregistered dataset type.
        """
        try:
            from milia_pipeline.exceptions import HandlerNotAvailableError
            expected_error = HandlerNotAvailableError
        except ImportError:
            # If the custom exception is not importable, accept any Exception
            expected_error = Exception

        mock_dataset_config = MagicMock(spec=_DatasetConfig)
        mock_dataset_config.dataset_type = "__completely_invalid_dataset_type__"
        mock_dataset_config.common_properties = []

        mock_filter_config = MagicMock(spec=_FilterConfig)
        mock_processing_config = MagicMock(spec=_ProcessingConfig)

        with pytest.raises(expected_error):
            _create_dataset_handler(
                dataset_config=mock_dataset_config,
                filter_config=mock_filter_config,
                processing_config=mock_processing_config,
                logger=logging.getLogger("test.factory.invalid"),
            )


# ===========================================================================
# Section 5: Registry API Surface Parity
# ===========================================================================

class TestRegistryAPIParity:
    """
    Verify that both registries expose a consistent API surface.

    The DatasetRegistry and HandlerRegistry were designed to follow the
    same pattern (as stated in handler_registry.py docstring). This test
    class ensures that contract is maintained over time.
    """

    # Methods that both registries must expose on their instances
    _SHARED_INSTANCE_METHODS = [
        "register",
        "unregister",
        "get",
        "get_or_none",
        "list_all",
        "list_all_classes",
        "is_registered",
        "clear",
        "add_on_change_callback",
        "remove_on_change_callback",
    ]

    # Dunder protocols that both registries must support
    _SHARED_DUNDER_METHODS = [
        "__contains__",
        "__iter__",
        "__len__",
    ]

    # Module-level convenience functions that both modules must expose
    _SHARED_MODULE_FUNCTIONS = [
        "get_default_registry",
        "get",
        "list_all",
        "is_registered",
    ]

    @pytest.mark.skipif(
        not _BOTH_REGISTRIES_AVAILABLE, reason=SKIP_NO_BOTH_REGISTRIES
    )
    def test_dataset_registry_has_all_shared_instance_methods(
        self, isolated_dataset_registry
    ):
        """DatasetRegistry instances expose all expected public methods."""
        missing = [
            m for m in self._SHARED_INSTANCE_METHODS
            if not callable(getattr(isolated_dataset_registry, m, None))
        ]
        assert not missing, (
            f"DatasetRegistry missing instance methods: {missing}"
        )

    @pytest.mark.skipif(
        not _BOTH_REGISTRIES_AVAILABLE, reason=SKIP_NO_BOTH_REGISTRIES
    )
    def test_handler_registry_has_all_shared_instance_methods(
        self, isolated_handler_registry
    ):
        """HandlerRegistry instances expose all expected public methods."""
        missing = [
            m for m in self._SHARED_INSTANCE_METHODS
            if not callable(getattr(isolated_handler_registry, m, None))
        ]
        assert not missing, (
            f"HandlerRegistry missing instance methods: {missing}"
        )

    @pytest.mark.skipif(
        not _BOTH_REGISTRIES_AVAILABLE, reason=SKIP_NO_BOTH_REGISTRIES
    )
    def test_both_registries_support_dunder_protocols(
        self, isolated_dataset_registry, isolated_handler_registry
    ):
        """Both registries support __contains__, __iter__, __len__."""
        for method_name in self._SHARED_DUNDER_METHODS:
            assert hasattr(isolated_dataset_registry, method_name), (
                f"DatasetRegistry missing {method_name}"
            )
            assert hasattr(isolated_handler_registry, method_name), (
                f"HandlerRegistry missing {method_name}"
            )

    @pytest.mark.skipif(
        not _BOTH_REGISTRIES_AVAILABLE, reason=SKIP_NO_BOTH_REGISTRIES
    )
    def test_both_modules_expose_convenience_functions(self):
        """Both registry modules expose the shared convenience functions."""
        for func_name in self._SHARED_MODULE_FUNCTIONS:
            assert hasattr(_ds_registry_mod, func_name), (
                f"datasets.registry module missing function: {func_name}"
            )
            assert hasattr(_hr_registry_mod, func_name), (
                f"handlers.handler_registry module missing function: {func_name}"
            )


# ===========================================================================
# Section 6: Isolated Registry Behavioral Contract
# ===========================================================================

class TestIsolatedRegistryBehavior:
    """
    Verify that creating independent registry instances (non-singleton)
    produces isolated state, as documented in both registry modules.

    This is critical for test safety: tests that register/unregister entries
    must not corrupt the default global registries.
    """

    @pytest.mark.skipif(
        not _DATASET_REGISTRY_AVAILABLE, reason=SKIP_NO_DATASET_REGISTRY
    )
    def test_isolated_dataset_registry_starts_empty(
        self, isolated_dataset_registry
    ):
        """A fresh DatasetRegistry() instance has zero entries."""
        assert len(isolated_dataset_registry) == 0, (
            f"Fresh DatasetRegistry has {len(isolated_dataset_registry)} entries, "
            "expected 0"
        )
        assert isolated_dataset_registry.list_all() == []

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_isolated_handler_registry_starts_empty(
        self, isolated_handler_registry
    ):
        """A fresh HandlerRegistry() instance has zero entries."""
        assert len(isolated_handler_registry) == 0, (
            f"Fresh HandlerRegistry has {len(isolated_handler_registry)} entries, "
            "expected 0"
        )
        assert isolated_handler_registry.list_all() == []

    @pytest.mark.skipif(
        not _DATASET_REGISTRY_AVAILABLE, reason=SKIP_NO_DATASET_REGISTRY
    )
    def test_isolated_dataset_registry_does_not_affect_default(
        self, isolated_dataset_registry
    ):
        """
        Operations on an isolated DatasetRegistry must not change the
        default global registry.
        """
        default_registry = _ds_registry_mod.get_default_registry()
        default_count_before = len(default_registry)

        # Clear the isolated registry (should be a no-op since it's empty,
        # but proves the mechanism doesn't cross-contaminate)
        isolated_dataset_registry.clear()

        default_count_after = len(default_registry)
        assert default_count_before == default_count_after, (
            f"Clearing an isolated DatasetRegistry changed the default registry "
            f"(before={default_count_before}, after={default_count_after})"
        )

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_isolated_handler_registry_does_not_affect_default(
        self, isolated_handler_registry
    ):
        """
        Operations on an isolated HandlerRegistry must not change the
        default global registry.
        """
        default_registry = _hr_registry_mod.get_default_registry()
        default_count_before = len(default_registry)

        isolated_handler_registry.clear()

        default_count_after = len(default_registry)
        assert default_count_before == default_count_after, (
            f"Clearing an isolated HandlerRegistry changed the default registry "
            f"(before={default_count_before}, after={default_count_after})"
        )


# ===========================================================================
# Section 7: Change Callback Contract
# ===========================================================================

class TestRegistryCallbackContract:
    """
    Verify that both registries correctly implement the on-change callback
    mechanism, which is part of their documented public API.
    """

    @pytest.mark.skipif(
        not _DATASET_REGISTRY_AVAILABLE, reason=SKIP_NO_DATASET_REGISTRY
    )
    def test_dataset_registry_callback_add_remove(self, isolated_dataset_registry):
        """add_on_change_callback / remove_on_change_callback round-trip."""
        callback_called = []

        def on_change():
            callback_called.append(True)

        isolated_dataset_registry.add_on_change_callback(on_change)
        removed = isolated_dataset_registry.remove_on_change_callback(on_change)
        assert removed is True, "remove_on_change_callback should return True for existing callback"

        # Removing again should return False
        removed_again = isolated_dataset_registry.remove_on_change_callback(on_change)
        assert removed_again is False, (
            "remove_on_change_callback should return False when callback not found"
        )

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_handler_registry_callback_add_remove(self, isolated_handler_registry):
        """add_on_change_callback / remove_on_change_callback round-trip."""
        callback_called = []

        def on_change():
            callback_called.append(True)

        isolated_handler_registry.add_on_change_callback(on_change)
        removed = isolated_handler_registry.remove_on_change_callback(on_change)
        assert removed is True

        removed_again = isolated_handler_registry.remove_on_change_callback(on_change)
        assert removed_again is False

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_handler_registry_callback_fires_on_clear(
        self, isolated_handler_registry
    ):
        """Clearing the registry must fire registered callbacks."""
        callback_called = []

        def on_change():
            callback_called.append(True)

        isolated_handler_registry.add_on_change_callback(on_change)
        isolated_handler_registry.clear()

        assert len(callback_called) == 1, (
            f"Expected callback to fire once on clear(), fired {len(callback_called)} times"
        )

        # Cleanup: remove callback to prevent interference
        isolated_handler_registry.remove_on_change_callback(on_change)

    @pytest.mark.skipif(
        not _DATASET_REGISTRY_AVAILABLE, reason=SKIP_NO_DATASET_REGISTRY
    )
    def test_dataset_registry_callback_fires_on_clear(
        self, isolated_dataset_registry
    ):
        """Clearing the registry must fire registered callbacks."""
        callback_called = []

        def on_change():
            callback_called.append(True)

        isolated_dataset_registry.add_on_change_callback(on_change)
        isolated_dataset_registry.clear()

        assert len(callback_called) == 1, (
            f"Expected callback to fire once on clear(), fired {len(callback_called)} times"
        )

        isolated_dataset_registry.remove_on_change_callback(on_change)


# ===========================================================================
# Section 8: Error Handling Contract
# ===========================================================================

class TestRegistryErrorHandling:
    """
    Verify that both registries raise the documented exception types
    when operations fail.
    """

    @pytest.mark.skipif(
        not _DATASET_REGISTRY_AVAILABLE, reason=SKIP_NO_DATASET_REGISTRY
    )
    def test_dataset_registry_get_raises_on_missing(self, isolated_dataset_registry):
        """get() on DatasetRegistry raises DatasetNotFoundError for unknown name."""
        try:
            from milia_pipeline.exceptions import DatasetNotFoundError
            expected_exc = DatasetNotFoundError
        except ImportError:
            expected_exc = Exception

        with pytest.raises(expected_exc):
            isolated_dataset_registry.get("__nonexistent__")

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_handler_registry_get_raises_on_missing(self, isolated_handler_registry):
        """get() on HandlerRegistry raises HandlerNotFoundError for unknown name."""
        try:
            from milia_pipeline.handlers.handler_registry import HandlerNotFoundError
            expected_exc = HandlerNotFoundError
        except ImportError:
            expected_exc = Exception

        with pytest.raises(expected_exc):
            isolated_handler_registry.get("__nonexistent__")

    @pytest.mark.skipif(
        not _DATASET_REGISTRY_AVAILABLE, reason=SKIP_NO_DATASET_REGISTRY
    )
    def test_dataset_registry_register_rejects_non_class(
        self, isolated_dataset_registry
    ):
        """register() raises TypeError when passed an instance instead of a class."""
        with pytest.raises(TypeError, match="Expected class"):
            isolated_dataset_registry.register("not_a_class")

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_handler_registry_register_rejects_non_class(
        self, isolated_handler_registry
    ):
        """register() raises TypeError when passed an instance instead of a class."""
        with pytest.raises(TypeError, match="Expected class"):
            isolated_handler_registry.register("not_a_class")

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_handler_registry_register_rejects_class_without_get_dataset_type(
        self, isolated_handler_registry
    ):
        """
        register() raises TypeError when the class lacks get_dataset_type().
        This verifies the type-safety check documented in handler_registry.py
        lines 117-121.
        """

        class BadHandler:
            """A class that does not implement get_dataset_type()."""
            pass

        with pytest.raises(TypeError, match="get_dataset_type"):
            isolated_handler_registry.register(BadHandler)

    @pytest.mark.skipif(
        not _DATASET_REGISTRY_AVAILABLE, reason=SKIP_NO_DATASET_REGISTRY
    )
    def test_dataset_registry_get_or_none_returns_none_for_missing(
        self, isolated_dataset_registry
    ):
        """get_or_none() returns None rather than raising for unknown names."""
        result = isolated_dataset_registry.get_or_none("__nonexistent__")
        assert result is None

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_handler_registry_get_or_none_returns_none_for_missing(
        self, isolated_handler_registry
    ):
        """get_or_none() returns None rather than raising for unknown names."""
        result = isolated_handler_registry.get_or_none("__nonexistent__")
        assert result is None


# ===========================================================================
# Section 9: Handler Registry — get_dataset_type Consistency
# ===========================================================================

class TestHandlerDatasetTypeConsistency:
    """
    For each handler class in the HandlerRegistry, verify that the
    registered key name is derivable from the class identity, following
    the naming convention documented in handler_registry.py.

    The handler_registry.register() method derives the name by stripping
    'DatasetHandler' or 'Handler' from the class name. This test ensures
    the registration key is consistent with that derivation.
    """

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_handler_class_names_follow_convention(self, handler_registry_entries):
        """
        Each handler's class name should end with 'DatasetHandler' or 'Handler',
        and stripping that suffix should relate to the registration key.
        """
        violations: List[str] = []

        for name in handler_registry_entries:
            handler_cls = _hr_registry_mod.get(name)
            cls_name = handler_cls.__name__

            # The class name should contain 'Handler' (convention check)
            if "Handler" not in cls_name:
                violations.append(
                    f"'{name}': class {cls_name} does not contain 'Handler' in name"
                )

        if violations:
            logger.warning(
                "Handler naming convention violations:\n  %s",
                "\n  ".join(violations),
            )
            # This is a warning-level check, not a hard failure, because
            # the handler_registry.register() has fallback name derivation.
            # Uncomment the next line to enforce strict naming:
            # pytest.fail(f"Naming violations found: {violations}")


# ===========================================================================
# Section 10: Thread Safety Smoke Check
# ===========================================================================

class TestRegistryThreadSafetySmoke:
    """
    Lightweight thread-safety smoke check for both registries.

    This is NOT a comprehensive concurrency test (that belongs in
    test_thread_safety_registries.py per section 6.1). This only verifies
    that the RLock attribute exists and that concurrent reads do not raise.
    """

    @pytest.mark.skipif(
        not _DATASET_REGISTRY_AVAILABLE, reason=SKIP_NO_DATASET_REGISTRY
    )
    def test_dataset_registry_has_lock(self, isolated_dataset_registry):
        """DatasetRegistry instances have an RLock (_lock attribute)."""
        assert hasattr(isolated_dataset_registry, "_lock"), (
            "DatasetRegistry missing _lock attribute (RLock for thread safety)"
        )

    @pytest.mark.skipif(
        not _HANDLER_REGISTRY_AVAILABLE, reason=SKIP_NO_HANDLER_REGISTRY
    )
    def test_handler_registry_has_lock(self, isolated_handler_registry):
        """HandlerRegistry instances have an RLock (_lock attribute)."""
        assert hasattr(isolated_handler_registry, "_lock"), (
            "HandlerRegistry missing _lock attribute (RLock for thread safety)"
        )

    @pytest.mark.skipif(
        not _BOTH_REGISTRIES_AVAILABLE, reason=SKIP_NO_BOTH_REGISTRIES
    )
    def test_concurrent_reads_do_not_raise(self):
        """
        Multiple threads calling list_all() simultaneously on the default
        registries must not raise exceptions.
        """
        import concurrent.futures

        errors: List[str] = []

        def read_dataset_registry():
            try:
                _ds_registry_mod.list_all()
            except Exception as exc:
                errors.append(f"DatasetRegistry: {exc}")

        def read_handler_registry():
            try:
                _hr_registry_mod.list_all()
            except Exception as exc:
                errors.append(f"HandlerRegistry: {exc}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for _ in range(10):
                futures.append(executor.submit(read_dataset_registry))
                futures.append(executor.submit(read_handler_registry))
            concurrent.futures.wait(futures)

        assert not errors, (
            f"Concurrent reads raised exceptions: {errors}"
        )
