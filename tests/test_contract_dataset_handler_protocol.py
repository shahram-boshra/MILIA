# tests/test_contract_dataset_handler_protocol.py

"""
Contract Tests — DatasetHandlerProtocol
========================================

Verifies that every registered handler implementation satisfies all 11 methods
of `DatasetHandlerProtocol` with correct return types.

Test Scope (from MILIA_Test_Recommendations.md §2.1):
- Iterates over all 10 registered handlers
- For each handler, verifies:
    1. Method existence (all 11 protocol methods present)
    2. Return type correctness (e.g., get_dataset_type() -> str)
    3. runtime_checkable protocol conformance via isinstance()
- Uses real handler classes with mocked config dependencies

Modules Exercised:
- milia_pipeline/datasets/protocols.py — DatasetHandlerProtocol (11 methods)
- milia_pipeline/handlers/handler_registry.py — HandlerRegistry.list_all()
- milia_pipeline/handlers/implementations/*.py — All 10 handler implementations

Design Decisions:
- No sys.modules pollution: all mocking is test-scoped via @patch or fixtures
- Handlers are instantiated with minimal mock configs to exercise real method logic
- Parameterized tests ensure coverage of every handler uniformly
- Separated into: structural (method existence), behavioral (return types),
  protocol (isinstance), and cross-handler consistency tests

References:
- MILIA_Pipeline_Project_Structure.md: handler module (§5), DatasetHandler ABC
  constructor: __init__(dataset_config, filter_config, processing_config, logger,
  experimental_setup=None), 12 abstract methods
- protocols.py: DatasetHandlerProtocol with 11 @runtime_checkable methods
- handler_registry.py: HandlerRegistry, @register_handler, get_default_registry()
- All 10 handler implementations: DFT, DMC, Wavefunction, QM9, ANI1x, ANI1ccx,
  ANI2x, RMD17, XXMD, QDPi

Execution:
    pytest tests/test_contract_dataset_handler_protocol.py -v
    pytest tests/test_contract_dataset_handler_protocol.py -v -m contract
"""

import sys
import os
import inspect
import logging
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Imports from MILIA project
# ---------------------------------------------------------------------------
from milia_pipeline.datasets.protocols import (
    DatasetHandlerProtocol,
    DatasetConverterProtocol,
    DatasetValidatorProtocol,
)
from milia_pipeline.handlers.handler_registry import (
    HandlerRegistry,
    get_default_registry,
    register_handler,
    HandlerRegistrationError,
    HandlerNotFoundError,
)

# Import all handler implementations to trigger @register_handler decorators.
# The implementations/__init__.py performs dynamic discovery, but explicit
# import guarantees all handlers are registered before parameterisation.
import milia_pipeline.handlers.implementations  # noqa: F401


# ---------------------------------------------------------------------------
# Constants derived from evidence
# ---------------------------------------------------------------------------

# All 10 handler classes and their expected dataset_type strings.
# Source: grep of @register_handler + get_dataset_type() across all 10 files.
EXPECTED_HANDLERS: Dict[str, str] = {
    "DFT": "DFTDatasetHandler",
    "DMC": "DMCDatasetHandler",
    "Wavefunction": "WavefunctionDatasetHandler",
    "QM9": "QM9DatasetHandler",
    "ANI1x": "ANI1xDatasetHandler",
    "ANI1ccx": "ANI1ccxDatasetHandler",
    "ANI2x": "ANI2xDatasetHandler",
    "RMD17": "RMD17DatasetHandler",
    "XXMD": "XXMDDatasetHandler",
    "QDPi": "QDPiDatasetHandler",
}

# The 11 protocol methods from DatasetHandlerProtocol (protocols.py).
PROTOCOL_METHODS: List[str] = [
    "get_dataset_type",
    "validate_molecule_data",
    "get_required_properties",
    "process_property_value",
    "enrich_pyg_data",
    "get_processing_statistics",
    "get_supported_structural_features",
    "get_molecular_charge",
    "get_molecule_creation_strategy",
    "get_transform_recommendations",
    "get_supported_descriptors",
]

# Valid molecule creation strategies from project structure documentation.
# Source: MILIA_Pipeline_Project_Structure.md line 2945.
VALID_CREATION_STRATEGIES = {"identifier_coordinate_based", "coordinate_based"}

# Pytest marker for selective CI execution.
pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_processing_config():
    """
    Create a minimal mock ProcessingConfig matching attributes accessed by handlers.

    Evidence (from dft.py, dmc.py, qm9.py, etc.):
    - self.processing_config.scalar_graph_targets  (list)
    - self.processing_config.node_features          (list)
    - self.processing_config.vector_graph_properties (list)
    - self.processing_config.variable_len_graph_properties (list)
    - self.processing_config.calculate_atomization_energy_from (str or None)
    - self.processing_config.atomization_energy_key_name (str or None)
    - self.processing_config.vibration_refinement   (dict or None)
    """
    pc = MagicMock(name="ProcessingConfig")
    pc.scalar_graph_targets = []
    pc.node_features = []
    pc.vector_graph_properties = []
    pc.variable_len_graph_properties = []
    pc.calculate_atomization_energy_from = None
    pc.atomization_energy_key_name = None
    pc.vibration_refinement = None
    return pc


def _make_mock_dataset_config(dataset_type: str = "DFT"):
    """
    Create a minimal mock DatasetConfig matching attributes accessed by handlers.

    Evidence (from dmc.py):
    - self.dataset_config.is_uncertainty_enabled  (bool)
    - self.dataset_config.uncertainty_config       (dict or None)
    """
    dc = MagicMock(name="DatasetConfig")
    dc.dataset_type = dataset_type
    dc.is_uncertainty_enabled = False
    dc.uncertainty_config = None
    dc.handler_config = None
    dc.validation_config = None
    dc.migration_config = None
    return dc


def _make_mock_filter_config():
    """
    Create a minimal mock FilterConfig.

    Evidence (from project structure):
    - FilterConfig fields: max_atoms, min_atoms, heavy_atom_filter, etc.
    """
    fc = MagicMock(name="FilterConfig")
    fc.max_atoms = None
    fc.min_atoms = None
    fc.heavy_atom_filter = None
    fc.dmc_uncertainty_filter = None
    fc.handler_filters = {}
    return fc


@pytest.fixture
def mock_logger():
    """Provide a standard Python logger for handler instantiation."""
    return logging.getLogger("test_contract_handler_protocol")


@pytest.fixture
def mock_configs():
    """
    Bundle of minimal mock configs for constructing any handler.

    Returns:
        tuple: (dataset_config, filter_config, processing_config)
    """
    return (
        _make_mock_dataset_config(),
        _make_mock_filter_config(),
        _make_mock_processing_config(),
    )


@pytest.fixture
def default_registry():
    """
    Return the default global HandlerRegistry (already populated by the
    module-level import of milia_pipeline.handlers.implementations).
    """
    return get_default_registry()


def _instantiate_handler(handler_class, dataset_type: str, mock_logger):
    """
    Instantiate a handler with the correct dataset_type in its DatasetConfig.

    The DatasetHandler ABC constructor signature (from project structure §5):
        __init__(dataset_config, filter_config, processing_config, logger,
                 experimental_setup=None)
    """
    dc = _make_mock_dataset_config(dataset_type)
    fc = _make_mock_filter_config()
    pc = _make_mock_processing_config()
    return handler_class(dc, fc, pc, mock_logger)


@pytest.fixture
def all_handler_instances(default_registry, mock_logger):
    """
    Instantiate every registered handler with minimal mock configs.

    Returns:
        Dict[str, handler_instance]: Mapping from dataset_type to live instance.
    """
    instances = {}
    for name in default_registry.list_all():
        handler_class = default_registry.get(name)
        instances[name] = _instantiate_handler(handler_class, name, mock_logger)
    return instances


# ---------------------------------------------------------------------------
# Helper: Collect handler classes from registry for parametrize
# ---------------------------------------------------------------------------

def _get_registered_handler_names() -> List[str]:
    """
    Collect all handler names from the default registry.

    This is called at module load time for @pytest.mark.parametrize.
    The import of milia_pipeline.handlers.implementations above ensures
    all handlers are registered before this runs.
    """
    registry = get_default_registry()
    return sorted(registry.list_all())


# Dynamic parametrization over all registered handlers.
REGISTERED_HANDLER_NAMES = _get_registered_handler_names()


# ============================================================================
# TEST CLASS 1: Registry Population Verification
# ============================================================================

class TestRegistryPopulation:
    """Verify the handler registry is correctly populated with all 10 handlers."""

    def test_registry_is_not_empty(self, default_registry):
        """The default registry must contain at least one handler."""
        assert len(default_registry) > 0, (
            "HandlerRegistry is empty — handler implementations were not discovered"
        )

    def test_all_expected_handlers_registered(self, default_registry):
        """Every expected handler must be present in the registry."""
        registered = set(default_registry.list_all())
        expected = set(EXPECTED_HANDLERS.keys())
        missing = expected - registered
        assert not missing, (
            f"Expected handlers missing from registry: {missing}. "
            f"Registered: {sorted(registered)}"
        )

    def test_no_unexpected_handlers(self, default_registry):
        """
        Registry should not contain unknown handlers.

        This is a soft check — new handlers are fine, but it flags
        accidental registrations during development.
        """
        registered = set(default_registry.list_all())
        expected = set(EXPECTED_HANDLERS.keys())
        extra = registered - expected
        if extra:
            # Warn rather than fail — extensibility is by design
            pytest.skip(
                f"Registry contains additional handlers not in expected set: {extra}. "
                f"This may be intentional (new handler added)."
            )

    def test_handler_count(self, default_registry):
        """Registry should contain at least 10 handlers (the known set)."""
        count = len(default_registry)
        assert count >= 10, (
            f"Expected at least 10 handlers, found {count}: "
            f"{default_registry.list_all()}"
        )

    def test_registry_info_structure(self, default_registry):
        """get_registry_info() must return a well-formed diagnostic dict."""
        info = default_registry.get_registry_info()
        assert isinstance(info, dict)
        assert "total_handlers" in info
        assert "registered_handlers" in info
        assert "handler_classes" in info
        assert "callback_count" in info
        assert isinstance(info["total_handlers"], int)
        assert isinstance(info["registered_handlers"], list)
        assert isinstance(info["handler_classes"], dict)


# ============================================================================
# TEST CLASS 2: Structural Contract — Method Existence
# ============================================================================

class TestStructuralContract:
    """
    Every handler class must define all 11 protocol methods as callable attributes.

    This is a compile-time-like check: does the class have the right shape?
    """

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_handler_has_all_protocol_methods(self, handler_name, default_registry):
        """Handler class has all 11 DatasetHandlerProtocol methods."""
        handler_class = default_registry.get(handler_name)
        missing = []
        for method_name in PROTOCOL_METHODS:
            if not hasattr(handler_class, method_name):
                missing.append(method_name)
            elif not callable(getattr(handler_class, method_name)):
                missing.append(f"{method_name} (not callable)")
        assert not missing, (
            f"Handler '{handler_name}' ({handler_class.__name__}) missing protocol "
            f"methods: {missing}"
        )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_handler_is_not_abstract(self, handler_name, default_registry):
        """Registered handlers must not have unimplemented abstract methods."""
        handler_class = default_registry.get(handler_name)
        abstract_methods = getattr(handler_class, "__abstractmethods__", frozenset())
        assert not abstract_methods, (
            f"Handler '{handler_name}' has unimplemented abstract methods: "
            f"{abstract_methods}"
        )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_handler_class_naming_convention(self, handler_name, default_registry):
        """Handler classes should end with 'DatasetHandler' or 'Handler'."""
        handler_class = default_registry.get(handler_name)
        class_name = handler_class.__name__
        assert class_name.endswith("Handler"), (
            f"Handler class '{class_name}' does not follow naming convention "
            f"(expected suffix 'Handler' or 'DatasetHandler')"
        )

    @pytest.mark.parametrize("method_name", PROTOCOL_METHODS)
    def test_protocol_method_is_defined_in_protocol(self, method_name):
        """Each expected method must actually exist in the Protocol definition."""
        assert hasattr(DatasetHandlerProtocol, method_name), (
            f"Method '{method_name}' is not defined in DatasetHandlerProtocol"
        )


# ============================================================================
# TEST CLASS 3: Behavioral Contract — Return Types
# ============================================================================

class TestBehavioralContract:
    """
    Call each protocol method on live handler instances and verify return types.

    Each test creates a handler with minimal mock configs, calls a single
    protocol method, and asserts the return type matches the protocol spec.
    """

    # ------------------------------------------------------------------
    # 1. get_dataset_type() -> str
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_dataset_type_returns_str(self, handler_name, default_registry, mock_logger):
        """get_dataset_type() must return a non-empty str."""
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        result = handler.get_dataset_type()
        assert isinstance(result, str), (
            f"{handler_name}.get_dataset_type() returned {type(result)}, expected str"
        )
        assert len(result) > 0, (
            f"{handler_name}.get_dataset_type() returned empty string"
        )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_dataset_type_matches_registry_name(
        self, handler_name, default_registry, mock_logger
    ):
        """
        The value returned by get_dataset_type() should correspond to the
        registry key under which this handler is registered.

        Note: The handler_registry.py register() method derives the name from
        the class name if get_dataset_type() cannot be called as a classmethod.
        We verify that the instance method returns a consistent value.
        """
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        dataset_type = handler.get_dataset_type()
        # The dataset_type should be present in the expected handlers mapping
        if handler_name in EXPECTED_HANDLERS:
            assert dataset_type == handler_name, (
                f"Handler registered as '{handler_name}' but get_dataset_type() "
                f"returns '{dataset_type}'"
            )

    # ------------------------------------------------------------------
    # 2. get_required_properties() -> List[str]
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_required_properties_returns_list_of_str(
        self, handler_name, default_registry, mock_logger
    ):
        """get_required_properties() must return List[str]."""
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        result = handler.get_required_properties()
        assert isinstance(result, list), (
            f"{handler_name}.get_required_properties() returned {type(result)}, "
            f"expected list"
        )
        for item in result:
            assert isinstance(item, str), (
                f"{handler_name}.get_required_properties() contains non-str "
                f"item: {item!r} ({type(item)})"
            )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_required_properties_non_empty(
        self, handler_name, default_registry, mock_logger
    ):
        """Every handler should require at least one property."""
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        result = handler.get_required_properties()
        assert len(result) > 0, (
            f"{handler_name}.get_required_properties() returned empty list — "
            f"every dataset type requires at least some properties"
        )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_required_properties_no_duplicates(
        self, handler_name, default_registry, mock_logger
    ):
        """Required properties list should have no duplicates."""
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        result = handler.get_required_properties()
        duplicates = [p for p in result if result.count(p) > 1]
        assert not duplicates, (
            f"{handler_name}.get_required_properties() contains duplicates: "
            f"{set(duplicates)}"
        )

    # ------------------------------------------------------------------
    # 3. get_molecule_creation_strategy() -> str
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_molecule_creation_strategy_returns_str(
        self, handler_name, default_registry, mock_logger
    ):
        """get_molecule_creation_strategy() must return a str."""
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        result = handler.get_molecule_creation_strategy()
        assert isinstance(result, str), (
            f"{handler_name}.get_molecule_creation_strategy() returned "
            f"{type(result)}, expected str"
        )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_molecule_creation_strategy_valid_value(
        self, handler_name, default_registry, mock_logger
    ):
        """
        Strategy must be one of the known valid values.

        Evidence: MILIA_Pipeline_Project_Structure.md line 2945 documents
        'identifier_coordinate_based' and 'coordinate_based' as the two
        valid strategies.
        """
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        result = handler.get_molecule_creation_strategy()
        assert result in VALID_CREATION_STRATEGIES, (
            f"{handler_name}.get_molecule_creation_strategy() returned "
            f"'{result}', expected one of {VALID_CREATION_STRATEGIES}"
        )

    # ------------------------------------------------------------------
    # 4. get_transform_recommendations() -> Dict[str, List[str]]
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_transform_recommendations_returns_dict(
        self, handler_name, default_registry, mock_logger
    ):
        """get_transform_recommendations() must return Dict[str, List[str]]."""
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        result = handler.get_transform_recommendations()
        assert isinstance(result, dict), (
            f"{handler_name}.get_transform_recommendations() returned "
            f"{type(result)}, expected dict"
        )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_transform_recommendations_has_standard_keys(
        self, handler_name, default_registry, mock_logger
    ):
        """
        Recommendations dict should contain standard keys.

        Evidence: dft.py, qm9.py, qdpi.py all return dicts with keys
        'recommended', 'avoid', 'warnings'.
        """
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        result = handler.get_transform_recommendations()
        expected_keys = {"recommended", "avoid", "warnings"}
        actual_keys = set(result.keys())
        missing_keys = expected_keys - actual_keys
        assert not missing_keys, (
            f"{handler_name}.get_transform_recommendations() missing keys: "
            f"{missing_keys}. Got keys: {actual_keys}"
        )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_transform_recommendations_values_are_lists(
        self, handler_name, default_registry, mock_logger
    ):
        """Each value in the recommendations dict must be a list."""
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        result = handler.get_transform_recommendations()
        for key, value in result.items():
            assert isinstance(value, list), (
                f"{handler_name}.get_transform_recommendations()['{key}'] "
                f"is {type(value)}, expected list"
            )

    # ------------------------------------------------------------------
    # 5. get_supported_structural_features() -> Dict[str, List[str]]
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_supported_structural_features_returns_dict(
        self, handler_name, default_registry, mock_logger
    ):
        """get_supported_structural_features() must return a dict."""
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        result = handler.get_supported_structural_features()
        assert isinstance(result, dict), (
            f"{handler_name}.get_supported_structural_features() returned "
            f"{type(result)}, expected dict"
        )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_supported_structural_features_has_atom_and_bond(
        self, handler_name, default_registry, mock_logger
    ):
        """
        Features dict should contain 'atom' and 'bond' keys.

        Evidence: dft.py and dmc.py both return {'atom': [...], 'bond': [...]}.
        """
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        result = handler.get_supported_structural_features()
        assert "atom" in result, (
            f"{handler_name}.get_supported_structural_features() missing 'atom' key"
        )
        assert "bond" in result, (
            f"{handler_name}.get_supported_structural_features() missing 'bond' key"
        )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_supported_structural_features_values_are_lists_of_str(
        self, handler_name, default_registry, mock_logger
    ):
        """'atom' and 'bond' values must be lists of strings."""
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        result = handler.get_supported_structural_features()
        for key in ("atom", "bond"):
            if key in result:
                assert isinstance(result[key], list), (
                    f"{handler_name} structural features['{key}'] is "
                    f"{type(result[key])}, expected list"
                )
                for item in result[key]:
                    assert isinstance(item, str), (
                        f"{handler_name} structural features['{key}'] contains "
                        f"non-str item: {item!r}"
                    )

    # ------------------------------------------------------------------
    # 6. get_supported_descriptors() -> Dict[str, List[str]]
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_supported_descriptors_returns_dict(
        self, handler_name, default_registry, mock_logger
    ):
        """get_supported_descriptors() must return a dict."""
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        result = handler.get_supported_descriptors()
        assert isinstance(result, dict), (
            f"{handler_name}.get_supported_descriptors() returned "
            f"{type(result)}, expected dict"
        )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_supported_descriptors_has_categories(
        self, handler_name, default_registry, mock_logger
    ):
        """
        Descriptor dict should contain a 'categories' key.

        Evidence: dft.py, qm9.py, qdpi.py all include 'categories' key
        with list of descriptor category names.
        """
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        result = handler.get_supported_descriptors()
        assert "categories" in result, (
            f"{handler_name}.get_supported_descriptors() missing 'categories' key. "
            f"Got keys: {list(result.keys())}"
        )
        assert isinstance(result["categories"], list), (
            f"{handler_name} descriptors['categories'] is "
            f"{type(result['categories'])}, expected list"
        )

    # ------------------------------------------------------------------
    # 7. get_molecular_charge() -> int
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_molecular_charge_returns_int(
        self, handler_name, default_registry, mock_logger
    ):
        """
        get_molecular_charge() must return an int when given minimal valid input.

        Evidence: All handlers have signature
            get_molecular_charge(raw_properties_dict, atomic_numbers, mol_identifier)
        and return int (0 as default for neutral molecules).
        """
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        # Provide minimal inputs — empty dict, small atomic number array, no identifier
        raw_props = {}
        atomic_numbers = np.array([6, 1, 1, 1, 1])  # Methane-like
        result = handler.get_molecular_charge(raw_props, atomic_numbers)
        assert isinstance(result, int), (
            f"{handler_name}.get_molecular_charge() returned {type(result)}, "
            f"expected int"
        )

    # ------------------------------------------------------------------
    # 8. get_processing_statistics() -> Dict[str, Any]
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_processing_statistics_returns_dict(
        self, handler_name, default_registry, mock_logger
    ):
        """get_processing_statistics() must return a dict."""
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        # Pass empty list of processed molecules
        result = handler.get_processing_statistics([])
        assert isinstance(result, dict), (
            f"{handler_name}.get_processing_statistics() returned "
            f"{type(result)}, expected dict"
        )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_processing_statistics_has_dataset_type(
        self, handler_name, default_registry, mock_logger
    ):
        """
        Statistics dict should include 'dataset_type' key.

        Evidence: dft.py get_processing_statistics() sets
            stats['dataset_type'] = 'DFT'
        DMC does similarly.
        """
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        result = handler.get_processing_statistics([])
        assert "dataset_type" in result, (
            f"{handler_name}.get_processing_statistics() missing 'dataset_type' key"
        )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_processing_statistics_has_total_count(
        self, handler_name, default_registry, mock_logger
    ):
        """
        Statistics dict should include a molecule count key.

        Evidence:
        - dft.py sets stats['total_processed'] = len(processed_molecules)
        - wavefunction.py sets stats['total_molecules'] = len(processed_molecules)

        Both naming variants are acceptable; the contract requires that at
        least one total-count key is present and reports 0 for empty input.
        """
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        result = handler.get_processing_statistics([])
        # Accept either naming convention
        count_keys = {"total_processed", "total_molecules"}
        found_keys = count_keys & set(result.keys())
        assert found_keys, (
            f"{handler_name}.get_processing_statistics() missing a total count key. "
            f"Expected one of {count_keys}, got keys: {list(result.keys())}"
        )
        count_key = found_keys.pop()
        assert result[count_key] == 0, (
            f"{handler_name}.get_processing_statistics([]) should report "
            f"{count_key}=0, got {result[count_key]}"
        )


# ============================================================================
# TEST CLASS 4: Protocol Conformance (isinstance checks)
# ============================================================================

class TestProtocolConformance:
    """
    Verify runtime_checkable protocol isinstance() conformance.

    DatasetHandlerProtocol is decorated with @runtime_checkable, enabling
    isinstance() checks at runtime (protocols.py).
    """

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_handler_instance_satisfies_protocol(
        self, handler_name, default_registry, mock_logger
    ):
        """
        isinstance(handler_instance, DatasetHandlerProtocol) must be True.

        This is the core contract test: runtime_checkable verifies that all
        protocol methods exist on the instance.
        """
        handler_class = default_registry.get(handler_name)
        handler = _instantiate_handler(handler_class, handler_name, mock_logger)
        assert isinstance(handler, DatasetHandlerProtocol), (
            f"Handler instance for '{handler_name}' "
            f"({handler_class.__name__}) does NOT satisfy "
            f"DatasetHandlerProtocol — missing method(s)"
        )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_handler_class_satisfies_protocol_structurally(
        self, handler_name, default_registry
    ):
        """
        The handler class itself should have all protocol methods.

        Note: runtime_checkable only checks for method existence, not
        signatures. This test confirms the class (not instance) has the
        right shape.
        """
        handler_class = default_registry.get(handler_name)
        for method_name in PROTOCOL_METHODS:
            assert hasattr(handler_class, method_name), (
                f"Handler class '{handler_class.__name__}' missing "
                f"protocol method '{method_name}'"
            )


# ============================================================================
# TEST CLASS 5: Method Signature Contract
# ============================================================================

class TestMethodSignatures:
    """
    Verify that protocol method signatures match the expected parameter lists.

    Evidence: protocols.py defines exact signatures; handler implementations
    in dft.py, dmc.py, etc. must match.
    """

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_dataset_type_signature(self, handler_name, default_registry):
        """get_dataset_type(self) -> str — no parameters beyond self."""
        handler_class = default_registry.get(handler_name)
        sig = inspect.signature(handler_class.get_dataset_type)
        # Filter out 'self'
        params = [
            p for p in sig.parameters.values()
            if p.name != "self"
        ]
        assert len(params) == 0, (
            f"{handler_name}.get_dataset_type() has unexpected parameters: "
            f"{[p.name for p in params]}"
        )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_validate_molecule_data_signature(self, handler_name, default_registry):
        """
        validate_molecule_data(self, raw_properties_dict, molecule_index, identifier)
        """
        handler_class = default_registry.get(handler_name)
        sig = inspect.signature(handler_class.validate_molecule_data)
        param_names = [
            p.name for p in sig.parameters.values() if p.name != "self"
        ]
        # Must have at least raw_properties_dict and molecule_index
        assert "raw_properties_dict" in param_names, (
            f"{handler_name}.validate_molecule_data() missing "
            f"'raw_properties_dict' parameter"
        )
        assert "molecule_index" in param_names, (
            f"{handler_name}.validate_molecule_data() missing "
            f"'molecule_index' parameter"
        )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_get_molecular_charge_signature(self, handler_name, default_registry):
        """
        get_molecular_charge(self, raw_properties_dict, atomic_numbers, mol_identifier)
        """
        handler_class = default_registry.get(handler_name)
        sig = inspect.signature(handler_class.get_molecular_charge)
        param_names = [
            p.name for p in sig.parameters.values() if p.name != "self"
        ]
        assert "raw_properties_dict" in param_names, (
            f"{handler_name}.get_molecular_charge() missing "
            f"'raw_properties_dict' parameter"
        )
        assert "atomic_numbers" in param_names, (
            f"{handler_name}.get_molecular_charge() missing "
            f"'atomic_numbers' parameter"
        )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_enrich_pyg_data_signature(self, handler_name, default_registry):
        """
        enrich_pyg_data(self, pyg_data, raw_properties_dict, molecule_index, identifier)
        """
        handler_class = default_registry.get(handler_name)
        sig = inspect.signature(handler_class.enrich_pyg_data)
        param_names = [
            p.name for p in sig.parameters.values() if p.name != "self"
        ]
        assert "pyg_data" in param_names, (
            f"{handler_name}.enrich_pyg_data() missing 'pyg_data' parameter"
        )
        assert "raw_properties_dict" in param_names, (
            f"{handler_name}.enrich_pyg_data() missing "
            f"'raw_properties_dict' parameter"
        )

    @pytest.mark.parametrize("handler_name", REGISTERED_HANDLER_NAMES)
    def test_process_property_value_signature(self, handler_name, default_registry):
        """
        process_property_value(self, key, value, molecule_index, identifier)
        """
        handler_class = default_registry.get(handler_name)
        sig = inspect.signature(handler_class.process_property_value)
        param_names = [
            p.name for p in sig.parameters.values() if p.name != "self"
        ]
        assert "key" in param_names, (
            f"{handler_name}.process_property_value() missing 'key' parameter"
        )
        assert "value" in param_names, (
            f"{handler_name}.process_property_value() missing 'value' parameter"
        )


# ============================================================================
# TEST CLASS 6: Cross-Handler Consistency
# ============================================================================

class TestCrossHandlerConsistency:
    """
    Verify that all handlers behave consistently for shared contract properties.
    """

    def test_all_handlers_return_unique_dataset_types(
        self, all_handler_instances
    ):
        """No two handlers should return the same dataset_type string."""
        type_to_handler = {}
        for name, handler in all_handler_instances.items():
            dtype = handler.get_dataset_type()
            if dtype in type_to_handler:
                pytest.fail(
                    f"Duplicate dataset_type '{dtype}' returned by both "
                    f"'{type_to_handler[dtype]}' and '{name}'"
                )
            type_to_handler[dtype] = name

    def test_all_handlers_have_non_empty_structural_features(
        self, all_handler_instances
    ):
        """Every handler must support at least some structural features."""
        for name, handler in all_handler_instances.items():
            features = handler.get_supported_structural_features()
            atom_features = features.get("atom", [])
            bond_features = features.get("bond", [])
            total = len(atom_features) + len(bond_features)
            assert total > 0, (
                f"Handler '{name}' has zero supported structural features"
            )

    def test_all_handlers_have_at_least_one_descriptor_category(
        self, all_handler_instances
    ):
        """Every handler should support at least one descriptor category."""
        for name, handler in all_handler_instances.items():
            descriptors = handler.get_supported_descriptors()
            categories = descriptors.get("categories", [])
            assert len(categories) > 0, (
                f"Handler '{name}' has zero descriptor categories"
            )

    def test_creation_strategy_consistency_within_handler(
        self, all_handler_instances
    ):
        """Calling get_molecule_creation_strategy() twice returns same value."""
        for name, handler in all_handler_instances.items():
            first_call = handler.get_molecule_creation_strategy()
            second_call = handler.get_molecule_creation_strategy()
            assert first_call == second_call, (
                f"Handler '{name}' returned inconsistent creation strategies: "
                f"'{first_call}' vs '{second_call}'"
            )

    def test_molecular_charge_default_for_empty_data(
        self, all_handler_instances
    ):
        """
        Given empty raw_properties_dict, all handlers should return an int
        without raising an exception. Most handlers default to 0 (neutral).
        """
        atomic_numbers = np.array([6, 1, 1, 1, 1])
        for name, handler in all_handler_instances.items():
            try:
                charge = handler.get_molecular_charge({}, atomic_numbers)
                assert isinstance(charge, int), (
                    f"Handler '{name}' returned non-int charge: "
                    f"{type(charge)}"
                )
            except Exception as e:
                pytest.fail(
                    f"Handler '{name}' raised {type(e).__name__} when given "
                    f"empty raw_properties_dict: {e}"
                )

    def test_processing_statistics_with_sample_data(
        self, all_handler_instances
    ):
        """
        get_processing_statistics() should handle a list with one minimal
        molecule dict without raising an exception.
        """
        sample_molecules = [{"molecule_index": 0, "identifier": "test"}]
        count_keys = {"total_processed", "total_molecules"}
        for name, handler in all_handler_instances.items():
            try:
                stats = handler.get_processing_statistics(sample_molecules)
                assert isinstance(stats, dict), (
                    f"Handler '{name}' returned non-dict stats: {type(stats)}"
                )
                found_keys = count_keys & set(stats.keys())
                assert found_keys, (
                    f"Handler '{name}' missing total count key. "
                    f"Expected one of {count_keys}, got: {list(stats.keys())}"
                )
                count_key = found_keys.pop()
                assert stats[count_key] == 1, (
                    f"Handler '{name}' reported wrong {count_key}: "
                    f"{stats[count_key]}"
                )
            except Exception as e:
                pytest.fail(
                    f"Handler '{name}' raised {type(e).__name__} in "
                    f"get_processing_statistics(): {e}"
                )


# ============================================================================
# TEST CLASS 7: Isolated Registry Tests (non-global)
# ============================================================================

class TestIsolatedRegistry:
    """
    Verify HandlerRegistry behavior with an isolated (non-singleton) instance.

    This ensures contract tests don't pollute the global registry and validates
    the registry's API independently.
    """

    def test_isolated_registry_starts_empty(self):
        """A fresh HandlerRegistry instance should have zero entries."""
        registry = HandlerRegistry()
        assert len(registry) == 0
        assert registry.list_all() == []

    def test_register_and_retrieve(self, default_registry, mock_logger):
        """Register a handler in an isolated registry and retrieve it."""
        registry = HandlerRegistry()
        # Pick one handler class from the global registry
        some_name = default_registry.list_all()[0]
        handler_class = default_registry.get(some_name)
        registry.register(handler_class)
        assert len(registry) == 1
        retrieved = registry.get(registry.list_all()[0])
        assert retrieved is handler_class

    def test_get_nonexistent_handler_raises(self):
        """Requesting a non-registered handler must raise HandlerNotFoundError."""
        registry = HandlerRegistry()
        with pytest.raises(HandlerNotFoundError) as exc_info:
            registry.get("NonExistentHandler")
        assert exc_info.value.handler_name == "NonExistentHandler"

    def test_unregister(self, default_registry):
        """unregister() should remove a handler and return True."""
        registry = HandlerRegistry()
        some_name = default_registry.list_all()[0]
        handler_class = default_registry.get(some_name)
        registry.register(handler_class)
        registered_name = registry.list_all()[0]
        assert registry.unregister(registered_name) is True
        assert len(registry) == 0

    def test_unregister_nonexistent_returns_false(self):
        """unregister() for non-existent name returns False."""
        registry = HandlerRegistry()
        assert registry.unregister("NoSuchHandler") is False

    def test_clear_empties_registry(self, default_registry):
        """clear() should remove all registrations."""
        registry = HandlerRegistry()
        for name in default_registry.list_all()[:3]:
            handler_class = default_registry.get(name)
            registry.register(handler_class)
        assert len(registry) > 0
        registry.clear()
        assert len(registry) == 0

    def test_contains_operator(self, default_registry):
        """'in' operator should work with HandlerRegistry."""
        registry = HandlerRegistry()
        some_name = default_registry.list_all()[0]
        handler_class = default_registry.get(some_name)
        registry.register(handler_class)
        registered_name = registry.list_all()[0]
        assert registered_name in registry
        assert "UnknownHandler" not in registry

    def test_iter_over_registry(self, default_registry):
        """Iterating a registry should yield handler names."""
        registry = HandlerRegistry()
        for name in default_registry.list_all()[:3]:
            handler_class = default_registry.get(name)
            registry.register(handler_class)
        names = list(registry)
        assert len(names) == 3
        for n in names:
            assert isinstance(n, str)

    def test_on_change_callback_fires(self):
        """Registering a handler should trigger change callbacks."""
        registry = HandlerRegistry()
        callback_calls = []
        registry.add_on_change_callback(lambda: callback_calls.append(True))

        # Get a real handler class to register
        global_reg = get_default_registry()
        some_name = global_reg.list_all()[0]
        handler_class = global_reg.get(some_name)

        registry.register(handler_class)
        assert len(callback_calls) == 1, (
            "Change callback was not invoked on register()"
        )

    def test_remove_on_change_callback(self):
        """Removed callbacks should no longer fire."""
        registry = HandlerRegistry()
        calls = []
        cb = lambda: calls.append(True)  # noqa: E731
        registry.add_on_change_callback(cb)
        removed = registry.remove_on_change_callback(cb)
        assert removed is True

        # Register something — callback should NOT fire
        global_reg = get_default_registry()
        some_name = global_reg.list_all()[0]
        handler_class = global_reg.get(some_name)
        registry.register(handler_class)
        assert len(calls) == 0, "Removed callback was still invoked"


# ============================================================================
# TEST CLASS 8: Converter and Validator Protocols (Smoke)
# ============================================================================

class TestAuxiliaryProtocols:
    """
    Smoke checks for DatasetConverterProtocol and DatasetValidatorProtocol.

    These are secondary protocols from protocols.py. We verify they are
    importable and that isinstance() works correctly with mock objects.
    """

    def test_converter_protocol_is_importable(self):
        """DatasetConverterProtocol can be imported."""
        assert DatasetConverterProtocol is not None

    def test_validator_protocol_is_importable(self):
        """DatasetValidatorProtocol can be imported."""
        assert DatasetValidatorProtocol is not None

    def test_converter_protocol_isinstance_with_conforming_mock(self):
        """A mock with convert() and supports_format() satisfies converter protocol."""
        mock_converter = MagicMock()
        mock_converter.convert = MagicMock(return_value=MagicMock())
        mock_converter.supports_format = MagicMock(return_value=True)
        assert isinstance(mock_converter, DatasetConverterProtocol)

    def test_validator_protocol_isinstance_with_conforming_mock(self):
        """A mock with validate() and get_validation_rules() satisfies validator protocol."""
        mock_validator = MagicMock()
        mock_validator.validate = MagicMock(return_value={})
        mock_validator.get_validation_rules = MagicMock(return_value={})
        assert isinstance(mock_validator, DatasetValidatorProtocol)

    def test_handler_protocol_rejects_non_conforming_object(self):
        """An object missing protocol methods should fail isinstance()."""

        class IncompleteHandler:
            def get_dataset_type(self):
                return "test"
            # Missing 10 other methods

        obj = IncompleteHandler()
        # runtime_checkable checks for method existence — this should fail
        # because many required methods are missing
        assert not isinstance(obj, DatasetHandlerProtocol), (
            "Object with only get_dataset_type() should not satisfy the "
            "full DatasetHandlerProtocol"
        )
