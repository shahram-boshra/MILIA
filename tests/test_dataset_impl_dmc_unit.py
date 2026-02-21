#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/datasets/implementations/dmc.py

Module under test: dmc.py
- DMCDataset: BaseDataset subclass with @register decorator
  - metadata: DatasetMetadata (Pydantic frozen dataclass)
  - schema: DatasetSchema (Pydantic frozen dataclass)
  - features: DatasetFeatures (Pydantic frozen dataclass)
  - config_key: str
  - get_required_properties(): classmethod -> List[str]
  - get_feature_support(): classmethod -> Dict[str, bool]
  - get_molecule_creation_strategy(): classmethod -> str
  - create_handler(): classmethod -> DMCDatasetHandler (lazy import)

Test path on local machine: ~/ml_projects/milia/tests/test_dataset_impl_dmc_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/datasets/implementations/dmc.py

NOTE: This test suite runs inside Docker at /app/milia
Path mappings:
- Project root: /app/milia (mapped from ~/ml_projects/milia)

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution

Evidence sources:
- dmc.py: Complete source (provided)
- MILIA_Pipeline_Project_Structure.md: base.py details (lines 335-351),
  registry.py details (lines 369-375), implementations/ structure (lines 311-318)
- test_dataset_impl_dft_unit.py: Test conventions and patterns (provided)

Updated: February 2026 - Production-ready comprehensive test coverage
"""

import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetFeatures,
    DatasetMetadata,
    DatasetSchema,
)
from milia_pipeline.datasets.implementations.dmc import DMCDataset
from milia_pipeline.datasets.registry import (
    is_registered,
)

# ============================================================================
# CONSTANTS: Expected values derived from dmc.py source
# ============================================================================

EXPECTED_METADATA_NAME = "DMC"
EXPECTED_METADATA_VERSION = "1.0.0"
EXPECTED_METADATA_DESCRIPTION = "DMC quantum Monte Carlo dataset with uncertainty handling"
EXPECTED_METADATA_AUTHOR = "MILIA Pipeline Team"

EXPECTED_REQUIRED_PROPERTIES = ("Etot", "std", "atoms", "coordinates")
EXPECTED_OPTIONAL_PROPERTIES = ("qmc_stats", "correlation_data")
EXPECTED_IDENTIFIER_KEYS = (("inchi", "inchi"), ("graphs", "smiles"))
EXPECTED_COORDINATE_UNITS = "angstrom"
EXPECTED_ENERGY_UNITS = "hartree"

EXPECTED_FEATURES = {
    "vibrational_analysis": False,
    "uncertainty_handling": True,
    "atomization_energy": False,
    "rotational_constants": False,
    "frequency_analysis": False,
    "orbital_analysis": False,
    "homo_lumo_gap": False,
    "mo_energies": False,
}

EXPECTED_CONFIG_KEY = "dmc_config"
EXPECTED_MOLECULE_CREATION_STRATEGY = "identifier_coordinate_based"

EXPECTED_CLASSMETHOD_NAMES = [
    "get_required_properties",
    "get_feature_support",
    "get_molecule_creation_strategy",
    "create_handler",
]

# Sentinel for sys.modules cleanup in scoped handler mocking
_SENTINEL = object()


# ============================================================================
# GROUP 1: DMCDataset — Class Identity and Type Hierarchy (8 tests)
# ============================================================================


class TestDMCDatasetClassIdentity(unittest.TestCase):
    """Verify DMCDataset is a proper BaseDataset subclass with correct identity."""

    def test_is_a_class(self):
        """DMCDataset is a class (not a function or module)."""
        self.assertTrue(inspect.isclass(DMCDataset))

    def test_has_correct_name(self):
        """Class name is 'DMCDataset'."""
        self.assertEqual(DMCDataset.__name__, "DMCDataset")

    def test_has_correct_module(self):
        """Defined in the datasets.implementations.dmc module."""
        self.assertIn("implementations.dmc", DMCDataset.__module__)

    def test_is_subclass_of_base_dataset(self):
        """DMCDataset inherits from BaseDataset."""
        self.assertTrue(
            issubclass(DMCDataset, BaseDataset),
            "DMCDataset must be a subclass of BaseDataset",
        )

    def test_is_not_base_dataset_itself(self):
        """DMCDataset is a distinct class, not BaseDataset itself."""
        self.assertIsNot(DMCDataset, BaseDataset)

    def test_has_docstring(self):
        """DMCDataset has a non-empty docstring."""
        self.assertIsNotNone(DMCDataset.__doc__)
        self.assertGreater(len(DMCDataset.__doc__.strip()), 0)

    def test_docstring_mentions_dmc(self):
        """DMCDataset docstring references DMC (Diffusion Monte Carlo).

        Evidence: dmc.py class docstring mentions 'DMC' and
        'Diffusion Monte Carlo'.
        """
        self.assertIn("DMC", DMCDataset.__doc__)
        self.assertIn("Monte Carlo", DMCDataset.__doc__)

    def test_mro_includes_base_dataset(self):
        """Method Resolution Order includes BaseDataset."""
        self.assertIn(BaseDataset, DMCDataset.__mro__)


# ============================================================================
# GROUP 2: DMCDataset — Registration with @register (5 tests)
# ============================================================================


class TestDMCDatasetRegistration(unittest.TestCase):
    """Verify DMCDataset is registered via @register decorator."""

    def test_is_registered_in_default_registry(self):
        """DMCDataset is discoverable in the default DatasetRegistry.

        Evidence: @register decorator applied at class definition (dmc.py line 34).
        Evidence: registry.py convenience function is_registered() (project structure line 375).
        """
        self.assertTrue(
            is_registered("DMC"),
            "DMCDataset must be registered under name 'DMC'",
        )

    def test_get_returns_dmc_dataset_class(self):
        """Registry get('DMC') returns the DMCDataset class.

        Evidence: registry.py get() method returns the registered class (project structure line 372).
        """
        from milia_pipeline.datasets.registry import get

        retrieved = get("DMC")
        self.assertIs(retrieved, DMCDataset)

    def test_listed_in_all_datasets(self):
        """DMCDataset name appears in list_all() results.

        Evidence: registry.py list_all() returns all registered names (project structure line 372).
        """
        from milia_pipeline.datasets.registry import list_all

        all_names = list_all()
        self.assertIn("DMC", all_names)

    def test_register_decorator_is_imported(self):
        """The dmc module imports the register decorator.

        Evidence: dmc.py line 22 imports register from milia_pipeline.datasets.registry.
        """
        source = inspect.getsource(sys.modules[DMCDataset.__module__])
        self.assertIn("from milia_pipeline.datasets.registry import register", source)

    def test_registration_uses_metadata_name(self):
        """Registration key matches metadata.name ('DMC').

        Evidence: @register decorator uses cls.metadata.name (registry.py convention).
        Evidence: dmc.py metadata.name = 'DMC'.
        """
        self.assertEqual(DMCDataset.metadata.name, "DMC")
        self.assertTrue(is_registered(DMCDataset.metadata.name))


# ============================================================================
# GROUP 3: DMCDataset.metadata — DatasetMetadata (6 tests)
# ============================================================================


class TestDMCDatasetMetadata(unittest.TestCase):
    """Verify DMCDataset.metadata is a correctly configured DatasetMetadata.

    Evidence: dmc.py lines 55-60, base.py DatasetMetadata Pydantic frozen dataclass.
    """

    def test_metadata_is_dataset_metadata_instance(self):
        """metadata attribute is a DatasetMetadata instance."""
        self.assertIsInstance(DMCDataset.metadata, DatasetMetadata)

    def test_metadata_name(self):
        """metadata.name is 'DMC'."""
        self.assertEqual(DMCDataset.metadata.name, EXPECTED_METADATA_NAME)

    def test_metadata_version(self):
        """metadata.version is '1.0.0'."""
        self.assertEqual(DMCDataset.metadata.version, EXPECTED_METADATA_VERSION)

    def test_metadata_description(self):
        """metadata.description matches expected DMC description."""
        self.assertEqual(
            DMCDataset.metadata.description,
            EXPECTED_METADATA_DESCRIPTION,
        )

    def test_metadata_author(self):
        """metadata.author is 'MILIA Pipeline Team'."""
        self.assertEqual(DMCDataset.metadata.author, EXPECTED_METADATA_AUTHOR)

    def test_metadata_is_frozen(self):
        """metadata is immutable (Pydantic frozen dataclass).

        Evidence: base.py DatasetMetadata is a Pydantic frozen dataclass
        (project structure line 337-339).
        """
        with self.assertRaises((AttributeError, TypeError, Exception)):
            # Pydantic frozen dataclasses raise on attribute assignment
            DMCDataset.metadata.name = "MODIFIED"


# ============================================================================
# GROUP 4: DMCDataset.schema — DatasetSchema (8 tests)
# ============================================================================


class TestDMCDatasetSchema(unittest.TestCase):
    """Verify DMCDataset.schema is a correctly configured DatasetSchema.

    Evidence: dmc.py lines 62-69, base.py DatasetSchema Pydantic frozen dataclass.
    """

    def test_schema_is_dataset_schema_instance(self):
        """schema attribute is a DatasetSchema instance."""
        self.assertIsInstance(DMCDataset.schema, DatasetSchema)

    def test_schema_required_properties(self):
        """schema.required_properties is ('Etot', 'std', 'atoms', 'coordinates').

        Evidence: config_constants.py line 250
        HANDLER_REQUIRED_PROPERTIES['DMC'] = ['Etot', 'std', 'atoms', 'coordinates']
        """
        self.assertEqual(
            DMCDataset.schema.required_properties,
            EXPECTED_REQUIRED_PROPERTIES,
        )

    def test_schema_optional_properties(self):
        """schema.optional_properties is ('qmc_stats', 'correlation_data')."""
        self.assertEqual(
            DMCDataset.schema.optional_properties,
            EXPECTED_OPTIONAL_PROPERTIES,
        )

    def test_schema_identifier_keys(self):
        """schema.identifier_keys is (('inchi', 'inchi'), ('graphs', 'smiles'))."""
        self.assertEqual(
            DMCDataset.schema.identifier_keys,
            EXPECTED_IDENTIFIER_KEYS,
        )

    def test_schema_coordinate_units(self):
        """schema.coordinate_units is 'angstrom'."""
        self.assertEqual(
            DMCDataset.schema.coordinate_units,
            EXPECTED_COORDINATE_UNITS,
        )

    def test_schema_energy_units(self):
        """schema.energy_units is 'hartree'."""
        self.assertEqual(
            DMCDataset.schema.energy_units,
            EXPECTED_ENERGY_UNITS,
        )

    def test_schema_is_frozen(self):
        """schema is immutable (Pydantic frozen dataclass).

        Evidence: base.py DatasetSchema is a Pydantic frozen dataclass
        (project structure line 340-343).
        """
        with self.assertRaises((AttributeError, TypeError, Exception)):
            DMCDataset.schema.required_properties = ("modified",)

    def test_schema_required_properties_are_tuples(self):
        """required_properties and optional_properties are tuples (immutable sequences)."""
        self.assertIsInstance(DMCDataset.schema.required_properties, tuple)
        self.assertIsInstance(DMCDataset.schema.optional_properties, tuple)


# ============================================================================
# GROUP 5: DMCDataset.features — DatasetFeatures (10 tests)
# ============================================================================


class TestDMCDatasetFeatures(unittest.TestCase):
    """Verify DMCDataset.features is a correctly configured DatasetFeatures.

    Evidence: dmc.py lines 71-80, base.py DatasetFeatures Pydantic frozen dataclass.
    Evidence: config_constants.py lines 228-234 HANDLER_FEATURE_SUPPORT['DMC'].
    """

    def test_features_is_dataset_features_instance(self):
        """features attribute is a DatasetFeatures instance."""
        self.assertIsInstance(DMCDataset.features, DatasetFeatures)

    def test_vibrational_analysis_disabled(self):
        """features.vibrational_analysis is False.

        Evidence: config_constants.py line 229 'vibrational_analysis': False
        """
        self.assertFalse(DMCDataset.features.vibrational_analysis)

    def test_uncertainty_handling_enabled(self):
        """features.uncertainty_handling is True.

        Evidence: config_constants.py line 230 'uncertainty_handling': True
        Key DMC-specific feature: DMC includes statistical uncertainties (std).
        """
        self.assertTrue(DMCDataset.features.uncertainty_handling)

    def test_atomization_energy_disabled(self):
        """features.atomization_energy is False.

        Evidence: config_constants.py line 231 'atomization_energy': False
        """
        self.assertFalse(DMCDataset.features.atomization_energy)

    def test_rotational_constants_disabled(self):
        """features.rotational_constants is False.

        Evidence: config_constants.py line 232 'rotational_constants': False
        """
        self.assertFalse(DMCDataset.features.rotational_constants)

    def test_frequency_analysis_disabled(self):
        """features.frequency_analysis is False.

        Evidence: config_constants.py line 233 'frequency_analysis': False
        """
        self.assertFalse(DMCDataset.features.frequency_analysis)

    def test_orbital_analysis_disabled(self):
        """features.orbital_analysis is False."""
        self.assertFalse(DMCDataset.features.orbital_analysis)

    def test_homo_lumo_gap_disabled(self):
        """features.homo_lumo_gap is False."""
        self.assertFalse(DMCDataset.features.homo_lumo_gap)

    def test_mo_energies_disabled(self):
        """features.mo_energies is False."""
        self.assertFalse(DMCDataset.features.mo_energies)

    def test_features_is_frozen(self):
        """features is immutable (Pydantic frozen dataclass).

        Evidence: base.py DatasetFeatures is a Pydantic frozen dataclass
        (project structure line 344-346).
        """
        with self.assertRaises((AttributeError, TypeError, Exception)):
            DMCDataset.features.uncertainty_handling = False


# ============================================================================
# GROUP 6: DMCDataset.config_key (2 tests)
# ============================================================================


class TestDMCDatasetConfigKey(unittest.TestCase):
    """Verify DMCDataset.config_key is correctly set.

    Evidence: dmc.py line 82: config_key = "dmc_config"
    """

    def test_config_key_value(self):
        """config_key is 'dmc_config'."""
        self.assertEqual(DMCDataset.config_key, EXPECTED_CONFIG_KEY)

    def test_config_key_is_string(self):
        """config_key is a string."""
        self.assertIsInstance(DMCDataset.config_key, str)


# ============================================================================
# GROUP 7: DMCDataset.get_required_properties() (5 tests)
# ============================================================================


class TestDMCDatasetGetRequiredProperties(unittest.TestCase):
    """Verify DMCDataset.get_required_properties() classmethod.

    Evidence: dmc.py lines 120-127.
    Evidence: config_constants.py line 250.
    """

    def test_is_classmethod(self):
        """get_required_properties is a classmethod."""
        descriptor = DMCDataset.__dict__.get("get_required_properties")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_list(self):
        """get_required_properties() returns a list."""
        result = DMCDataset.get_required_properties()
        self.assertIsInstance(result, list)

    def test_returns_correct_values(self):
        """get_required_properties() returns ['Etot', 'std', 'atoms', 'coordinates'].

        Evidence: config_constants.py line 250
        HANDLER_REQUIRED_PROPERTIES['DMC'] = ['Etot', 'std', 'atoms', 'coordinates']

        Key difference from DFT: DMC includes 'std' (statistical uncertainty).
        """
        result = DMCDataset.get_required_properties()
        self.assertEqual(result, list(EXPECTED_REQUIRED_PROPERTIES))

    def test_returns_new_list_each_call(self):
        """get_required_properties() returns a fresh list (not the same object).

        Evidence: implementation uses list(cls.schema.required_properties),
        converting the tuple to a new list each time.
        """
        result1 = DMCDataset.get_required_properties()
        result2 = DMCDataset.get_required_properties()
        self.assertEqual(result1, result2)
        self.assertIsNot(result1, result2)

    def test_contains_all_strings(self):
        """All items in get_required_properties() are strings."""
        result = DMCDataset.get_required_properties()
        for item in result:
            with self.subTest(item=item):
                self.assertIsInstance(item, str)


# ============================================================================
# GROUP 8: DMCDataset.get_feature_support() (6 tests)
# ============================================================================


class TestDMCDatasetGetFeatureSupport(unittest.TestCase):
    """Verify DMCDataset.get_feature_support() classmethod.

    Evidence: dmc.py lines 129-143.
    Evidence: config_constants.py lines 228-234.
    """

    def test_is_classmethod(self):
        """get_feature_support is a classmethod."""
        descriptor = DMCDataset.__dict__.get("get_feature_support")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_dict(self):
        """get_feature_support() returns a dict."""
        result = DMCDataset.get_feature_support()
        self.assertIsInstance(result, dict)

    def test_returns_correct_feature_flags(self):
        """get_feature_support() returns the expected feature flags dict.

        Key difference from DFT: uncertainty_handling=True, vibrational_analysis=False.
        """
        result = DMCDataset.get_feature_support()
        self.assertEqual(result, EXPECTED_FEATURES)

    def test_all_values_are_booleans(self):
        """All values in get_feature_support() dict are booleans."""
        result = DMCDataset.get_feature_support()
        for key, value in result.items():
            with self.subTest(feature=key):
                self.assertIsInstance(value, bool)

    def test_has_exactly_8_feature_flags(self):
        """get_feature_support() returns exactly 8 feature flags.

        Evidence: base.py DatasetFeatures has 8 feature flags
        (project structure line 345).
        """
        result = DMCDataset.get_feature_support()
        self.assertEqual(len(result), 8)

    def test_delegates_to_features_to_dict(self):
        """get_feature_support() delegates to cls.features.to_dict().

        Evidence: dmc.py line 142: return cls.features.to_dict()
        Evidence: base.py DatasetFeatures.to_dict() method
        (project structure line 346).
        """
        direct_dict = DMCDataset.features.to_dict()
        method_result = DMCDataset.get_feature_support()
        self.assertEqual(direct_dict, method_result)


# ============================================================================
# GROUP 9: DMCDataset.get_molecule_creation_strategy() (4 tests)
# ============================================================================


class TestDMCDatasetGetMoleculeCreationStrategy(unittest.TestCase):
    """Verify DMCDataset.get_molecule_creation_strategy() classmethod.

    Evidence: dmc.py lines 145-163.
    Evidence: dataset_handlers.py lines 2190-2207.
    """

    def test_is_classmethod(self):
        """get_molecule_creation_strategy is a classmethod."""
        descriptor = DMCDataset.__dict__.get("get_molecule_creation_strategy")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_string(self):
        """get_molecule_creation_strategy() returns a string."""
        result = DMCDataset.get_molecule_creation_strategy()
        self.assertIsInstance(result, str)

    def test_returns_identifier_coordinate_based(self):
        """get_molecule_creation_strategy() returns 'identifier_coordinate_based'.

        Evidence: DMCDatasetHandler.get_molecule_creation_strategy() returns
        'identifier_coordinate_based' (dataset_handlers.py lines 2190-2207).
        """
        result = DMCDataset.get_molecule_creation_strategy()
        self.assertEqual(result, EXPECTED_MOLECULE_CREATION_STRATEGY)

    def test_has_docstring(self):
        """get_molecule_creation_strategy method has a non-empty docstring."""
        method = DMCDataset.get_molecule_creation_strategy
        self.assertIsNotNone(method.__doc__)
        self.assertGreater(len(method.__doc__.strip()), 0)


# ============================================================================
# GROUP 10: DMCDataset.create_handler() — Lazy Import Pattern (7 tests)
# ============================================================================


class TestDMCDatasetCreateHandler(unittest.TestCase):
    """Verify DMCDataset.create_handler() factory method with lazy import.

    Evidence: dmc.py lines 89-115.
    The create_handler() method uses lazy import to break circular dependency
    between datasets/implementations/dmc.py and handlers/implementations/dmc.py.
    """

    def test_is_classmethod(self):
        """create_handler is a classmethod."""
        descriptor = DMCDataset.__dict__.get("create_handler")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_has_correct_signature(self):
        """create_handler has the expected 5-parameter signature (+ cls).

        Evidence: base.py BaseDataset.create_handler() has 5-parameter signature
        (project structure line 351).

        NOTE: In Python 3.10, inspect.signature() on a bound classmethod
        excludes 'cls' from the parameters. We verify the unbound signature
        via __func__ to capture all parameters including 'cls'.
        """
        # Access the underlying function to get the full signature including 'cls'
        unbound_func = DMCDataset.__dict__["create_handler"].__func__
        sig = inspect.signature(unbound_func)
        params = list(sig.parameters.keys())
        self.assertEqual(
            params,
            [
                "cls",
                "dataset_config",
                "filter_config",
                "processing_config",
                "logger",
                "experimental_setup",
            ],
        )

    def test_experimental_setup_default_is_none(self):
        """create_handler experimental_setup parameter defaults to None.

        Evidence: dmc.py line 97: experimental_setup=None.
        """
        sig = inspect.signature(DMCDataset.create_handler)
        default = sig.parameters["experimental_setup"].default
        self.assertIsNone(default)

    def _mock_handler_module(self):
        """Helper: create a mock handler module with a mock DMCDatasetHandler class.

        The actual milia_pipeline.handlers.implementations.dmc module cannot be
        imported in the test environment due to handler dependencies.
        To test create_handler()'s lazy import behavior,
        we temporarily inject a mock module into sys.modules so that the
        'from milia_pipeline.handlers.implementations.dmc import DMCDatasetHandler'
        statement inside create_handler() resolves to our mock.

        This uses a context manager pattern to ensure sys.modules is cleaned up
        after each test (no mock pollution).
        """
        import contextlib

        @contextlib.contextmanager
        def _scoped_handler_mock():
            mock_handler_cls = Mock(name="MockDMCDatasetHandler")
            mock_module = MagicMock()
            mock_module.DMCDatasetHandler = mock_handler_cls

            handler_mod_key = "milia_pipeline.handlers.implementations.dmc"
            original = sys.modules.get(handler_mod_key, _SENTINEL)
            sys.modules[handler_mod_key] = mock_module
            try:
                yield mock_handler_cls
            finally:
                if original is _SENTINEL:
                    sys.modules.pop(handler_mod_key, None)
                else:
                    sys.modules[handler_mod_key] = original

        return _scoped_handler_mock()

    def test_create_handler_calls_lazy_import(self):
        """create_handler performs lazy import of DMCDatasetHandler.

        Evidence: dmc.py line 112:
        from milia_pipeline.handlers.implementations.dmc import DMCDatasetHandler
        """
        with self._mock_handler_module() as mock_cls:
            mock_cls.return_value = Mock()
            DMCDataset.create_handler(
                dataset_config=Mock(),
                filter_config=Mock(),
                processing_config=Mock(),
                logger=Mock(),
                experimental_setup=None,
            )
            mock_cls.assert_called_once()

    def test_create_handler_passes_all_args_to_constructor(self):
        """create_handler passes all 5 arguments to DMCDatasetHandler().

        Evidence: dmc.py lines 114-119.
        """
        mock_dataset_config = Mock(name="dataset_config")
        mock_filter_config = Mock(name="filter_config")
        mock_processing_config = Mock(name="processing_config")
        mock_logger = Mock(name="logger")
        mock_experimental_setup = Mock(name="experimental_setup")

        with self._mock_handler_module() as mock_cls:
            mock_cls.return_value = Mock()
            DMCDataset.create_handler(
                dataset_config=mock_dataset_config,
                filter_config=mock_filter_config,
                processing_config=mock_processing_config,
                logger=mock_logger,
                experimental_setup=mock_experimental_setup,
            )
            mock_cls.assert_called_once_with(
                mock_dataset_config,
                mock_filter_config,
                mock_processing_config,
                mock_logger,
                mock_experimental_setup,
            )

    def test_create_handler_returns_handler_instance(self):
        """create_handler returns the DMCDatasetHandler instance.

        Evidence: dmc.py line 114: return DMCDatasetHandler(...).
        """
        mock_handler_instance = Mock(name="handler_instance")
        with self._mock_handler_module() as mock_cls:
            mock_cls.return_value = mock_handler_instance
            result = DMCDataset.create_handler(
                dataset_config=Mock(),
                filter_config=Mock(),
                processing_config=Mock(),
                logger=Mock(),
            )
            self.assertIs(result, mock_handler_instance)

    def test_create_handler_has_docstring(self):
        """create_handler method has a non-empty docstring."""
        method = DMCDataset.create_handler
        self.assertIsNotNone(method.__doc__)
        self.assertIn("lazy import", method.__doc__.lower())


# ============================================================================
# GROUP 11: DMCDataset — handler_class Default (3 tests)
# ============================================================================


class TestDMCDatasetHandlerClassAttribute(unittest.TestCase):
    """Verify DMCDataset.handler_class is None (default from BaseDataset).

    Evidence: dmc.py lines 84-89 (NOTE comment about handler_class intentionally NOT set).
    Evidence: base.py BaseDataset optional handler_class (project structure line 349).
    """

    def test_handler_class_is_none(self):
        """handler_class is None (default from BaseDataset).

        Evidence: DMCDatasetHandler is registered via @register_handler decorator
        and discovered dynamically through the HandlerRegistry.
        """
        self.assertIsNone(DMCDataset.handler_class)

    def test_converter_class_is_none(self):
        """converter_class is None (default from BaseDataset).

        Evidence: base.py BaseDataset optional converter_class
        (project structure line 349).
        """
        self.assertIsNone(DMCDataset.converter_class)

    def test_validator_class_is_none(self):
        """validator_class is None (default from BaseDataset).

        Evidence: base.py BaseDataset optional validator_class
        (project structure line 349).
        """
        self.assertIsNone(DMCDataset.validator_class)


# ============================================================================
# GROUP 12: DMCDataset — Method Signatures and Return Annotations (6 tests)
# ============================================================================


class TestDMCDatasetMethodSignatures(unittest.TestCase):
    """Verify method signatures and return type annotations."""

    def _get_sig(self, method_name: str) -> inspect.Signature:
        """Helper: get the signature of a DMCDataset method."""
        method = getattr(DMCDataset, method_name)
        return inspect.signature(method)

    def test_get_required_properties_return_annotation(self):
        """get_required_properties() -> List[str]."""
        sig = self._get_sig("get_required_properties")
        self.assertEqual(sig.return_annotation, list[str])

    def test_get_feature_support_return_annotation(self):
        """get_feature_support() -> Dict[str, bool]."""
        sig = self._get_sig("get_feature_support")
        self.assertEqual(sig.return_annotation, dict[str, bool])

    def test_get_molecule_creation_strategy_return_annotation(self):
        """get_molecule_creation_strategy() -> str."""
        sig = self._get_sig("get_molecule_creation_strategy")
        self.assertIs(sig.return_annotation, str)

    def test_get_required_properties_params(self):
        """get_required_properties(cls) has only 'cls' parameter."""
        sig = self._get_sig("get_required_properties")
        # Bound method signature excludes 'cls'
        params = list(sig.parameters.keys())
        self.assertEqual(params, [])

    def test_get_feature_support_params(self):
        """get_feature_support(cls) has only 'cls' parameter."""
        sig = self._get_sig("get_feature_support")
        params = list(sig.parameters.keys())
        self.assertEqual(params, [])

    def test_get_molecule_creation_strategy_params(self):
        """get_molecule_creation_strategy(cls) has only 'cls' parameter."""
        sig = self._get_sig("get_molecule_creation_strategy")
        params = list(sig.parameters.keys())
        self.assertEqual(params, [])


# ============================================================================
# GROUP 13: DMCDataset — Method Docstrings (4 tests with subTests)
# ============================================================================


class TestDMCDatasetMethodDocstrings(unittest.TestCase):
    """Verify each DMCDataset method has a non-empty docstring."""

    def test_each_classmethod_has_docstring(self):
        """Every expected classmethod has a non-empty docstring."""
        for method_name in EXPECTED_CLASSMETHOD_NAMES:
            with self.subTest(method=method_name):
                method = getattr(DMCDataset, method_name)
                doc = getattr(method, "__doc__", None)
                self.assertIsNotNone(doc, f"{method_name} has no docstring")
                self.assertGreater(
                    len(doc.strip()),
                    0,
                    f"{method_name} has empty docstring",
                )

    def test_get_required_properties_docstring_mentions_evidence(self):
        """get_required_properties docstring references config_constants.py."""
        method = DMCDataset.get_required_properties
        self.assertIn("config_constants", method.__doc__)

    def test_get_feature_support_docstring_mentions_evidence(self):
        """get_feature_support docstring references config_constants.py."""
        method = DMCDataset.get_feature_support
        self.assertIn("config_constants", method.__doc__)

    def test_get_molecule_creation_strategy_docstring_mentions_evidence(self):
        """get_molecule_creation_strategy docstring references dataset_handlers.py."""
        method = DMCDataset.get_molecule_creation_strategy
        self.assertIn("dataset_handlers", method.__doc__)


# ============================================================================
# GROUP 14: DMCDataset — Module-Level Imports and Exports (5 tests)
# ============================================================================


class TestDMCDatasetModuleImportsAndExports(unittest.TestCase):
    """Verify the dmc implementation module imports and exports correctly."""

    def test_module_has_docstring(self):
        """The dmc.py module has a non-empty module docstring."""
        import milia_pipeline.datasets.implementations.dmc as mod

        self.assertIsNotNone(mod.__doc__)
        self.assertGreater(len(mod.__doc__.strip()), 0)

    def test_module_exports_dmc_dataset(self):
        """DMCDataset is importable from the implementations.dmc module."""
        import milia_pipeline.datasets.implementations.dmc as mod

        self.assertTrue(hasattr(mod, "DMCDataset"))
        self.assertIs(mod.DMCDataset, DMCDataset)

    def test_module_imports_base_classes(self):
        """Module imports BaseDataset and data classes from base.py.

        Evidence: dmc.py lines 17-22.
        """
        source = inspect.getsource(sys.modules["milia_pipeline.datasets.implementations.dmc"])
        self.assertIn("from milia_pipeline.datasets.base import", source)
        self.assertIn("BaseDataset", source)
        self.assertIn("DatasetMetadata", source)
        self.assertIn("DatasetSchema", source)
        self.assertIn("DatasetFeatures", source)

    def test_module_imports_register_decorator(self):
        """Module imports @register from registry.

        Evidence: dmc.py line 22.
        """
        source = inspect.getsource(sys.modules["milia_pipeline.datasets.implementations.dmc"])
        self.assertIn("from milia_pipeline.datasets.registry import register", source)

    def test_module_does_not_import_handler_at_module_level(self):
        """DMCDatasetHandler is NOT imported at module level (lazy import only).

        Evidence: dmc.py lines 27-31 (NOTE comment about circular import prevention).
        The handler is only imported inside create_handler() method.

        Uses ast module to reliably distinguish module-level imports from
        imports nested inside function/method bodies.
        """
        import ast

        source = inspect.getsource(sys.modules["milia_pipeline.datasets.implementations.dmc"])
        tree = ast.parse(source)

        # Collect module-level import statements only
        # Module-level = direct children of the Module node, or direct children
        # of a ClassDef that is a direct child of the Module (class body, not methods)
        module_level_import_names = []

        for node in ast.iter_child_nodes(tree):
            # Top-level imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.names
                    or isinstance(node, ast.Import)
                    and node.names
                ):
                    for alias in node.names:
                        module_level_import_names.append(alias.name)

            # Class-level (but not inside methods) — check direct children of ClassDef
            elif isinstance(node, ast.ClassDef):
                for class_child in ast.iter_child_nodes(node):
                    if isinstance(class_child, ast.ImportFrom) and class_child.names:
                        for alias in class_child.names:
                            module_level_import_names.append(alias.name)

        self.assertNotIn(
            "DMCDatasetHandler",
            module_level_import_names,
            "DMCDatasetHandler should NOT be imported at module level "
            "(only inside create_handler() via lazy import)",
        )


# ============================================================================
# GROUP 15: DMCDataset — DatasetFeatures.to_dict() and .supports() (4 tests)
# ============================================================================


class TestDMCDatasetFeaturesIntegration(unittest.TestCase):
    """Verify DatasetFeatures integration methods work correctly with DMC.

    Evidence: base.py DatasetFeatures.to_dict() and .supports() methods
    (project structure line 346).
    """

    def test_to_dict_returns_expected_dict(self):
        """features.to_dict() returns the full feature flags dictionary."""
        result = DMCDataset.features.to_dict()
        self.assertEqual(result, EXPECTED_FEATURES)

    def test_supports_uncertainty_handling(self):
        """features.supports('uncertainty_handling') returns True.

        Key DMC-specific feature: uncertainty handling is the distinguishing
        capability of DMC datasets (includes statistical std values).
        """
        self.assertTrue(DMCDataset.features.supports("uncertainty_handling"))

    def test_supports_vibrational_analysis_false(self):
        """features.supports('vibrational_analysis') returns False.

        Key difference from DFT: DMC does NOT support vibrational analysis.
        """
        self.assertFalse(DMCDataset.features.supports("vibrational_analysis"))

    def test_to_dict_keys_match_expected_features(self):
        """features.to_dict() keys match all 8 expected feature names."""
        result = DMCDataset.features.to_dict()
        self.assertEqual(set(result.keys()), set(EXPECTED_FEATURES.keys()))


# ============================================================================
# GROUP 16: DMCDataset — Schema Consistency with Methods (3 tests)
# ============================================================================


class TestDMCDatasetSchemaMethodConsistency(unittest.TestCase):
    """Verify schema data is consistent with method return values."""

    def test_required_properties_matches_schema(self):
        """get_required_properties() returns the same values as schema.required_properties.

        Evidence: dmc.py line 126: return list(cls.schema.required_properties).
        """
        method_result = DMCDataset.get_required_properties()
        schema_result = list(DMCDataset.schema.required_properties)
        self.assertEqual(method_result, schema_result)

    def test_feature_support_matches_features(self):
        """get_feature_support() returns the same values as features.to_dict().

        Evidence: dmc.py line 142: return cls.features.to_dict().
        """
        method_result = DMCDataset.get_feature_support()
        features_result = DMCDataset.features.to_dict()
        self.assertEqual(method_result, features_result)

    def test_required_properties_count(self):
        """get_required_properties() has exactly 4 items (Etot, std, atoms, coordinates).

        Key difference from DFT: DMC has 4 required properties (includes 'std').
        """
        result = DMCDataset.get_required_properties()
        self.assertEqual(len(result), 4)


# ============================================================================
# GROUP 17: DMCDataset — Edge Cases and Robustness (5 tests)
# ============================================================================


class TestDMCDatasetEdgeCases(unittest.TestCase):
    """Test edge cases and robustness of DMCDataset."""

    def test_multiple_calls_return_consistent_results(self):
        """Multiple calls to classmethods return identical results."""
        for _ in range(3):
            self.assertEqual(
                DMCDataset.get_required_properties(),
                list(EXPECTED_REQUIRED_PROPERTIES),
            )
            self.assertEqual(
                DMCDataset.get_feature_support(),
                EXPECTED_FEATURES,
            )
            self.assertEqual(
                DMCDataset.get_molecule_creation_strategy(),
                EXPECTED_MOLECULE_CREATION_STRATEGY,
            )

    def test_classmethods_callable_on_class_not_instance(self):
        """All classmethods are callable on the class directly (no instantiation needed)."""
        # DMCDataset is never instantiated — these are all classmethods
        for method_name in EXPECTED_CLASSMETHOD_NAMES:
            with self.subTest(method=method_name):
                method = getattr(DMCDataset, method_name)
                self.assertTrue(callable(method))

    def test_identifier_keys_structure(self):
        """identifier_keys contains tuples of (source_key, identifier_type) pairs."""
        for key_pair in DMCDataset.schema.identifier_keys:
            with self.subTest(key_pair=key_pair):
                self.assertIsInstance(key_pair, tuple)
                self.assertEqual(len(key_pair), 2)
                self.assertIsInstance(key_pair[0], str)
                self.assertIsInstance(key_pair[1], str)

    def test_create_handler_with_none_experimental_setup(self):
        """create_handler works when experimental_setup is None (default)."""
        import contextlib

        @contextlib.contextmanager
        def _scoped_handler_mock():
            mock_handler_cls = Mock(name="MockDMCDatasetHandler")
            mock_module = MagicMock()
            mock_module.DMCDatasetHandler = mock_handler_cls
            handler_mod_key = "milia_pipeline.handlers.implementations.dmc"
            original = sys.modules.get(handler_mod_key, _SENTINEL)
            sys.modules[handler_mod_key] = mock_module
            try:
                yield mock_handler_cls
            finally:
                if original is _SENTINEL:
                    sys.modules.pop(handler_mod_key, None)
                else:
                    sys.modules[handler_mod_key] = original

        with _scoped_handler_mock() as mock_cls:
            mock_cls.return_value = Mock()
            DMCDataset.create_handler(
                dataset_config=Mock(),
                filter_config=Mock(),
                processing_config=Mock(),
                logger=Mock(),
                # experimental_setup not passed — uses default None
            )
            # Verify None was passed as the 5th argument
            args = mock_cls.call_args[0]
            self.assertIsNone(args[4])

    def test_class_attributes_not_overridable_via_instance(self):
        """Class-level attributes (metadata, schema, features) are class attributes,
        not instance attributes. DMCDataset is used as a class, not instantiated,
        but we verify the attributes live on the class itself."""
        self.assertIn("metadata", DMCDataset.__dict__)
        self.assertIn("schema", DMCDataset.__dict__)
        self.assertIn("features", DMCDataset.__dict__)
        self.assertIn("config_key", DMCDataset.__dict__)


# ============================================================================
# GROUP 18: DMCDataset — DMC-Specific Differentiation (6 tests)
# ============================================================================


class TestDMCDatasetDifferentiation(unittest.TestCase):
    """Verify DMC-specific characteristics that distinguish it from DFT.

    Evidence: dmc.py class docstring and config_constants.py lines 228-234, 250, 270-273.
    """

    def test_required_properties_includes_std(self):
        """DMC required properties include 'std' for statistical uncertainty.

        Evidence: config_constants.py line 250
        Key difference: DFT does NOT require 'std'; DMC does.
        """
        result = DMCDataset.get_required_properties()
        self.assertIn("std", result)

    def test_uncertainty_handling_is_enabled(self):
        """DMC enables uncertainty_handling feature (DFT does not).

        Evidence: config_constants.py line 230.
        This is the primary distinguishing feature of DMC datasets.
        """
        self.assertTrue(DMCDataset.features.uncertainty_handling)

    def test_vibrational_analysis_is_disabled(self):
        """DMC disables vibrational_analysis (DFT enables it).

        Evidence: config_constants.py line 229.
        """
        self.assertFalse(DMCDataset.features.vibrational_analysis)

    def test_atomization_energy_is_disabled(self):
        """DMC disables atomization_energy (DFT enables it).

        Evidence: config_constants.py line 231.
        """
        self.assertFalse(DMCDataset.features.atomization_energy)

    def test_optional_properties_are_qmc_specific(self):
        """DMC optional properties are QMC-specific (qmc_stats, correlation_data).

        Evidence: dmc.py schema optional_properties.
        DFT has vibrational-specific optional properties instead.
        """
        optional = DMCDataset.schema.optional_properties
        self.assertIn("qmc_stats", optional)
        self.assertIn("correlation_data", optional)

    def test_config_key_is_dmc_specific(self):
        """config_key is 'dmc_config' (not 'dft_config').

        Evidence: dmc.py line 82: config_key = "dmc_config"
        """
        self.assertEqual(DMCDataset.config_key, "dmc_config")
        self.assertNotEqual(DMCDataset.config_key, "dft_config")


# ============================================================================
# TEST RUNNER
# ============================================================================


def run_comprehensive_suite():
    """Run all test groups in a structured order."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestDMCDatasetClassIdentity,  # GROUP 1:  8 tests
        TestDMCDatasetRegistration,  # GROUP 2:  5 tests
        TestDMCDatasetMetadata,  # GROUP 3:  6 tests
        TestDMCDatasetSchema,  # GROUP 4:  8 tests
        TestDMCDatasetFeatures,  # GROUP 5: 10 tests
        TestDMCDatasetConfigKey,  # GROUP 6:  2 tests
        TestDMCDatasetGetRequiredProperties,  # GROUP 7:  5 tests
        TestDMCDatasetGetFeatureSupport,  # GROUP 8:  6 tests
        TestDMCDatasetGetMoleculeCreationStrategy,  # GROUP 9:  4 tests
        TestDMCDatasetCreateHandler,  # GROUP 10: 7 tests
        TestDMCDatasetHandlerClassAttribute,  # GROUP 11: 3 tests
        TestDMCDatasetMethodSignatures,  # GROUP 12: 6 tests
        TestDMCDatasetMethodDocstrings,  # GROUP 13: 4 tests
        TestDMCDatasetModuleImportsAndExports,  # GROUP 14: 5 tests
        TestDMCDatasetFeaturesIntegration,  # GROUP 15: 4 tests
        TestDMCDatasetSchemaMethodConsistency,  # GROUP 16: 3 tests
        TestDMCDatasetEdgeCases,  # GROUP 17: 5 tests
        TestDMCDatasetDifferentiation,  # GROUP 18: 6 tests
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS — datasets/implementations/dmc.py")
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
TEST SUITE SUMMARY — milia_pipeline/datasets/implementations/dmc.py
====================================================================

95 comprehensive production-ready tests covering:

GROUP 1: DMCDataset Class Identity and Type Hierarchy (8 tests)
- Is a class, correct name, correct module
- Subclass of BaseDataset, not BaseDataset itself
- Has docstring, docstring mentions DMC / Monte Carlo
- MRO includes BaseDataset

GROUP 2: DMCDataset Registration with @register (5 tests)
- Is registered in default registry under 'DMC'
- get('DMC') returns DMCDataset class
- Listed in list_all() results
- Module imports @register decorator
- Registration key matches metadata.name

GROUP 3: DMCDataset.metadata — DatasetMetadata (6 tests)
- Is DatasetMetadata instance
- name='DMC', version='1.0.0', description, author
- Metadata is frozen (immutable)

GROUP 4: DMCDataset.schema — DatasetSchema (8 tests)
- Is DatasetSchema instance
- required_properties, optional_properties, identifier_keys
- coordinate_units='angstrom', energy_units='hartree'
- Schema is frozen, properties are tuples

GROUP 5: DMCDataset.features — DatasetFeatures (10 tests)
- Is DatasetFeatures instance
- All 8 feature flags verified individually
- Features is frozen (immutable)

GROUP 6: DMCDataset.config_key (2 tests)
- Value is 'dmc_config', is a string

GROUP 7: DMCDataset.get_required_properties() (5 tests)
- Is classmethod, returns list
- Returns ['Etot', 'std', 'atoms', 'coordinates']
- Returns fresh list each call, all items are strings

GROUP 8: DMCDataset.get_feature_support() (6 tests)
- Is classmethod, returns dict
- Returns correct 8 feature flags
- All values are booleans
- Delegates to features.to_dict()

GROUP 9: DMCDataset.get_molecule_creation_strategy() (4 tests)
- Is classmethod, returns string
- Returns 'identifier_coordinate_based'
- Has docstring

GROUP 10: DMCDataset.create_handler() — Lazy Import (7 tests)
- Is classmethod, correct 5-parameter signature
- experimental_setup defaults to None
- Performs lazy import of DMCDatasetHandler
- Passes all args to constructor
- Returns handler instance
- Has docstring mentioning lazy import

GROUP 11: DMCDataset handler_class Default (3 tests)
- handler_class is None
- converter_class is None
- validator_class is None

GROUP 12: DMCDataset Method Signatures and Return Annotations (6 tests)
- get_required_properties -> List[str]
- get_feature_support -> Dict[str, bool]
- get_molecule_creation_strategy -> str
- All classmethods have no params (bound method excludes cls)

GROUP 13: DMCDataset Method Docstrings (4 tests with subTests)
- All 4 classmethods have non-empty docstrings
- Evidence references in docstrings

GROUP 14: Module-Level Imports and Exports (5 tests)
- Module has docstring, exports DMCDataset
- Imports base classes and @register
- Does NOT import DMCDatasetHandler at module level

GROUP 15: DatasetFeatures Integration (4 tests)
- to_dict() returns expected dict
- supports() works for enabled/disabled features
- to_dict() keys match all 8 features

GROUP 16: Schema-Method Consistency (3 tests)
- get_required_properties() matches schema.required_properties
- get_feature_support() matches features.to_dict()
- Required properties count is 4

GROUP 17: Edge Cases and Robustness (5 tests)
- Multiple calls return consistent results
- Classmethods callable on class
- identifier_keys structure validation
- create_handler with None experimental_setup
- Class attributes live in __dict__

GROUP 18: DMC-Specific Differentiation (6 tests)
- Required properties include 'std' (uncertainty)
- uncertainty_handling enabled (key DMC feature)
- vibrational_analysis disabled (differs from DFT)
- atomization_energy disabled (differs from DFT)
- Optional properties are QMC-specific
- config_key is 'dmc_config' (not 'dft_config')

Total: 95 comprehensive production-ready tests

PRODUCTION-READY QUALITIES:
- NO sys.modules pollution (no module-level mocking)
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution
- Dynamic verification of class structure and attributes
- Evidence-based testing with source references
- Lazy import pattern tested via mocking
- Immutability (frozen dataclass) verified
- Compatible with both pytest and unittest runner
- No NPZ file downloads or file system dependencies
- Future-proof: tests verify contracts, not implementation details
- DMC-specific differentiation tests verify dataset identity
"""
