#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/datasets/implementations/wavefunction.py

Module under test: wavefunction.py
- WavefunctionDataset: BaseDataset subclass with @register decorator
  - metadata: DatasetMetadata (Pydantic frozen dataclass)
  - schema: DatasetSchema (Pydantic frozen dataclass)
  - features: DatasetFeatures (Pydantic frozen dataclass)
  - config_key: str
  - get_required_properties(): classmethod -> List[str]
  - get_feature_support(): classmethod -> Dict[str, bool]
  - get_molecule_creation_strategy(): classmethod -> str
  - create_handler(): classmethod -> WavefunctionDatasetHandler (lazy import)

Test path on local machine: ~/ml_projects/milia/tests/test_dataset_impl_wavefunction_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/datasets/implementations/wavefunction.py

NOTE: This test suite runs inside Docker at /app/milia
Path mappings:
- Project root: /app/milia (mapped from ~/ml_projects/milia)

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution

Evidence sources:
- wavefunction.py: Complete source (provided)
- MILIA_Pipeline_Project_Structure.md: base.py details (lines 335-351),
  registry.py details (lines 369-375), implementations/ structure (lines 377-380)
- test_dataset_impl_dft_unit.py: Test conventions and patterns (provided)

CRITICAL DIFFERENCES FROM DFT tested here:
1. Molecule creation strategy: 'coordinate_based' (NOT 'identifier_coordinate_based')
2. Coordinate units: 'bohr' (NOT 'angstrom')
3. Energy units: 'eV' (NOT 'hartree')
4. Required properties: ('atoms', 'coordinates', 'compounds') — NO 'Etot'
5. Identifier keys: (('compounds', 'compound_id'),) — single key, label only
6. Features: orbital_analysis=True, homo_lumo_gap=True, mo_energies=True
   (all vibrational/thermodynamic features False)
7. Handler lazy import: WavefunctionDatasetHandler from handlers.implementations.wavefunction

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
from milia_pipeline.datasets.implementations.wavefunction import WavefunctionDataset
from milia_pipeline.datasets.registry import (
    is_registered,
)

# ============================================================================
# CONSTANTS: Expected values derived from wavefunction.py source
# ============================================================================

EXPECTED_METADATA_NAME = "Wavefunction"
EXPECTED_METADATA_VERSION = "1.0.0"
EXPECTED_METADATA_DESCRIPTION = (
    "Quantum mechanical wavefunction dataset from .molden files with orbital analysis"
)
EXPECTED_METADATA_AUTHOR = "MILIA Pipeline Team"

EXPECTED_REQUIRED_PROPERTIES = ("atoms", "coordinates", "compounds")
EXPECTED_OPTIONAL_PROPERTIES = ("mo_energies", "mo_occupations", "homo_lumo_gap_eV", "total_energy")
EXPECTED_IDENTIFIER_KEYS = (("compounds", "compound_id"),)
EXPECTED_COORDINATE_UNITS = "bohr"
EXPECTED_ENERGY_UNITS = "eV"

EXPECTED_FEATURES = {
    "vibrational_analysis": False,
    "uncertainty_handling": False,
    "atomization_energy": False,
    "rotational_constants": False,
    "frequency_analysis": False,
    "orbital_analysis": True,
    "homo_lumo_gap": True,
    "mo_energies": True,
}

EXPECTED_CONFIG_KEY = "wavefunction_config"
EXPECTED_MOLECULE_CREATION_STRATEGY = "coordinate_based"

EXPECTED_CLASSMETHOD_NAMES = [
    "get_required_properties",
    "get_feature_support",
    "get_molecule_creation_strategy",
    "create_handler",
]

# Sentinel for sys.modules cleanup in scoped handler mocking
_SENTINEL = object()


# ============================================================================
# GROUP 1: WavefunctionDataset — Class Identity and Type Hierarchy (8 tests)
# ============================================================================


class TestWavefunctionDatasetClassIdentity(unittest.TestCase):
    """Verify WavefunctionDataset is a proper BaseDataset subclass with correct identity."""

    def test_is_a_class(self):
        """WavefunctionDataset is a class (not a function or module)."""
        self.assertTrue(inspect.isclass(WavefunctionDataset))

    def test_has_correct_name(self):
        """Class name is 'WavefunctionDataset'."""
        self.assertEqual(WavefunctionDataset.__name__, "WavefunctionDataset")

    def test_has_correct_module(self):
        """Defined in the datasets.implementations.wavefunction module."""
        self.assertIn("implementations.wavefunction", WavefunctionDataset.__module__)

    def test_is_subclass_of_base_dataset(self):
        """WavefunctionDataset inherits from BaseDataset."""
        self.assertTrue(
            issubclass(WavefunctionDataset, BaseDataset),
            "WavefunctionDataset must be a subclass of BaseDataset",
        )

    def test_is_not_base_dataset_itself(self):
        """WavefunctionDataset is a distinct class, not BaseDataset itself."""
        self.assertIsNot(WavefunctionDataset, BaseDataset)

    def test_has_docstring(self):
        """WavefunctionDataset has a non-empty docstring."""
        self.assertIsNotNone(WavefunctionDataset.__doc__)
        self.assertGreater(len(WavefunctionDataset.__doc__.strip()), 0)

    def test_docstring_mentions_wavefunction(self):
        """WavefunctionDataset docstring references Wavefunction.

        Evidence: wavefunction.py class docstring mentions 'Wavefunction' and
        'coordinate_based' strategy.
        """
        self.assertIn("Wavefunction", WavefunctionDataset.__doc__)
        self.assertIn("coordinate_based", WavefunctionDataset.__doc__)

    def test_mro_includes_base_dataset(self):
        """Method Resolution Order includes BaseDataset."""
        self.assertIn(BaseDataset, WavefunctionDataset.__mro__)


# ============================================================================
# GROUP 2: WavefunctionDataset — Registration with @register (5 tests)
# ============================================================================


class TestWavefunctionDatasetRegistration(unittest.TestCase):
    """Verify WavefunctionDataset is registered via @register decorator."""

    def test_is_registered_in_default_registry(self):
        """WavefunctionDataset is discoverable in the default DatasetRegistry.

        Evidence: @register decorator applied at class definition (wavefunction.py line 39).
        Evidence: registry.py convenience function is_registered() (project structure line 375).
        """
        self.assertTrue(
            is_registered("Wavefunction"),
            "WavefunctionDataset must be registered under name 'Wavefunction'",
        )

    def test_get_returns_wavefunction_dataset_class(self):
        """Registry get('Wavefunction') returns the WavefunctionDataset class.

        Evidence: registry.py get() method returns the registered class (project structure line 372).
        """
        from milia_pipeline.datasets.registry import get

        retrieved = get("Wavefunction")
        self.assertIs(retrieved, WavefunctionDataset)

    def test_listed_in_all_datasets(self):
        """WavefunctionDataset name appears in list_all() results.

        Evidence: registry.py list_all() returns all registered names (project structure line 372).
        """
        from milia_pipeline.datasets.registry import list_all

        all_names = list_all()
        self.assertIn("Wavefunction", all_names)

    def test_register_decorator_is_imported(self):
        """The wavefunction module imports the register decorator.

        Evidence: wavefunction.py line 27 imports register from milia_pipeline.datasets.registry.
        """
        source = inspect.getsource(sys.modules[WavefunctionDataset.__module__])
        self.assertIn("from milia_pipeline.datasets.registry import register", source)

    def test_registration_uses_metadata_name(self):
        """Registration key matches metadata.name ('Wavefunction').

        Evidence: @register decorator uses cls.metadata.name (registry.py convention).
        Evidence: wavefunction.py metadata.name = 'Wavefunction' (line 65).
        """
        self.assertEqual(WavefunctionDataset.metadata.name, "Wavefunction")
        self.assertTrue(is_registered(WavefunctionDataset.metadata.name))


# ============================================================================
# GROUP 3: WavefunctionDataset.metadata — DatasetMetadata (6 tests)
# ============================================================================


class TestWavefunctionDatasetMetadata(unittest.TestCase):
    """Verify WavefunctionDataset.metadata is a correctly configured DatasetMetadata.

    Evidence: wavefunction.py lines 63-68, base.py DatasetMetadata Pydantic frozen dataclass.
    """

    def test_metadata_is_dataset_metadata_instance(self):
        """metadata attribute is a DatasetMetadata instance."""
        self.assertIsInstance(WavefunctionDataset.metadata, DatasetMetadata)

    def test_metadata_name(self):
        """metadata.name is 'Wavefunction'."""
        self.assertEqual(WavefunctionDataset.metadata.name, EXPECTED_METADATA_NAME)

    def test_metadata_version(self):
        """metadata.version is '1.0.0'."""
        self.assertEqual(WavefunctionDataset.metadata.version, EXPECTED_METADATA_VERSION)

    def test_metadata_description(self):
        """metadata.description matches expected Wavefunction description."""
        self.assertEqual(
            WavefunctionDataset.metadata.description,
            EXPECTED_METADATA_DESCRIPTION,
        )

    def test_metadata_author(self):
        """metadata.author is 'MILIA Pipeline Team'."""
        self.assertEqual(WavefunctionDataset.metadata.author, EXPECTED_METADATA_AUTHOR)

    def test_metadata_is_frozen(self):
        """metadata is immutable (Pydantic frozen dataclass).

        Evidence: base.py DatasetMetadata is a Pydantic frozen dataclass
        (project structure line 337-339).
        """
        with self.assertRaises((AttributeError, TypeError, Exception)):
            # Pydantic frozen dataclasses raise on attribute assignment
            WavefunctionDataset.metadata.name = "MODIFIED"


# ============================================================================
# GROUP 4: WavefunctionDataset.schema — DatasetSchema (8 tests)
# ============================================================================


class TestWavefunctionDatasetSchema(unittest.TestCase):
    """Verify WavefunctionDataset.schema is a correctly configured DatasetSchema.

    Evidence: wavefunction.py lines 70-78, base.py DatasetSchema Pydantic frozen dataclass.

    CRITICAL DIFFERENCES FROM DFT:
    - required_properties: ('atoms', 'coordinates', 'compounds') — NO 'Etot'
    - identifier_keys: (('compounds', 'compound_id'),) — single key, label only
    - coordinate_units: 'bohr' (NOT 'angstrom')
    - energy_units: 'eV' (NOT 'hartree')
    """

    def test_schema_is_dataset_schema_instance(self):
        """schema attribute is a DatasetSchema instance."""
        self.assertIsInstance(WavefunctionDataset.schema, DatasetSchema)

    def test_schema_required_properties(self):
        """schema.required_properties is ('atoms', 'coordinates', 'compounds').

        Evidence: config_constants.py line 251
        HANDLER_REQUIRED_PROPERTIES['Wavefunction'] = ['atoms', 'coordinates', 'compounds']

        CRITICAL: Unlike DFT, Wavefunction does NOT require 'Etot'.
        """
        self.assertEqual(
            WavefunctionDataset.schema.required_properties,
            EXPECTED_REQUIRED_PROPERTIES,
        )

    def test_schema_optional_properties(self):
        """schema.optional_properties is ('mo_energies', 'mo_occupations', 'homo_lumo_gap_eV', 'total_energy').

        Evidence: wavefunction.py lines 73-74.
        """
        self.assertEqual(
            WavefunctionDataset.schema.optional_properties,
            EXPECTED_OPTIONAL_PROPERTIES,
        )

    def test_schema_identifier_keys(self):
        """schema.identifier_keys is (('compounds', 'compound_id'),).

        CRITICAL: Unlike DFT (which uses InChI/SMILES), Wavefunction compound IDs
        are labels only, not parseable chemical identifiers.

        Evidence: wavefunction.py line 75.
        Evidence: dataset_handlers.py lines 3036-3069.
        """
        self.assertEqual(
            WavefunctionDataset.schema.identifier_keys,
            EXPECTED_IDENTIFIER_KEYS,
        )

    def test_schema_coordinate_units(self):
        """schema.coordinate_units is 'bohr'.

        CRITICAL DIFFERENCE FROM DFT: Wavefunction uses atomic units (Bohr),
        NOT Angstrom. Automatic conversion to Angstrom required during processing.

        Evidence: wavefunction.py line 76.
        Evidence: config_constants.py line 274-276.
        """
        self.assertEqual(
            WavefunctionDataset.schema.coordinate_units,
            EXPECTED_COORDINATE_UNITS,
        )

    def test_schema_energy_units(self):
        """schema.energy_units is 'eV'.

        Evidence: wavefunction.py line 77.
        """
        self.assertEqual(
            WavefunctionDataset.schema.energy_units,
            EXPECTED_ENERGY_UNITS,
        )

    def test_schema_is_frozen(self):
        """schema is immutable (Pydantic frozen dataclass).

        Evidence: base.py DatasetSchema is a Pydantic frozen dataclass
        (project structure line 340-343).
        """
        with self.assertRaises((AttributeError, TypeError, Exception)):
            WavefunctionDataset.schema.required_properties = ("modified",)

    def test_schema_required_properties_are_tuples(self):
        """required_properties and optional_properties are tuples (immutable sequences)."""
        self.assertIsInstance(WavefunctionDataset.schema.required_properties, tuple)
        self.assertIsInstance(WavefunctionDataset.schema.optional_properties, tuple)


# ============================================================================
# GROUP 5: WavefunctionDataset.features — DatasetFeatures (10 tests)
# ============================================================================


class TestWavefunctionDatasetFeatures(unittest.TestCase):
    """Verify WavefunctionDataset.features is a correctly configured DatasetFeatures.

    Evidence: wavefunction.py lines 80-89, base.py DatasetFeatures Pydantic frozen dataclass.
    Evidence: config_constants.py lines 235-244 HANDLER_FEATURE_SUPPORT['Wavefunction'].

    CRITICAL DIFFERENCE FROM DFT:
    - Wavefunction: orbital_analysis=True, homo_lumo_gap=True, mo_energies=True
    - Wavefunction: vibrational_analysis=False, atomization_energy=False,
      rotational_constants=False, frequency_analysis=False
    - DFT is the inverse pattern.
    """

    def test_features_is_dataset_features_instance(self):
        """features attribute is a DatasetFeatures instance."""
        self.assertIsInstance(WavefunctionDataset.features, DatasetFeatures)

    def test_vibrational_analysis_disabled(self):
        """features.vibrational_analysis is False.

        Evidence: Wavefunction datasets do not include vibrational modes/frequencies.
        """
        self.assertFalse(WavefunctionDataset.features.vibrational_analysis)

    def test_uncertainty_handling_disabled(self):
        """features.uncertainty_handling is False."""
        self.assertFalse(WavefunctionDataset.features.uncertainty_handling)

    def test_atomization_energy_disabled(self):
        """features.atomization_energy is False."""
        self.assertFalse(WavefunctionDataset.features.atomization_energy)

    def test_rotational_constants_disabled(self):
        """features.rotational_constants is False."""
        self.assertFalse(WavefunctionDataset.features.rotational_constants)

    def test_frequency_analysis_disabled(self):
        """features.frequency_analysis is False."""
        self.assertFalse(WavefunctionDataset.features.frequency_analysis)

    def test_orbital_analysis_enabled(self):
        """features.orbital_analysis is True.

        Evidence: config_constants.py line 240
        HANDLER_FEATURE_SUPPORT['Wavefunction']['orbital_analysis'] = True
        """
        self.assertTrue(WavefunctionDataset.features.orbital_analysis)

    def test_homo_lumo_gap_enabled(self):
        """features.homo_lumo_gap is True.

        Evidence: config_constants.py line 241
        HANDLER_FEATURE_SUPPORT['Wavefunction']['homo_lumo_gap'] = True
        """
        self.assertTrue(WavefunctionDataset.features.homo_lumo_gap)

    def test_mo_energies_enabled(self):
        """features.mo_energies is True.

        Evidence: config_constants.py line 242
        HANDLER_FEATURE_SUPPORT['Wavefunction']['mo_energies'] = True
        """
        self.assertTrue(WavefunctionDataset.features.mo_energies)

    def test_features_is_frozen(self):
        """features is immutable (Pydantic frozen dataclass).

        Evidence: base.py DatasetFeatures is a Pydantic frozen dataclass
        (project structure line 344-346).
        """
        with self.assertRaises((AttributeError, TypeError, Exception)):
            WavefunctionDataset.features.orbital_analysis = False


# ============================================================================
# GROUP 6: WavefunctionDataset.config_key (2 tests)
# ============================================================================


class TestWavefunctionDatasetConfigKey(unittest.TestCase):
    """Verify WavefunctionDataset.config_key is correctly set.

    Evidence: wavefunction.py line 91.
    """

    def test_config_key_value(self):
        """config_key is 'wavefunction_config'."""
        self.assertEqual(WavefunctionDataset.config_key, EXPECTED_CONFIG_KEY)

    def test_config_key_is_string(self):
        """config_key is a string."""
        self.assertIsInstance(WavefunctionDataset.config_key, str)


# ============================================================================
# GROUP 7: WavefunctionDataset.get_required_properties() (5 tests)
# ============================================================================


class TestWavefunctionDatasetGetRequiredProperties(unittest.TestCase):
    """Verify WavefunctionDataset.get_required_properties() classmethod.

    Evidence: wavefunction.py lines 119-133.
    Evidence: config_constants.py line 251.

    CRITICAL DIFFERENCE FROM DFT:
    - Wavefunction: ['atoms', 'coordinates', 'compounds'] — NO 'Etot'
    - DFT: ['Etot', 'atoms', 'coordinates']
    """

    def test_is_classmethod(self):
        """get_required_properties is a classmethod."""
        descriptor = WavefunctionDataset.__dict__.get("get_required_properties")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_list(self):
        """get_required_properties() returns a list."""
        result = WavefunctionDataset.get_required_properties()
        self.assertIsInstance(result, list)

    def test_returns_correct_values(self):
        """get_required_properties() returns ['atoms', 'coordinates', 'compounds'].

        CRITICAL: 'Etot' is NOT required for Wavefunction datasets.
        Energy information comes from orbital analysis (mo_energies, total_energy).
        """
        result = WavefunctionDataset.get_required_properties()
        self.assertEqual(result, list(EXPECTED_REQUIRED_PROPERTIES))

    def test_returns_new_list_each_call(self):
        """get_required_properties() returns a fresh list (not the same object).

        Evidence: implementation uses list(cls.schema.required_properties),
        converting the tuple to a new list each time.
        """
        result1 = WavefunctionDataset.get_required_properties()
        result2 = WavefunctionDataset.get_required_properties()
        self.assertEqual(result1, result2)
        self.assertIsNot(result1, result2)

    def test_contains_all_strings(self):
        """All items in get_required_properties() are strings."""
        result = WavefunctionDataset.get_required_properties()
        for item in result:
            with self.subTest(item=item):
                self.assertIsInstance(item, str)


# ============================================================================
# GROUP 8: WavefunctionDataset.get_feature_support() (6 tests)
# ============================================================================


class TestWavefunctionDatasetGetFeatureSupport(unittest.TestCase):
    """Verify WavefunctionDataset.get_feature_support() classmethod.

    Evidence: wavefunction.py lines 135-153.
    Evidence: config_constants.py lines 235-244.
    """

    def test_is_classmethod(self):
        """get_feature_support is a classmethod."""
        descriptor = WavefunctionDataset.__dict__.get("get_feature_support")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_dict(self):
        """get_feature_support() returns a dict."""
        result = WavefunctionDataset.get_feature_support()
        self.assertIsInstance(result, dict)

    def test_returns_correct_feature_flags(self):
        """get_feature_support() returns the expected feature flags dict."""
        result = WavefunctionDataset.get_feature_support()
        self.assertEqual(result, EXPECTED_FEATURES)

    def test_all_values_are_booleans(self):
        """All values in get_feature_support() dict are booleans."""
        result = WavefunctionDataset.get_feature_support()
        for key, value in result.items():
            with self.subTest(feature=key):
                self.assertIsInstance(value, bool)

    def test_has_exactly_8_feature_flags(self):
        """get_feature_support() returns exactly 8 feature flags.

        Evidence: base.py DatasetFeatures has 8 feature flags
        (project structure line 345).
        """
        result = WavefunctionDataset.get_feature_support()
        self.assertEqual(len(result), 8)

    def test_delegates_to_features_to_dict(self):
        """get_feature_support() delegates to cls.features.to_dict().

        Evidence: wavefunction.py line 152: return cls.features.to_dict()
        Evidence: base.py DatasetFeatures.to_dict() method
        (project structure line 346).
        """
        direct_dict = WavefunctionDataset.features.to_dict()
        method_result = WavefunctionDataset.get_feature_support()
        self.assertEqual(direct_dict, method_result)


# ============================================================================
# GROUP 9: WavefunctionDataset.get_molecule_creation_strategy() (4 tests)
# ============================================================================


class TestWavefunctionDatasetGetMoleculeCreationStrategy(unittest.TestCase):
    """Verify WavefunctionDataset.get_molecule_creation_strategy() classmethod.

    Evidence: wavefunction.py lines 155-200.
    Evidence: dataset_handlers.py lines 3071-3091.

    CRITICAL DIFFERENCE FROM DFT:
    - Wavefunction: 'coordinate_based' — compound IDs are NOT parseable
    - DFT: 'identifier_coordinate_based' — uses InChI/SMILES identifiers
    """

    def test_is_classmethod(self):
        """get_molecule_creation_strategy is a classmethod."""
        descriptor = WavefunctionDataset.__dict__.get("get_molecule_creation_strategy")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_string(self):
        """get_molecule_creation_strategy() returns a string."""
        result = WavefunctionDataset.get_molecule_creation_strategy()
        self.assertIsInstance(result, str)

    def test_returns_coordinate_based(self):
        """get_molecule_creation_strategy() returns 'coordinate_based'.

        CRITICAL: Unlike DFT ('identifier_coordinate_based'), Wavefunction uses
        'coordinate_based' because compound IDs (e.g., 'BrCPxSiSxH4_331') are
        NOT parseable chemical identifiers. Molecular connectivity must be
        inferred directly from 3D coordinates using rdDetermineBonds.

        Evidence: dataset_handlers.py lines 3071-3091
        WavefunctionDatasetHandler.get_molecule_creation_strategy() returns 'coordinate_based'
        """
        result = WavefunctionDataset.get_molecule_creation_strategy()
        self.assertEqual(result, EXPECTED_MOLECULE_CREATION_STRATEGY)

    def test_has_docstring(self):
        """get_molecule_creation_strategy method has a non-empty docstring."""
        method = WavefunctionDataset.get_molecule_creation_strategy
        self.assertIsNotNone(method.__doc__)
        self.assertGreater(len(method.__doc__.strip()), 0)


# ============================================================================
# GROUP 10: WavefunctionDataset.create_handler() — Lazy Import Pattern (7 tests)
# ============================================================================


class TestWavefunctionDatasetCreateHandler(unittest.TestCase):
    """Verify WavefunctionDataset.create_handler() factory method with lazy import.

    Evidence: wavefunction.py lines 99-117.
    The create_handler() method uses lazy import to break circular dependency
    between datasets/implementations/wavefunction.py and handlers/implementations/wavefunction.py.
    """

    def test_is_classmethod(self):
        """create_handler is a classmethod."""
        descriptor = WavefunctionDataset.__dict__.get("create_handler")
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
        unbound_func = WavefunctionDataset.__dict__["create_handler"].__func__
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

        Evidence: wavefunction.py line 106: experimental_setup=None.
        """
        sig = inspect.signature(WavefunctionDataset.create_handler)
        default = sig.parameters["experimental_setup"].default
        self.assertIsNone(default)

    def _mock_handler_module(self):
        """Helper: create a mock handler module with a mock WavefunctionDatasetHandler class.

        The actual milia_pipeline.handlers.implementations.wavefunction module cannot be
        imported in the test environment due to missing dependencies.
        To test create_handler()'s lazy import behavior, we temporarily inject a
        mock module into sys.modules so that the
        'from milia_pipeline.handlers.implementations.wavefunction import WavefunctionDatasetHandler'
        statement inside create_handler() resolves to our mock.

        This uses a context manager pattern to ensure sys.modules is cleaned up
        after each test (no mock pollution).
        """
        import contextlib

        @contextlib.contextmanager
        def _scoped_handler_mock():
            mock_handler_cls = Mock(name="MockWavefunctionDatasetHandler")
            mock_module = MagicMock()
            mock_module.WavefunctionDatasetHandler = mock_handler_cls

            handler_mod_key = "milia_pipeline.handlers.implementations.wavefunction"
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
        """create_handler performs lazy import of WavefunctionDatasetHandler.

        Evidence: wavefunction.py line 114:
        from milia_pipeline.handlers.implementations.wavefunction import WavefunctionDatasetHandler
        """
        with self._mock_handler_module() as mock_cls:
            mock_cls.return_value = Mock()
            WavefunctionDataset.create_handler(
                dataset_config=Mock(),
                filter_config=Mock(),
                processing_config=Mock(),
                logger=Mock(),
                experimental_setup=None,
            )
            mock_cls.assert_called_once()

    def test_create_handler_passes_all_args_to_constructor(self):
        """create_handler passes all 5 arguments to WavefunctionDatasetHandler().

        Evidence: wavefunction.py lines 116-121.
        """
        mock_dataset_config = Mock(name="dataset_config")
        mock_filter_config = Mock(name="filter_config")
        mock_processing_config = Mock(name="processing_config")
        mock_logger = Mock(name="logger")
        mock_experimental_setup = Mock(name="experimental_setup")

        with self._mock_handler_module() as mock_cls:
            mock_cls.return_value = Mock()
            WavefunctionDataset.create_handler(
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
        """create_handler returns the WavefunctionDatasetHandler instance.

        Evidence: wavefunction.py line 116: return WavefunctionDatasetHandler(...).
        """
        mock_handler_instance = Mock(name="handler_instance")
        with self._mock_handler_module() as mock_cls:
            mock_cls.return_value = mock_handler_instance
            result = WavefunctionDataset.create_handler(
                dataset_config=Mock(),
                filter_config=Mock(),
                processing_config=Mock(),
                logger=Mock(),
            )
            self.assertIs(result, mock_handler_instance)

    def test_create_handler_has_docstring(self):
        """create_handler method has a non-empty docstring."""
        method = WavefunctionDataset.create_handler
        self.assertIsNotNone(method.__doc__)
        self.assertIn("lazy import", method.__doc__.lower())


# ============================================================================
# GROUP 11: WavefunctionDataset — handler_class Default (3 tests)
# ============================================================================


class TestWavefunctionDatasetHandlerClassAttribute(unittest.TestCase):
    """Verify WavefunctionDataset.handler_class is None (default from BaseDataset).

    Evidence: wavefunction.py lines 93-97 (NOTE comment about handler_class intentionally NOT set).
    Evidence: base.py BaseDataset optional handler_class (project structure line 349).
    """

    def test_handler_class_is_none(self):
        """handler_class is None (default from BaseDataset).

        Evidence: WavefunctionDatasetHandler is registered via @register_handler decorator
        and discovered dynamically through the HandlerRegistry.
        """
        self.assertIsNone(WavefunctionDataset.handler_class)

    def test_converter_class_is_none(self):
        """converter_class is None (default from BaseDataset).

        Evidence: base.py BaseDataset optional converter_class
        (project structure line 349).
        """
        self.assertIsNone(WavefunctionDataset.converter_class)

    def test_validator_class_is_none(self):
        """validator_class is None (default from BaseDataset).

        Evidence: base.py BaseDataset optional validator_class
        (project structure line 349).
        """
        self.assertIsNone(WavefunctionDataset.validator_class)


# ============================================================================
# GROUP 12: WavefunctionDataset — Method Signatures and Return Annotations (6 tests)
# ============================================================================


class TestWavefunctionDatasetMethodSignatures(unittest.TestCase):
    """Verify method signatures and return type annotations."""

    def _get_sig(self, method_name: str) -> inspect.Signature:
        """Helper: get the signature of a WavefunctionDataset method."""
        method = getattr(WavefunctionDataset, method_name)
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
# GROUP 13: WavefunctionDataset — Method Docstrings (4 tests with subTests)
# ============================================================================


class TestWavefunctionDatasetMethodDocstrings(unittest.TestCase):
    """Verify each WavefunctionDataset method has a non-empty docstring."""

    def test_each_classmethod_has_docstring(self):
        """Every expected classmethod has a non-empty docstring."""
        for method_name in EXPECTED_CLASSMETHOD_NAMES:
            with self.subTest(method=method_name):
                method = getattr(WavefunctionDataset, method_name)
                doc = getattr(method, "__doc__", None)
                self.assertIsNotNone(doc, f"{method_name} has no docstring")
                self.assertGreater(
                    len(doc.strip()),
                    0,
                    f"{method_name} has empty docstring",
                )

    def test_get_required_properties_docstring_mentions_evidence(self):
        """get_required_properties docstring references config_constants.py."""
        method = WavefunctionDataset.get_required_properties
        self.assertIn("config_constants", method.__doc__)

    def test_get_feature_support_docstring_mentions_evidence(self):
        """get_feature_support docstring references config_constants.py."""
        method = WavefunctionDataset.get_feature_support
        self.assertIn("config_constants", method.__doc__)

    def test_get_molecule_creation_strategy_docstring_mentions_evidence(self):
        """get_molecule_creation_strategy docstring references dataset_handlers.py."""
        method = WavefunctionDataset.get_molecule_creation_strategy
        self.assertIn("dataset_handlers", method.__doc__)


# ============================================================================
# GROUP 14: WavefunctionDataset — Module-Level Imports and Exports (5 tests)
# ============================================================================


class TestWavefunctionDatasetModuleImportsAndExports(unittest.TestCase):
    """Verify the wavefunction implementation module imports and exports correctly."""

    def test_module_has_docstring(self):
        """The wavefunction.py module has a non-empty module docstring."""
        import milia_pipeline.datasets.implementations.wavefunction as mod

        self.assertIsNotNone(mod.__doc__)
        self.assertGreater(len(mod.__doc__.strip()), 0)

    def test_module_exports_wavefunction_dataset(self):
        """WavefunctionDataset is importable from the implementations.wavefunction module."""
        import milia_pipeline.datasets.implementations.wavefunction as mod

        self.assertTrue(hasattr(mod, "WavefunctionDataset"))
        self.assertIs(mod.WavefunctionDataset, WavefunctionDataset)

    def test_module_imports_base_classes(self):
        """Module imports BaseDataset and data classes from base.py.

        Evidence: wavefunction.py lines 22-27.
        """
        source = inspect.getsource(
            sys.modules["milia_pipeline.datasets.implementations.wavefunction"]
        )
        self.assertIn("from milia_pipeline.datasets.base import", source)
        self.assertIn("BaseDataset", source)
        self.assertIn("DatasetMetadata", source)
        self.assertIn("DatasetSchema", source)
        self.assertIn("DatasetFeatures", source)

    def test_module_imports_register_decorator(self):
        """Module imports @register from registry.

        Evidence: wavefunction.py line 27.
        """
        source = inspect.getsource(
            sys.modules["milia_pipeline.datasets.implementations.wavefunction"]
        )
        self.assertIn("from milia_pipeline.datasets.registry import register", source)

    def test_module_does_not_import_handler_at_module_level(self):
        """WavefunctionDatasetHandler is NOT imported at module level (lazy import only).

        Evidence: wavefunction.py lines 30-35 (NOTE comment about circular import prevention).
        The handler is only imported inside create_handler() method.

        Uses ast module to reliably distinguish module-level imports from
        imports nested inside function/method bodies.
        """
        import ast

        source = inspect.getsource(
            sys.modules["milia_pipeline.datasets.implementations.wavefunction"]
        )
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
                    if isinstance(class_child, (ast.Import, ast.ImportFrom)):
                        if isinstance(class_child, ast.ImportFrom) and class_child.names:
                            for alias in class_child.names:
                                module_level_import_names.append(alias.name)

        self.assertNotIn(
            "WavefunctionDatasetHandler",
            module_level_import_names,
            "WavefunctionDatasetHandler should NOT be imported at module level "
            "(only inside create_handler() via lazy import)",
        )


# ============================================================================
# GROUP 15: WavefunctionDataset — DatasetFeatures.to_dict() and .supports() (4 tests)
# ============================================================================


class TestWavefunctionDatasetFeaturesIntegration(unittest.TestCase):
    """Verify DatasetFeatures integration methods work correctly with Wavefunction.

    Evidence: base.py DatasetFeatures.to_dict() and .supports() methods
    (project structure line 346).
    """

    def test_to_dict_returns_expected_dict(self):
        """features.to_dict() returns the full feature flags dictionary."""
        result = WavefunctionDataset.features.to_dict()
        self.assertEqual(result, EXPECTED_FEATURES)

    def test_supports_orbital_analysis(self):
        """features.supports('orbital_analysis') returns True.

        CRITICAL: This is the key distinguishing feature of Wavefunction datasets.
        """
        self.assertTrue(WavefunctionDataset.features.supports("orbital_analysis"))

    def test_supports_vibrational_analysis_false(self):
        """features.supports('vibrational_analysis') returns False.

        CRITICAL DIFFERENCE FROM DFT: DFT supports vibrational_analysis, Wavefunction does not.
        """
        self.assertFalse(WavefunctionDataset.features.supports("vibrational_analysis"))

    def test_to_dict_keys_match_expected_features(self):
        """features.to_dict() keys match all 8 expected feature names."""
        result = WavefunctionDataset.features.to_dict()
        self.assertEqual(set(result.keys()), set(EXPECTED_FEATURES.keys()))


# ============================================================================
# GROUP 16: WavefunctionDataset — Schema Consistency with Methods (3 tests)
# ============================================================================


class TestWavefunctionDatasetSchemaMethodConsistency(unittest.TestCase):
    """Verify schema data is consistent with method return values."""

    def test_required_properties_matches_schema(self):
        """get_required_properties() returns the same values as schema.required_properties.

        Evidence: wavefunction.py line 132: return list(cls.schema.required_properties).
        """
        method_result = WavefunctionDataset.get_required_properties()
        schema_result = list(WavefunctionDataset.schema.required_properties)
        self.assertEqual(method_result, schema_result)

    def test_feature_support_matches_features(self):
        """get_feature_support() returns the same values as features.to_dict().

        Evidence: wavefunction.py line 152: return cls.features.to_dict().
        """
        method_result = WavefunctionDataset.get_feature_support()
        features_result = WavefunctionDataset.features.to_dict()
        self.assertEqual(method_result, features_result)

    def test_required_properties_count(self):
        """get_required_properties() has exactly 3 items (atoms, coordinates, compounds).

        CRITICAL: Unlike DFT (Etot, atoms, coordinates), Wavefunction requires
        atoms, coordinates, and compounds.
        """
        result = WavefunctionDataset.get_required_properties()
        self.assertEqual(len(result), 3)


# ============================================================================
# GROUP 17: WavefunctionDataset — Edge Cases and Robustness (5 tests)
# ============================================================================


class TestWavefunctionDatasetEdgeCases(unittest.TestCase):
    """Test edge cases and robustness of WavefunctionDataset."""

    def test_multiple_calls_return_consistent_results(self):
        """Multiple calls to classmethods return identical results."""
        for _ in range(3):
            self.assertEqual(
                WavefunctionDataset.get_required_properties(),
                list(EXPECTED_REQUIRED_PROPERTIES),
            )
            self.assertEqual(
                WavefunctionDataset.get_feature_support(),
                EXPECTED_FEATURES,
            )
            self.assertEqual(
                WavefunctionDataset.get_molecule_creation_strategy(),
                EXPECTED_MOLECULE_CREATION_STRATEGY,
            )

    def test_classmethods_callable_on_class_not_instance(self):
        """All classmethods are callable on the class directly (no instantiation needed)."""
        # WavefunctionDataset is never instantiated — these are all classmethods
        for method_name in EXPECTED_CLASSMETHOD_NAMES:
            with self.subTest(method=method_name):
                method = getattr(WavefunctionDataset, method_name)
                self.assertTrue(callable(method))

    def test_identifier_keys_structure(self):
        """identifier_keys contains tuples of (source_key, identifier_type) pairs.

        CRITICAL: Wavefunction has only ONE identifier key pair: ('compounds', 'compound_id').
        This is a label-only identifier — compound IDs are NOT parseable chemical identifiers.
        """
        for key_pair in WavefunctionDataset.schema.identifier_keys:
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
            mock_handler_cls = Mock(name="MockWavefunctionDatasetHandler")
            mock_module = MagicMock()
            mock_module.WavefunctionDatasetHandler = mock_handler_cls
            handler_mod_key = "milia_pipeline.handlers.implementations.wavefunction"
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
            WavefunctionDataset.create_handler(
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
        not instance attributes. WavefunctionDataset is used as a class, not instantiated,
        but we verify the attributes live on the class itself."""
        self.assertIn("metadata", WavefunctionDataset.__dict__)
        self.assertIn("schema", WavefunctionDataset.__dict__)
        self.assertIn("features", WavefunctionDataset.__dict__)
        self.assertIn("config_key", WavefunctionDataset.__dict__)


# ============================================================================
# GROUP 18: WavefunctionDataset — Critical Differences from DFT/DMC (6 tests)
# ============================================================================


class TestWavefunctionDatasetCriticalDifferences(unittest.TestCase):
    """Verify the critical behavioral differences between Wavefunction and DFT/DMC.

    These tests explicitly validate the 4 key differences documented in wavefunction.py:
    1. Molecule creation strategy: 'coordinate_based'
    2. Coordinate units: 'bohr'
    3. Charge determination: from n_electrons (not InChI)
    4. Identifier keys: compound_id (label only)

    Evidence: wavefunction.py class docstring, config_constants.py lines 235-284,
    dataset_handlers.py lines 3036-3091.
    """

    def test_strategy_is_coordinate_based_not_identifier(self):
        """Molecule creation strategy is 'coordinate_based', NOT 'identifier_coordinate_based'.

        Evidence: dataset_handlers.py lines 3071-3091
        Compound IDs like 'BrCPxSiSxH4_331' are NOT parseable chemical identifiers.
        """
        strategy = WavefunctionDataset.get_molecule_creation_strategy()
        self.assertEqual(strategy, "coordinate_based")
        self.assertNotEqual(strategy, "identifier_coordinate_based")

    def test_coordinates_in_bohr_not_angstrom(self):
        """Coordinate units are 'bohr', NOT 'angstrom'.

        Evidence: config_constants.py line 274-276.
        Wavefunction data uses atomic units (Bohr); automatic conversion to Angstrom
        is required during processing.
        """
        self.assertEqual(WavefunctionDataset.schema.coordinate_units, "bohr")
        self.assertNotEqual(WavefunctionDataset.schema.coordinate_units, "angstrom")

    def test_identifier_keys_single_compound_id(self):
        """Identifier keys use ('compounds', 'compound_id') — a single label-only key.

        CRITICAL: Unlike DFT which uses (('inchi', 'inchi'), ('graphs', 'smiles')),
        Wavefunction has only one identifier key pair, and compound_id is a label
        that is NOT parsed for molecular structure.
        """
        self.assertEqual(len(WavefunctionDataset.schema.identifier_keys), 1)
        self.assertEqual(
            WavefunctionDataset.schema.identifier_keys[0],
            ("compounds", "compound_id"),
        )

    def test_etot_not_in_required_properties(self):
        """'Etot' is NOT in required properties.

        CRITICAL: Unlike DFT, Wavefunction does NOT require 'Etot' as a core property.
        Energy information comes from orbital analysis (mo_energies, total_energy).
        """
        required = WavefunctionDataset.get_required_properties()
        self.assertNotIn("Etot", required)

    def test_orbital_features_enabled(self):
        """Orbital analysis features (orbital_analysis, homo_lumo_gap, mo_energies) are all True.

        These are the distinguishing features of Wavefunction datasets.
        """
        features = WavefunctionDataset.get_feature_support()
        self.assertTrue(features["orbital_analysis"])
        self.assertTrue(features["homo_lumo_gap"])
        self.assertTrue(features["mo_energies"])

    def test_vibrational_features_disabled(self):
        """Vibrational/thermodynamic features are all False.

        CRITICAL DIFFERENCE: DFT has vibrational_analysis=True, atomization_energy=True,
        rotational_constants=True, frequency_analysis=True. Wavefunction has all False.
        """
        features = WavefunctionDataset.get_feature_support()
        self.assertFalse(features["vibrational_analysis"])
        self.assertFalse(features["atomization_energy"])
        self.assertFalse(features["rotational_constants"])
        self.assertFalse(features["frequency_analysis"])


# ============================================================================
# TEST RUNNER
# ============================================================================


def run_comprehensive_suite():
    """Run all test groups in a structured order."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestWavefunctionDatasetClassIdentity,  # GROUP 1:  8 tests
        TestWavefunctionDatasetRegistration,  # GROUP 2:  5 tests
        TestWavefunctionDatasetMetadata,  # GROUP 3:  6 tests
        TestWavefunctionDatasetSchema,  # GROUP 4:  8 tests
        TestWavefunctionDatasetFeatures,  # GROUP 5: 10 tests
        TestWavefunctionDatasetConfigKey,  # GROUP 6:  2 tests
        TestWavefunctionDatasetGetRequiredProperties,  # GROUP 7:  5 tests
        TestWavefunctionDatasetGetFeatureSupport,  # GROUP 8:  6 tests
        TestWavefunctionDatasetGetMoleculeCreationStrategy,  # GROUP 9:  4 tests
        TestWavefunctionDatasetCreateHandler,  # GROUP 10: 7 tests
        TestWavefunctionDatasetHandlerClassAttribute,  # GROUP 11: 3 tests
        TestWavefunctionDatasetMethodSignatures,  # GROUP 12: 6 tests
        TestWavefunctionDatasetMethodDocstrings,  # GROUP 13: 4 tests
        TestWavefunctionDatasetModuleImportsAndExports,  # GROUP 14: 5 tests
        TestWavefunctionDatasetFeaturesIntegration,  # GROUP 15: 4 tests
        TestWavefunctionDatasetSchemaMethodConsistency,  # GROUP 16: 3 tests
        TestWavefunctionDatasetEdgeCases,  # GROUP 17: 5 tests
        TestWavefunctionDatasetCriticalDifferences,  # GROUP 18: 6 tests
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS — datasets/implementations/wavefunction.py")
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
TEST SUITE SUMMARY — milia_pipeline/datasets/implementations/wavefunction.py
=============================================================================

95 comprehensive production-ready tests covering:

GROUP 1: WavefunctionDataset Class Identity and Type Hierarchy (8 tests)
- Is a class, correct name, correct module
- Subclass of BaseDataset, not BaseDataset itself
- Has docstring, docstring mentions Wavefunction and coordinate_based
- MRO includes BaseDataset

GROUP 2: WavefunctionDataset Registration with @register (5 tests)
- Is registered in default registry under 'Wavefunction'
- get('Wavefunction') returns WavefunctionDataset class
- Listed in list_all() results
- Module imports @register decorator
- Registration key matches metadata.name

GROUP 3: WavefunctionDataset.metadata — DatasetMetadata (6 tests)
- Is DatasetMetadata instance
- name='Wavefunction', version='1.0.0', description, author
- Metadata is frozen (immutable)

GROUP 4: WavefunctionDataset.schema — DatasetSchema (8 tests)
- Is DatasetSchema instance
- required_properties=('atoms', 'coordinates', 'compounds') — NO 'Etot'
- optional_properties=('mo_energies', 'mo_occupations', 'homo_lumo_gap_eV', 'total_energy')
- identifier_keys=(('compounds', 'compound_id'),) — single label-only key
- coordinate_units='bohr' (NOT 'angstrom')
- energy_units='eV'
- Schema is frozen, properties are tuples

GROUP 5: WavefunctionDataset.features — DatasetFeatures (10 tests)
- Is DatasetFeatures instance
- All 8 feature flags verified individually
  - vibrational_analysis=False, uncertainty_handling=False
  - atomization_energy=False, rotational_constants=False, frequency_analysis=False
  - orbital_analysis=True, homo_lumo_gap=True, mo_energies=True
- Features is frozen (immutable)

GROUP 6: WavefunctionDataset.config_key (2 tests)
- Value is 'wavefunction_config', is a string

GROUP 7: WavefunctionDataset.get_required_properties() (5 tests)
- Is classmethod, returns list
- Returns ['atoms', 'coordinates', 'compounds']
- Returns fresh list each call, all items are strings

GROUP 8: WavefunctionDataset.get_feature_support() (6 tests)
- Is classmethod, returns dict
- Returns correct 8 feature flags
- All values are booleans
- Delegates to features.to_dict()

GROUP 9: WavefunctionDataset.get_molecule_creation_strategy() (4 tests)
- Is classmethod, returns string
- Returns 'coordinate_based' (NOT 'identifier_coordinate_based')
- Has docstring

GROUP 10: WavefunctionDataset.create_handler() — Lazy Import (7 tests)
- Is classmethod, correct 5-parameter signature
- experimental_setup defaults to None
- Performs lazy import of WavefunctionDatasetHandler
- Passes all args to constructor
- Returns handler instance
- Has docstring mentioning lazy import

GROUP 11: WavefunctionDataset handler_class Default (3 tests)
- handler_class is None
- converter_class is None
- validator_class is None

GROUP 12: WavefunctionDataset Method Signatures and Return Annotations (6 tests)
- get_required_properties -> List[str]
- get_feature_support -> Dict[str, bool]
- get_molecule_creation_strategy -> str
- All classmethods have no params (bound method excludes cls)

GROUP 13: WavefunctionDataset Method Docstrings (4 tests with subTests)
- All 4 classmethods have non-empty docstrings
- Evidence references in docstrings

GROUP 14: Module-Level Imports and Exports (5 tests)
- Module has docstring, exports WavefunctionDataset
- Imports base classes and @register
- Does NOT import WavefunctionDatasetHandler at module level

GROUP 15: DatasetFeatures Integration (4 tests)
- to_dict() returns expected dict
- supports() works for enabled features (orbital_analysis)
- supports() works for disabled features (vibrational_analysis)
- to_dict() keys match all 8 features

GROUP 16: Schema-Method Consistency (3 tests)
- get_required_properties() matches schema.required_properties
- get_feature_support() matches features.to_dict()
- Required properties count is 3

GROUP 17: Edge Cases and Robustness (5 tests)
- Multiple calls return consistent results
- Classmethods callable on class
- identifier_keys structure validation (single key pair)
- create_handler with None experimental_setup
- Class attributes live in __dict__

GROUP 18: Critical Differences from DFT/DMC (6 tests)
- Strategy is 'coordinate_based' not 'identifier_coordinate_based'
- Coordinates in 'bohr' not 'angstrom'
- Single identifier key ('compounds', 'compound_id')
- 'Etot' not in required properties
- Orbital features enabled, vibrational features disabled

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
- Critical Wavefunction-specific differences from DFT/DMC explicitly tested
"""
