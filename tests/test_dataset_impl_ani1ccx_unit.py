#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/datasets/implementations/ani1ccx.py

Module under test: ani1ccx.py
- ANI1ccxDataset: BaseDataset subclass with @register decorator
  - metadata: DatasetMetadata (Pydantic frozen dataclass)
  - schema: DatasetSchema (Pydantic frozen dataclass)
  - features: DatasetFeatures (Pydantic frozen dataclass)
  - config_key: str
  - get_required_properties(): classmethod -> List[str]
  - get_feature_support(): classmethod -> Dict[str, bool]
  - get_molecule_creation_strategy(): classmethod -> str
  - create_handler(): classmethod -> ANI1ccxDatasetHandler (lazy import)

Test path on local machine: ~/ml_projects/milia/tests/test_dataset_impl_ani1ccx_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/datasets/implementations/ani1ccx.py

NOTE: This test suite runs inside Docker at /app/milia
Path mappings:
- Project root: /app/milia (mapped from ~/ml_projects/milia)

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution

Evidence sources:
- ani1ccx.py: Complete source (provided)
- ani1x.py: Sister dataset implementation (provided)
- MILIA_Pipeline_Project_Structure.md: base.py details (lines 335-351),
  registry.py details (lines 369-375), implementations/ structure (lines 311-318)
- test_dataset_impl_ani1x_unit.py: Test conventions and patterns (provided)

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
from milia_pipeline.datasets.implementations.ani1ccx import ANI1ccxDataset
from milia_pipeline.datasets.registry import (
    is_registered,
)

# ============================================================================
# CONSTANTS: Expected values derived from ani1ccx.py source
# ============================================================================

EXPECTED_METADATA_NAME = "ANI1ccx"
EXPECTED_METADATA_VERSION = "1.0.0"
EXPECTED_METADATA_DESCRIPTION = (
    "ANI-1ccx dataset with ~500k coupled-cluster conformations for organic "
    "molecules (CHNO). Properties computed at CCSD(T)/CBS level using transfer "
    "learning from \u03c9B97x/6-31G* DFT calculations."
)
EXPECTED_METADATA_AUTHOR = "Smith, Nebgen, Lubbers, Isayev, Roitberg"
EXPECTED_METADATA_LICENSE = "CC0"

# ANI-1ccx uses 'ccsd_energy' as its primary energy target (mapped from ccsd(t)_cbs.energy)
EXPECTED_REQUIRED_PROPERTIES = ("ccsd_energy", "atoms", "coordinates")
EXPECTED_OPTIONAL_PROPERTIES = (
    "dft_energy",  # wb97x_dz.energy (Hartree) - for comparison
    "forces",  # wb97x_dz.forces (Hartree/Angstrom)
    "hirshfeld_charges",  # wb97x_dz.hirshfeld_charges (e)
    "cm5_charges",  # wb97x_dz.cm5_charges (e)
    "dipole",  # wb97x_dz.dipole (Debye)
    "molecule_id",  # Molecule group identifier from HDF5
)
# CRITICAL: ANI-1ccx has NO parseable chemical identifiers — empty tuple
EXPECTED_IDENTIFIER_KEYS = ()
EXPECTED_COORDINATE_UNITS = "angstrom"
EXPECTED_ENERGY_UNITS = "hartree"

EXPECTED_FEATURES = {
    "vibrational_analysis": False,
    "uncertainty_handling": False,
    "atomization_energy": True,
    "rotational_constants": False,
    "frequency_analysis": False,
    "orbital_analysis": False,
    "homo_lumo_gap": False,
    "mo_energies": False,
}

EXPECTED_CONFIG_KEY = "ani1ccx_config"
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
# GROUP 1: ANI1ccxDataset — Class Identity and Type Hierarchy (8 tests)
# ============================================================================


class TestANI1ccxDatasetClassIdentity(unittest.TestCase):
    """Verify ANI1ccxDataset is a proper BaseDataset subclass with correct identity."""

    def test_is_a_class(self):
        """ANI1ccxDataset is a class (not a function or module)."""
        self.assertTrue(inspect.isclass(ANI1ccxDataset))

    def test_has_correct_name(self):
        """Class name is 'ANI1ccxDataset'."""
        self.assertEqual(ANI1ccxDataset.__name__, "ANI1ccxDataset")

    def test_has_correct_module(self):
        """Defined in the datasets.implementations.ani1ccx module."""
        self.assertIn("implementations.ani1ccx", ANI1ccxDataset.__module__)

    def test_is_subclass_of_base_dataset(self):
        """ANI1ccxDataset inherits from BaseDataset."""
        self.assertTrue(
            issubclass(ANI1ccxDataset, BaseDataset),
            "ANI1ccxDataset must be a subclass of BaseDataset",
        )

    def test_is_not_base_dataset_itself(self):
        """ANI1ccxDataset is a distinct class, not BaseDataset itself."""
        self.assertIsNot(ANI1ccxDataset, BaseDataset)

    def test_has_docstring(self):
        """ANI1ccxDataset has a non-empty docstring."""
        self.assertIsNotNone(ANI1ccxDataset.__doc__)
        self.assertGreater(len(ANI1ccxDataset.__doc__.strip()), 0)

    def test_docstring_mentions_ani1ccx(self):
        """ANI1ccxDataset docstring references ANI-1ccx dataset and coordinate_based.

        Evidence: ani1ccx.py class docstring mentions 'ANI-1ccx' and 'coordinate_based'.
        """
        self.assertIn("ANI-1ccx", ANI1ccxDataset.__doc__)
        self.assertIn("coordinate_based", ANI1ccxDataset.__doc__)

    def test_mro_includes_base_dataset(self):
        """Method Resolution Order includes BaseDataset."""
        self.assertIn(BaseDataset, ANI1ccxDataset.__mro__)


# ============================================================================
# GROUP 2: ANI1ccxDataset — Registration with @register (5 tests)
# ============================================================================


class TestANI1ccxDatasetRegistration(unittest.TestCase):
    """Verify ANI1ccxDataset is registered via @register decorator."""

    def test_is_registered_in_default_registry(self):
        """ANI1ccxDataset is discoverable in the default DatasetRegistry.

        Evidence: @register decorator applied at class definition (ani1ccx.py).
        Evidence: registry.py convenience function is_registered().
        """
        self.assertTrue(
            is_registered("ANI1ccx"),
            "ANI1ccxDataset must be registered under name 'ANI1ccx'",
        )

    def test_get_returns_ani1ccx_dataset_class(self):
        """Registry get('ANI1ccx') returns the ANI1ccxDataset class.

        Evidence: registry.py get() method returns the registered class.
        """
        from milia_pipeline.datasets.registry import get

        retrieved = get("ANI1ccx")
        self.assertIs(retrieved, ANI1ccxDataset)

    def test_listed_in_all_datasets(self):
        """ANI1ccxDataset name appears in list_all() results.

        Evidence: registry.py list_all() returns all registered names.
        """
        from milia_pipeline.datasets.registry import list_all

        all_names = list_all()
        self.assertIn("ANI1ccx", all_names)

    def test_register_decorator_is_imported(self):
        """The ani1ccx module imports the register decorator.

        Evidence: ani1ccx.py imports register from milia_pipeline.datasets.registry.
        """
        source = inspect.getsource(sys.modules[ANI1ccxDataset.__module__])
        self.assertIn("from milia_pipeline.datasets.registry import register", source)

    def test_registration_uses_metadata_name(self):
        """Registration key matches metadata.name ('ANI1ccx').

        Evidence: @register decorator uses cls.metadata.name (registry.py convention).
        Evidence: ani1ccx.py metadata.name = 'ANI1ccx'.
        """
        self.assertEqual(ANI1ccxDataset.metadata.name, "ANI1ccx")
        self.assertTrue(is_registered(ANI1ccxDataset.metadata.name))


# ============================================================================
# GROUP 3: ANI1ccxDataset.metadata — DatasetMetadata (7 tests)
# ============================================================================


class TestANI1ccxDatasetMetadata(unittest.TestCase):
    """Verify ANI1ccxDataset.metadata is a correctly configured DatasetMetadata.

    Evidence: ani1ccx.py metadata definition, base.py DatasetMetadata Pydantic frozen dataclass.
    """

    def test_metadata_is_dataset_metadata_instance(self):
        """metadata attribute is a DatasetMetadata instance."""
        self.assertIsInstance(ANI1ccxDataset.metadata, DatasetMetadata)

    def test_metadata_name(self):
        """metadata.name is 'ANI1ccx'."""
        self.assertEqual(ANI1ccxDataset.metadata.name, EXPECTED_METADATA_NAME)

    def test_metadata_version(self):
        """metadata.version is '1.0.0'."""
        self.assertEqual(ANI1ccxDataset.metadata.version, EXPECTED_METADATA_VERSION)

    def test_metadata_description(self):
        """metadata.description matches expected ANI-1ccx description."""
        self.assertEqual(
            ANI1ccxDataset.metadata.description,
            EXPECTED_METADATA_DESCRIPTION,
        )

    def test_metadata_author(self):
        """metadata.author is 'Smith, Nebgen, Lubbers, Isayev, Roitberg'."""
        self.assertEqual(ANI1ccxDataset.metadata.author, EXPECTED_METADATA_AUTHOR)

    def test_metadata_license(self):
        """metadata.license is 'CC0'.

        Evidence: ani1ccx.py metadata license="CC0".
        """
        self.assertEqual(ANI1ccxDataset.metadata.license, EXPECTED_METADATA_LICENSE)

    def test_metadata_is_frozen(self):
        """metadata is immutable (Pydantic frozen dataclass).

        Evidence: base.py DatasetMetadata is a Pydantic frozen dataclass
        (project structure line 337-339).
        """
        with self.assertRaises((AttributeError, TypeError, Exception)):
            # Pydantic frozen dataclasses raise on attribute assignment
            ANI1ccxDataset.metadata.name = "MODIFIED"


# ============================================================================
# GROUP 4: ANI1ccxDataset.schema — DatasetSchema (9 tests)
# ============================================================================


class TestANI1ccxDatasetSchema(unittest.TestCase):
    """Verify ANI1ccxDataset.schema is a correctly configured DatasetSchema.

    Evidence: ani1ccx.py schema definition, base.py DatasetSchema Pydantic frozen dataclass.
    """

    def test_schema_is_dataset_schema_instance(self):
        """schema attribute is a DatasetSchema instance."""
        self.assertIsInstance(ANI1ccxDataset.schema, DatasetSchema)

    def test_schema_required_properties(self):
        """schema.required_properties is ('ccsd_energy', 'atoms', 'coordinates').

        Evidence: ani1ccx.py schema required_properties definition.
        ANI-1ccx uses 'ccsd_energy' (mapped from ccsd(t)_cbs.energy), 'atoms', 'coordinates'.

        KEY DIFFERENCE FROM ANI-1x: ANI-1x uses 'energy' (from wb97x_dz.energy),
        while ANI-1ccx uses 'ccsd_energy' (from ccsd(t)_cbs.energy).
        """
        self.assertEqual(
            ANI1ccxDataset.schema.required_properties,
            EXPECTED_REQUIRED_PROPERTIES,
        )

    def test_schema_optional_properties(self):
        """schema.optional_properties contains dft_energy, forces, charges, dipole, molecule_id.

        Evidence: ani1ccx.py schema optional_properties definition.

        KEY DIFFERENCE FROM ANI-1x: ANI-1ccx has 'dft_energy' (wb97x_dz.energy for comparison)
        as an optional property, while ANI-1x has it as the required 'energy'.
        """
        self.assertEqual(
            ANI1ccxDataset.schema.optional_properties,
            EXPECTED_OPTIONAL_PROPERTIES,
        )

    def test_schema_identifier_keys_empty(self):
        """schema.identifier_keys is an empty tuple.

        CRITICAL: ANI-1ccx has NO parseable chemical identifiers.
        The HDF5 structure contains only atomic_numbers and coordinates.
        Evidence: ani1ccx.py schema identifier_keys=() and extensive comments.
        """
        self.assertEqual(
            ANI1ccxDataset.schema.identifier_keys,
            EXPECTED_IDENTIFIER_KEYS,
        )
        self.assertEqual(len(ANI1ccxDataset.schema.identifier_keys), 0)

    def test_schema_coordinate_units(self):
        """schema.coordinate_units is 'angstrom'.

        Evidence: ANI-1ccx coordinates are in Angstrom (DFT-optimized geometries).
        """
        self.assertEqual(
            ANI1ccxDataset.schema.coordinate_units,
            EXPECTED_COORDINATE_UNITS,
        )

    def test_schema_energy_units(self):
        """schema.energy_units is 'hartree'.

        Evidence: ANI-1ccx energies are in Hartree (standard quantum chemistry output).
        """
        self.assertEqual(
            ANI1ccxDataset.schema.energy_units,
            EXPECTED_ENERGY_UNITS,
        )

    def test_schema_is_frozen(self):
        """schema is immutable (Pydantic frozen dataclass).

        Evidence: base.py DatasetSchema is a Pydantic frozen dataclass
        (project structure line 340-343).
        """
        with self.assertRaises((AttributeError, TypeError, Exception)):
            ANI1ccxDataset.schema.required_properties = ("modified",)

    def test_schema_required_properties_are_tuples(self):
        """required_properties and optional_properties are tuples (immutable sequences)."""
        self.assertIsInstance(ANI1ccxDataset.schema.required_properties, tuple)
        self.assertIsInstance(ANI1ccxDataset.schema.optional_properties, tuple)

    def test_schema_identifier_keys_is_tuple(self):
        """identifier_keys is a tuple (even though empty).

        Evidence: ani1ccx.py schema identifier_keys=() — empty tuple, not None or list.
        """
        self.assertIsInstance(ANI1ccxDataset.schema.identifier_keys, tuple)


# ============================================================================
# GROUP 5: ANI1ccxDataset.features — DatasetFeatures (10 tests)
# ============================================================================


class TestANI1ccxDatasetFeatures(unittest.TestCase):
    """Verify ANI1ccxDataset.features is a correctly configured DatasetFeatures.

    Evidence: ani1ccx.py features definition, base.py DatasetFeatures Pydantic frozen dataclass.
    ANI-1ccx is a coupled-cluster dataset — only atomization_energy is True.
    """

    def test_features_is_dataset_features_instance(self):
        """features attribute is a DatasetFeatures instance."""
        self.assertIsInstance(ANI1ccxDataset.features, DatasetFeatures)

    def test_vibrational_analysis_disabled(self):
        """features.vibrational_analysis is False.

        Evidence: ANI-1ccx does not have vibrational frequencies.
        """
        self.assertFalse(ANI1ccxDataset.features.vibrational_analysis)

    def test_uncertainty_handling_disabled(self):
        """features.uncertainty_handling is False.

        Evidence: ANI-1ccx is deterministic coupled-cluster — no statistical uncertainties.
        """
        self.assertFalse(ANI1ccxDataset.features.uncertainty_handling)

    def test_atomization_energy_enabled(self):
        """features.atomization_energy is True.

        Evidence: Atomization energies can be calculated from ccsd(t)_cbs.energy.
        """
        self.assertTrue(ANI1ccxDataset.features.atomization_energy)

    def test_rotational_constants_disabled(self):
        """features.rotational_constants is False.

        Evidence: ANI-1ccx does not have rotational constants.
        """
        self.assertFalse(ANI1ccxDataset.features.rotational_constants)

    def test_frequency_analysis_disabled(self):
        """features.frequency_analysis is False.

        Evidence: ANI-1ccx does not have frequency analysis.
        """
        self.assertFalse(ANI1ccxDataset.features.frequency_analysis)

    def test_orbital_analysis_disabled(self):
        """features.orbital_analysis is False.

        Evidence: ANI-1ccx does not have orbital analysis.
        """
        self.assertFalse(ANI1ccxDataset.features.orbital_analysis)

    def test_homo_lumo_gap_disabled(self):
        """features.homo_lumo_gap is False.

        Evidence: ANI-1ccx does not have HOMO-LUMO gap.
        """
        self.assertFalse(ANI1ccxDataset.features.homo_lumo_gap)

    def test_mo_energies_disabled(self):
        """features.mo_energies is False.

        Evidence: ANI-1ccx does not have MO energies.
        """
        self.assertFalse(ANI1ccxDataset.features.mo_energies)

    def test_features_is_frozen(self):
        """features is immutable (Pydantic frozen dataclass).

        Evidence: base.py DatasetFeatures is a Pydantic frozen dataclass
        (project structure line 344-346).
        """
        with self.assertRaises((AttributeError, TypeError, Exception)):
            ANI1ccxDataset.features.vibrational_analysis = True


# ============================================================================
# GROUP 6: ANI1ccxDataset.config_key (2 tests)
# ============================================================================


class TestANI1ccxDatasetConfigKey(unittest.TestCase):
    """Verify ANI1ccxDataset.config_key is correctly set.

    Evidence: ani1ccx.py config_key = "ani1ccx_config".
    """

    def test_config_key_value(self):
        """config_key is 'ani1ccx_config'."""
        self.assertEqual(ANI1ccxDataset.config_key, EXPECTED_CONFIG_KEY)

    def test_config_key_is_string(self):
        """config_key is a string."""
        self.assertIsInstance(ANI1ccxDataset.config_key, str)


# ============================================================================
# GROUP 7: ANI1ccxDataset.get_required_properties() (5 tests)
# ============================================================================


class TestANI1ccxDatasetGetRequiredProperties(unittest.TestCase):
    """Verify ANI1ccxDataset.get_required_properties() classmethod.

    Evidence: ani1ccx.py get_required_properties() implementation.
    Property names mapped from HDF5 keys during preprocessing:
    - 'ccsd(t)_cbs.energy' → 'ccsd_energy'
    - 'atomic_numbers' → 'atoms' (converted to symbols)
    - 'coordinates' → 'coordinates'
    """

    def test_is_classmethod(self):
        """get_required_properties is a classmethod."""
        descriptor = ANI1ccxDataset.__dict__.get("get_required_properties")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_list(self):
        """get_required_properties() returns a list."""
        result = ANI1ccxDataset.get_required_properties()
        self.assertIsInstance(result, list)

    def test_returns_correct_values(self):
        """get_required_properties() returns ['ccsd_energy', 'atoms', 'coordinates'].

        KEY DIFFERENCE FROM ANI-1x: ANI-1x returns ['energy', 'atoms', 'coordinates'].
        ANI-1ccx returns ['ccsd_energy', 'atoms', 'coordinates'] because
        the primary energy target is ccsd(t)_cbs.energy, not wb97x_dz.energy.
        """
        result = ANI1ccxDataset.get_required_properties()
        self.assertEqual(result, list(EXPECTED_REQUIRED_PROPERTIES))

    def test_returns_new_list_each_call(self):
        """get_required_properties() returns a fresh list (not the same object).

        Evidence: implementation uses list(cls.schema.required_properties),
        converting the tuple to a new list each time.
        """
        result1 = ANI1ccxDataset.get_required_properties()
        result2 = ANI1ccxDataset.get_required_properties()
        self.assertEqual(result1, result2)
        self.assertIsNot(result1, result2)

    def test_contains_all_strings(self):
        """All items in get_required_properties() are strings."""
        result = ANI1ccxDataset.get_required_properties()
        for item in result:
            with self.subTest(item=item):
                self.assertIsInstance(item, str)


# ============================================================================
# GROUP 8: ANI1ccxDataset.get_feature_support() (6 tests)
# ============================================================================


class TestANI1ccxDatasetGetFeatureSupport(unittest.TestCase):
    """Verify ANI1ccxDataset.get_feature_support() classmethod.

    Evidence: ani1ccx.py get_feature_support() implementation.
    ANI-1ccx: Only atomization_energy is True, all other 7 flags are False.
    """

    def test_is_classmethod(self):
        """get_feature_support is a classmethod."""
        descriptor = ANI1ccxDataset.__dict__.get("get_feature_support")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_dict(self):
        """get_feature_support() returns a dict."""
        result = ANI1ccxDataset.get_feature_support()
        self.assertIsInstance(result, dict)

    def test_returns_correct_feature_flags(self):
        """get_feature_support() returns the expected feature flags dict.

        ANI-1ccx specifics: Only atomization_energy=True, everything else False.
        """
        result = ANI1ccxDataset.get_feature_support()
        self.assertEqual(result, EXPECTED_FEATURES)

    def test_all_values_are_booleans(self):
        """All values in get_feature_support() dict are booleans."""
        result = ANI1ccxDataset.get_feature_support()
        for key, value in result.items():
            with self.subTest(feature=key):
                self.assertIsInstance(value, bool)

    def test_has_exactly_8_feature_flags(self):
        """get_feature_support() returns exactly 8 feature flags.

        Evidence: base.py DatasetFeatures has 8 feature flags
        (project structure line 345).
        """
        result = ANI1ccxDataset.get_feature_support()
        self.assertEqual(len(result), 8)

    def test_delegates_to_features_to_dict(self):
        """get_feature_support() delegates to cls.features.to_dict().

        Evidence: ani1ccx.py: return cls.features.to_dict()
        """
        direct_dict = ANI1ccxDataset.features.to_dict()
        method_result = ANI1ccxDataset.get_feature_support()
        self.assertEqual(direct_dict, method_result)


# ============================================================================
# GROUP 9: ANI1ccxDataset.get_molecule_creation_strategy() (4 tests)
# ============================================================================


class TestANI1ccxDatasetGetMoleculeCreationStrategy(unittest.TestCase):
    """Verify ANI1ccxDataset.get_molecule_creation_strategy() classmethod.

    Evidence: ani1ccx.py get_molecule_creation_strategy() implementation.
    CRITICAL: ANI-1ccx uses 'coordinate_based' — NO parseable identifiers available.
    """

    def test_is_classmethod(self):
        """get_molecule_creation_strategy is a classmethod."""
        descriptor = ANI1ccxDataset.__dict__.get("get_molecule_creation_strategy")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_string(self):
        """get_molecule_creation_strategy() returns a string."""
        result = ANI1ccxDataset.get_molecule_creation_strategy()
        self.assertIsInstance(result, str)

    def test_returns_coordinate_based(self):
        """get_molecule_creation_strategy() returns 'coordinate_based'.

        CRITICAL DIFFERENCE FROM DFT/QM9: DFT uses 'identifier_coordinate_based'
        because it has InChI/SMILES identifiers. ANI-1ccx uses 'coordinate_based'
        because its HDF5 structure contains only atomic_numbers and coordinates.

        Evidence: ani1ccx.py docstring and Scientific Data paper Table 1.
        """
        result = ANI1ccxDataset.get_molecule_creation_strategy()
        self.assertEqual(result, EXPECTED_MOLECULE_CREATION_STRATEGY)

    def test_has_docstring(self):
        """get_molecule_creation_strategy method has a non-empty docstring."""
        method = ANI1ccxDataset.get_molecule_creation_strategy
        self.assertIsNotNone(method.__doc__)
        self.assertGreater(len(method.__doc__.strip()), 0)


# ============================================================================
# GROUP 10: ANI1ccxDataset.create_handler() — Lazy Import Pattern (7 tests)
# ============================================================================


class TestANI1ccxDatasetCreateHandler(unittest.TestCase):
    """Verify ANI1ccxDataset.create_handler() factory method with lazy import.

    Evidence: ani1ccx.py create_handler() implementation.
    The create_handler() method uses lazy import to break circular dependency
    between datasets/implementations/ani1ccx.py and handlers/implementations/ani1ccx.py.
    """

    def test_is_classmethod(self):
        """create_handler is a classmethod."""
        descriptor = ANI1ccxDataset.__dict__.get("create_handler")
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
        unbound_func = ANI1ccxDataset.__dict__["create_handler"].__func__
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

        Evidence: ani1ccx.py create_handler signature: experimental_setup=None.
        """
        sig = inspect.signature(ANI1ccxDataset.create_handler)
        default = sig.parameters["experimental_setup"].default
        self.assertIsNone(default)

    def _mock_handler_module(self):
        """Helper: create a mock handler module with a mock ANI1ccxDatasetHandler class.

        The actual milia_pipeline.handlers.implementations.ani1ccx module cannot be
        imported in the test environment due to handler dependencies.
        To test create_handler()'s lazy import behavior, we temporarily inject
        a mock module into sys.modules so that the
        'from milia_pipeline.handlers.implementations.ani1ccx import ANI1ccxDatasetHandler'
        statement inside create_handler() resolves to our mock.

        This uses a context manager pattern to ensure sys.modules is cleaned up
        after each test (no mock pollution).
        """
        import contextlib

        @contextlib.contextmanager
        def _scoped_handler_mock():
            mock_handler_cls = Mock(name="MockANI1ccxDatasetHandler")
            mock_module = MagicMock()
            mock_module.ANI1ccxDatasetHandler = mock_handler_cls

            handler_mod_key = "milia_pipeline.handlers.implementations.ani1ccx"
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
        """create_handler performs lazy import of ANI1ccxDatasetHandler.

        Evidence: ani1ccx.py create_handler():
        from milia_pipeline.handlers.implementations.ani1ccx import ANI1ccxDatasetHandler
        """
        with self._mock_handler_module() as mock_cls:
            mock_cls.return_value = Mock()
            ANI1ccxDataset.create_handler(
                dataset_config=Mock(),
                filter_config=Mock(),
                processing_config=Mock(),
                logger=Mock(),
                experimental_setup=None,
            )
            mock_cls.assert_called_once()

    def test_create_handler_passes_all_args_to_constructor(self):
        """create_handler passes all 5 arguments to ANI1ccxDatasetHandler().

        Evidence: ani1ccx.py create_handler() return ANI1ccxDatasetHandler(...).
        """
        mock_dataset_config = Mock(name="dataset_config")
        mock_filter_config = Mock(name="filter_config")
        mock_processing_config = Mock(name="processing_config")
        mock_logger = Mock(name="logger")
        mock_experimental_setup = Mock(name="experimental_setup")

        with self._mock_handler_module() as mock_cls:
            mock_cls.return_value = Mock()
            ANI1ccxDataset.create_handler(
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
        """create_handler returns the ANI1ccxDatasetHandler instance.

        Evidence: ani1ccx.py: return ANI1ccxDatasetHandler(...).
        """
        mock_handler_instance = Mock(name="handler_instance")
        with self._mock_handler_module() as mock_cls:
            mock_cls.return_value = mock_handler_instance
            result = ANI1ccxDataset.create_handler(
                dataset_config=Mock(),
                filter_config=Mock(),
                processing_config=Mock(),
                logger=Mock(),
            )
            self.assertIs(result, mock_handler_instance)

    def test_create_handler_has_docstring(self):
        """create_handler method has a non-empty docstring mentioning lazy import."""
        method = ANI1ccxDataset.create_handler
        self.assertIsNotNone(method.__doc__)
        self.assertIn("lazy import", method.__doc__.lower())


# ============================================================================
# GROUP 11: ANI1ccxDataset — handler_class Default (3 tests)
# ============================================================================


class TestANI1ccxDatasetHandlerClassAttribute(unittest.TestCase):
    """Verify ANI1ccxDataset.handler_class is None (default from BaseDataset).

    Evidence: ani1ccx.py NOTE comment about handler_class intentionally NOT set.
    Evidence: base.py BaseDataset optional handler_class (project structure line 349).
    """

    def test_handler_class_is_none(self):
        """handler_class is None (default from BaseDataset).

        Evidence: ANI1ccxDatasetHandler is registered via @register_handler decorator
        and discovered dynamically through the HandlerRegistry.
        """
        self.assertIsNone(ANI1ccxDataset.handler_class)

    def test_converter_class_is_none(self):
        """converter_class is None (default from BaseDataset).

        Evidence: base.py BaseDataset optional converter_class
        (project structure line 349).
        """
        self.assertIsNone(ANI1ccxDataset.converter_class)

    def test_validator_class_is_none(self):
        """validator_class is None (default from BaseDataset).

        Evidence: base.py BaseDataset optional validator_class
        (project structure line 349).
        """
        self.assertIsNone(ANI1ccxDataset.validator_class)


# ============================================================================
# GROUP 12: ANI1ccxDataset — Method Signatures and Return Annotations (6 tests)
# ============================================================================


class TestANI1ccxDatasetMethodSignatures(unittest.TestCase):
    """Verify method signatures and return type annotations."""

    def _get_sig(self, method_name: str) -> inspect.Signature:
        """Helper: get the signature of an ANI1ccxDataset method."""
        method = getattr(ANI1ccxDataset, method_name)
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
# GROUP 13: ANI1ccxDataset — Method Docstrings (4 tests with subTests)
# ============================================================================


class TestANI1ccxDatasetMethodDocstrings(unittest.TestCase):
    """Verify each ANI1ccxDataset method has a non-empty docstring."""

    def test_each_classmethod_has_docstring(self):
        """Every expected classmethod has a non-empty docstring."""
        for method_name in EXPECTED_CLASSMETHOD_NAMES:
            with self.subTest(method=method_name):
                method = getattr(ANI1ccxDataset, method_name)
                doc = getattr(method, "__doc__", None)
                self.assertIsNotNone(doc, f"{method_name} has no docstring")
                self.assertGreater(
                    len(doc.strip()),
                    0,
                    f"{method_name} has empty docstring",
                )

    def test_get_required_properties_docstring_mentions_mapping(self):
        """get_required_properties docstring references HDF5 key mapping.

        Evidence: ani1ccx.py get_required_properties docstring mentions
        'ccsd(t)_cbs.energy' → 'ccsd_energy' mapping.
        """
        method = ANI1ccxDataset.get_required_properties
        doc = method.__doc__
        self.assertIn("ccsd_energy", doc)

    def test_get_feature_support_docstring_mentions_features(self):
        """get_feature_support docstring references feature flags.

        Evidence: ani1ccx.py get_feature_support docstring lists available features.
        """
        method = ANI1ccxDataset.get_feature_support
        doc = method.__doc__
        self.assertIn("vibrational_analysis", doc)

    def test_get_molecule_creation_strategy_docstring_mentions_coordinate_based(self):
        """get_molecule_creation_strategy docstring references coordinate_based strategy.

        Evidence: ani1ccx.py get_molecule_creation_strategy docstring explains why
        coordinate_based is used (no parseable identifiers in HDF5).
        """
        method = ANI1ccxDataset.get_molecule_creation_strategy
        doc = method.__doc__
        self.assertIn("coordinate_based", doc)


# ============================================================================
# GROUP 14: ANI1ccxDataset — Module-Level Imports and Exports (5 tests)
# ============================================================================


class TestANI1ccxDatasetModuleImportsAndExports(unittest.TestCase):
    """Verify the ani1ccx implementation module imports and exports correctly."""

    def test_module_has_docstring(self):
        """The ani1ccx.py module has a non-empty module docstring."""
        import milia_pipeline.datasets.implementations.ani1ccx as mod

        self.assertIsNotNone(mod.__doc__)
        self.assertGreater(len(mod.__doc__.strip()), 0)

    def test_module_exports_ani1ccx_dataset(self):
        """ANI1ccxDataset is importable from the implementations.ani1ccx module."""
        import milia_pipeline.datasets.implementations.ani1ccx as mod

        self.assertTrue(hasattr(mod, "ANI1ccxDataset"))
        self.assertIs(mod.ANI1ccxDataset, ANI1ccxDataset)

    def test_module_imports_base_classes(self):
        """Module imports BaseDataset and data classes from base.py.

        Evidence: ani1ccx.py imports from milia_pipeline.datasets.base.
        """
        source = inspect.getsource(sys.modules["milia_pipeline.datasets.implementations.ani1ccx"])
        self.assertIn("from milia_pipeline.datasets.base import", source)
        self.assertIn("BaseDataset", source)
        self.assertIn("DatasetMetadata", source)
        self.assertIn("DatasetSchema", source)
        self.assertIn("DatasetFeatures", source)

    def test_module_imports_register_decorator(self):
        """Module imports @register from registry.

        Evidence: ani1ccx.py imports register from milia_pipeline.datasets.registry.
        """
        source = inspect.getsource(sys.modules["milia_pipeline.datasets.implementations.ani1ccx"])
        self.assertIn("from milia_pipeline.datasets.registry import register", source)

    def test_module_does_not_import_handler_at_module_level(self):
        """ANI1ccxDatasetHandler is NOT imported at module level (lazy import only).

        Evidence: ani1ccx.py NOTE comment about circular import prevention.
        The handler is only imported inside create_handler() method.

        Uses ast module to reliably distinguish module-level imports from
        imports nested inside function/method bodies.
        """
        import ast

        source = inspect.getsource(sys.modules["milia_pipeline.datasets.implementations.ani1ccx"])
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
            "ANI1ccxDatasetHandler",
            module_level_import_names,
            "ANI1ccxDatasetHandler should NOT be imported at module level "
            "(only inside create_handler() via lazy import)",
        )


# ============================================================================
# GROUP 15: ANI1ccxDataset — DatasetFeatures.to_dict() and .supports() (4 tests)
# ============================================================================


class TestANI1ccxDatasetFeaturesIntegration(unittest.TestCase):
    """Verify DatasetFeatures integration methods work correctly with ANI-1ccx.

    Evidence: base.py DatasetFeatures.to_dict() and .supports() methods
    (project structure line 346).
    """

    def test_to_dict_returns_expected_dict(self):
        """features.to_dict() returns the full feature flags dictionary."""
        result = ANI1ccxDataset.features.to_dict()
        self.assertEqual(result, EXPECTED_FEATURES)

    def test_supports_atomization_energy(self):
        """features.supports('atomization_energy') returns True.

        ANI-1ccx specific: atomization_energy is the only enabled feature.
        Can compute from ccsd(t)_cbs.energy.
        """
        self.assertTrue(ANI1ccxDataset.features.supports("atomization_energy"))

    def test_supports_vibrational_analysis_false(self):
        """features.supports('vibrational_analysis') returns False.

        ANI-1ccx specific: no vibrational frequencies available.
        """
        self.assertFalse(ANI1ccxDataset.features.supports("vibrational_analysis"))

    def test_to_dict_keys_match_expected_features(self):
        """features.to_dict() keys match all 8 expected feature names."""
        result = ANI1ccxDataset.features.to_dict()
        self.assertEqual(set(result.keys()), set(EXPECTED_FEATURES.keys()))


# ============================================================================
# GROUP 16: ANI1ccxDataset — Schema Consistency with Methods (3 tests)
# ============================================================================


class TestANI1ccxDatasetSchemaMethodConsistency(unittest.TestCase):
    """Verify schema data is consistent with method return values."""

    def test_required_properties_matches_schema(self):
        """get_required_properties() returns the same values as schema.required_properties.

        Evidence: ani1ccx.py: return list(cls.schema.required_properties).
        """
        method_result = ANI1ccxDataset.get_required_properties()
        schema_result = list(ANI1ccxDataset.schema.required_properties)
        self.assertEqual(method_result, schema_result)

    def test_feature_support_matches_features(self):
        """get_feature_support() returns the same values as features.to_dict().

        Evidence: ani1ccx.py: return cls.features.to_dict().
        """
        method_result = ANI1ccxDataset.get_feature_support()
        features_result = ANI1ccxDataset.features.to_dict()
        self.assertEqual(method_result, features_result)

    def test_required_properties_count(self):
        """get_required_properties() has exactly 3 items (ccsd_energy, atoms, coordinates)."""
        result = ANI1ccxDataset.get_required_properties()
        self.assertEqual(len(result), 3)


# ============================================================================
# GROUP 17: ANI1ccxDataset — Edge Cases and Robustness (6 tests)
# ============================================================================


class TestANI1ccxDatasetEdgeCases(unittest.TestCase):
    """Test edge cases and robustness of ANI1ccxDataset."""

    def test_multiple_calls_return_consistent_results(self):
        """Multiple calls to classmethods return identical results."""
        for _ in range(3):
            self.assertEqual(
                ANI1ccxDataset.get_required_properties(),
                list(EXPECTED_REQUIRED_PROPERTIES),
            )
            self.assertEqual(
                ANI1ccxDataset.get_feature_support(),
                EXPECTED_FEATURES,
            )
            self.assertEqual(
                ANI1ccxDataset.get_molecule_creation_strategy(),
                EXPECTED_MOLECULE_CREATION_STRATEGY,
            )

    def test_classmethods_callable_on_class_not_instance(self):
        """All classmethods are callable on the class directly (no instantiation needed)."""
        # ANI1ccxDataset is never instantiated — these are all classmethods
        for method_name in EXPECTED_CLASSMETHOD_NAMES:
            with self.subTest(method=method_name):
                method = getattr(ANI1ccxDataset, method_name)
                self.assertTrue(callable(method))

    def test_identifier_keys_is_empty_tuple(self):
        """identifier_keys is an empty tuple — ANI-1ccx has no chemical identifiers.

        CRITICAL DIFFERENCE FROM DFT: DFT has (('inchi', 'inchi'), ('graphs', 'smiles')).
        ANI-1ccx has () because the HDF5 file has no parseable identifiers.
        Same as ANI-1x — both are in the same HDF5 file (ani1x-release.h5).
        """
        self.assertEqual(ANI1ccxDataset.schema.identifier_keys, ())
        self.assertIsInstance(ANI1ccxDataset.schema.identifier_keys, tuple)
        self.assertEqual(len(ANI1ccxDataset.schema.identifier_keys), 0)

    def test_create_handler_with_none_experimental_setup(self):
        """create_handler works when experimental_setup is None (default)."""
        import contextlib

        @contextlib.contextmanager
        def _scoped_handler_mock():
            mock_handler_cls = Mock(name="MockANI1ccxDatasetHandler")
            mock_module = MagicMock()
            mock_module.ANI1ccxDatasetHandler = mock_handler_cls
            handler_mod_key = "milia_pipeline.handlers.implementations.ani1ccx"
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
            ANI1ccxDataset.create_handler(
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
        not instance attributes. ANI1ccxDataset is used as a class, not instantiated,
        but we verify the attributes live on the class itself."""
        self.assertIn("metadata", ANI1ccxDataset.__dict__)
        self.assertIn("schema", ANI1ccxDataset.__dict__)
        self.assertIn("features", ANI1ccxDataset.__dict__)
        self.assertIn("config_key", ANI1ccxDataset.__dict__)

    def test_strategy_is_not_identifier_based(self):
        """get_molecule_creation_strategy() does NOT return 'identifier_coordinate_based'.

        CRITICAL: ANI-1ccx differs from DFT/QM9 which use 'identifier_coordinate_based'.
        ANI-1ccx uses 'coordinate_based' because it has no chemical identifiers.
        """
        result = ANI1ccxDataset.get_molecule_creation_strategy()
        self.assertNotEqual(result, "identifier_coordinate_based")
        self.assertEqual(result, "coordinate_based")


# ============================================================================
# GROUP 18: ANI1ccxDataset — ANI-1ccx vs ANI-1x Differentiation (7 tests)
# ============================================================================


class TestANI1ccxVsANI1xDifferentiation(unittest.TestCase):
    """Verify ANI-1ccx is correctly differentiated from ANI-1x.

    Evidence: ani1ccx.py module docstring 'KEY DIFFERENCE FROM ANI-1x' section.
    ANI-1ccx is a SUBSET of ANI-1x with coupled-cluster energies.
    Both share the same HDF5 file but have different primary energy targets.
    """

    def test_different_metadata_name(self):
        """ANI1ccxDataset.metadata.name is 'ANI1ccx', not 'ANI1x'.

        Evidence: ani1ccx.py metadata name="ANI1ccx".
        """
        self.assertEqual(ANI1ccxDataset.metadata.name, "ANI1ccx")
        self.assertNotEqual(ANI1ccxDataset.metadata.name, "ANI1x")

    def test_different_config_key(self):
        """ANI1ccxDataset.config_key is 'ani1ccx_config', not 'ani1x_config'.

        Evidence: ani1ccx.py config_key = "ani1ccx_config".
        """
        self.assertEqual(ANI1ccxDataset.config_key, "ani1ccx_config")
        self.assertNotEqual(ANI1ccxDataset.config_key, "ani1x_config")

    def test_required_properties_use_ccsd_energy_not_energy(self):
        """ANI-1ccx uses 'ccsd_energy' as primary energy, not 'energy'.

        KEY DIFFERENCE: ANI-1x maps wb97x_dz.energy → 'energy'.
        ANI-1ccx maps ccsd(t)_cbs.energy → 'ccsd_energy'.
        """
        required = ANI1ccxDataset.get_required_properties()
        self.assertIn("ccsd_energy", required)
        self.assertNotIn("energy", required)

    def test_optional_properties_include_dft_energy(self):
        """ANI-1ccx has 'dft_energy' as optional (DFT energy for comparison).

        KEY DIFFERENCE: In ANI-1x, wb97x_dz.energy is the required primary energy.
        In ANI-1ccx, it becomes the optional 'dft_energy' for comparison purposes.
        """
        optional = ANI1ccxDataset.schema.optional_properties
        self.assertIn("dft_energy", optional)

    def test_same_molecule_creation_strategy_as_ani1x(self):
        """ANI-1ccx uses the same coordinate_based strategy as ANI-1x.

        Both datasets are in the same HDF5 file with no parseable identifiers.
        """
        self.assertEqual(
            ANI1ccxDataset.get_molecule_creation_strategy(),
            "coordinate_based",
        )

    def test_same_identifier_keys_as_ani1x(self):
        """ANI-1ccx has the same empty identifier_keys as ANI-1x.

        Both datasets share the same HDF5 structure with no chemical identifiers.
        """
        self.assertEqual(ANI1ccxDataset.schema.identifier_keys, ())

    def test_description_mentions_coupled_cluster(self):
        """ANI-1ccx metadata description mentions coupled-cluster.

        This distinguishes it from ANI-1x which mentions DFT conformations.
        """
        self.assertIn("coupled-cluster", ANI1ccxDataset.metadata.description)


# ============================================================================
# TEST RUNNER
# ============================================================================


def run_comprehensive_suite():
    """Run all test groups in a structured order."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestANI1ccxDatasetClassIdentity,  # GROUP 1:  8 tests
        TestANI1ccxDatasetRegistration,  # GROUP 2:  5 tests
        TestANI1ccxDatasetMetadata,  # GROUP 3:  7 tests
        TestANI1ccxDatasetSchema,  # GROUP 4:  9 tests
        TestANI1ccxDatasetFeatures,  # GROUP 5: 10 tests
        TestANI1ccxDatasetConfigKey,  # GROUP 6:  2 tests
        TestANI1ccxDatasetGetRequiredProperties,  # GROUP 7:  5 tests
        TestANI1ccxDatasetGetFeatureSupport,  # GROUP 8:  6 tests
        TestANI1ccxDatasetGetMoleculeCreationStrategy,  # GROUP 9:  4 tests
        TestANI1ccxDatasetCreateHandler,  # GROUP 10: 7 tests
        TestANI1ccxDatasetHandlerClassAttribute,  # GROUP 11: 3 tests
        TestANI1ccxDatasetMethodSignatures,  # GROUP 12: 6 tests
        TestANI1ccxDatasetMethodDocstrings,  # GROUP 13: 4 tests
        TestANI1ccxDatasetModuleImportsAndExports,  # GROUP 14: 5 tests
        TestANI1ccxDatasetFeaturesIntegration,  # GROUP 15: 4 tests
        TestANI1ccxDatasetSchemaMethodConsistency,  # GROUP 16: 3 tests
        TestANI1ccxDatasetEdgeCases,  # GROUP 17: 6 tests
        TestANI1ccxVsANI1xDifferentiation,  # GROUP 18: 7 tests
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS — datasets/implementations/ani1ccx.py")
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
TEST SUITE SUMMARY — milia_pipeline/datasets/implementations/ani1ccx.py
====================================================================

101 comprehensive production-ready tests covering:

GROUP 1: ANI1ccxDataset Class Identity and Type Hierarchy (8 tests)
- Is a class, correct name, correct module
- Subclass of BaseDataset, not BaseDataset itself
- Has docstring, docstring mentions ANI-1ccx and coordinate_based
- MRO includes BaseDataset

GROUP 2: ANI1ccxDataset Registration with @register (5 tests)
- Is registered in default registry under 'ANI1ccx'
- get('ANI1ccx') returns ANI1ccxDataset class
- Listed in list_all() results
- Module imports @register decorator
- Registration key matches metadata.name

GROUP 3: ANI1ccxDataset.metadata — DatasetMetadata (7 tests)
- Is DatasetMetadata instance
- name='ANI1ccx', version='1.0.0', description, author, license='CC0'
- Metadata is frozen (immutable)

GROUP 4: ANI1ccxDataset.schema — DatasetSchema (9 tests)
- Is DatasetSchema instance
- required_properties=('ccsd_energy', 'atoms', 'coordinates')
- optional_properties (dft_energy, forces, charges, dipole, molecule_id)
- identifier_keys=() — EMPTY (no parseable identifiers)
- coordinate_units='angstrom', energy_units='hartree'
- Schema is frozen, properties are tuples, identifier_keys is tuple

GROUP 5: ANI1ccxDataset.features — DatasetFeatures (10 tests)
- Is DatasetFeatures instance
- All 8 feature flags verified individually
- Only atomization_energy=True, all others False
- Features is frozen (immutable)

GROUP 6: ANI1ccxDataset.config_key (2 tests)
- Value is 'ani1ccx_config', is a string

GROUP 7: ANI1ccxDataset.get_required_properties() (5 tests)
- Is classmethod, returns list
- Returns ['ccsd_energy', 'atoms', 'coordinates']
- Returns fresh list each call, all items are strings

GROUP 8: ANI1ccxDataset.get_feature_support() (6 tests)
- Is classmethod, returns dict
- Returns correct 8 feature flags (only atomization_energy True)
- All values are booleans
- Delegates to features.to_dict()

GROUP 9: ANI1ccxDataset.get_molecule_creation_strategy() (4 tests)
- Is classmethod, returns string
- Returns 'coordinate_based' (NOT 'identifier_coordinate_based')
- Has docstring

GROUP 10: ANI1ccxDataset.create_handler() — Lazy Import (7 tests)
- Is classmethod, correct 5-parameter signature
- experimental_setup defaults to None
- Performs lazy import of ANI1ccxDatasetHandler
- Passes all args to constructor
- Returns handler instance
- Has docstring mentioning lazy import

GROUP 11: ANI1ccxDataset handler_class Default (3 tests)
- handler_class is None
- converter_class is None
- validator_class is None

GROUP 12: ANI1ccxDataset Method Signatures and Return Annotations (6 tests)
- get_required_properties -> List[str]
- get_feature_support -> Dict[str, bool]
- get_molecule_creation_strategy -> str
- All classmethods have no params (bound method excludes cls)

GROUP 13: ANI1ccxDataset Method Docstrings (4 tests with subTests)
- All 4 classmethods have non-empty docstrings
- Docstrings reference HDF5 mapping (ccsd_energy), features, coordinate_based

GROUP 14: Module-Level Imports and Exports (5 tests)
- Module has docstring, exports ANI1ccxDataset
- Imports base classes and @register
- Does NOT import ANI1ccxDatasetHandler at module level

GROUP 15: DatasetFeatures Integration (4 tests)
- to_dict() returns expected dict
- supports() works for enabled/disabled features
- to_dict() keys match all 8 features

GROUP 16: Schema-Method Consistency (3 tests)
- get_required_properties() matches schema.required_properties
- get_feature_support() matches features.to_dict()
- Required properties count is 3

GROUP 17: Edge Cases and Robustness (6 tests)
- Multiple calls return consistent results
- Classmethods callable on class
- identifier_keys is empty tuple (ANI-1ccx specific)
- create_handler with None experimental_setup
- Class attributes live in __dict__
- Strategy is NOT identifier_coordinate_based (ANI-1ccx vs DFT distinction)

GROUP 18: ANI-1ccx vs ANI-1x Differentiation (7 tests)
- Different metadata name (ANI1ccx vs ANI1x)
- Different config_key (ani1ccx_config vs ani1x_config)
- Uses ccsd_energy (not energy) as required primary energy
- Has dft_energy as optional property
- Same coordinate_based strategy as ANI-1x
- Same empty identifier_keys as ANI-1x
- Description mentions coupled-cluster

Total: 101 comprehensive production-ready tests

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
- ANI-1ccx-specific distinctions tested (coordinate_based, empty identifier_keys)
- ANI-1ccx vs ANI-1x differentiation tested (ccsd_energy, dft_energy, coupled-cluster)
"""
