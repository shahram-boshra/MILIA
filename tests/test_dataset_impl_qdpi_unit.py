#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/datasets/implementations/qdpi.py

Module under test: qdpi.py
- QDPiDataset: BaseDataset subclass with @register decorator
  - metadata: DatasetMetadata (Pydantic frozen dataclass)
  - schema: DatasetSchema (Pydantic frozen dataclass)
  - features: DatasetFeatures (Pydantic frozen dataclass)
  - config_key: str
  - get_required_properties(): classmethod -> List[str]
  - get_feature_support(): classmethod -> Dict[str, bool]
  - get_molecule_creation_strategy(): classmethod -> str
  - get_supported_elements(): classmethod -> List[int]
  - get_supported_element_symbols(): classmethod -> List[str]
  - supports_charged_molecules(): classmethod -> bool
  - get_source_subsets(): classmethod -> List[str]
  - create_handler(): classmethod -> QDPiDatasetHandler (lazy import)

Test path on local machine: ~/ml_projects/milia/tests/test_dataset_impl_qdpi_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/datasets/implementations/qdpi.py

NOTE: This test suite runs inside Docker at /app/milia
Path mappings:
- Project root: /app/milia (mapped from ~/ml_projects/milia)

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution

Evidence sources:
- qdpi.py: Complete source (provided)
- MILIA_Pipeline_Project_Structure.md: base.py details (lines 335-351),
  registry.py details (lines 369-375), implementations/ structure (lines 377-380)
- test_dataset_impl_xxmd_unit.py: Test conventions and patterns (provided)

QDPi-specific characteristics (from qdpi.py source):
- ~1.6 million structures for drug discovery ML potentials
- 13 elements: H, Li, C, N, O, F, Na, P, S, Cl, K, Br, I
- Contains BOTH neutral AND charged molecules (unlike ANI-2x)
- No parseable chemical identifiers — coordinate_based strategy
- identifier_keys = () — empty tuple
- Energy: eV → Hartree conversion during preprocessing
- Forces: eV/Angstrom → Hartree/Angstrom conversion
- HDF5 format (DeePMD-kit)
- 7 source subsets: spice, ani, geom, freesolvmd, re, remd, comp6
- DFT functional: ωB97M-D3(BJ)/def2-TZVPPD
- License: CC BY 4.0
- Additional classmethods: get_supported_elements, get_supported_element_symbols,
  supports_charged_molecules, get_source_subsets

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
from milia_pipeline.datasets.implementations.qdpi import (
    QDPiDataset,
)
from milia_pipeline.datasets.registry import (
    is_registered,
)

# ============================================================================
# CONSTANTS: Expected values derived from qdpi.py source
# ============================================================================

EXPECTED_METADATA_NAME = "QDPi"
EXPECTED_METADATA_VERSION = "1.0.0"
EXPECTED_METADATA_DESCRIPTION = (
    "Quantum Deep Potential Interaction dataset for drug discovery. "
    "Contains ~1.6 million structures of drug-like molecules and biopolymer "
    "fragments at \u03c9B97M-D3(BJ)/def2-TZVPPD level. Includes both neutral and "
    "charged molecules for comprehensive protonation state coverage."
)
EXPECTED_METADATA_AUTHOR = "Zeng, Giese, G\u00f6tz, York (Rutgers LBSR)"
EXPECTED_METADATA_LICENSE = "CC BY 4.0"

EXPECTED_REQUIRED_PROPERTIES = ("atoms", "coordinates", "energy")
EXPECTED_OPTIONAL_PROPERTIES = (
    "forces",  # Atomic forces in Hartree/Angstrom (converted from eV/Angstrom)
    "formula",  # Chemical formula group identifier from HDF5
    "molecular_charge",  # CRITICAL: Molecular charge for charged molecules
    "subset",  # Source subset identifier (spice, ani, geom, etc.)
    "charge_type",  # 'neutral' or 'charged' (from directory structure)
)
# CRITICAL: QDPi has NO parseable chemical identifiers — empty tuple
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

EXPECTED_CONFIG_KEY = "qdpi_config"
EXPECTED_MOLECULE_CREATION_STRATEGY = "coordinate_based"

# QDPi-specific: 13 supported elements (atomic numbers)
EXPECTED_SUPPORTED_ELEMENTS = [1, 3, 6, 7, 8, 9, 11, 15, 16, 17, 19, 35, 53]
EXPECTED_SUPPORTED_ELEMENT_SYMBOLS = [
    "H",
    "Li",
    "C",
    "N",
    "O",
    "F",
    "Na",
    "P",
    "S",
    "Cl",
    "K",
    "Br",
    "I",
]

# QDPi-specific: 7 source subsets
EXPECTED_SOURCE_SUBSETS = ["spice", "ani", "geom", "freesolvmd", "re", "remd", "comp6"]

EXPECTED_CLASSMETHOD_NAMES = [
    "get_required_properties",
    "get_feature_support",
    "get_molecule_creation_strategy",
    "create_handler",
    "get_supported_elements",
    "get_supported_element_symbols",
    "supports_charged_molecules",
    "get_source_subsets",
]

# Sentinel for sys.modules cleanup in scoped handler mocking
_SENTINEL = object()


# ============================================================================
# GROUP 1: QDPiDataset — Class Identity and Type Hierarchy (8 tests)
# ============================================================================


class TestQDPiDatasetClassIdentity(unittest.TestCase):
    """Verify QDPiDataset is a proper BaseDataset subclass with correct identity."""

    def test_is_a_class(self):
        """QDPiDataset is a class (not a function or module)."""
        self.assertTrue(inspect.isclass(QDPiDataset))

    def test_has_correct_name(self):
        """Class name is 'QDPiDataset'."""
        self.assertEqual(QDPiDataset.__name__, "QDPiDataset")

    def test_has_correct_module(self):
        """Defined in the datasets.implementations.qdpi module."""
        self.assertIn("implementations.qdpi", QDPiDataset.__module__)

    def test_is_subclass_of_base_dataset(self):
        """QDPiDataset inherits from BaseDataset."""
        self.assertTrue(
            issubclass(QDPiDataset, BaseDataset),
            "QDPiDataset must be a subclass of BaseDataset",
        )

    def test_is_not_base_dataset_itself(self):
        """QDPiDataset is a distinct class, not BaseDataset itself."""
        self.assertIsNot(QDPiDataset, BaseDataset)

    def test_has_docstring(self):
        """QDPiDataset has a non-empty docstring."""
        self.assertIsNotNone(QDPiDataset.__doc__)
        self.assertGreater(len(QDPiDataset.__doc__.strip()), 0)

    def test_docstring_mentions_qdpi_and_coordinate_based(self):
        """QDPiDataset docstring references QDπ dataset and coordinate_based.

        Evidence: qdpi.py class docstring mentions 'QDπ' and 'coordinate_based'.
        """
        doc = QDPiDataset.__doc__
        # The docstring uses the Unicode π character
        self.assertTrue(
            "QD" in doc and "coordinate_based" in doc,
            "Docstring must mention QD(π) and coordinate_based",
        )

    def test_mro_includes_base_dataset(self):
        """Method Resolution Order includes BaseDataset."""
        self.assertIn(BaseDataset, QDPiDataset.__mro__)


# ============================================================================
# GROUP 2: QDPiDataset — Registration with @register (5 tests)
# ============================================================================


class TestQDPiDatasetRegistration(unittest.TestCase):
    """Verify QDPiDataset is registered via @register decorator."""

    def test_is_registered_in_default_registry(self):
        """QDPiDataset is discoverable in the default DatasetRegistry.

        Evidence: @register decorator applied at class definition (qdpi.py).
        Evidence: registry.py convenience function is_registered().
        """
        self.assertTrue(
            is_registered("QDPi"),
            "QDPiDataset must be registered under name 'QDPi'",
        )

    def test_get_returns_qdpi_dataset_class(self):
        """Registry get('QDPi') returns the QDPiDataset class.

        Evidence: registry.py get() method returns the registered class.
        """
        from milia_pipeline.datasets.registry import get

        retrieved = get("QDPi")
        self.assertIs(retrieved, QDPiDataset)

    def test_listed_in_all_datasets(self):
        """QDPiDataset name appears in list_all() results.

        Evidence: registry.py list_all() returns all registered names.
        """
        from milia_pipeline.datasets.registry import list_all

        all_names = list_all()
        self.assertIn("QDPi", all_names)

    def test_register_decorator_is_imported(self):
        """The qdpi module imports the register decorator.

        Evidence: qdpi.py imports register from milia_pipeline.datasets.registry.
        """
        source = inspect.getsource(sys.modules[QDPiDataset.__module__])
        self.assertIn("from milia_pipeline.datasets.registry import register", source)

    def test_registration_uses_metadata_name(self):
        """Registration key matches metadata.name ('QDPi').

        Evidence: @register decorator uses cls.metadata.name (registry.py convention).
        Evidence: qdpi.py metadata.name = 'QDPi'.
        """
        self.assertEqual(QDPiDataset.metadata.name, "QDPi")
        self.assertTrue(is_registered(QDPiDataset.metadata.name))


# ============================================================================
# GROUP 3: QDPiDataset.metadata — DatasetMetadata (7 tests)
# ============================================================================


class TestQDPiDatasetMetadata(unittest.TestCase):
    """Verify QDPiDataset.metadata is a correctly configured DatasetMetadata.

    Evidence: qdpi.py metadata definition, base.py DatasetMetadata Pydantic frozen dataclass.
    """

    def test_metadata_is_dataset_metadata_instance(self):
        """metadata attribute is a DatasetMetadata instance."""
        self.assertIsInstance(QDPiDataset.metadata, DatasetMetadata)

    def test_metadata_name(self):
        """metadata.name is 'QDPi'."""
        self.assertEqual(QDPiDataset.metadata.name, EXPECTED_METADATA_NAME)

    def test_metadata_version(self):
        """metadata.version is '1.0.0'."""
        self.assertEqual(QDPiDataset.metadata.version, EXPECTED_METADATA_VERSION)

    def test_metadata_description(self):
        """metadata.description matches expected QDPi description."""
        self.assertEqual(
            QDPiDataset.metadata.description,
            EXPECTED_METADATA_DESCRIPTION,
        )

    def test_metadata_author(self):
        """metadata.author is 'Zeng, Giese, Götz, York (Rutgers LBSR)'."""
        self.assertEqual(QDPiDataset.metadata.author, EXPECTED_METADATA_AUTHOR)

    def test_metadata_license(self):
        """metadata.license is 'CC BY 4.0'.

        Evidence: qdpi.py metadata license="CC BY 4.0".
        """
        self.assertEqual(QDPiDataset.metadata.license, EXPECTED_METADATA_LICENSE)

    def test_metadata_is_frozen(self):
        """metadata is immutable (Pydantic frozen dataclass).

        Evidence: base.py DatasetMetadata is a Pydantic frozen dataclass
        (project structure line 337-339).
        """
        with self.assertRaises((AttributeError, TypeError, Exception)):
            # Pydantic frozen dataclasses raise on attribute assignment
            QDPiDataset.metadata.name = "MODIFIED"


# ============================================================================
# GROUP 4: QDPiDataset.schema — DatasetSchema (9 tests)
# ============================================================================


class TestQDPiDatasetSchema(unittest.TestCase):
    """Verify QDPiDataset.schema is a correctly configured DatasetSchema.

    Evidence: qdpi.py schema definition, base.py DatasetSchema Pydantic frozen dataclass.
    """

    def test_schema_is_dataset_schema_instance(self):
        """schema attribute is a DatasetSchema instance."""
        self.assertIsInstance(QDPiDataset.schema, DatasetSchema)

    def test_schema_required_properties(self):
        """schema.required_properties is ('atoms', 'coordinates', 'energy').

        Evidence: qdpi.py schema required_properties definition.
        QDPi uses 'energy' (singular), same as xxMD.
        """
        self.assertEqual(
            QDPiDataset.schema.required_properties,
            EXPECTED_REQUIRED_PROPERTIES,
        )

    def test_schema_optional_properties(self):
        """schema.optional_properties contains forces, formula, molecular_charge, subset, charge_type.

        Evidence: qdpi.py schema optional_properties definition.
        CRITICAL: molecular_charge is essential for charged molecule bond order determination.
        """
        self.assertEqual(
            QDPiDataset.schema.optional_properties,
            EXPECTED_OPTIONAL_PROPERTIES,
        )

    def test_schema_identifier_keys_empty(self):
        """schema.identifier_keys is an empty tuple.

        CRITICAL: QDPi has NO parseable chemical identifiers.
        HDF5 format contains only elements, atomic_types, coordinates, energies, forces.
        Evidence: qdpi.py schema identifier_keys=() and extensive comments.
        """
        self.assertEqual(
            QDPiDataset.schema.identifier_keys,
            EXPECTED_IDENTIFIER_KEYS,
        )
        self.assertEqual(len(QDPiDataset.schema.identifier_keys), 0)

    def test_schema_coordinate_units(self):
        """schema.coordinate_units is 'angstrom'.

        Evidence: QDPi coordinates are in Angstrom.
        """
        self.assertEqual(
            QDPiDataset.schema.coordinate_units,
            EXPECTED_COORDINATE_UNITS,
        )

    def test_schema_energy_units(self):
        """schema.energy_units is 'hartree'.

        Evidence: QDPi energies are originally in eV but converted
        to Hartree during preprocessing for consistency with MILIA standardization.
        The schema reflects POST-PREPROCESSING units.
        """
        self.assertEqual(
            QDPiDataset.schema.energy_units,
            EXPECTED_ENERGY_UNITS,
        )

    def test_schema_is_frozen(self):
        """schema is immutable (Pydantic frozen dataclass).

        Evidence: base.py DatasetSchema is a Pydantic frozen dataclass
        (project structure line 340-343).
        """
        with self.assertRaises((AttributeError, TypeError, Exception)):
            QDPiDataset.schema.required_properties = ("modified",)

    def test_schema_required_properties_are_tuples(self):
        """required_properties and optional_properties are tuples (immutable sequences)."""
        self.assertIsInstance(QDPiDataset.schema.required_properties, tuple)
        self.assertIsInstance(QDPiDataset.schema.optional_properties, tuple)

    def test_schema_identifier_keys_is_tuple(self):
        """identifier_keys is a tuple (even though empty).

        Evidence: qdpi.py schema identifier_keys=() — empty tuple, not None or list.
        """
        self.assertIsInstance(QDPiDataset.schema.identifier_keys, tuple)


# ============================================================================
# GROUP 5: QDPiDataset.features — DatasetFeatures (10 tests)
# ============================================================================


class TestQDPiDatasetFeatures(unittest.TestCase):
    """Verify QDPiDataset.features is a correctly configured DatasetFeatures.

    Evidence: qdpi.py features definition, base.py DatasetFeatures Pydantic frozen dataclass.
    QDPi is a DFT dataset with limited analysis features — only atomization_energy is True.
    """

    def test_features_is_dataset_features_instance(self):
        """features attribute is a DatasetFeatures instance."""
        self.assertIsInstance(QDPiDataset.features, DatasetFeatures)

    def test_vibrational_analysis_disabled(self):
        """features.vibrational_analysis is False.

        Evidence: QDPi does not have vibrational frequencies.
        """
        self.assertFalse(QDPiDataset.features.vibrational_analysis)

    def test_uncertainty_handling_disabled(self):
        """features.uncertainty_handling is False.

        Evidence: QDPi is deterministic DFT — no statistical uncertainties.
        """
        self.assertFalse(QDPiDataset.features.uncertainty_handling)

    def test_atomization_energy_enabled(self):
        """features.atomization_energy is True.

        Evidence: Atomization energies can be calculated from total DFT energy.
        """
        self.assertTrue(QDPiDataset.features.atomization_energy)

    def test_rotational_constants_disabled(self):
        """features.rotational_constants is False.

        Evidence: QDPi does not have rotational constants.
        """
        self.assertFalse(QDPiDataset.features.rotational_constants)

    def test_frequency_analysis_disabled(self):
        """features.frequency_analysis is False.

        Evidence: QDPi does not have frequency analysis.
        """
        self.assertFalse(QDPiDataset.features.frequency_analysis)

    def test_orbital_analysis_disabled(self):
        """features.orbital_analysis is False.

        Evidence: QDPi does not have orbital analysis.
        """
        self.assertFalse(QDPiDataset.features.orbital_analysis)

    def test_homo_lumo_gap_disabled(self):
        """features.homo_lumo_gap is False.

        Evidence: QDPi does not have HOMO-LUMO gap.
        """
        self.assertFalse(QDPiDataset.features.homo_lumo_gap)

    def test_mo_energies_disabled(self):
        """features.mo_energies is False.

        Evidence: QDPi does not have MO energies.
        """
        self.assertFalse(QDPiDataset.features.mo_energies)

    def test_features_is_frozen(self):
        """features is immutable (Pydantic frozen dataclass).

        Evidence: base.py DatasetFeatures is a Pydantic frozen dataclass
        (project structure line 344-346).
        """
        with self.assertRaises((AttributeError, TypeError, Exception)):
            QDPiDataset.features.vibrational_analysis = True


# ============================================================================
# GROUP 6: QDPiDataset.config_key (2 tests)
# ============================================================================


class TestQDPiDatasetConfigKey(unittest.TestCase):
    """Verify QDPiDataset.config_key is correctly set.

    Evidence: qdpi.py config_key = "qdpi_config".
    """

    def test_config_key_value(self):
        """config_key is 'qdpi_config'."""
        self.assertEqual(QDPiDataset.config_key, EXPECTED_CONFIG_KEY)

    def test_config_key_is_string(self):
        """config_key is a string."""
        self.assertIsInstance(QDPiDataset.config_key, str)


# ============================================================================
# GROUP 7: QDPiDataset.get_required_properties() (5 tests)
# ============================================================================


class TestQDPiDatasetGetRequiredProperties(unittest.TestCase):
    """Verify QDPiDataset.get_required_properties() classmethod.

    Evidence: qdpi.py get_required_properties() implementation.
    QDPi required properties: atoms, coordinates, energy.
    """

    def test_is_classmethod(self):
        """get_required_properties is a classmethod."""
        descriptor = QDPiDataset.__dict__.get("get_required_properties")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_list(self):
        """get_required_properties() returns a list."""
        result = QDPiDataset.get_required_properties()
        self.assertIsInstance(result, list)

    def test_returns_correct_values(self):
        """get_required_properties() returns ['atoms', 'coordinates', 'energy'].

        Evidence: qdpi.py schema required_properties=('atoms', 'coordinates', 'energy').
        """
        result = QDPiDataset.get_required_properties()
        self.assertEqual(result, list(EXPECTED_REQUIRED_PROPERTIES))

    def test_returns_new_list_each_call(self):
        """get_required_properties() returns a fresh list (not the same object).

        Evidence: implementation uses list(cls.schema.required_properties),
        converting the tuple to a new list each time.
        """
        result1 = QDPiDataset.get_required_properties()
        result2 = QDPiDataset.get_required_properties()
        self.assertEqual(result1, result2)
        self.assertIsNot(result1, result2)

    def test_contains_all_strings(self):
        """All items in get_required_properties() are strings."""
        result = QDPiDataset.get_required_properties()
        for item in result:
            with self.subTest(item=item):
                self.assertIsInstance(item, str)


# ============================================================================
# GROUP 8: QDPiDataset.get_feature_support() (6 tests)
# ============================================================================


class TestQDPiDatasetGetFeatureSupport(unittest.TestCase):
    """Verify QDPiDataset.get_feature_support() classmethod.

    Evidence: qdpi.py get_feature_support() implementation.
    QDPi: Only atomization_energy is True, all other 7 flags are False.
    """

    def test_is_classmethod(self):
        """get_feature_support is a classmethod."""
        descriptor = QDPiDataset.__dict__.get("get_feature_support")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_dict(self):
        """get_feature_support() returns a dict."""
        result = QDPiDataset.get_feature_support()
        self.assertIsInstance(result, dict)

    def test_returns_correct_feature_flags(self):
        """get_feature_support() returns the expected feature flags dict.

        QDPi specifics: Only atomization_energy=True, everything else False.
        """
        result = QDPiDataset.get_feature_support()
        self.assertEqual(result, EXPECTED_FEATURES)

    def test_all_values_are_booleans(self):
        """All values in get_feature_support() dict are booleans."""
        result = QDPiDataset.get_feature_support()
        for key, value in result.items():
            with self.subTest(feature=key):
                self.assertIsInstance(value, bool)

    def test_has_exactly_8_feature_flags(self):
        """get_feature_support() returns exactly 8 feature flags.

        Evidence: base.py DatasetFeatures has 8 feature flags
        (project structure line 345).
        """
        result = QDPiDataset.get_feature_support()
        self.assertEqual(len(result), 8)

    def test_delegates_to_features_to_dict(self):
        """get_feature_support() delegates to cls.features.to_dict().

        Evidence: qdpi.py: return cls.features.to_dict()
        """
        direct_dict = QDPiDataset.features.to_dict()
        method_result = QDPiDataset.get_feature_support()
        self.assertEqual(direct_dict, method_result)


# ============================================================================
# GROUP 9: QDPiDataset.get_molecule_creation_strategy() (4 tests)
# ============================================================================


class TestQDPiDatasetGetMoleculeCreationStrategy(unittest.TestCase):
    """Verify QDPiDataset.get_molecule_creation_strategy() classmethod.

    Evidence: qdpi.py get_molecule_creation_strategy() implementation.
    CRITICAL: QDPi uses 'coordinate_based' — NO parseable identifiers available.
    """

    def test_is_classmethod(self):
        """get_molecule_creation_strategy is a classmethod."""
        descriptor = QDPiDataset.__dict__.get("get_molecule_creation_strategy")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_string(self):
        """get_molecule_creation_strategy() returns a string."""
        result = QDPiDataset.get_molecule_creation_strategy()
        self.assertIsInstance(result, str)

    def test_returns_coordinate_based(self):
        """get_molecule_creation_strategy() returns 'coordinate_based'.

        CRITICAL: QDPi HDF5 structure contains NO parseable chemical identifiers.
        Only elements, atomic_types, coordinates, energies, forces are available.
        Molecular connectivity is inferred from 3D coordinates using rdDetermineBonds.

        Evidence: qdpi.py docstring and Zeng et al., Scientific Data 12, 693 (2025).
        """
        result = QDPiDataset.get_molecule_creation_strategy()
        self.assertEqual(result, EXPECTED_MOLECULE_CREATION_STRATEGY)

    def test_has_docstring(self):
        """get_molecule_creation_strategy method has a non-empty docstring."""
        method = QDPiDataset.get_molecule_creation_strategy
        self.assertIsNotNone(method.__doc__)
        self.assertGreater(len(method.__doc__.strip()), 0)


# ============================================================================
# GROUP 10: QDPiDataset.create_handler() — Lazy Import Pattern (7 tests)
# ============================================================================


class TestQDPiDatasetCreateHandler(unittest.TestCase):
    """Verify QDPiDataset.create_handler() factory method with lazy import.

    Evidence: qdpi.py create_handler() implementation.
    The create_handler() method uses lazy import to break circular dependency
    between datasets/implementations/qdpi.py and handlers/implementations/qdpi.py.
    """

    def test_is_classmethod(self):
        """create_handler is a classmethod."""
        descriptor = QDPiDataset.__dict__.get("create_handler")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_has_correct_signature(self):
        """create_handler has the expected 5-parameter signature (+ cls).

        Evidence: base.py BaseDataset.create_handler() has 5-parameter signature
        (project structure line 351).

        NOTE: In Python 3.10+, inspect.signature() on a bound classmethod
        excludes 'cls' from the parameters. We verify the unbound signature
        via __func__ to capture all parameters including 'cls'.
        """
        # Access the underlying function to get the full signature including 'cls'
        unbound_func = QDPiDataset.__dict__["create_handler"].__func__
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

        Evidence: qdpi.py create_handler signature: experimental_setup=None.
        """
        sig = inspect.signature(QDPiDataset.create_handler)
        default = sig.parameters["experimental_setup"].default
        self.assertIsNone(default)

    def _mock_handler_module(self):
        """Helper: create a mock handler module with a mock QDPiDatasetHandler class.

        The actual milia_pipeline.handlers.implementations.qdpi module cannot be
        imported in the test environment due to handler dependencies.
        To test create_handler()'s lazy import behavior, we temporarily inject
        a mock module into sys.modules so that the
        'from milia_pipeline.handlers.implementations.qdpi import QDPiDatasetHandler'
        statement inside create_handler() resolves to our mock.

        This uses a context manager pattern to ensure sys.modules is cleaned up
        after each test (no mock pollution).
        """
        import contextlib

        @contextlib.contextmanager
        def _scoped_handler_mock():
            mock_handler_cls = Mock(name="MockQDPiDatasetHandler")
            mock_module = MagicMock()
            mock_module.QDPiDatasetHandler = mock_handler_cls

            handler_mod_key = "milia_pipeline.handlers.implementations.qdpi"
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
        """create_handler performs lazy import of QDPiDatasetHandler.

        Evidence: qdpi.py create_handler():
        from milia_pipeline.handlers.implementations.qdpi import QDPiDatasetHandler
        """
        with self._mock_handler_module() as mock_cls:
            mock_cls.return_value = Mock()
            QDPiDataset.create_handler(
                dataset_config=Mock(),
                filter_config=Mock(),
                processing_config=Mock(),
                logger=Mock(),
                experimental_setup=None,
            )
            mock_cls.assert_called_once()

    def test_create_handler_passes_all_args_to_constructor(self):
        """create_handler passes all 5 arguments to QDPiDatasetHandler().

        Evidence: qdpi.py create_handler() return QDPiDatasetHandler(...).
        """
        mock_dataset_config = Mock(name="dataset_config")
        mock_filter_config = Mock(name="filter_config")
        mock_processing_config = Mock(name="processing_config")
        mock_logger = Mock(name="logger")
        mock_experimental_setup = Mock(name="experimental_setup")

        with self._mock_handler_module() as mock_cls:
            mock_cls.return_value = Mock()
            QDPiDataset.create_handler(
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
        """create_handler returns the QDPiDatasetHandler instance.

        Evidence: qdpi.py: return QDPiDatasetHandler(...).
        """
        mock_handler_instance = Mock(name="handler_instance")
        with self._mock_handler_module() as mock_cls:
            mock_cls.return_value = mock_handler_instance
            result = QDPiDataset.create_handler(
                dataset_config=Mock(),
                filter_config=Mock(),
                processing_config=Mock(),
                logger=Mock(),
            )
            self.assertIs(result, mock_handler_instance)

    def test_create_handler_has_docstring(self):
        """create_handler method has a non-empty docstring mentioning lazy import."""
        method = QDPiDataset.create_handler
        self.assertIsNotNone(method.__doc__)
        self.assertIn("lazy import", method.__doc__.lower())


# ============================================================================
# GROUP 11: QDPiDataset — handler_class Default (3 tests)
# ============================================================================


class TestQDPiDatasetHandlerClassAttribute(unittest.TestCase):
    """Verify QDPiDataset.handler_class is None (default from BaseDataset).

    Evidence: qdpi.py NOTE comment about handler_class intentionally NOT set.
    Evidence: base.py BaseDataset optional handler_class (project structure line 349).
    """

    def test_handler_class_is_none(self):
        """handler_class is None (default from BaseDataset).

        Evidence: QDPiDatasetHandler is registered via @register_handler decorator
        and discovered dynamically through the HandlerRegistry.
        """
        self.assertIsNone(QDPiDataset.handler_class)

    def test_converter_class_is_none(self):
        """converter_class is None (default from BaseDataset).

        Evidence: base.py BaseDataset optional converter_class
        (project structure line 349).
        """
        self.assertIsNone(QDPiDataset.converter_class)

    def test_validator_class_is_none(self):
        """validator_class is None (default from BaseDataset).

        Evidence: base.py BaseDataset optional validator_class
        (project structure line 349).
        """
        self.assertIsNone(QDPiDataset.validator_class)


# ============================================================================
# GROUP 12: QDPiDataset — Method Signatures and Return Annotations (7 tests)
# ============================================================================


class TestQDPiDatasetMethodSignatures(unittest.TestCase):
    """Verify method signatures and return type annotations.

    QDPi has 8 classmethods including QDPi-specific ones.
    """

    def _get_sig(self, method_name: str) -> inspect.Signature:
        """Helper: get the signature of a QDPiDataset method."""
        method = getattr(QDPiDataset, method_name)
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

    def test_create_handler_has_5_params_excluding_cls(self):
        """create_handler has 5 parameters (excluding cls in bound signature).

        Evidence: base.py BaseDataset.create_handler() with 5-parameter signature.
        """
        sig = self._get_sig("create_handler")
        params = list(sig.parameters.keys())
        self.assertEqual(len(params), 5)
        self.assertEqual(
            params,
            [
                "dataset_config",
                "filter_config",
                "processing_config",
                "logger",
                "experimental_setup",
            ],
        )


# ============================================================================
# GROUP 13: QDPiDataset — Method Docstrings (4 tests with subTests)
# ============================================================================


class TestQDPiDatasetMethodDocstrings(unittest.TestCase):
    """Verify each QDPiDataset method has a non-empty docstring."""

    def test_each_classmethod_has_docstring(self):
        """Every expected classmethod has a non-empty docstring."""
        for method_name in EXPECTED_CLASSMETHOD_NAMES:
            with self.subTest(method=method_name):
                method = getattr(QDPiDataset, method_name)
                doc = getattr(method, "__doc__", None)
                self.assertIsNotNone(doc, f"{method_name} has no docstring")
                self.assertGreater(
                    len(doc.strip()),
                    0,
                    f"{method_name} has empty docstring",
                )

    def test_get_required_properties_docstring_mentions_properties(self):
        """get_required_properties docstring references QDPi properties.

        Evidence: qdpi.py get_required_properties docstring.
        """
        method = QDPiDataset.get_required_properties
        doc = method.__doc__
        self.assertIn("propert", doc.lower())

    def test_get_feature_support_docstring_mentions_feature(self):
        """get_feature_support docstring references feature support.

        Evidence: qdpi.py get_feature_support docstring.
        """
        method = QDPiDataset.get_feature_support
        doc = method.__doc__
        self.assertIn("feature", doc.lower())

    def test_get_molecule_creation_strategy_docstring_mentions_coordinate_based(self):
        """get_molecule_creation_strategy docstring references coordinate_based strategy.

        Evidence: qdpi.py get_molecule_creation_strategy docstring explains why
        coordinate_based is used (no parseable identifiers in HDF5).
        """
        method = QDPiDataset.get_molecule_creation_strategy
        doc = method.__doc__
        self.assertIn("coordinate_based", doc)


# ============================================================================
# GROUP 14: QDPiDataset — Module-Level Imports and Exports (5 tests)
# ============================================================================


class TestQDPiDatasetModuleImportsAndExports(unittest.TestCase):
    """Verify the qdpi implementation module imports and exports correctly."""

    def test_module_has_docstring(self):
        """The qdpi.py module has a non-empty module docstring."""
        import milia_pipeline.datasets.implementations.qdpi as mod

        self.assertIsNotNone(mod.__doc__)
        self.assertGreater(len(mod.__doc__.strip()), 0)

    def test_module_exports_qdpi_dataset(self):
        """QDPiDataset is importable from the implementations.qdpi module."""
        import milia_pipeline.datasets.implementations.qdpi as mod

        self.assertTrue(hasattr(mod, "QDPiDataset"))
        self.assertIs(mod.QDPiDataset, QDPiDataset)

    def test_module_imports_base_classes(self):
        """Module imports BaseDataset and data classes from base.py.

        Evidence: qdpi.py imports from milia_pipeline.datasets.base.
        """
        source = inspect.getsource(sys.modules["milia_pipeline.datasets.implementations.qdpi"])
        self.assertIn("from milia_pipeline.datasets.base import", source)
        self.assertIn("BaseDataset", source)
        self.assertIn("DatasetMetadata", source)
        self.assertIn("DatasetSchema", source)
        self.assertIn("DatasetFeatures", source)

    def test_module_imports_register_decorator(self):
        """Module imports @register from registry.

        Evidence: qdpi.py imports register from milia_pipeline.datasets.registry.
        """
        source = inspect.getsource(sys.modules["milia_pipeline.datasets.implementations.qdpi"])
        self.assertIn("from milia_pipeline.datasets.registry import register", source)

    def test_module_does_not_import_handler_at_module_level(self):
        """QDPiDatasetHandler is NOT imported at module level (lazy import only).

        Evidence: qdpi.py NOTE comment about circular import prevention.
        The handler is only imported inside create_handler() method.

        Uses ast module to reliably distinguish module-level imports from
        imports nested inside function/method bodies.
        """
        import ast

        source = inspect.getsource(sys.modules["milia_pipeline.datasets.implementations.qdpi"])
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
            "QDPiDatasetHandler",
            module_level_import_names,
            "QDPiDatasetHandler should NOT be imported at module level "
            "(only inside create_handler() via lazy import)",
        )


# ============================================================================
# GROUP 15: QDPiDataset — DatasetFeatures.to_dict() and .supports() (4 tests)
# ============================================================================


class TestQDPiDatasetFeaturesIntegration(unittest.TestCase):
    """Verify DatasetFeatures integration methods work correctly with QDPi.

    Evidence: base.py DatasetFeatures.to_dict() and .supports() methods
    (project structure line 346).
    """

    def test_to_dict_returns_expected_dict(self):
        """features.to_dict() returns the full feature flags dictionary."""
        result = QDPiDataset.features.to_dict()
        self.assertEqual(result, EXPECTED_FEATURES)

    def test_supports_atomization_energy(self):
        """features.supports('atomization_energy') returns True.

        QDPi specific: atomization_energy is the only enabled feature.
        """
        self.assertTrue(QDPiDataset.features.supports("atomization_energy"))

    def test_supports_vibrational_analysis_false(self):
        """features.supports('vibrational_analysis') returns False.

        QDPi specific: no vibrational frequencies available.
        """
        self.assertFalse(QDPiDataset.features.supports("vibrational_analysis"))

    def test_to_dict_keys_match_expected_features(self):
        """features.to_dict() keys match all 8 expected feature names."""
        result = QDPiDataset.features.to_dict()
        self.assertEqual(set(result.keys()), set(EXPECTED_FEATURES.keys()))


# ============================================================================
# GROUP 16: QDPiDataset — Schema Consistency with Methods (3 tests)
# ============================================================================


class TestQDPiDatasetSchemaMethodConsistency(unittest.TestCase):
    """Verify schema data is consistent with method return values."""

    def test_required_properties_matches_schema(self):
        """get_required_properties() returns the same values as schema.required_properties.

        Evidence: qdpi.py: return list(cls.schema.required_properties).
        """
        method_result = QDPiDataset.get_required_properties()
        schema_result = list(QDPiDataset.schema.required_properties)
        self.assertEqual(method_result, schema_result)

    def test_feature_support_matches_features(self):
        """get_feature_support() returns the same values as features.to_dict().

        Evidence: qdpi.py: return cls.features.to_dict().
        """
        method_result = QDPiDataset.get_feature_support()
        features_result = QDPiDataset.features.to_dict()
        self.assertEqual(method_result, features_result)

    def test_required_properties_count(self):
        """get_required_properties() has exactly 3 items (atoms, coordinates, energy)."""
        result = QDPiDataset.get_required_properties()
        self.assertEqual(len(result), 3)


# ============================================================================
# GROUP 17: QDPiDataset — Edge Cases and Robustness (9 tests)
# ============================================================================


class TestQDPiDatasetEdgeCases(unittest.TestCase):
    """Test edge cases and robustness of QDPiDataset."""

    def test_multiple_calls_return_consistent_results(self):
        """Multiple calls to classmethods return identical results."""
        for _ in range(3):
            self.assertEqual(
                QDPiDataset.get_required_properties(),
                list(EXPECTED_REQUIRED_PROPERTIES),
            )
            self.assertEqual(
                QDPiDataset.get_feature_support(),
                EXPECTED_FEATURES,
            )
            self.assertEqual(
                QDPiDataset.get_molecule_creation_strategy(),
                EXPECTED_MOLECULE_CREATION_STRATEGY,
            )

    def test_classmethods_callable_on_class_not_instance(self):
        """All classmethods are callable on the class directly (no instantiation needed)."""
        for method_name in EXPECTED_CLASSMETHOD_NAMES:
            with self.subTest(method=method_name):
                method = getattr(QDPiDataset, method_name)
                self.assertTrue(callable(method))

    def test_identifier_keys_is_empty_tuple(self):
        """identifier_keys is an empty tuple — QDPi has no chemical identifiers.

        CRITICAL: QDPi HDF5 files contain only elements, atomic_types, coordinates,
        energies, and forces. No SMILES, InChI, or other parseable identifiers.
        """
        self.assertEqual(QDPiDataset.schema.identifier_keys, ())
        self.assertIsInstance(QDPiDataset.schema.identifier_keys, tuple)
        self.assertEqual(len(QDPiDataset.schema.identifier_keys), 0)

    def test_create_handler_with_none_experimental_setup(self):
        """create_handler works when experimental_setup is None (default)."""
        import contextlib

        @contextlib.contextmanager
        def _scoped_handler_mock():
            mock_handler_cls = Mock(name="MockQDPiDatasetHandler")
            mock_module = MagicMock()
            mock_module.QDPiDatasetHandler = mock_handler_cls
            handler_mod_key = "milia_pipeline.handlers.implementations.qdpi"
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
            QDPiDataset.create_handler(
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
        not instance attributes. QDPiDataset is used as a class, not instantiated,
        but we verify the attributes live on the class itself."""
        self.assertIn("metadata", QDPiDataset.__dict__)
        self.assertIn("schema", QDPiDataset.__dict__)
        self.assertIn("features", QDPiDataset.__dict__)
        self.assertIn("config_key", QDPiDataset.__dict__)

    def test_strategy_is_not_identifier_based(self):
        """get_molecule_creation_strategy() does NOT return 'identifier_coordinate_based'.

        CRITICAL: QDPi differs from DFT/QM9 which use 'identifier_coordinate_based'.
        QDPi uses 'coordinate_based' because it has no chemical identifiers.
        """
        result = QDPiDataset.get_molecule_creation_strategy()
        self.assertNotEqual(result, "identifier_coordinate_based")
        self.assertEqual(result, "coordinate_based")

    def test_required_properties_uses_energy_not_energies(self):
        """QDPi required_properties uses 'energy' (singular), NOT 'energies' (plural).

        Evidence: qdpi.py schema required_properties=('atoms', 'coordinates', 'energy').
        The 'energies' key from HDF5 is mapped to 'energy' during preprocessing.
        """
        result = QDPiDataset.get_required_properties()
        self.assertIn("energy", result)
        self.assertNotIn("energies", result)

    def test_no_molecules_class_attribute(self):
        """QDPi does NOT have a MOLECULES class attribute in its own __dict__.

        QDPi is a large-scale dataset (~1.6M structures) without a fixed
        molecule list, unlike rMD17 which defines MOLECULES.
        """
        self.assertNotIn("MOLECULES", QDPiDataset.__dict__)

    def test_optional_properties_includes_molecular_charge(self):
        """QDPi optional_properties includes 'molecular_charge'.

        CRITICAL: QDPi contains both neutral AND charged molecules.
        molecular_charge is essential for correct bond order determination
        via rdDetermineBonds in charged species.
        Evidence: qdpi.py schema optional_properties definition.
        """
        self.assertIn("molecular_charge", QDPiDataset.schema.optional_properties)


# ============================================================================
# GROUP 18: QDPiDataset — QDPi-Specific Distinctions (10 tests)
# ============================================================================


class TestQDPiDatasetSpecificDistinctions(unittest.TestCase):
    """Test QDPi-specific characteristics that distinguish it from other datasets.

    Evidence: qdpi.py module docstring and class docstring describe QDPi
    characteristics: charged molecules, 13 elements, HDF5 format,
    ωB97M-D3(BJ)/def2-TZVPPD functional, drug discovery focus.
    """

    def test_metadata_description_mentions_drug_discovery(self):
        """metadata.description mentions 'drug discovery'.

        Evidence: qdpi.py metadata.description references drug discovery.
        """
        self.assertIn("drug discovery", QDPiDataset.metadata.description)

    def test_metadata_description_mentions_1_6_million(self):
        """metadata.description mentions '~1.6 million structures'.

        Evidence: qdpi.py metadata.description states ~1.6 million structures.
        """
        self.assertIn("1.6 million", QDPiDataset.metadata.description)

    def test_metadata_description_mentions_charged_molecules(self):
        """metadata.description mentions charged molecules.

        CRITICAL: QDPi supports both neutral and charged molecules,
        unlike ANI-2x which is neutral-only.
        Evidence: qdpi.py metadata.description.
        """
        self.assertIn("charged", QDPiDataset.metadata.description)

    def test_metadata_license_is_cc_by_4(self):
        """QDPi uses CC BY 4.0 license.

        Evidence: qdpi.py metadata license="CC BY 4.0".
        """
        self.assertEqual(QDPiDataset.metadata.license, "CC BY 4.0")

    def test_optional_properties_has_forces(self):
        """QDPi optional_properties includes 'forces'.

        Evidence: qdpi.py schema optional_properties includes 'forces'
        for atomic forces (eV/Angstrom → Hartree/Angstrom).
        """
        self.assertIn("forces", QDPiDataset.schema.optional_properties)

    def test_optional_properties_has_formula(self):
        """QDPi optional_properties includes 'formula'.

        Evidence: qdpi.py schema optional_properties includes 'formula'
        for chemical formula group identifier from HDF5.
        """
        self.assertIn("formula", QDPiDataset.schema.optional_properties)

    def test_optional_properties_has_subset(self):
        """QDPi optional_properties includes 'subset'.

        Evidence: qdpi.py schema optional_properties includes 'subset'
        for source dataset identifier (spice, ani, geom, etc.).
        """
        self.assertIn("subset", QDPiDataset.schema.optional_properties)

    def test_optional_properties_has_charge_type(self):
        """QDPi optional_properties includes 'charge_type'.

        Evidence: qdpi.py schema optional_properties includes 'charge_type'
        for 'neutral' or 'charged' (from directory structure).
        """
        self.assertIn("charge_type", QDPiDataset.schema.optional_properties)

    def test_metadata_description_mentions_dft_functional(self):
        """metadata.description mentions the DFT functional level.

        Evidence: qdpi.py metadata.description references ωB97M-D3(BJ)/def2-TZVPPD.
        """
        desc = QDPiDataset.metadata.description
        # Check for the functional name (may use Unicode ω or encoded form)
        self.assertIn("def2-TZVPPD", desc)

    def test_metadata_description_mentions_biopolymer(self):
        """metadata.description mentions 'biopolymer fragments'.

        Evidence: qdpi.py metadata.description.
        """
        self.assertIn("biopolymer", QDPiDataset.metadata.description)


# ============================================================================
# GROUP 19: QDPiDataset — get_supported_elements() (6 tests)
# ============================================================================


class TestQDPiDatasetGetSupportedElements(unittest.TestCase):
    """Verify QDPiDataset.get_supported_elements() classmethod.

    Evidence: qdpi.py get_supported_elements() implementation.
    QDPi supports 13 elements: H(1), Li(3), C(6), N(7), O(8), F(9),
    Na(11), P(15), S(16), Cl(17), K(19), Br(35), I(53).
    """

    def test_is_classmethod(self):
        """get_supported_elements is a classmethod."""
        descriptor = QDPiDataset.__dict__.get("get_supported_elements")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_list(self):
        """get_supported_elements() returns a list."""
        result = QDPiDataset.get_supported_elements()
        self.assertIsInstance(result, list)

    def test_returns_correct_atomic_numbers(self):
        """get_supported_elements() returns the correct 13 atomic numbers.

        Evidence: qdpi.py lists [1, 3, 6, 7, 8, 9, 11, 15, 16, 17, 19, 35, 53].
        """
        result = QDPiDataset.get_supported_elements()
        self.assertEqual(result, EXPECTED_SUPPORTED_ELEMENTS)

    def test_has_13_elements(self):
        """get_supported_elements() returns exactly 13 elements.

        Evidence: qdpi.py docstring states QDPi supports 13 elements.
        """
        result = QDPiDataset.get_supported_elements()
        self.assertEqual(len(result), 13)

    def test_all_values_are_integers(self):
        """All values in get_supported_elements() are integers."""
        result = QDPiDataset.get_supported_elements()
        for z in result:
            with self.subTest(atomic_number=z):
                self.assertIsInstance(z, int)

    def test_has_docstring(self):
        """get_supported_elements method has a non-empty docstring."""
        method = QDPiDataset.get_supported_elements
        self.assertIsNotNone(method.__doc__)
        self.assertGreater(len(method.__doc__.strip()), 0)


# ============================================================================
# GROUP 20: QDPiDataset — get_supported_element_symbols() (5 tests)
# ============================================================================


class TestQDPiDatasetGetSupportedElementSymbols(unittest.TestCase):
    """Verify QDPiDataset.get_supported_element_symbols() classmethod.

    Evidence: qdpi.py get_supported_element_symbols() implementation.
    """

    def test_is_classmethod(self):
        """get_supported_element_symbols is a classmethod."""
        descriptor = QDPiDataset.__dict__.get("get_supported_element_symbols")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_list(self):
        """get_supported_element_symbols() returns a list."""
        result = QDPiDataset.get_supported_element_symbols()
        self.assertIsInstance(result, list)

    def test_returns_correct_symbols(self):
        """get_supported_element_symbols() returns the correct 13 element symbols.

        Evidence: qdpi.py lists ['H', 'Li', 'C', 'N', 'O', 'F', 'Na', 'P', 'S', 'Cl', 'K', 'Br', 'I'].
        """
        result = QDPiDataset.get_supported_element_symbols()
        self.assertEqual(result, EXPECTED_SUPPORTED_ELEMENT_SYMBOLS)

    def test_has_13_symbols(self):
        """get_supported_element_symbols() returns exactly 13 symbols."""
        result = QDPiDataset.get_supported_element_symbols()
        self.assertEqual(len(result), 13)

    def test_all_values_are_strings(self):
        """All values in get_supported_element_symbols() are strings."""
        result = QDPiDataset.get_supported_element_symbols()
        for sym in result:
            with self.subTest(symbol=sym):
                self.assertIsInstance(sym, str)


# ============================================================================
# GROUP 21: QDPiDataset — supports_charged_molecules() (4 tests)
# ============================================================================


class TestQDPiDatasetSupportsChargedMolecules(unittest.TestCase):
    """Verify QDPiDataset.supports_charged_molecules() classmethod.

    Evidence: qdpi.py supports_charged_molecules() implementation.
    CRITICAL: QDPi supports BOTH neutral AND charged molecules.
    This is a key difference from ANI-2x (neutral-only).
    """

    def test_is_classmethod(self):
        """supports_charged_molecules is a classmethod."""
        descriptor = QDPiDataset.__dict__.get("supports_charged_molecules")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_bool(self):
        """supports_charged_molecules() returns a boolean."""
        result = QDPiDataset.supports_charged_molecules()
        self.assertIsInstance(result, bool)

    def test_returns_true(self):
        """supports_charged_molecules() returns True.

        CRITICAL: QDPi contains both neutral and charged molecules including
        ion pairs, protonated amino acids, deprotonated species, and
        tautomers/protonation states from the RE dataset.
        Evidence: qdpi.py supports_charged_molecules() returns True.
        """
        self.assertTrue(QDPiDataset.supports_charged_molecules())

    def test_has_docstring(self):
        """supports_charged_molecules method has a non-empty docstring."""
        method = QDPiDataset.supports_charged_molecules
        self.assertIsNotNone(method.__doc__)
        self.assertGreater(len(method.__doc__.strip()), 0)


# ============================================================================
# GROUP 22: QDPiDataset — get_source_subsets() (5 tests)
# ============================================================================


class TestQDPiDatasetGetSourceSubsets(unittest.TestCase):
    """Verify QDPiDataset.get_source_subsets() classmethod.

    Evidence: qdpi.py get_source_subsets() implementation.
    QDPi contains 7 source subsets selected via active learning.
    """

    def test_is_classmethod(self):
        """get_source_subsets is a classmethod."""
        descriptor = QDPiDataset.__dict__.get("get_source_subsets")
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_list(self):
        """get_source_subsets() returns a list."""
        result = QDPiDataset.get_source_subsets()
        self.assertIsInstance(result, list)

    def test_returns_correct_subsets(self):
        """get_source_subsets() returns the correct 7 source subsets.

        Evidence: qdpi.py lists ['spice', 'ani', 'geom', 'freesolvmd', 're', 'remd', 'comp6'].
        """
        result = QDPiDataset.get_source_subsets()
        self.assertEqual(result, EXPECTED_SOURCE_SUBSETS)

    def test_has_7_subsets(self):
        """get_source_subsets() returns exactly 7 subsets."""
        result = QDPiDataset.get_source_subsets()
        self.assertEqual(len(result), 7)

    def test_all_values_are_strings(self):
        """All values in get_source_subsets() are strings."""
        result = QDPiDataset.get_source_subsets()
        for subset in result:
            with self.subTest(subset=subset):
                self.assertIsInstance(subset, str)


# ============================================================================
# GROUP 23: QDPiDataset — Element Count Consistency (3 tests)
# ============================================================================


class TestQDPiDatasetElementConsistency(unittest.TestCase):
    """Verify consistency between get_supported_elements() and
    get_supported_element_symbols().

    Evidence: qdpi.py both methods must return the same number of elements.
    """

    def test_element_count_matches_symbol_count(self):
        """get_supported_elements() and get_supported_element_symbols()
        return the same number of items."""
        elements = QDPiDataset.get_supported_elements()
        symbols = QDPiDataset.get_supported_element_symbols()
        self.assertEqual(len(elements), len(symbols))

    def test_elements_are_sorted_by_atomic_number(self):
        """get_supported_elements() returns atomic numbers in ascending order.

        Evidence: qdpi.py lists elements as [1, 3, 6, 7, 8, 9, 11, 15, 16, 17, 19, 35, 53].
        """
        elements = QDPiDataset.get_supported_elements()
        self.assertEqual(elements, sorted(elements))

    def test_all_atomic_numbers_are_positive(self):
        """All atomic numbers in get_supported_elements() are positive integers."""
        elements = QDPiDataset.get_supported_elements()
        for z in elements:
            with self.subTest(atomic_number=z):
                self.assertGreater(z, 0)


# ============================================================================
# TEST RUNNER
# ============================================================================


def run_comprehensive_suite():
    """Run all test groups in a structured order."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestQDPiDatasetClassIdentity,  # GROUP 1:  8 tests
        TestQDPiDatasetRegistration,  # GROUP 2:  5 tests
        TestQDPiDatasetMetadata,  # GROUP 3:  7 tests
        TestQDPiDatasetSchema,  # GROUP 4:  9 tests
        TestQDPiDatasetFeatures,  # GROUP 5: 10 tests
        TestQDPiDatasetConfigKey,  # GROUP 6:  2 tests
        TestQDPiDatasetGetRequiredProperties,  # GROUP 7:  5 tests
        TestQDPiDatasetGetFeatureSupport,  # GROUP 8:  6 tests
        TestQDPiDatasetGetMoleculeCreationStrategy,  # GROUP 9:  4 tests
        TestQDPiDatasetCreateHandler,  # GROUP 10: 7 tests
        TestQDPiDatasetHandlerClassAttribute,  # GROUP 11: 3 tests
        TestQDPiDatasetMethodSignatures,  # GROUP 12: 7 tests
        TestQDPiDatasetMethodDocstrings,  # GROUP 13: 4 tests
        TestQDPiDatasetModuleImportsAndExports,  # GROUP 14: 5 tests
        TestQDPiDatasetFeaturesIntegration,  # GROUP 15: 4 tests
        TestQDPiDatasetSchemaMethodConsistency,  # GROUP 16: 3 tests
        TestQDPiDatasetEdgeCases,  # GROUP 17: 9 tests
        TestQDPiDatasetSpecificDistinctions,  # GROUP 18: 10 tests
        TestQDPiDatasetGetSupportedElements,  # GROUP 19: 6 tests
        TestQDPiDatasetGetSupportedElementSymbols,  # GROUP 20: 5 tests
        TestQDPiDatasetSupportsChargedMolecules,  # GROUP 21: 4 tests
        TestQDPiDatasetGetSourceSubsets,  # GROUP 22: 5 tests
        TestQDPiDatasetElementConsistency,  # GROUP 23: 3 tests
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS — datasets/implementations/qdpi.py")
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
TEST SUITE SUMMARY — milia_pipeline/datasets/implementations/qdpi.py
====================================================================

117 comprehensive production-ready tests covering:

GROUP 1: QDPiDataset Class Identity and Type Hierarchy (8 tests)
- Is a class, correct name, correct module
- Subclass of BaseDataset, not BaseDataset itself
- Has docstring, docstring mentions QDπ and coordinate_based
- MRO includes BaseDataset

GROUP 2: QDPiDataset Registration with @register (5 tests)
- Is registered in default registry under 'QDPi'
- get('QDPi') returns QDPiDataset class
- Listed in list_all() results
- Module imports @register decorator
- Registration key matches metadata.name

GROUP 3: QDPiDataset.metadata — DatasetMetadata (7 tests)
- Is DatasetMetadata instance
- name='QDPi', version='1.0.0', description, author, license='CC BY 4.0'
- Metadata is frozen (immutable)

GROUP 4: QDPiDataset.schema — DatasetSchema (9 tests)
- Is DatasetSchema instance
- required_properties=('atoms', 'coordinates', 'energy')
- optional_properties (forces, formula, molecular_charge, subset, charge_type)
- identifier_keys=() — EMPTY (no parseable identifiers)
- coordinate_units='angstrom', energy_units='hartree'
- Schema is frozen, properties are tuples, identifier_keys is tuple

GROUP 5: QDPiDataset.features — DatasetFeatures (10 tests)
- Is DatasetFeatures instance
- All 8 feature flags verified individually
- Only atomization_energy=True, all others False
- Features is frozen (immutable)

GROUP 6: QDPiDataset.config_key (2 tests)
- Value is 'qdpi_config', is a string

GROUP 7: QDPiDataset.get_required_properties() (5 tests)
- Is classmethod, returns list
- Returns ['atoms', 'coordinates', 'energy']
- Returns fresh list each call, all items are strings

GROUP 8: QDPiDataset.get_feature_support() (6 tests)
- Is classmethod, returns dict
- Returns correct 8 feature flags (only atomization_energy True)
- All values are booleans
- Delegates to features.to_dict()

GROUP 9: QDPiDataset.get_molecule_creation_strategy() (4 tests)
- Is classmethod, returns string
- Returns 'coordinate_based' (NOT 'identifier_coordinate_based')
- Has docstring

GROUP 10: QDPiDataset.create_handler() — Lazy Import (7 tests)
- Is classmethod, correct 5-parameter signature
- experimental_setup defaults to None
- Performs lazy import of QDPiDatasetHandler
- Passes all args to constructor
- Returns handler instance
- Has docstring mentioning lazy import

GROUP 11: QDPiDataset handler_class Default (3 tests)
- handler_class is None
- converter_class is None
- validator_class is None

GROUP 12: QDPiDataset Method Signatures and Return Annotations (7 tests)
- get_required_properties -> List[str]
- get_feature_support -> Dict[str, bool]
- get_molecule_creation_strategy -> str
- All classmethods have no params (bound method excludes cls)
- create_handler has 5 params excluding cls

GROUP 13: QDPiDataset Method Docstrings (4 tests with subTests)
- All 8 classmethods have non-empty docstrings
- Docstrings reference properties, features, coordinate_based

GROUP 14: Module-Level Imports and Exports (5 tests)
- Module has docstring, exports QDPiDataset
- Imports base classes and @register
- Does NOT import QDPiDatasetHandler at module level

GROUP 15: DatasetFeatures Integration (4 tests)
- to_dict() returns expected dict
- supports() works for enabled/disabled features
- to_dict() keys match all 8 features

GROUP 16: Schema-Method Consistency (3 tests)
- get_required_properties() matches schema.required_properties
- get_feature_support() matches features.to_dict()
- Required properties count is 3

GROUP 17: Edge Cases and Robustness (9 tests)
- Multiple calls return consistent results
- Classmethods callable on class
- identifier_keys is empty tuple (QDPi specific)
- create_handler with None experimental_setup
- Class attributes live in __dict__
- Strategy is NOT identifier_coordinate_based (QDPi vs DFT distinction)
- Required properties uses 'energy' not 'energies'
- No MOLECULES class attribute
- optional_properties includes 'molecular_charge' (QDPi-specific)

GROUP 18: QDPi-Specific Distinctions (10 tests)
- Description mentions drug discovery, ~1.6 million, charged molecules
- License is CC BY 4.0
- Optional properties includes forces, formula, subset, charge_type
- Description mentions DFT functional and biopolymer fragments

GROUP 19: QDPiDataset.get_supported_elements() (6 tests)
- Is classmethod, returns list of 13 atomic numbers
- Correct values [1, 3, 6, 7, 8, 9, 11, 15, 16, 17, 19, 35, 53]
- All values are integers, has docstring

GROUP 20: QDPiDataset.get_supported_element_symbols() (5 tests)
- Is classmethod, returns list of 13 element symbols
- Correct values ['H', 'Li', 'C', 'N', 'O', 'F', 'Na', 'P', 'S', 'Cl', 'K', 'Br', 'I']
- All values are strings

GROUP 21: QDPiDataset.supports_charged_molecules() (4 tests)
- Is classmethod, returns bool
- Returns True (CRITICAL: QDPi supports charged molecules)
- Has docstring

GROUP 22: QDPiDataset.get_source_subsets() (5 tests)
- Is classmethod, returns list of 7 subsets
- Correct values ['spice', 'ani', 'geom', 'freesolvmd', 're', 'remd', 'comp6']
- All values are strings

GROUP 23: QDPiDataset Element Count Consistency (3 tests)
- Element count matches symbol count (13)
- Elements sorted by atomic number
- All atomic numbers are positive

Total: 117 comprehensive production-ready tests

PRODUCTION-READY QUALITIES:
- NO sys.modules pollution (no module-level mocking)
- All mocking via context managers (test-level only)
- No teardown_module needed since no global mock pollution
- Dynamic verification of class structure and attributes
- Evidence-based testing with source references
- Lazy import pattern tested via scoped mocking
- Immutability (frozen dataclass) verified
- Compatible with both pytest and unittest runner
- No file downloads or file system dependencies
- Future-proof: tests verify contracts, not implementation details
- QDPi-specific distinctions tested (coordinate_based, empty identifier_keys,
  'energy' not 'energies', no MOLECULES attribute, molecular_charge, charged
  molecules support, 13 elements, 7 source subsets, CC BY 4.0 license,
  drug discovery, biopolymer fragments, ωB97M-D3(BJ)/def2-TZVPPD functional)
"""
