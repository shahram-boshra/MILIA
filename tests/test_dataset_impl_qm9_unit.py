#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/datasets/implementations/qm9.py

Module under test: qm9.py
- QM9Dataset: BaseDataset subclass with @register decorator
  - metadata: DatasetMetadata (Pydantic frozen dataclass)
  - schema: DatasetSchema (Pydantic frozen dataclass)
  - features: DatasetFeatures (Pydantic frozen dataclass)
  - config_key: str
  - get_required_properties(): classmethod -> List[str]
  - get_feature_support(): classmethod -> Dict[str, bool]
  - get_molecule_creation_strategy(): classmethod -> str
  - create_handler(): classmethod -> QM9DatasetHandler (lazy import)

Test path on local machine: ~/ml_projects/milia/tests/test_dataset_impl_qm9_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/datasets/implementations/qm9.py

NOTE: This test suite runs inside Docker at /app/milia
Path mappings:
- Project root: /app/milia (mapped from ~/ml_projects/milia)

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution

Evidence sources:
- qm9.py: Complete source (provided)
- MILIA_Pipeline_Project_Structure.md: base.py details (lines 335-351),
  registry.py details (lines 369-375), implementations/ structure (lines 311-318)
- test_dataset_impl_dft_unit.py: Test conventions and patterns (provided)

Updated: February 2026 - Production-ready comprehensive test coverage
"""

import sys
import os
from pathlib import Path
import unittest
from unittest.mock import patch, Mock, MagicMock, call
import inspect
from typing import Dict, List, Any

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from milia_pipeline.datasets.implementations.qm9 import QM9Dataset
from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetMetadata,
    DatasetSchema,
    DatasetFeatures,
)
from milia_pipeline.datasets.registry import (
    register,
    is_registered,
    get_default_registry,
)


# ============================================================================
# CONSTANTS: Expected values derived from qm9.py source
# ============================================================================

EXPECTED_METADATA_NAME = "QM9"
EXPECTED_METADATA_VERSION = "1.0.0"
EXPECTED_METADATA_DESCRIPTION = (
    "QM9 quantum chemistry dataset with 133,885 stable small organic molecules "
    "(CHONF, up to 9 heavy atoms). Properties computed at B3LYP/6-31G(2df,p) level."
)
EXPECTED_METADATA_AUTHOR = "Ramakrishnan, Dral, Rupp, von Lilienfeld"
EXPECTED_METADATA_LICENSE = "CC0"

EXPECTED_REQUIRED_PROPERTIES = ('U0', 'atoms', 'coordinates')
EXPECTED_OPTIONAL_PROPERTIES = (
    'A', 'B', 'C',
    'mu',
    'alpha',
    'homo', 'lumo', 'gap',
    'r2',
    'zpve',
    'U', 'H', 'G',
    'Cv',
    'Qmulliken',
    'freqs',
    'smiles_relaxed',
    'inchi_relaxed',
)
EXPECTED_IDENTIFIER_KEYS = (
    ('inchi', 'inchi'),
    ('smiles', 'smiles'),
)
EXPECTED_COORDINATE_UNITS = 'angstrom'
EXPECTED_ENERGY_UNITS = 'hartree'

EXPECTED_FEATURES = {
    'vibrational_analysis': True,
    'uncertainty_handling': False,
    'atomization_energy': True,
    'rotational_constants': True,
    'frequency_analysis': True,
    'orbital_analysis': False,
    'homo_lumo_gap': True,
    'mo_energies': False,
}

EXPECTED_CONFIG_KEY = "qm9_config"
EXPECTED_MOLECULE_CREATION_STRATEGY = 'identifier_coordinate_based'

EXPECTED_CLASSMETHOD_NAMES = [
    'get_required_properties',
    'get_feature_support',
    'get_molecule_creation_strategy',
    'create_handler',
]

# Sentinel for sys.modules cleanup in scoped handler mocking
_SENTINEL = object()


# ============================================================================
# GROUP 1: QM9Dataset — Class Identity and Type Hierarchy (8 tests)
# ============================================================================

class TestQM9DatasetClassIdentity(unittest.TestCase):
    """Verify QM9Dataset is a proper BaseDataset subclass with correct identity."""

    def test_is_a_class(self):
        """QM9Dataset is a class (not a function or module)."""
        self.assertTrue(inspect.isclass(QM9Dataset))

    def test_has_correct_name(self):
        """Class name is 'QM9Dataset'."""
        self.assertEqual(QM9Dataset.__name__, "QM9Dataset")

    def test_has_correct_module(self):
        """Defined in the datasets.implementations.qm9 module."""
        self.assertIn("implementations.qm9", QM9Dataset.__module__)

    def test_is_subclass_of_base_dataset(self):
        """QM9Dataset inherits from BaseDataset."""
        self.assertTrue(
            issubclass(QM9Dataset, BaseDataset),
            "QM9Dataset must be a subclass of BaseDataset",
        )

    def test_is_not_base_dataset_itself(self):
        """QM9Dataset is a distinct class, not BaseDataset itself."""
        self.assertIsNot(QM9Dataset, BaseDataset)

    def test_has_docstring(self):
        """QM9Dataset has a non-empty docstring."""
        self.assertIsNotNone(QM9Dataset.__doc__)
        self.assertGreater(len(QM9Dataset.__doc__.strip()), 0)

    def test_docstring_mentions_qm9(self):
        """QM9Dataset docstring references QM9 dataset specifics."""
        self.assertIn("QM9", QM9Dataset.__doc__)
        self.assertIn("133,885", QM9Dataset.__doc__)

    def test_mro_includes_base_dataset(self):
        """Method Resolution Order includes BaseDataset."""
        self.assertIn(BaseDataset, QM9Dataset.__mro__)


# ============================================================================
# GROUP 2: QM9Dataset — Registration with @register (5 tests)
# ============================================================================

class TestQM9DatasetRegistration(unittest.TestCase):
    """Verify QM9Dataset is registered via @register decorator."""

    def test_is_registered_in_default_registry(self):
        """QM9Dataset is discoverable in the default DatasetRegistry.

        Evidence: @register decorator applied at class definition (qm9.py line ~42).
        Evidence: registry.py convenience function is_registered() (project structure line 375).
        """
        self.assertTrue(
            is_registered("QM9"),
            "QM9Dataset must be registered under name 'QM9'",
        )

    def test_get_returns_qm9_dataset_class(self):
        """Registry get('QM9') returns the QM9Dataset class.

        Evidence: registry.py get() method returns the registered class (project structure line 372).
        """
        from milia_pipeline.datasets.registry import get
        retrieved = get("QM9")
        self.assertIs(retrieved, QM9Dataset)

    def test_listed_in_all_datasets(self):
        """QM9Dataset name appears in list_all() results.

        Evidence: registry.py list_all() returns all registered names (project structure line 372).
        """
        from milia_pipeline.datasets.registry import list_all
        all_names = list_all()
        self.assertIn("QM9", all_names)

    def test_register_decorator_is_imported(self):
        """The qm9 module imports the register decorator.

        Evidence: qm9.py imports register from milia_pipeline.datasets.registry.
        """
        source = inspect.getsource(
            sys.modules[QM9Dataset.__module__]
        )
        self.assertIn("from milia_pipeline.datasets.registry import register", source)

    def test_registration_uses_metadata_name(self):
        """Registration key matches metadata.name ('QM9').

        Evidence: @register decorator uses cls.metadata.name (registry.py convention).
        Evidence: qm9.py metadata.name = 'QM9'.
        """
        self.assertEqual(QM9Dataset.metadata.name, "QM9")
        self.assertTrue(is_registered(QM9Dataset.metadata.name))


# ============================================================================
# GROUP 3: QM9Dataset.metadata — DatasetMetadata (7 tests)
# ============================================================================

class TestQM9DatasetMetadata(unittest.TestCase):
    """Verify QM9Dataset.metadata is a correctly configured DatasetMetadata.

    Evidence: qm9.py metadata definition, base.py DatasetMetadata Pydantic frozen dataclass.
    """

    def test_metadata_is_dataset_metadata_instance(self):
        """metadata attribute is a DatasetMetadata instance."""
        self.assertIsInstance(QM9Dataset.metadata, DatasetMetadata)

    def test_metadata_name(self):
        """metadata.name is 'QM9'."""
        self.assertEqual(QM9Dataset.metadata.name, EXPECTED_METADATA_NAME)

    def test_metadata_version(self):
        """metadata.version is '1.0.0'."""
        self.assertEqual(QM9Dataset.metadata.version, EXPECTED_METADATA_VERSION)

    def test_metadata_description(self):
        """metadata.description matches expected QM9 description."""
        self.assertEqual(
            QM9Dataset.metadata.description,
            EXPECTED_METADATA_DESCRIPTION,
        )

    def test_metadata_author(self):
        """metadata.author is 'Ramakrishnan, Dral, Rupp, von Lilienfeld'."""
        self.assertEqual(QM9Dataset.metadata.author, EXPECTED_METADATA_AUTHOR)

    def test_metadata_license(self):
        """metadata.license is 'CC0'."""
        self.assertEqual(QM9Dataset.metadata.license, EXPECTED_METADATA_LICENSE)

    def test_metadata_is_frozen(self):
        """metadata is immutable (Pydantic frozen dataclass).

        Evidence: base.py DatasetMetadata is a Pydantic frozen dataclass
        (project structure line 337-339).
        """
        with self.assertRaises((AttributeError, TypeError, Exception)):
            # Pydantic frozen dataclasses raise on attribute assignment
            QM9Dataset.metadata.name = "MODIFIED"


# ============================================================================
# GROUP 4: QM9Dataset.schema — DatasetSchema (8 tests)
# ============================================================================

class TestQM9DatasetSchema(unittest.TestCase):
    """Verify QM9Dataset.schema is a correctly configured DatasetSchema.

    Evidence: qm9.py schema definition, base.py DatasetSchema Pydantic frozen dataclass.
    """

    def test_schema_is_dataset_schema_instance(self):
        """schema attribute is a DatasetSchema instance."""
        self.assertIsInstance(QM9Dataset.schema, DatasetSchema)

    def test_schema_required_properties(self):
        """schema.required_properties is ('U0', 'atoms', 'coordinates').

        Evidence: qm9.py schema required_properties definition.
        U0 (internal energy at 0K) is the primary energy target for QM9.
        """
        self.assertEqual(
            QM9Dataset.schema.required_properties,
            EXPECTED_REQUIRED_PROPERTIES,
        )

    def test_schema_optional_properties(self):
        """schema.optional_properties contains all 18 QM9 optional properties.

        Evidence: qm9.py schema optional_properties definition.
        Includes rotational constants, dipole moment, polarizability, orbital energies,
        electronic spatial extent, ZPVE, thermodynamic energies, heat capacity,
        Mulliken charges, frequencies, relaxed SMILES and InChI.
        """
        self.assertEqual(
            QM9Dataset.schema.optional_properties,
            EXPECTED_OPTIONAL_PROPERTIES,
        )

    def test_schema_identifier_keys(self):
        """schema.identifier_keys is (('inchi', 'inchi'), ('smiles', 'smiles')).

        Evidence: qm9.py schema identifier_keys definition.
        InChI is tried FIRST as MILIA's primary molecular scheme.
        """
        self.assertEqual(
            QM9Dataset.schema.identifier_keys,
            EXPECTED_IDENTIFIER_KEYS,
        )

    def test_schema_coordinate_units(self):
        """schema.coordinate_units is 'angstrom'."""
        self.assertEqual(
            QM9Dataset.schema.coordinate_units,
            EXPECTED_COORDINATE_UNITS,
        )

    def test_schema_energy_units(self):
        """schema.energy_units is 'hartree'."""
        self.assertEqual(
            QM9Dataset.schema.energy_units,
            EXPECTED_ENERGY_UNITS,
        )

    def test_schema_is_frozen(self):
        """schema is immutable (Pydantic frozen dataclass).

        Evidence: base.py DatasetSchema is a Pydantic frozen dataclass
        (project structure line 340-343).
        """
        with self.assertRaises((AttributeError, TypeError, Exception)):
            QM9Dataset.schema.required_properties = ('modified',)

    def test_schema_required_properties_are_tuples(self):
        """required_properties and optional_properties are tuples (immutable sequences)."""
        self.assertIsInstance(QM9Dataset.schema.required_properties, tuple)
        self.assertIsInstance(QM9Dataset.schema.optional_properties, tuple)


# ============================================================================
# GROUP 5: QM9Dataset.features — DatasetFeatures (10 tests)
# ============================================================================

class TestQM9DatasetFeatures(unittest.TestCase):
    """Verify QM9Dataset.features is a correctly configured DatasetFeatures.

    Evidence: qm9.py features definition, base.py DatasetFeatures Pydantic frozen dataclass.
    """

    def test_features_is_dataset_features_instance(self):
        """features attribute is a DatasetFeatures instance."""
        self.assertIsInstance(QM9Dataset.features, DatasetFeatures)

    def test_vibrational_analysis_enabled(self):
        """features.vibrational_analysis is True (QM9 has vibrational frequencies)."""
        self.assertTrue(QM9Dataset.features.vibrational_analysis)

    def test_uncertainty_handling_disabled(self):
        """features.uncertainty_handling is False (QM9 is deterministic DFT)."""
        self.assertFalse(QM9Dataset.features.uncertainty_handling)

    def test_atomization_energy_enabled(self):
        """features.atomization_energy is True (can compute from U0 and atomic references)."""
        self.assertTrue(QM9Dataset.features.atomization_energy)

    def test_rotational_constants_enabled(self):
        """features.rotational_constants is True (QM9 has A, B, C)."""
        self.assertTrue(QM9Dataset.features.rotational_constants)

    def test_frequency_analysis_enabled(self):
        """features.frequency_analysis is True (QM9 has vibrational frequencies)."""
        self.assertTrue(QM9Dataset.features.frequency_analysis)

    def test_orbital_analysis_disabled(self):
        """features.orbital_analysis is False (QM9 has only HOMO/LUMO, not full spectrum)."""
        self.assertFalse(QM9Dataset.features.orbital_analysis)

    def test_homo_lumo_gap_enabled(self):
        """features.homo_lumo_gap is True (QM9 has HOMO-LUMO gap property).

        Evidence: qm9.py features homo_lumo_gap=True.
        QM9 provides homo, lumo, and gap properties.
        """
        self.assertTrue(QM9Dataset.features.homo_lumo_gap)

    def test_mo_energies_disabled(self):
        """features.mo_energies is False (QM9 has only HOMO/LUMO energies)."""
        self.assertFalse(QM9Dataset.features.mo_energies)

    def test_features_is_frozen(self):
        """features is immutable (Pydantic frozen dataclass).

        Evidence: base.py DatasetFeatures is a Pydantic frozen dataclass
        (project structure line 344-346).
        """
        with self.assertRaises((AttributeError, TypeError, Exception)):
            QM9Dataset.features.vibrational_analysis = False


# ============================================================================
# GROUP 6: QM9Dataset.config_key (2 tests)
# ============================================================================

class TestQM9DatasetConfigKey(unittest.TestCase):
    """Verify QM9Dataset.config_key is correctly set.

    Evidence: qm9.py config_key = "qm9_config".
    """

    def test_config_key_value(self):
        """config_key is 'qm9_config'."""
        self.assertEqual(QM9Dataset.config_key, EXPECTED_CONFIG_KEY)

    def test_config_key_is_string(self):
        """config_key is a string."""
        self.assertIsInstance(QM9Dataset.config_key, str)


# ============================================================================
# GROUP 7: QM9Dataset.get_required_properties() (5 tests)
# ============================================================================

class TestQM9DatasetGetRequiredProperties(unittest.TestCase):
    """Verify QM9Dataset.get_required_properties() classmethod.

    Evidence: qm9.py get_required_properties method.
    """

    def test_is_classmethod(self):
        """get_required_properties is a classmethod."""
        descriptor = QM9Dataset.__dict__.get('get_required_properties')
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_list(self):
        """get_required_properties() returns a list."""
        result = QM9Dataset.get_required_properties()
        self.assertIsInstance(result, list)

    def test_returns_correct_values(self):
        """get_required_properties() returns ['U0', 'atoms', 'coordinates']."""
        result = QM9Dataset.get_required_properties()
        self.assertEqual(result, list(EXPECTED_REQUIRED_PROPERTIES))

    def test_returns_new_list_each_call(self):
        """get_required_properties() returns a fresh list (not the same object).

        Evidence: implementation uses list(cls.schema.required_properties),
        converting the tuple to a new list each time.
        """
        result1 = QM9Dataset.get_required_properties()
        result2 = QM9Dataset.get_required_properties()
        self.assertEqual(result1, result2)
        self.assertIsNot(result1, result2)

    def test_contains_all_strings(self):
        """All items in get_required_properties() are strings."""
        result = QM9Dataset.get_required_properties()
        for item in result:
            with self.subTest(item=item):
                self.assertIsInstance(item, str)


# ============================================================================
# GROUP 8: QM9Dataset.get_feature_support() (6 tests)
# ============================================================================

class TestQM9DatasetGetFeatureSupport(unittest.TestCase):
    """Verify QM9Dataset.get_feature_support() classmethod.

    Evidence: qm9.py get_feature_support method.
    """

    def test_is_classmethod(self):
        """get_feature_support is a classmethod."""
        descriptor = QM9Dataset.__dict__.get('get_feature_support')
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_dict(self):
        """get_feature_support() returns a dict."""
        result = QM9Dataset.get_feature_support()
        self.assertIsInstance(result, dict)

    def test_returns_correct_feature_flags(self):
        """get_feature_support() returns the expected feature flags dict."""
        result = QM9Dataset.get_feature_support()
        self.assertEqual(result, EXPECTED_FEATURES)

    def test_all_values_are_booleans(self):
        """All values in get_feature_support() dict are booleans."""
        result = QM9Dataset.get_feature_support()
        for key, value in result.items():
            with self.subTest(feature=key):
                self.assertIsInstance(value, bool)

    def test_has_exactly_8_feature_flags(self):
        """get_feature_support() returns exactly 8 feature flags.

        Evidence: base.py DatasetFeatures has 8 feature flags
        (project structure line 345).
        """
        result = QM9Dataset.get_feature_support()
        self.assertEqual(len(result), 8)

    def test_delegates_to_features_to_dict(self):
        """get_feature_support() delegates to cls.features.to_dict().

        Evidence: qm9.py get_feature_support: return cls.features.to_dict()
        Evidence: base.py DatasetFeatures.to_dict() method
        (project structure line 346).
        """
        direct_dict = QM9Dataset.features.to_dict()
        method_result = QM9Dataset.get_feature_support()
        self.assertEqual(direct_dict, method_result)


# ============================================================================
# GROUP 9: QM9Dataset.get_molecule_creation_strategy() (4 tests)
# ============================================================================

class TestQM9DatasetGetMoleculeCreationStrategy(unittest.TestCase):
    """Verify QM9Dataset.get_molecule_creation_strategy() classmethod.

    Evidence: qm9.py get_molecule_creation_strategy method.
    """

    def test_is_classmethod(self):
        """get_molecule_creation_strategy is a classmethod."""
        descriptor = QM9Dataset.__dict__.get('get_molecule_creation_strategy')
        self.assertIsNotNone(descriptor)
        self.assertIsInstance(descriptor, classmethod)

    def test_returns_string(self):
        """get_molecule_creation_strategy() returns a string."""
        result = QM9Dataset.get_molecule_creation_strategy()
        self.assertIsInstance(result, str)

    def test_returns_identifier_coordinate_based(self):
        """get_molecule_creation_strategy() returns 'identifier_coordinate_based'.

        Evidence: qm9.py get_molecule_creation_strategy returns 'identifier_coordinate_based'.
        QM9 provides SMILES/InChI identifiers + B3LYP-optimized Cartesian coordinates.
        """
        result = QM9Dataset.get_molecule_creation_strategy()
        self.assertEqual(result, EXPECTED_MOLECULE_CREATION_STRATEGY)

    def test_has_docstring(self):
        """get_molecule_creation_strategy method has a non-empty docstring."""
        method = getattr(QM9Dataset, 'get_molecule_creation_strategy')
        self.assertIsNotNone(method.__doc__)
        self.assertGreater(len(method.__doc__.strip()), 0)


# ============================================================================
# GROUP 10: QM9Dataset.create_handler() — Lazy Import Pattern (7 tests)
# ============================================================================

class TestQM9DatasetCreateHandler(unittest.TestCase):
    """Verify QM9Dataset.create_handler() factory method with lazy import.

    Evidence: qm9.py create_handler method.
    The create_handler() method uses lazy import to break circular dependency
    between datasets/implementations/qm9.py and handlers/implementations/qm9.py.
    """

    def test_is_classmethod(self):
        """create_handler is a classmethod."""
        descriptor = QM9Dataset.__dict__.get('create_handler')
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
        unbound_func = QM9Dataset.__dict__['create_handler'].__func__
        sig = inspect.signature(unbound_func)
        params = list(sig.parameters.keys())
        self.assertEqual(
            params,
            ['cls', 'dataset_config', 'filter_config',
             'processing_config', 'logger', 'experimental_setup'],
        )

    def test_experimental_setup_default_is_none(self):
        """create_handler experimental_setup parameter defaults to None.

        Evidence: qm9.py create_handler: experimental_setup=None.
        """
        sig = inspect.signature(QM9Dataset.create_handler)
        default = sig.parameters['experimental_setup'].default
        self.assertIsNone(default)

    def _mock_handler_module(self):
        """Helper: create a mock handler module with a mock QM9DatasetHandler class.

        The actual milia_pipeline.handlers.implementations.qm9 module cannot be
        imported in the test environment due to deep dependency chains.
        To test create_handler()'s lazy import behavior, we temporarily inject
        a mock module into sys.modules so that the
        'from milia_pipeline.handlers.implementations.qm9 import QM9DatasetHandler'
        statement inside create_handler() resolves to our mock.

        This uses a context manager pattern to ensure sys.modules is cleaned up
        after each test (no mock pollution).
        """
        import contextlib

        @contextlib.contextmanager
        def _scoped_handler_mock():
            mock_handler_cls = Mock(name='MockQM9DatasetHandler')
            mock_module = MagicMock()
            mock_module.QM9DatasetHandler = mock_handler_cls

            handler_mod_key = 'milia_pipeline.handlers.implementations.qm9'
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
        """create_handler performs lazy import of QM9DatasetHandler.

        Evidence: qm9.py create_handler:
        from milia_pipeline.handlers.implementations.qm9 import QM9DatasetHandler
        """
        with self._mock_handler_module() as mock_cls:
            mock_cls.return_value = Mock()
            QM9Dataset.create_handler(
                dataset_config=Mock(),
                filter_config=Mock(),
                processing_config=Mock(),
                logger=Mock(),
                experimental_setup=None,
            )
            mock_cls.assert_called_once()

    def test_create_handler_passes_all_args_to_constructor(self):
        """create_handler passes all 5 arguments to QM9DatasetHandler().

        Evidence: qm9.py create_handler returns QM9DatasetHandler(...).
        """
        mock_dataset_config = Mock(name='dataset_config')
        mock_filter_config = Mock(name='filter_config')
        mock_processing_config = Mock(name='processing_config')
        mock_logger = Mock(name='logger')
        mock_experimental_setup = Mock(name='experimental_setup')

        with self._mock_handler_module() as mock_cls:
            mock_cls.return_value = Mock()
            QM9Dataset.create_handler(
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
        """create_handler returns the QM9DatasetHandler instance.

        Evidence: qm9.py create_handler: return QM9DatasetHandler(...).
        """
        mock_handler_instance = Mock(name='handler_instance')
        with self._mock_handler_module() as mock_cls:
            mock_cls.return_value = mock_handler_instance
            result = QM9Dataset.create_handler(
                dataset_config=Mock(),
                filter_config=Mock(),
                processing_config=Mock(),
                logger=Mock(),
            )
            self.assertIs(result, mock_handler_instance)

    def test_create_handler_has_docstring(self):
        """create_handler method has a non-empty docstring mentioning lazy import."""
        method = getattr(QM9Dataset, 'create_handler')
        self.assertIsNotNone(method.__doc__)
        self.assertIn("lazy import", method.__doc__.lower())


# ============================================================================
# GROUP 11: QM9Dataset — handler_class Default (3 tests)
# ============================================================================

class TestQM9DatasetHandlerClassAttribute(unittest.TestCase):
    """Verify QM9Dataset.handler_class is None (default from BaseDataset).

    Evidence: qm9.py NOTE comment about handler_class intentionally NOT set.
    Evidence: base.py BaseDataset optional handler_class (project structure line 349).
    """

    def test_handler_class_is_none(self):
        """handler_class is None (default from BaseDataset).

        Evidence: QM9DatasetHandler is registered via @register_handler decorator
        and discovered dynamically through the HandlerRegistry.
        """
        self.assertIsNone(QM9Dataset.handler_class)

    def test_converter_class_is_none(self):
        """converter_class is None (default from BaseDataset).

        Evidence: base.py BaseDataset optional converter_class
        (project structure line 349).
        """
        self.assertIsNone(QM9Dataset.converter_class)

    def test_validator_class_is_none(self):
        """validator_class is None (default from BaseDataset).

        Evidence: base.py BaseDataset optional validator_class
        (project structure line 349).
        """
        self.assertIsNone(QM9Dataset.validator_class)


# ============================================================================
# GROUP 12: QM9Dataset — Method Signatures and Return Annotations (6 tests)
# ============================================================================

class TestQM9DatasetMethodSignatures(unittest.TestCase):
    """Verify method signatures and return type annotations."""

    def _get_sig(self, method_name: str) -> inspect.Signature:
        """Helper: get the signature of a QM9Dataset method."""
        method = getattr(QM9Dataset, method_name)
        return inspect.signature(method)

    def test_get_required_properties_return_annotation(self):
        """get_required_properties() -> List[str]."""
        sig = self._get_sig('get_required_properties')
        self.assertEqual(sig.return_annotation, List[str])

    def test_get_feature_support_return_annotation(self):
        """get_feature_support() -> Dict[str, bool]."""
        sig = self._get_sig('get_feature_support')
        self.assertEqual(sig.return_annotation, Dict[str, bool])

    def test_get_molecule_creation_strategy_return_annotation(self):
        """get_molecule_creation_strategy() -> str."""
        sig = self._get_sig('get_molecule_creation_strategy')
        self.assertIs(sig.return_annotation, str)

    def test_get_required_properties_params(self):
        """get_required_properties(cls) has only 'cls' parameter."""
        sig = self._get_sig('get_required_properties')
        # Bound method signature excludes 'cls'
        params = list(sig.parameters.keys())
        self.assertEqual(params, [])

    def test_get_feature_support_params(self):
        """get_feature_support(cls) has only 'cls' parameter."""
        sig = self._get_sig('get_feature_support')
        params = list(sig.parameters.keys())
        self.assertEqual(params, [])

    def test_get_molecule_creation_strategy_params(self):
        """get_molecule_creation_strategy(cls) has only 'cls' parameter."""
        sig = self._get_sig('get_molecule_creation_strategy')
        params = list(sig.parameters.keys())
        self.assertEqual(params, [])


# ============================================================================
# GROUP 13: QM9Dataset — Method Docstrings (4 tests with subTests)
# ============================================================================

class TestQM9DatasetMethodDocstrings(unittest.TestCase):
    """Verify each QM9Dataset method has a non-empty docstring."""

    def test_each_classmethod_has_docstring(self):
        """Every expected classmethod has a non-empty docstring."""
        for method_name in EXPECTED_CLASSMETHOD_NAMES:
            with self.subTest(method=method_name):
                method = getattr(QM9Dataset, method_name)
                doc = getattr(method, "__doc__", None)
                self.assertIsNotNone(
                    doc, f"{method_name} has no docstring"
                )
                self.assertGreater(
                    len(doc.strip()), 0,
                    f"{method_name} has empty docstring",
                )

    def test_get_required_properties_docstring_mentions_qm9(self):
        """get_required_properties docstring references QM9."""
        method = getattr(QM9Dataset, 'get_required_properties')
        self.assertIn("QM9", method.__doc__)

    def test_get_feature_support_docstring_mentions_qm9(self):
        """get_feature_support docstring references QM9."""
        method = getattr(QM9Dataset, 'get_feature_support')
        self.assertIn("QM9", method.__doc__)

    def test_get_molecule_creation_strategy_docstring_mentions_strategy(self):
        """get_molecule_creation_strategy docstring references the strategy type."""
        method = getattr(QM9Dataset, 'get_molecule_creation_strategy')
        self.assertIn("identifier_coordinate_based", method.__doc__)


# ============================================================================
# GROUP 14: QM9Dataset — Module-Level Imports and Exports (5 tests)
# ============================================================================

class TestQM9DatasetModuleImportsAndExports(unittest.TestCase):
    """Verify the qm9 implementation module imports and exports correctly."""

    def test_module_has_docstring(self):
        """The qm9.py module has a non-empty module docstring."""
        import milia_pipeline.datasets.implementations.qm9 as mod
        self.assertIsNotNone(mod.__doc__)
        self.assertGreater(len(mod.__doc__.strip()), 0)

    def test_module_exports_qm9_dataset(self):
        """QM9Dataset is importable from the implementations.qm9 module."""
        import milia_pipeline.datasets.implementations.qm9 as mod
        self.assertTrue(hasattr(mod, "QM9Dataset"))
        self.assertIs(mod.QM9Dataset, QM9Dataset)

    def test_module_imports_base_classes(self):
        """Module imports BaseDataset and data classes from base.py.

        Evidence: qm9.py imports from milia_pipeline.datasets.base.
        """
        source = inspect.getsource(
            sys.modules['milia_pipeline.datasets.implementations.qm9']
        )
        self.assertIn("from milia_pipeline.datasets.base import", source)
        self.assertIn("BaseDataset", source)
        self.assertIn("DatasetMetadata", source)
        self.assertIn("DatasetSchema", source)
        self.assertIn("DatasetFeatures", source)

    def test_module_imports_register_decorator(self):
        """Module imports @register from registry.

        Evidence: qm9.py imports register from milia_pipeline.datasets.registry.
        """
        source = inspect.getsource(
            sys.modules['milia_pipeline.datasets.implementations.qm9']
        )
        self.assertIn("from milia_pipeline.datasets.registry import register", source)

    def test_module_does_not_import_handler_at_module_level(self):
        """QM9DatasetHandler is NOT imported at module level (lazy import only).

        Evidence: qm9.py NOTE comment about circular import prevention.
        The handler is only imported inside create_handler() method.

        Uses ast module to reliably distinguish module-level imports from
        imports nested inside function/method bodies.
        """
        import ast

        source = inspect.getsource(
            sys.modules['milia_pipeline.datasets.implementations.qm9']
        )
        tree = ast.parse(source)

        # Collect module-level import statements only
        # Module-level = direct children of the Module node, or direct children
        # of a ClassDef that is a direct child of the Module (class body, not methods)
        module_level_import_names = []

        for node in ast.iter_child_nodes(tree):
            # Top-level imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.names:
                    for alias in node.names:
                        module_level_import_names.append(alias.name)
                elif isinstance(node, ast.Import) and node.names:
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
            "QM9DatasetHandler",
            module_level_import_names,
            "QM9DatasetHandler should NOT be imported at module level "
            "(only inside create_handler() via lazy import)",
        )


# ============================================================================
# GROUP 15: QM9Dataset — DatasetFeatures.to_dict() and .supports() (4 tests)
# ============================================================================

class TestQM9DatasetFeaturesIntegration(unittest.TestCase):
    """Verify DatasetFeatures integration methods work correctly with QM9.

    Evidence: base.py DatasetFeatures.to_dict() and .supports() methods
    (project structure line 346).
    """

    def test_to_dict_returns_expected_dict(self):
        """features.to_dict() returns the full feature flags dictionary."""
        result = QM9Dataset.features.to_dict()
        self.assertEqual(result, EXPECTED_FEATURES)

    def test_supports_vibrational_analysis(self):
        """features.supports('vibrational_analysis') returns True."""
        self.assertTrue(QM9Dataset.features.supports('vibrational_analysis'))

    def test_supports_uncertainty_handling_false(self):
        """features.supports('uncertainty_handling') returns False."""
        self.assertFalse(QM9Dataset.features.supports('uncertainty_handling'))

    def test_to_dict_keys_match_expected_features(self):
        """features.to_dict() keys match all 8 expected feature names."""
        result = QM9Dataset.features.to_dict()
        self.assertEqual(set(result.keys()), set(EXPECTED_FEATURES.keys()))


# ============================================================================
# GROUP 16: QM9Dataset — Schema Consistency with Methods (3 tests)
# ============================================================================

class TestQM9DatasetSchemaMethodConsistency(unittest.TestCase):
    """Verify schema data is consistent with method return values."""

    def test_required_properties_matches_schema(self):
        """get_required_properties() returns the same values as schema.required_properties.

        Evidence: qm9.py get_required_properties: return list(cls.schema.required_properties).
        """
        method_result = QM9Dataset.get_required_properties()
        schema_result = list(QM9Dataset.schema.required_properties)
        self.assertEqual(method_result, schema_result)

    def test_feature_support_matches_features(self):
        """get_feature_support() returns the same values as features.to_dict().

        Evidence: qm9.py get_feature_support: return cls.features.to_dict().
        """
        method_result = QM9Dataset.get_feature_support()
        features_result = QM9Dataset.features.to_dict()
        self.assertEqual(method_result, features_result)

    def test_required_properties_count(self):
        """get_required_properties() has exactly 3 items (U0, atoms, coordinates)."""
        result = QM9Dataset.get_required_properties()
        self.assertEqual(len(result), 3)


# ============================================================================
# GROUP 17: QM9Dataset — QM9-Specific Schema Validation (6 tests)
# ============================================================================

class TestQM9DatasetSchemaSpecifics(unittest.TestCase):
    """Verify QM9-specific schema properties and their scientific correctness.

    Evidence: qm9.py schema definition and QM9 readme.txt property definitions.
    """

    def test_primary_energy_target_is_u0(self):
        """U0 (internal energy at 0K) is in required_properties as the primary target.

        Evidence: qm9.py schema comment: U0 (internal energy at 0K) is the primary
        energy target.
        """
        self.assertIn('U0', QM9Dataset.schema.required_properties)

    def test_orbital_energy_properties_in_optional(self):
        """HOMO, LUMO, and gap properties are in optional_properties.

        Evidence: qm9.py schema optional_properties includes homo, lumo, gap.
        These are orbital energies in Hartree from QM9.
        """
        optional = QM9Dataset.schema.optional_properties
        self.assertIn('homo', optional)
        self.assertIn('lumo', optional)
        self.assertIn('gap', optional)

    def test_thermodynamic_energies_in_optional(self):
        """U, H, G thermodynamic energies are in optional_properties.

        Evidence: qm9.py schema optional_properties includes U, H, G.
        U = internal energy at 298.15K, H = enthalpy, G = free energy (all Hartree).
        """
        optional = QM9Dataset.schema.optional_properties
        self.assertIn('U', optional)
        self.assertIn('H', optional)
        self.assertIn('G', optional)

    def test_rotational_constants_in_optional(self):
        """Rotational constants A, B, C are in optional_properties.

        Evidence: qm9.py schema optional_properties includes A, B, C (GHz).
        """
        optional = QM9Dataset.schema.optional_properties
        self.assertIn('A', optional)
        self.assertIn('B', optional)
        self.assertIn('C', optional)

    def test_inchi_is_primary_identifier(self):
        """InChI is the first identifier key (primary molecular scheme for MILIA).

        Evidence: qm9.py schema identifier_keys comment:
        InChI is tried FIRST as it is MILIA's primary molecular scheme.
        """
        first_key = QM9Dataset.schema.identifier_keys[0]
        self.assertEqual(first_key, ('inchi', 'inchi'))

    def test_smiles_is_fallback_identifier(self):
        """SMILES is the second identifier key (fallback).

        Evidence: qm9.py schema identifier_keys: SMILES is fallback only.
        """
        second_key = QM9Dataset.schema.identifier_keys[1]
        self.assertEqual(second_key, ('smiles', 'smiles'))


# ============================================================================
# GROUP 18: QM9Dataset — Edge Cases and Robustness (5 tests)
# ============================================================================

class TestQM9DatasetEdgeCases(unittest.TestCase):
    """Test edge cases and robustness of QM9Dataset."""

    def test_multiple_calls_return_consistent_results(self):
        """Multiple calls to classmethods return identical results."""
        for _ in range(3):
            self.assertEqual(
                QM9Dataset.get_required_properties(),
                list(EXPECTED_REQUIRED_PROPERTIES),
            )
            self.assertEqual(
                QM9Dataset.get_feature_support(),
                EXPECTED_FEATURES,
            )
            self.assertEqual(
                QM9Dataset.get_molecule_creation_strategy(),
                EXPECTED_MOLECULE_CREATION_STRATEGY,
            )

    def test_classmethods_callable_on_class_not_instance(self):
        """All classmethods are callable on the class directly (no instantiation needed)."""
        # QM9Dataset is never instantiated — these are all classmethods
        for method_name in EXPECTED_CLASSMETHOD_NAMES:
            with self.subTest(method=method_name):
                method = getattr(QM9Dataset, method_name)
                self.assertTrue(callable(method))

    def test_identifier_keys_structure(self):
        """identifier_keys contains tuples of (source_key, identifier_type) pairs."""
        for key_pair in QM9Dataset.schema.identifier_keys:
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
            mock_handler_cls = Mock(name='MockQM9DatasetHandler')
            mock_module = MagicMock()
            mock_module.QM9DatasetHandler = mock_handler_cls
            handler_mod_key = 'milia_pipeline.handlers.implementations.qm9'
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
            QM9Dataset.create_handler(
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
        not instance attributes. QM9Dataset is used as a class, not instantiated,
        but we verify the attributes live on the class itself."""
        self.assertIn('metadata', QM9Dataset.__dict__)
        self.assertIn('schema', QM9Dataset.__dict__)
        self.assertIn('features', QM9Dataset.__dict__)
        self.assertIn('config_key', QM9Dataset.__dict__)


# ============================================================================
# GROUP 19: QM9Dataset — Differences from DFT Dataset (4 tests)
# ============================================================================

class TestQM9DatasetDifferencesFromDFT(unittest.TestCase):
    """Verify QM9-specific properties that distinguish it from DFT dataset.

    These tests ensure QM9's unique characteristics are correctly encoded.
    """

    def test_homo_lumo_gap_is_enabled(self):
        """QM9 enables homo_lumo_gap (unlike DFT which disables it).

        Evidence: qm9.py features homo_lumo_gap=True.
        QM9 has HOMO, LUMO, and gap properties.
        """
        self.assertTrue(QM9Dataset.features.homo_lumo_gap)

    def test_required_energy_is_u0_not_etot(self):
        """QM9 uses U0 as primary energy target (not Etot like DFT).

        Evidence: qm9.py schema required_properties includes 'U0'.
        U0 = internal energy at 0K.
        """
        self.assertIn('U0', QM9Dataset.schema.required_properties)
        self.assertNotIn('Etot', QM9Dataset.schema.required_properties)

    def test_optional_properties_count(self):
        """QM9 has 18 optional properties (more than DFT's 4).

        Evidence: qm9.py schema optional_properties has 18 items covering
        rotational constants, dipole moment, polarizability, orbital energies,
        electronic spatial extent, ZPVE, thermodynamic energies, heat capacity,
        Mulliken charges, frequencies, relaxed SMILES and InChI.
        """
        self.assertEqual(len(QM9Dataset.schema.optional_properties), 18)

    def test_identifier_keys_use_smiles_not_graphs(self):
        """QM9 uses ('smiles', 'smiles') not ('graphs', 'smiles') for SMILES identifier.

        Evidence: qm9.py schema identifier_keys uses 'smiles' as the npz key,
        unlike DFT which uses 'graphs' as the npz key for SMILES.
        """
        # Find the SMILES identifier key
        smiles_keys = [k for k in QM9Dataset.schema.identifier_keys if k[1] == 'smiles']
        self.assertEqual(len(smiles_keys), 1)
        self.assertEqual(smiles_keys[0], ('smiles', 'smiles'))


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_comprehensive_suite():
    """Run all test groups in a structured order."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestQM9DatasetClassIdentity,                # GROUP 1:   8 tests
        TestQM9DatasetRegistration,                  # GROUP 2:   5 tests
        TestQM9DatasetMetadata,                      # GROUP 3:   7 tests
        TestQM9DatasetSchema,                        # GROUP 4:   8 tests
        TestQM9DatasetFeatures,                      # GROUP 5:  10 tests
        TestQM9DatasetConfigKey,                     # GROUP 6:   2 tests
        TestQM9DatasetGetRequiredProperties,         # GROUP 7:   5 tests
        TestQM9DatasetGetFeatureSupport,             # GROUP 8:   6 tests
        TestQM9DatasetGetMoleculeCreationStrategy,   # GROUP 9:   4 tests
        TestQM9DatasetCreateHandler,                 # GROUP 10:  7 tests
        TestQM9DatasetHandlerClassAttribute,         # GROUP 11:  3 tests
        TestQM9DatasetMethodSignatures,              # GROUP 12:  6 tests
        TestQM9DatasetMethodDocstrings,              # GROUP 13:  4 tests
        TestQM9DatasetModuleImportsAndExports,       # GROUP 14:  5 tests
        TestQM9DatasetFeaturesIntegration,           # GROUP 15:  4 tests
        TestQM9DatasetSchemaMethodConsistency,       # GROUP 16:  3 tests
        TestQM9DatasetSchemaSpecifics,               # GROUP 17:  6 tests
        TestQM9DatasetEdgeCases,                     # GROUP 18:  5 tests
        TestQM9DatasetDifferencesFromDFT,            # GROUP 19:  4 tests
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS — datasets/implementations/qm9.py")
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
TEST SUITE SUMMARY — milia_pipeline/datasets/implementations/qm9.py
====================================================================

97 comprehensive production-ready tests covering:

GROUP 1: QM9Dataset Class Identity and Type Hierarchy (8 tests)
- Is a class, correct name, correct module
- Subclass of BaseDataset, not BaseDataset itself
- Has docstring, docstring mentions QM9 and 133,885
- MRO includes BaseDataset

GROUP 2: QM9Dataset Registration with @register (5 tests)
- Is registered in default registry under 'QM9'
- get('QM9') returns QM9Dataset class
- Listed in list_all() results
- Module imports @register decorator
- Registration key matches metadata.name

GROUP 3: QM9Dataset.metadata — DatasetMetadata (7 tests)
- Is DatasetMetadata instance
- name='QM9', version='1.0.0', description, author, license='CC0'
- Metadata is frozen (immutable)

GROUP 4: QM9Dataset.schema — DatasetSchema (8 tests)
- Is DatasetSchema instance
- required_properties=('U0', 'atoms', 'coordinates')
- optional_properties (18 items), identifier_keys
- coordinate_units='angstrom', energy_units='hartree'
- Schema is frozen, properties are tuples

GROUP 5: QM9Dataset.features — DatasetFeatures (10 tests)
- Is DatasetFeatures instance
- All 8 feature flags verified individually
- homo_lumo_gap=True (unlike DFT)
- Features is frozen (immutable)

GROUP 6: QM9Dataset.config_key (2 tests)
- Value is 'qm9_config', is a string

GROUP 7: QM9Dataset.get_required_properties() (5 tests)
- Is classmethod, returns list
- Returns ['U0', 'atoms', 'coordinates']
- Returns fresh list each call, all items are strings

GROUP 8: QM9Dataset.get_feature_support() (6 tests)
- Is classmethod, returns dict
- Returns correct 8 feature flags
- All values are booleans
- Delegates to features.to_dict()

GROUP 9: QM9Dataset.get_molecule_creation_strategy() (4 tests)
- Is classmethod, returns string
- Returns 'identifier_coordinate_based'
- Has docstring

GROUP 10: QM9Dataset.create_handler() — Lazy Import (7 tests)
- Is classmethod, correct 5-parameter signature
- experimental_setup defaults to None
- Performs lazy import of QM9DatasetHandler
- Passes all args to constructor
- Returns handler instance
- Has docstring mentioning lazy import

GROUP 11: QM9Dataset handler_class Default (3 tests)
- handler_class is None
- converter_class is None
- validator_class is None

GROUP 12: QM9Dataset Method Signatures and Return Annotations (6 tests)
- get_required_properties -> List[str]
- get_feature_support -> Dict[str, bool]
- get_molecule_creation_strategy -> str
- All classmethods have no params (bound method excludes cls)

GROUP 13: QM9Dataset Method Docstrings (4 tests with subTests)
- All 4 classmethods have non-empty docstrings
- Docstrings reference QM9 and strategy type

GROUP 14: Module-Level Imports and Exports (5 tests)
- Module has docstring, exports QM9Dataset
- Imports base classes and @register
- Does NOT import QM9DatasetHandler at module level

GROUP 15: DatasetFeatures Integration (4 tests)
- to_dict() returns expected dict
- supports() works for enabled/disabled features
- to_dict() keys match all 8 features

GROUP 16: Schema-Method Consistency (3 tests)
- get_required_properties() matches schema.required_properties
- get_feature_support() matches features.to_dict()
- Required properties count is 3

GROUP 17: QM9-Specific Schema Validation (6 tests)
- U0 is primary energy target
- HOMO/LUMO/gap in optional properties
- Thermodynamic energies U/H/G in optional
- Rotational constants A/B/C in optional
- InChI is primary identifier, SMILES is fallback

GROUP 18: Edge Cases and Robustness (5 tests)
- Multiple calls return consistent results
- Classmethods callable on class
- identifier_keys structure validation
- create_handler with None experimental_setup
- Class attributes live in __dict__

GROUP 19: QM9 vs DFT Differences (4 tests)
- homo_lumo_gap is enabled (unlike DFT)
- U0 is primary energy (not Etot)
- 18 optional properties (more than DFT's 4)
- SMILES key is ('smiles', 'smiles') not ('graphs', 'smiles')

Total: 97 comprehensive production-ready tests

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
- QM9-specific scientific property validation
"""
