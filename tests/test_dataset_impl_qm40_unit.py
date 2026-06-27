#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/datasets/implementations/qm40.py

Module under test: qm40.py
- QM40Dataset: BaseDataset subclass with @register decorator
  - metadata: DatasetMetadata (frozen)
  - schema: DatasetSchema (frozen)
  - features: DatasetFeatures (frozen)
  - config_key: str ("qm40_config")
  - get_required_properties(): classmethod -> list[str]
  - get_feature_support(): classmethod -> dict[str, bool]
  - get_molecule_creation_strategy(): classmethod -> str ("coordinate_based")
  - get_supported_elements(): classmethod -> list[int]
  - get_supported_element_symbols(): classmethod -> list[str]
  - supports_charged_molecules(): classmethod -> bool (False — neutral-only)
  - create_handler(): classmethod -> QM40DatasetHandler (lazy import)

Test path on local machine: ~/ml_projects/milia/tests/test_dataset_impl_qm40_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/datasets/implementations/qm40.py

NOTE: This test suite runs inside Docker at /app/milia

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- The lazy-import factory is exercised by patching the handler class attribute
  on its real module (auto-restored), NOT by injecting into sys.modules.

QM40-specific characteristics (from qm40.py source):
- 162,954 neutral drug-like ZINC molecules; B3LYP/6-31G(2df,p)
- 7 elements: H, C, N, O, F, S, Cl
- NEUTRAL only (inverse of QDpi) -> supports_charged_molecules() is False
- No InChI -> coordinate_based strategy; identifier_keys = ()
- Primary energy target Internal_E_0K (not 'energy'/'Etot')
- features: atomization_energy, rotational_constants, homo_lumo_gap = True

Updated: June 2026 - Production-ready comprehensive test coverage
"""

import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import milia_pipeline
from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetFeatures,
    DatasetMetadata,
    DatasetSchema,
)
from milia_pipeline.datasets.implementations.qm40 import QM40Dataset
from milia_pipeline.datasets.registry import is_registered

# ============================================================================
# CONSTANTS: Expected values derived from qm40.py source
# ============================================================================

EXPECTED_METADATA_NAME = "QM40"
EXPECTED_METADATA_VERSION = milia_pipeline.__version__
EXPECTED_METADATA_AUTHOR = "Madushanka, Moura Jr., Kraka (SMU CATCO)"
EXPECTED_METADATA_LICENSE = "CC BY 4.0"

EXPECTED_REQUIRED_PROPERTIES = ("Internal_E_0K", "atoms", "coordinates")
EXPECTED_OPTIONAL_PROPERTIES = (
    "HOMO",
    "LUMO",
    "HL_gap",
    "Polarizability",
    "spatial_extent",
    "dipol_mom",
    "ZPE",
    "rot1",
    "rot2",
    "rot3",
    "Inter_E_298",
    "Enthalpy",
    "Free_E",
    "CV",
    "Entropy",
    "Qmulliken",
    "smiles",
    "initial_coordinates",
    "bond_atom1_idx",
    "bond_atom2_idx",
    "bond_tag",
    "bond_lmod_ka",
)
EXPECTED_IDENTIFIER_KEYS = ()
EXPECTED_COORDINATE_UNITS = "angstrom"
EXPECTED_ENERGY_UNITS = "hartree"

EXPECTED_FEATURES = {
    "vibrational_analysis": False,
    "uncertainty_handling": False,
    "atomization_energy": True,
    "rotational_constants": True,
    "frequency_analysis": False,
    "orbital_analysis": False,
    "homo_lumo_gap": True,
    "mo_energies": False,
}

EXPECTED_CONFIG_KEY = "qm40_config"
EXPECTED_MOLECULE_CREATION_STRATEGY = "coordinate_based"

EXPECTED_SUPPORTED_ELEMENTS = [1, 6, 7, 8, 9, 16, 17]
EXPECTED_SUPPORTED_ELEMENT_SYMBOLS = ["H", "C", "N", "O", "F", "S", "Cl"]

# Exceptions that frozen dataclass / frozen-pydantic models raise on mutation
_FROZEN_EXC = (AttributeError, TypeError, ValueError)


# ============================================================================
# GROUP 1: Class identity and type hierarchy
# ============================================================================


class TestClassIdentity(unittest.TestCase):
    def test_is_a_class(self):
        self.assertTrue(inspect.isclass(QM40Dataset))

    def test_has_correct_name(self):
        self.assertEqual(QM40Dataset.__name__, "QM40Dataset")

    def test_defined_in_qm40_module(self):
        self.assertIn("implementations.qm40", QM40Dataset.__module__)

    def test_is_subclass_of_base_dataset(self):
        self.assertTrue(issubclass(QM40Dataset, BaseDataset))

    def test_is_not_base_dataset_itself(self):
        self.assertIsNot(QM40Dataset, BaseDataset)

    def test_has_docstring(self):
        self.assertIsNotNone(QM40Dataset.__doc__)
        self.assertGreater(len(QM40Dataset.__doc__.strip()), 0)

    def test_mro_includes_base_dataset(self):
        self.assertIn(BaseDataset, QM40Dataset.__mro__)


# ============================================================================
# GROUP 2: Registration
# ============================================================================


class TestRegistration(unittest.TestCase):
    def test_registered_under_qm40(self):
        self.assertTrue(is_registered("QM40"))

    def test_registration_key_matches_metadata_name(self):
        self.assertTrue(is_registered(QM40Dataset.metadata.name))


# ============================================================================
# GROUP 3: metadata
# ============================================================================


class TestMetadata(unittest.TestCase):
    def test_is_dataset_metadata(self):
        self.assertIsInstance(QM40Dataset.metadata, DatasetMetadata)

    def test_name(self):
        self.assertEqual(QM40Dataset.metadata.name, EXPECTED_METADATA_NAME)

    def test_version(self):
        self.assertEqual(QM40Dataset.metadata.version, EXPECTED_METADATA_VERSION)

    def test_author(self):
        self.assertEqual(QM40Dataset.metadata.author, EXPECTED_METADATA_AUTHOR)

    def test_license(self):
        self.assertEqual(QM40Dataset.metadata.license, EXPECTED_METADATA_LICENSE)

    def test_description_nonempty(self):
        self.assertIsInstance(QM40Dataset.metadata.description, str)
        self.assertGreater(len(QM40Dataset.metadata.description), 0)

    def test_description_mentions_neutral_and_method(self):
        desc = QM40Dataset.metadata.description
        self.assertIn("neutral", desc.lower())
        self.assertIn("B3LYP", desc)

    def test_metadata_is_frozen(self):
        with self.assertRaises(_FROZEN_EXC):
            QM40Dataset.metadata.name = "MUTATED"


# ============================================================================
# GROUP 4: schema
# ============================================================================


class TestSchema(unittest.TestCase):
    def test_is_dataset_schema(self):
        self.assertIsInstance(QM40Dataset.schema, DatasetSchema)

    def test_required_properties_exact(self):
        self.assertEqual(QM40Dataset.schema.required_properties, EXPECTED_REQUIRED_PROPERTIES)

    def test_required_properties_is_tuple(self):
        self.assertIsInstance(QM40Dataset.schema.required_properties, tuple)

    def test_energy_key_is_internal_e_0k_and_first(self):
        """Primary energy target is Internal_E_0K (not 'energy'/'Etot') and listed first."""
        self.assertEqual(QM40Dataset.schema.required_properties[0], "Internal_E_0K")

    def test_optional_properties_exact(self):
        self.assertEqual(QM40Dataset.schema.optional_properties, EXPECTED_OPTIONAL_PROPERTIES)

    def test_optional_includes_qmulliken_and_bonds(self):
        opt = set(QM40Dataset.schema.optional_properties)
        for key in ("Qmulliken", "bond_atom1_idx", "bond_atom2_idx", "bond_tag", "bond_lmod_ka"):
            with self.subTest(key=key):
                self.assertIn(key, opt)

    def test_identifier_keys_empty(self):
        self.assertEqual(QM40Dataset.schema.identifier_keys, EXPECTED_IDENTIFIER_KEYS)

    def test_identifier_keys_is_tuple(self):
        self.assertIsInstance(QM40Dataset.schema.identifier_keys, tuple)

    def test_coordinate_units(self):
        self.assertEqual(QM40Dataset.schema.coordinate_units, EXPECTED_COORDINATE_UNITS)

    def test_energy_units(self):
        self.assertEqual(QM40Dataset.schema.energy_units, EXPECTED_ENERGY_UNITS)

    def test_schema_is_frozen(self):
        with self.assertRaises(_FROZEN_EXC):
            QM40Dataset.schema.energy_units = "eV"


# ============================================================================
# GROUP 5: features
# ============================================================================


class TestFeatures(unittest.TestCase):
    def test_is_dataset_features(self):
        self.assertIsInstance(QM40Dataset.features, DatasetFeatures)

    def test_all_flags(self):
        for flag, expected in EXPECTED_FEATURES.items():
            with self.subTest(flag=flag):
                self.assertEqual(getattr(QM40Dataset.features, flag), expected)

    def test_exactly_three_enabled(self):
        enabled = {k for k, v in EXPECTED_FEATURES.items() if v}
        self.assertEqual(enabled, {"atomization_energy", "rotational_constants", "homo_lumo_gap"})

    def test_features_is_frozen(self):
        with self.assertRaises(_FROZEN_EXC):
            QM40Dataset.features.atomization_energy = False


# ============================================================================
# GROUP 6: config_key
# ============================================================================


class TestConfigKey(unittest.TestCase):
    def test_value(self):
        self.assertEqual(QM40Dataset.config_key, EXPECTED_CONFIG_KEY)

    def test_is_string(self):
        self.assertIsInstance(QM40Dataset.config_key, str)


# ============================================================================
# GROUP 7: get_required_properties
# ============================================================================


class TestGetRequiredProperties(unittest.TestCase):
    def test_returns_list(self):
        self.assertIsInstance(QM40Dataset.get_required_properties(), list)

    def test_values(self):
        self.assertEqual(QM40Dataset.get_required_properties(), list(EXPECTED_REQUIRED_PROPERTIES))

    def test_all_strings(self):
        self.assertTrue(all(isinstance(p, str) for p in QM40Dataset.get_required_properties()))

    def test_returns_fresh_list(self):
        a = QM40Dataset.get_required_properties()
        a.append("x")
        self.assertNotIn("x", QM40Dataset.get_required_properties())

    def test_matches_schema(self):
        self.assertEqual(
            QM40Dataset.get_required_properties(), list(QM40Dataset.schema.required_properties)
        )


# ============================================================================
# GROUP 8: get_feature_support
# ============================================================================


class TestGetFeatureSupport(unittest.TestCase):
    def test_returns_dict(self):
        self.assertIsInstance(QM40Dataset.get_feature_support(), dict)

    def test_equals_expected(self):
        self.assertEqual(QM40Dataset.get_feature_support(), EXPECTED_FEATURES)

    def test_matches_features_to_dict(self):
        self.assertEqual(QM40Dataset.get_feature_support(), QM40Dataset.features.to_dict())

    def test_all_values_bool(self):
        self.assertTrue(
            all(isinstance(v, bool) for v in QM40Dataset.get_feature_support().values())
        )


# ============================================================================
# GROUP 9: get_molecule_creation_strategy
# ============================================================================


class TestMoleculeCreationStrategy(unittest.TestCase):
    def test_returns_coordinate_based(self):
        self.assertEqual(
            QM40Dataset.get_molecule_creation_strategy(), EXPECTED_MOLECULE_CREATION_STRATEGY
        )

    def test_is_string(self):
        self.assertIsInstance(QM40Dataset.get_molecule_creation_strategy(), str)

    def test_not_identifier_based(self):
        self.assertNotEqual(
            QM40Dataset.get_molecule_creation_strategy(), "identifier_coordinate_based"
        )


# ============================================================================
# GROUP 10: get_supported_elements / symbols
# ============================================================================


class TestSupportedElements(unittest.TestCase):
    def test_elements_values(self):
        self.assertEqual(QM40Dataset.get_supported_elements(), EXPECTED_SUPPORTED_ELEMENTS)

    def test_elements_all_ints(self):
        self.assertTrue(all(isinstance(z, int) for z in QM40Dataset.get_supported_elements()))

    def test_seven_elements(self):
        self.assertEqual(len(QM40Dataset.get_supported_elements()), 7)

    def test_symbols_values(self):
        self.assertEqual(
            QM40Dataset.get_supported_element_symbols(), EXPECTED_SUPPORTED_ELEMENT_SYMBOLS
        )

    def test_symbols_all_strings(self):
        self.assertTrue(
            all(isinstance(s, str) for s in QM40Dataset.get_supported_element_symbols())
        )

    def test_element_symbol_count_consistency(self):
        self.assertEqual(
            len(QM40Dataset.get_supported_elements()),
            len(QM40Dataset.get_supported_element_symbols()),
        )

    def test_no_non_qm40_elements(self):
        """QM40 excludes elements outside its 7-element set (e.g. Br=35, I=53, P=15)."""
        for z in (15, 35, 53):
            with self.subTest(z=z):
                self.assertNotIn(z, QM40Dataset.get_supported_elements())


# ============================================================================
# GROUP 11: supports_charged_molecules (CRITICAL — neutral only)
# ============================================================================


class TestSupportsChargedMolecules(unittest.TestCase):
    def test_returns_false(self):
        """QM40 is neutral-only (inverse of QDpi)."""
        self.assertIs(QM40Dataset.supports_charged_molecules(), False)

    def test_returns_bool(self):
        self.assertIsInstance(QM40Dataset.supports_charged_molecules(), bool)


# ============================================================================
# GROUP 12: create_handler — lazy import
# ============================================================================


class TestCreateHandler(unittest.TestCase):
    def test_signature_five_params_excluding_cls(self):
        sig = inspect.signature(QM40Dataset.create_handler)
        params = list(sig.parameters)
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

    def test_experimental_setup_defaults_none(self):
        sig = inspect.signature(QM40Dataset.create_handler)
        self.assertIsNone(sig.parameters["experimental_setup"].default)

    @patch("milia_pipeline.handlers.implementations.qm40.QM40DatasetHandler")
    def test_returns_handler_instance(self, mock_handler):
        instance = MagicMock()
        mock_handler.return_value = instance
        result = QM40Dataset.create_handler(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        self.assertIs(result, instance)
        mock_handler.assert_called_once()

    @patch("milia_pipeline.handlers.implementations.qm40.QM40DatasetHandler")
    def test_passes_all_args_positionally_with_experimental_none(self, mock_handler):
        QM40Dataset.create_handler("dc", "fc", "pc", "lg")
        args = mock_handler.call_args[0]
        self.assertEqual(args, ("dc", "fc", "pc", "lg", None))

    @patch("milia_pipeline.handlers.implementations.qm40.QM40DatasetHandler")
    def test_forwards_experimental_setup(self, mock_handler):
        sentinel = object()
        QM40Dataset.create_handler("dc", "fc", "pc", "lg", sentinel)
        self.assertIs(mock_handler.call_args[0][4], sentinel)

    def test_handler_not_imported_at_module_level(self):
        """Lazy import: QM40DatasetHandler is NOT bound in the dataset module namespace."""
        import milia_pipeline.datasets.implementations.qm40 as qm40_mod

        self.assertFalse(hasattr(qm40_mod, "QM40DatasetHandler"))


# ============================================================================
# GROUP 13: handler_class default
# ============================================================================


class TestHandlerClassDefault(unittest.TestCase):
    def test_handler_class_is_none(self):
        """handler_class is intentionally unset (None); the handler is discovered dynamically."""
        self.assertIsNone(getattr(QM40Dataset, "handler_class", None))


# ============================================================================
# GROUP 14: QM40-specific distinctions
# ============================================================================


class TestQM40Distinctions(unittest.TestCase):
    def test_energy_key_not_generic(self):
        req = QM40Dataset.schema.required_properties
        self.assertNotIn("energy", req)
        self.assertNotIn("Etot", req)
        self.assertIn("Internal_E_0K", req)

    def test_no_inchi_identifier(self):
        self.assertEqual(QM40Dataset.schema.identifier_keys, ())

    def test_distinct_from_charged_datasets(self):
        self.assertFalse(QM40Dataset.supports_charged_molecules())

    def test_atomization_and_homo_lumo_and_rot_enabled(self):
        feats = QM40Dataset.get_feature_support()
        self.assertTrue(feats["atomization_energy"])
        self.assertTrue(feats["homo_lumo_gap"])
        self.assertTrue(feats["rotational_constants"])

    def test_no_uncertainty_or_frequency(self):
        feats = QM40Dataset.get_feature_support()
        self.assertFalse(feats["uncertainty_handling"])
        self.assertFalse(feats["frequency_analysis"])
        self.assertFalse(feats["vibrational_analysis"])


# ============================================================================
# Comprehensive runner (pytest + unittest compatible)
# ============================================================================


def run_comprehensive_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    test_classes = [
        TestClassIdentity,
        TestRegistration,
        TestMetadata,
        TestSchema,
        TestFeatures,
        TestConfigKey,
        TestGetRequiredProperties,
        TestGetFeatureSupport,
        TestMoleculeCreationStrategy,
        TestSupportedElements,
        TestSupportsChargedMolecules,
        TestCreateHandler,
        TestHandlerClassDefault,
        TestQM40Distinctions,
    ]
    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS — datasets/implementations/qm40.py")
    print("=" * 80)
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "pytest" in sys.modules:
        pass
    else:
        sys.exit(run_comprehensive_suite())
