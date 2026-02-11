"""
Contract Tests for Dataset Base Classes (Section 2.2)

This module verifies that every registered dataset implementation (via @register
decorator) satisfies the BaseDataset ABC contract and has valid DatasetMetadata,
DatasetSchema, and DatasetFeatures.

Modules exercised:
- milia_pipeline/datasets/base.py — BaseDataset, DatasetMetadata, DatasetSchema, DatasetFeatures
- milia_pipeline/datasets/registry.py — DatasetRegistry, list_all(), get()
- milia_pipeline/datasets/implementations/ — All 10 registered dataset classes

Contract guarantees verified:
1. Every registered class is a proper BaseDataset subclass
2. metadata is a valid DatasetMetadata instance with non-empty fields
3. schema is a valid DatasetSchema instance with valid units and non-empty required_properties
4. features is a valid DatasetFeatures instance
5. config_key is a non-empty string
6. All abstract methods are implemented and return correct types
7. Pydantic V2 frozen dataclass immutability is enforced
8. DatasetFeatures.to_dict() and .supports() work correctly
9. Cross-field consistency (e.g., identifier_keys empty ↔ coordinate_based strategy)
10. __init_subclass__ validation catches malformed subclasses

Test execution:
    cd /app/milia
    python -m pytest tests/test_contract_dataset_base.py -v

Author: MILIA Pipeline Team
Date: February 2026
"""

import sys
import os
import copy
import logging
import pytest
from abc import ABC
from typing import Dict, List, Tuple, Type, Optional
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Imports from MILIA pipeline
# ---------------------------------------------------------------------------
from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetMetadata,
    DatasetSchema,
    DatasetFeatures,
)
from milia_pipeline.datasets.registry import (
    DatasetRegistry,
    get_default_registry,
    list_all,
    get,
    is_registered,
)

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.contract


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def default_registry() -> DatasetRegistry:
    """Return the default global registry (populated by @register decorators)."""
    return get_default_registry()


@pytest.fixture(scope="module")
def registered_names(default_registry: DatasetRegistry) -> List[str]:
    """Return all registered dataset names from the default registry."""
    names = default_registry.list_all()
    assert len(names) > 0, (
        "No datasets registered in default registry. "
        "Ensure milia_pipeline.datasets.implementations is imported."
    )
    return names


@pytest.fixture(scope="module")
def registered_classes(default_registry: DatasetRegistry, registered_names: List[str]):
    """Return mapping of name -> dataset class for all registered datasets."""
    return {name: default_registry.get(name) for name in registered_names}


@pytest.fixture
def isolated_registry() -> DatasetRegistry:
    """Create a fresh, isolated DatasetRegistry for mutation-safe tests."""
    return DatasetRegistry()


# ===========================================================================
# Known dataset catalog (from project structure documentation)
# Used for sanity checks — NOT for enforcing a closed set
# ===========================================================================

EXPECTED_DATASETS = {
    "DFT", "DMC", "Wavefunction", "QM9",
    "ANI1x", "ANI1ccx", "ANI2x",
    "RMD17", "XXMD", "QDPi",
}

VALID_COORDINATE_UNITS = ("angstrom", "bohr")
VALID_ENERGY_UNITS = ("hartree", "eV", "kcal/mol", "kJ/mol")
VALID_STRATEGIES = ("identifier_coordinate_based", "coordinate_based")


# ===========================================================================
# Section A: Registry Population Sanity Tests
# ===========================================================================

class TestRegistryPopulation:
    """Verify the default registry is populated with all expected datasets."""

    def test_registry_is_non_empty(self, registered_names: List[str]):
        """At least one dataset must be registered."""
        assert len(registered_names) >= 1

    def test_expected_datasets_are_registered(self, registered_names: List[str]):
        """All 10 known datasets from the project structure must be registered."""
        registered_set = set(registered_names)
        missing = EXPECTED_DATASETS - registered_set
        assert not missing, (
            f"Expected datasets missing from registry: {missing}. "
            f"Registered: {registered_set}"
        )

    def test_registry_list_all_matches_convenience_function(
        self, default_registry: DatasetRegistry
    ):
        """list_all() convenience function must match registry.list_all()."""
        assert set(list_all()) == set(default_registry.list_all())

    def test_registry_get_returns_class_for_each_name(
        self, default_registry: DatasetRegistry, registered_names: List[str]
    ):
        """get() must return a class (not instance) for each registered name."""
        for name in registered_names:
            cls = default_registry.get(name)
            assert isinstance(cls, type), (
                f"Registry.get('{name}') returned {type(cls).__name__}, expected a class"
            )

    def test_is_registered_returns_true_for_all(self, registered_names: List[str]):
        """is_registered() convenience function must return True for all known names."""
        for name in registered_names:
            assert is_registered(name), f"is_registered('{name}') returned False"


# ===========================================================================
# Section B: BaseDataset Subclass Contract
# ===========================================================================

class TestBaseDatasetSubclassContract:
    """Every registered dataset class must be a proper BaseDataset subclass."""

    def test_is_subclass_of_base_dataset(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """Each registered class must inherit from BaseDataset."""
        for name, cls in registered_classes.items():
            assert issubclass(cls, BaseDataset), (
                f"{cls.__name__} (registered as '{name}') does not inherit from BaseDataset"
            )

    def test_is_not_abstract(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """Registered classes must not have unresolved abstract methods."""
        for name, cls in registered_classes.items():
            abstract_methods = getattr(cls, '__abstractmethods__', frozenset())
            assert not abstract_methods, (
                f"{cls.__name__} has unresolved abstract methods: {abstract_methods}"
            )

    def test_has_required_class_attributes(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """Each class must have metadata, schema, features, config_key attributes."""
        required_attrs = ['metadata', 'schema', 'features', 'config_key']
        for name, cls in registered_classes.items():
            for attr in required_attrs:
                assert hasattr(cls, attr), (
                    f"{cls.__name__} missing required attribute '{attr}'"
                )


# ===========================================================================
# Section C: DatasetMetadata Contract
# ===========================================================================

class TestDatasetMetadataContract:
    """Validate DatasetMetadata for every registered dataset."""

    def test_metadata_is_dataset_metadata_instance(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """metadata must be a DatasetMetadata instance."""
        for name, cls in registered_classes.items():
            assert isinstance(cls.metadata, DatasetMetadata), (
                f"{cls.__name__}.metadata is {type(cls.metadata).__name__}, "
                f"expected DatasetMetadata"
            )

    def test_metadata_name_is_non_empty_string(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """metadata.name must be a non-empty string."""
        for name, cls in registered_classes.items():
            assert isinstance(cls.metadata.name, str), (
                f"{cls.__name__}.metadata.name is not a string"
            )
            assert len(cls.metadata.name) > 0, (
                f"{cls.__name__}.metadata.name is empty"
            )

    def test_metadata_name_matches_registry_key(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """metadata.name must match the key used for registry lookup."""
        for name, cls in registered_classes.items():
            assert cls.metadata.name == name, (
                f"{cls.__name__}.metadata.name = '{cls.metadata.name}' "
                f"does not match registry key '{name}'"
            )

    def test_metadata_version_is_non_empty_string(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """metadata.version must be a non-empty string."""
        for name, cls in registered_classes.items():
            assert isinstance(cls.metadata.version, str) and cls.metadata.version, (
                f"{cls.__name__}.metadata.version is empty or not a string"
            )

    def test_metadata_description_is_non_empty_string(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """metadata.description must be a non-empty string."""
        for name, cls in registered_classes.items():
            assert isinstance(cls.metadata.description, str) and cls.metadata.description, (
                f"{cls.__name__}.metadata.description is empty or not a string"
            )

    def test_metadata_is_frozen(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """DatasetMetadata must be immutable (frozen dataclass)."""
        for name, cls in registered_classes.items():
            metadata = cls.metadata
            with pytest.raises((AttributeError, TypeError, Exception)):
                # Pydantic V2 frozen dataclass raises on attribute assignment
                metadata.name = "MUTATED"  # type: ignore


# ===========================================================================
# Section D: DatasetSchema Contract
# ===========================================================================

class TestDatasetSchemaContract:
    """Validate DatasetSchema for every registered dataset."""

    def test_schema_is_dataset_schema_instance(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """schema must be a DatasetSchema instance."""
        for name, cls in registered_classes.items():
            assert isinstance(cls.schema, DatasetSchema), (
                f"{cls.__name__}.schema is {type(cls.schema).__name__}, "
                f"expected DatasetSchema"
            )

    def test_required_properties_is_non_empty_tuple(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """schema.required_properties must be a non-empty tuple of strings."""
        for name, cls in registered_classes.items():
            rp = cls.schema.required_properties
            assert isinstance(rp, tuple), (
                f"{cls.__name__}.schema.required_properties is {type(rp).__name__}, "
                f"expected tuple"
            )
            assert len(rp) > 0, (
                f"{cls.__name__}.schema.required_properties is empty"
            )
            for prop in rp:
                assert isinstance(prop, str) and prop, (
                    f"{cls.__name__}.schema.required_properties contains invalid entry: {prop!r}"
                )

    def test_optional_properties_is_tuple_of_strings(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """schema.optional_properties must be a tuple of strings (may be empty)."""
        for name, cls in registered_classes.items():
            op = cls.schema.optional_properties
            assert isinstance(op, tuple), (
                f"{cls.__name__}.schema.optional_properties is {type(op).__name__}, "
                f"expected tuple"
            )
            for prop in op:
                assert isinstance(prop, str) and prop, (
                    f"{cls.__name__}.schema.optional_properties contains invalid entry: {prop!r}"
                )

    def test_identifier_keys_is_tuple(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """schema.identifier_keys must be a tuple (may be empty for coordinate_based)."""
        for name, cls in registered_classes.items():
            ik = cls.schema.identifier_keys
            assert isinstance(ik, tuple), (
                f"{cls.__name__}.schema.identifier_keys is {type(ik).__name__}, "
                f"expected tuple"
            )

    def test_identifier_keys_entries_are_2_tuples(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """Each identifier_keys entry must be a 2-tuple of (npz_key, identifier_type)."""
        for name, cls in registered_classes.items():
            for entry in cls.schema.identifier_keys:
                assert isinstance(entry, tuple) and len(entry) == 2, (
                    f"{cls.__name__}.schema.identifier_keys entry {entry!r} "
                    f"is not a 2-tuple"
                )
                assert all(isinstance(s, str) and s for s in entry), (
                    f"{cls.__name__}.schema.identifier_keys entry {entry!r} "
                    f"contains non-string or empty value"
                )

    def test_coordinate_units_is_valid(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """schema.coordinate_units must be one of the valid unit strings."""
        for name, cls in registered_classes.items():
            cu = cls.schema.coordinate_units
            assert cu in VALID_COORDINATE_UNITS, (
                f"{cls.__name__}.schema.coordinate_units = '{cu}' "
                f"not in {VALID_COORDINATE_UNITS}"
            )

    def test_energy_units_is_valid(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """schema.energy_units must be one of the valid unit strings."""
        for name, cls in registered_classes.items():
            eu = cls.schema.energy_units
            assert eu in VALID_ENERGY_UNITS, (
                f"{cls.__name__}.schema.energy_units = '{eu}' "
                f"not in {VALID_ENERGY_UNITS}"
            )

    def test_no_overlap_between_required_and_optional(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """required_properties and optional_properties must not overlap."""
        for name, cls in registered_classes.items():
            required = set(cls.schema.required_properties)
            optional = set(cls.schema.optional_properties)
            overlap = required & optional
            assert not overlap, (
                f"{cls.__name__} has properties in both required and optional: {overlap}"
            )

    def test_schema_is_frozen(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """DatasetSchema must be immutable (frozen dataclass)."""
        for name, cls in registered_classes.items():
            schema = cls.schema
            with pytest.raises((AttributeError, TypeError, Exception)):
                schema.coordinate_units = "MUTATED"  # type: ignore


# ===========================================================================
# Section E: DatasetFeatures Contract
# ===========================================================================

class TestDatasetFeaturesContract:
    """Validate DatasetFeatures for every registered dataset."""

    EXPECTED_FEATURE_KEYS = {
        'vibrational_analysis',
        'uncertainty_handling',
        'atomization_energy',
        'rotational_constants',
        'frequency_analysis',
        'orbital_analysis',
        'homo_lumo_gap',
        'mo_energies',
    }

    def test_features_is_dataset_features_instance(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """features must be a DatasetFeatures instance."""
        for name, cls in registered_classes.items():
            assert isinstance(cls.features, DatasetFeatures), (
                f"{cls.__name__}.features is {type(cls.features).__name__}, "
                f"expected DatasetFeatures"
            )

    def test_all_feature_flags_are_booleans(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """Every feature flag must be a boolean value."""
        for name, cls in registered_classes.items():
            d = cls.features.to_dict()
            for key, val in d.items():
                assert isinstance(val, bool), (
                    f"{cls.__name__}.features.{key} = {val!r} (type {type(val).__name__}), "
                    f"expected bool"
                )

    def test_to_dict_returns_all_8_keys(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """to_dict() must return exactly the 8 expected feature keys."""
        for name, cls in registered_classes.items():
            d = cls.features.to_dict()
            assert isinstance(d, dict), (
                f"{cls.__name__}.features.to_dict() returned {type(d).__name__}, "
                f"expected dict"
            )
            assert set(d.keys()) == self.EXPECTED_FEATURE_KEYS, (
                f"{cls.__name__}.features.to_dict() keys mismatch. "
                f"Got: {set(d.keys())}, Expected: {self.EXPECTED_FEATURE_KEYS}"
            )

    def test_supports_method_agrees_with_to_dict(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """supports(key) must agree with to_dict()[key] for every feature."""
        for name, cls in registered_classes.items():
            d = cls.features.to_dict()
            for key, val in d.items():
                assert cls.features.supports(key) == val, (
                    f"{cls.__name__}.features.supports('{key}') = "
                    f"{cls.features.supports(key)}, but to_dict()['{key}'] = {val}"
                )

    def test_supports_returns_false_for_unknown_feature(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """supports() must return False for unrecognized feature names."""
        for name, cls in registered_classes.items():
            assert cls.features.supports('nonexistent_feature_xyz') is False

    def test_features_is_frozen(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """DatasetFeatures must be immutable (frozen dataclass)."""
        for name, cls in registered_classes.items():
            features = cls.features
            with pytest.raises((AttributeError, TypeError, Exception)):
                features.vibrational_analysis = True  # type: ignore


# ===========================================================================
# Section F: config_key Contract
# ===========================================================================

class TestConfigKeyContract:
    """Validate config_key for every registered dataset."""

    def test_config_key_is_non_empty_string(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """config_key must be a non-empty string."""
        for name, cls in registered_classes.items():
            assert isinstance(cls.config_key, str) and cls.config_key, (
                f"{cls.__name__}.config_key is empty or not a string"
            )

    def test_config_keys_are_unique(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """No two datasets should share the same config_key."""
        seen: Dict[str, str] = {}
        for name, cls in registered_classes.items():
            key = cls.config_key
            if key in seen:
                pytest.fail(
                    f"Duplicate config_key '{key}' shared by "
                    f"'{seen[key]}' and '{name}'"
                )
            seen[key] = name

    def test_config_key_ends_with_config(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """config_key should follow the naming convention *_config."""
        for name, cls in registered_classes.items():
            assert cls.config_key.endswith("_config"), (
                f"{cls.__name__}.config_key = '{cls.config_key}' "
                f"does not end with '_config'"
            )


# ===========================================================================
# Section G: Abstract Method Implementation Contract
# ===========================================================================

class TestAbstractMethodContract:
    """Verify all abstract methods are implemented and return correct types."""

    def test_get_required_properties_returns_list_of_strings(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """get_required_properties() must return List[str]."""
        for name, cls in registered_classes.items():
            result = cls.get_required_properties()
            assert isinstance(result, list), (
                f"{cls.__name__}.get_required_properties() returned "
                f"{type(result).__name__}, expected list"
            )
            assert len(result) > 0, (
                f"{cls.__name__}.get_required_properties() returned empty list"
            )
            for item in result:
                assert isinstance(item, str) and item, (
                    f"{cls.__name__}.get_required_properties() contains "
                    f"invalid entry: {item!r}"
                )

    def test_get_required_properties_matches_schema(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """get_required_properties() must match schema.required_properties."""
        for name, cls in registered_classes.items():
            from_method = cls.get_required_properties()
            from_schema = list(cls.schema.required_properties)
            assert from_method == from_schema, (
                f"{cls.__name__}.get_required_properties() = {from_method} "
                f"does not match schema.required_properties = {from_schema}"
            )

    def test_get_feature_support_returns_dict_of_str_bool(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """get_feature_support() must return Dict[str, bool]."""
        for name, cls in registered_classes.items():
            result = cls.get_feature_support()
            assert isinstance(result, dict), (
                f"{cls.__name__}.get_feature_support() returned "
                f"{type(result).__name__}, expected dict"
            )
            for k, v in result.items():
                assert isinstance(k, str), f"Key {k!r} is not a string"
                assert isinstance(v, bool), f"Value for '{k}' is not a bool"

    def test_get_feature_support_matches_features(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """get_feature_support() must match features.to_dict()."""
        for name, cls in registered_classes.items():
            from_method = cls.get_feature_support()
            from_features = cls.features.to_dict()
            assert from_method == from_features, (
                f"{cls.__name__}.get_feature_support() does not match "
                f"features.to_dict()"
            )

    def test_get_molecule_creation_strategy_returns_valid_string(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """get_molecule_creation_strategy() must return a valid strategy string."""
        for name, cls in registered_classes.items():
            result = cls.get_molecule_creation_strategy()
            assert result in VALID_STRATEGIES, (
                f"{cls.__name__}.get_molecule_creation_strategy() = '{result}' "
                f"not in {VALID_STRATEGIES}"
            )

    def test_get_optional_properties_returns_list_of_strings(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """get_optional_properties() must return List[str]."""
        for name, cls in registered_classes.items():
            result = cls.get_optional_properties()
            assert isinstance(result, list), (
                f"{cls.__name__}.get_optional_properties() returned "
                f"{type(result).__name__}, expected list"
            )
            for item in result:
                assert isinstance(item, str) and item, (
                    f"Invalid entry in get_optional_properties(): {item!r}"
                )

    def test_get_identifier_keys_returns_list_of_tuples(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """get_identifier_keys() must return List[Tuple[str, str]]."""
        for name, cls in registered_classes.items():
            result = cls.get_identifier_keys()
            assert isinstance(result, list), (
                f"{cls.__name__}.get_identifier_keys() returned "
                f"{type(result).__name__}, expected list"
            )
            for entry in result:
                assert isinstance(entry, tuple) and len(entry) == 2

    def test_get_coordinate_units_returns_string(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """get_coordinate_units() must return a valid coordinate unit string."""
        for name, cls in registered_classes.items():
            result = cls.get_coordinate_units()
            assert result in VALID_COORDINATE_UNITS

    def test_get_energy_units_returns_string(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """get_energy_units() must return a valid energy unit string."""
        for name, cls in registered_classes.items():
            result = cls.get_energy_units()
            assert result in VALID_ENERGY_UNITS


# ===========================================================================
# Section H: Cross-Field Consistency
# ===========================================================================

class TestCrossFieldConsistency:
    """Verify internal consistency across metadata, schema, features, and methods."""

    # Datasets that use coordinate_based strategy but carry label-only
    # identifier_keys (not parseable chemical identifiers).  Wavefunction's
    # compound_id is used purely for tracking/logging, NOT for molecular
    # connectivity — this is documented in wavefunction.py docstring.
    COORDINATE_BASED_WITH_LABEL_IDS = {"Wavefunction"}

    def test_coordinate_based_implies_empty_identifier_keys(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """If strategy is 'coordinate_based', identifier_keys should be empty
        (except for datasets with documented label-only identifiers)."""
        for name, cls in registered_classes.items():
            strategy = cls.get_molecule_creation_strategy()
            if strategy == 'coordinate_based':
                if name in self.COORDINATE_BASED_WITH_LABEL_IDS:
                    # Label-only identifiers are allowed — just verify they
                    # are well-formed tuples, not that they are empty
                    for entry in cls.schema.identifier_keys:
                        assert isinstance(entry, tuple) and len(entry) == 2
                else:
                    assert len(cls.schema.identifier_keys) == 0, (
                        f"{cls.__name__} uses 'coordinate_based' strategy but has "
                        f"non-empty identifier_keys: {cls.schema.identifier_keys}"
                    )

    def test_identifier_coordinate_based_implies_non_empty_identifier_keys(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """If strategy is 'identifier_coordinate_based', identifier_keys must not be empty."""
        for name, cls in registered_classes.items():
            strategy = cls.get_molecule_creation_strategy()
            if strategy == 'identifier_coordinate_based':
                assert len(cls.schema.identifier_keys) > 0, (
                    f"{cls.__name__} uses 'identifier_coordinate_based' strategy but has "
                    f"empty identifier_keys"
                )

    def test_uncertainty_handling_implies_std_in_required(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """If uncertainty_handling is True, 'std' should be in required_properties."""
        for name, cls in registered_classes.items():
            if cls.features.uncertainty_handling:
                assert 'std' in cls.schema.required_properties, (
                    f"{cls.__name__} has uncertainty_handling=True but 'std' "
                    f"is not in required_properties: {cls.schema.required_properties}"
                )

    def test_atoms_and_coordinates_always_required(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """Every dataset must require 'atoms' and 'coordinates'."""
        for name, cls in registered_classes.items():
            rp = cls.schema.required_properties
            assert 'atoms' in rp, (
                f"{cls.__name__} does not require 'atoms': {rp}"
            )
            assert 'coordinates' in rp, (
                f"{cls.__name__} does not require 'coordinates': {rp}"
            )

    def test_required_properties_has_at_least_three_entries(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """required_properties must have at least atoms, coordinates, and an energy/target."""
        for name, cls in registered_classes.items():
            assert len(cls.schema.required_properties) >= 3, (
                f"{cls.__name__}.schema.required_properties has only "
                f"{len(cls.schema.required_properties)} entries (need >= 3)"
            )


# ===========================================================================
# Section I: DatasetMetadata Pydantic V2 Validation
# ===========================================================================

class TestDatasetMetadataValidation:
    """Verify Pydantic V2 validation constraints on DatasetMetadata."""

    def test_creation_with_valid_data(self):
        """DatasetMetadata can be created with valid arguments."""
        m = DatasetMetadata(
            name="TestDS",
            version="1.0.0",
            description="A test dataset",
            author="Test Author",
            license="MIT",
        )
        assert m.name == "TestDS"
        assert m.version == "1.0.0"
        assert m.description == "A test dataset"
        assert m.author == "Test Author"
        assert m.license == "MIT"

    def test_creation_without_optional_fields(self):
        """DatasetMetadata can be created without author and license."""
        m = DatasetMetadata(name="Test", version="1.0", description="desc")
        assert m.author is None
        assert m.license is None

    def test_empty_name_raises(self):
        """DatasetMetadata with empty name must raise ValueError."""
        with pytest.raises((ValueError, Exception)):
            DatasetMetadata(name="", version="1.0", description="desc")

    def test_empty_version_raises(self):
        """DatasetMetadata with empty version must raise ValueError."""
        with pytest.raises((ValueError, Exception)):
            DatasetMetadata(name="Test", version="", description="desc")

    def test_empty_description_raises(self):
        """DatasetMetadata with empty description must raise ValueError."""
        with pytest.raises((ValueError, Exception)):
            DatasetMetadata(name="Test", version="1.0", description="")


# ===========================================================================
# Section J: DatasetSchema Pydantic V2 Validation
# ===========================================================================

class TestDatasetSchemaValidation:
    """Verify Pydantic V2 validation constraints on DatasetSchema."""

    def test_creation_with_valid_data(self):
        """DatasetSchema can be created with valid arguments."""
        s = DatasetSchema(
            required_properties=('energy', 'atoms', 'coordinates'),
            optional_properties=('forces',),
            identifier_keys=(),
            coordinate_units='angstrom',
            energy_units='hartree',
        )
        assert s.required_properties == ('energy', 'atoms', 'coordinates')

    def test_empty_required_properties_raises(self):
        """DatasetSchema with empty required_properties must raise ValueError."""
        with pytest.raises((ValueError, Exception)):
            DatasetSchema(required_properties=())

    def test_invalid_coordinate_units_raises(self):
        """DatasetSchema with invalid coordinate_units must raise ValueError."""
        with pytest.raises((ValueError, Exception)):
            DatasetSchema(
                required_properties=('a',),
                coordinate_units='lightyear',
            )

    def test_invalid_energy_units_raises(self):
        """DatasetSchema with invalid energy_units must raise ValueError."""
        with pytest.raises((ValueError, Exception)):
            DatasetSchema(
                required_properties=('a',),
                energy_units='joule',
            )

    def test_required_properties_must_be_tuple(self):
        """DatasetSchema coerces list input to tuple via Pydantic V2 validation."""
        # Pydantic V2 frozen dataclasses coerce compatible sequences to tuple
        # rather than raising TypeError.  Verify the coercion produces a tuple.
        s = DatasetSchema(required_properties=['a', 'b'])  # type: ignore
        assert isinstance(s.required_properties, tuple), (
            "Pydantic V2 should coerce list to tuple for required_properties"
        )
        assert s.required_properties == ('a', 'b')


# ===========================================================================
# Section K: DatasetFeatures Standalone Tests
# ===========================================================================

class TestDatasetFeaturesStandalone:
    """Test DatasetFeatures independently."""

    def test_default_all_false(self):
        """Default DatasetFeatures has all flags False."""
        f = DatasetFeatures()
        d = f.to_dict()
        assert all(v is False for v in d.values())

    def test_custom_flags(self):
        """DatasetFeatures respects custom flag values."""
        f = DatasetFeatures(
            vibrational_analysis=True,
            uncertainty_handling=True,
        )
        assert f.vibrational_analysis is True
        assert f.uncertainty_handling is True
        assert f.atomization_energy is False

    def test_to_dict_returns_new_dict_each_time(self):
        """to_dict() should return a new dict on each call."""
        f = DatasetFeatures()
        d1 = f.to_dict()
        d2 = f.to_dict()
        assert d1 == d2
        assert d1 is not d2  # different object identity


# ===========================================================================
# Section L: __init_subclass__ Validation
# ===========================================================================

class TestInitSubclassValidation:
    """Verify __init_subclass__ catches malformed subclasses at definition time."""

    def test_missing_metadata_raises_type_error(self):
        """Defining a subclass without metadata must raise TypeError."""
        with pytest.raises(TypeError, match="missing required class attributes"):
            class BadDataset(BaseDataset):
                # metadata is missing
                schema = DatasetSchema(required_properties=('a',))
                features = DatasetFeatures()
                config_key = "bad_config"

                @classmethod
                def get_required_properties(cls):
                    return ['a']

                @classmethod
                def get_feature_support(cls):
                    return {}

                @classmethod
                def get_molecule_creation_strategy(cls):
                    return 'coordinate_based'

    def test_missing_schema_raises_type_error(self):
        """Defining a subclass without schema must raise TypeError."""
        with pytest.raises(TypeError, match="missing required class attributes"):
            class BadDataset2(BaseDataset):
                metadata = DatasetMetadata(
                    name="Bad2", version="1.0", description="bad"
                )
                # schema is missing
                features = DatasetFeatures()
                config_key = "bad2_config"

                @classmethod
                def get_required_properties(cls):
                    return ['a']

                @classmethod
                def get_feature_support(cls):
                    return {}

                @classmethod
                def get_molecule_creation_strategy(cls):
                    return 'coordinate_based'

    def test_wrong_metadata_type_raises_type_error(self):
        """Defining a subclass with wrong metadata type must raise TypeError."""
        with pytest.raises(TypeError, match="must be a DatasetMetadata instance"):
            class BadDataset3(BaseDataset):
                metadata = {"name": "Bad3"}  # Not DatasetMetadata
                schema = DatasetSchema(required_properties=('a',))
                features = DatasetFeatures()
                config_key = "bad3_config"

                @classmethod
                def get_required_properties(cls):
                    return ['a']

                @classmethod
                def get_feature_support(cls):
                    return {}

                @classmethod
                def get_molecule_creation_strategy(cls):
                    return 'coordinate_based'

    def test_empty_config_key_raises_type_error(self):
        """Defining a subclass with empty config_key must raise TypeError."""
        with pytest.raises(TypeError, match="config_key"):
            class BadDataset4(BaseDataset):
                metadata = DatasetMetadata(
                    name="Bad4", version="1.0", description="bad"
                )
                schema = DatasetSchema(required_properties=('a',))
                features = DatasetFeatures()
                config_key = ""  # empty

                @classmethod
                def get_required_properties(cls):
                    return ['a']

                @classmethod
                def get_feature_support(cls):
                    return {}

                @classmethod
                def get_molecule_creation_strategy(cls):
                    return 'coordinate_based'


# ===========================================================================
# Section M: Isolated Registry Tests
# ===========================================================================

class TestIsolatedRegistry:
    """Test DatasetRegistry behavior with isolated instances (not polluting global)."""

    def test_fresh_registry_is_empty(self, isolated_registry: DatasetRegistry):
        """A fresh DatasetRegistry must be empty."""
        assert len(isolated_registry) == 0
        assert isolated_registry.list_all() == []

    def test_register_and_retrieve(self, isolated_registry: DatasetRegistry):
        """Register a valid dataset class and retrieve it."""
        # Use one of the already-imported classes
        from milia_pipeline.datasets.implementations.dft import DFTDataset
        isolated_registry.register(DFTDataset)
        assert isolated_registry.is_registered("DFT")
        assert isolated_registry.get("DFT") is DFTDataset

    def test_unregister(self, isolated_registry: DatasetRegistry):
        """Unregister a dataset and verify it's removed."""
        from milia_pipeline.datasets.implementations.dft import DFTDataset
        reg = DatasetRegistry()
        reg.register(DFTDataset)
        assert reg.is_registered("DFT")
        result = reg.unregister("DFT")
        assert result is True
        assert not reg.is_registered("DFT")

    def test_get_nonexistent_raises(self, isolated_registry: DatasetRegistry):
        """Getting a non-existent dataset must raise DatasetNotFoundError."""
        from milia_pipeline.exceptions import DatasetNotFoundError
        with pytest.raises(DatasetNotFoundError):
            isolated_registry.get("NonExistent")

    def test_get_or_none_returns_none(self, isolated_registry: DatasetRegistry):
        """get_or_none() returns None for non-existent datasets."""
        result = isolated_registry.get_or_none("NonExistent")
        assert result is None

    def test_clear(self):
        """clear() removes all registrations."""
        reg = DatasetRegistry()
        from milia_pipeline.datasets.implementations.dft import DFTDataset
        reg.register(DFTDataset)
        assert len(reg) > 0
        reg.clear()
        assert len(reg) == 0

    def test_duplicate_registration_same_class_is_idempotent(self):
        """Re-registering the same class should not raise."""
        reg = DatasetRegistry()
        from milia_pipeline.datasets.implementations.dft import DFTDataset
        reg.register(DFTDataset)
        # Should not raise
        reg.register(DFTDataset)
        assert len(reg) == 1

    def test_duplicate_registration_different_class_raises(self):
        """Registering a different class with the same name must raise."""
        from milia_pipeline.exceptions import DatasetRegistrationError
        reg = DatasetRegistry()
        from milia_pipeline.datasets.implementations.dft import DFTDataset
        reg.register(DFTDataset)

        # Create a minimal valid dataset with the same metadata.name
        class FakeDFT(BaseDataset):
            metadata = DatasetMetadata(name="DFT", version="2.0", description="fake")
            schema = DatasetSchema(required_properties=('x',))
            features = DatasetFeatures()
            config_key = "fake_dft_config"

            @classmethod
            def get_required_properties(cls):
                return ['x']

            @classmethod
            def get_feature_support(cls):
                return {}

            @classmethod
            def get_molecule_creation_strategy(cls):
                return 'coordinate_based'

        with pytest.raises(DatasetRegistrationError):
            reg.register(FakeDFT)

    def test_contains_operator(self):
        """The 'in' operator must work on the registry."""
        reg = DatasetRegistry()
        from milia_pipeline.datasets.implementations.qm9 import QM9Dataset
        reg.register(QM9Dataset)
        assert "QM9" in reg
        assert "NotHere" not in reg

    def test_iter_operator(self):
        """Iterating over the registry yields dataset names."""
        reg = DatasetRegistry()
        from milia_pipeline.datasets.implementations.qm9 import QM9Dataset
        from milia_pipeline.datasets.implementations.dft import DFTDataset
        reg.register(QM9Dataset)
        reg.register(DFTDataset)
        names = list(reg)
        assert set(names) == {"QM9", "DFT"}

    def test_on_change_callback(self):
        """Registering a dataset fires on_change_callback."""
        reg = DatasetRegistry()
        called = []
        reg.add_on_change_callback(lambda: called.append(True))

        from milia_pipeline.datasets.implementations.dft import DFTDataset
        reg.register(DFTDataset)
        assert len(called) == 1

    def test_remove_on_change_callback(self):
        """Removing a callback prevents it from being called."""
        reg = DatasetRegistry()
        called = []
        cb = lambda: called.append(True)
        reg.add_on_change_callback(cb)
        reg.remove_on_change_callback(cb)

        from milia_pipeline.datasets.implementations.dft import DFTDataset
        reg.register(DFTDataset)
        assert len(called) == 0

    def test_register_non_class_raises_type_error(self, isolated_registry):
        """Registering a non-class object must raise TypeError."""
        with pytest.raises(TypeError):
            isolated_registry.register("not_a_class")  # type: ignore

    def test_register_non_basedataset_subclass_raises_type_error(self, isolated_registry):
        """Registering a class not inheriting BaseDataset must raise TypeError."""
        class NotADataset:
            pass

        with pytest.raises(TypeError):
            isolated_registry.register(NotADataset)  # type: ignore


# ===========================================================================
# Section N: create_handler Contract (lazy import pattern)
# ===========================================================================

class TestCreateHandlerContract:
    """Verify create_handler() exists and uses lazy import pattern."""

    def test_create_handler_is_classmethod(
        self, registered_classes: Dict[str, Type[BaseDataset]]
    ):
        """create_handler must be a classmethod on every registered dataset."""
        for name, cls in registered_classes.items():
            assert hasattr(cls, 'create_handler'), (
                f"{cls.__name__} missing create_handler method"
            )
            # classmethod descriptors are stored as classmethod objects in __dict__
            # but accessible as bound methods on the class
            method = getattr(cls, 'create_handler')
            assert callable(method), (
                f"{cls.__name__}.create_handler is not callable"
            )


# ===========================================================================
# Section O: Dataset-Specific Property Validation
# ===========================================================================

class TestDatasetSpecificProperties:
    """Verify well-known properties of specific datasets."""

    def test_dft_uses_identifier_coordinate_based(self, default_registry):
        """DFT must use identifier_coordinate_based strategy."""
        cls = default_registry.get("DFT")
        assert cls.get_molecule_creation_strategy() == 'identifier_coordinate_based'

    def test_dmc_uses_identifier_coordinate_based(self, default_registry):
        """DMC must use identifier_coordinate_based strategy."""
        cls = default_registry.get("DMC")
        assert cls.get_molecule_creation_strategy() == 'identifier_coordinate_based'

    def test_dmc_has_uncertainty_handling(self, default_registry):
        """DMC must have uncertainty_handling=True."""
        cls = default_registry.get("DMC")
        assert cls.features.uncertainty_handling is True

    def test_qm9_uses_identifier_coordinate_based(self, default_registry):
        """QM9 must use identifier_coordinate_based strategy."""
        cls = default_registry.get("QM9")
        assert cls.get_molecule_creation_strategy() == 'identifier_coordinate_based'

    def test_qm9_has_homo_lumo_gap(self, default_registry):
        """QM9 must have homo_lumo_gap=True."""
        cls = default_registry.get("QM9")
        assert cls.features.homo_lumo_gap is True

    def test_wavefunction_uses_coordinate_based(self, default_registry):
        """Wavefunction must use coordinate_based strategy."""
        cls = default_registry.get("Wavefunction")
        assert cls.get_molecule_creation_strategy() == 'coordinate_based'

    def test_wavefunction_has_orbital_analysis(self, default_registry):
        """Wavefunction must have orbital_analysis=True."""
        cls = default_registry.get("Wavefunction")
        assert cls.features.orbital_analysis is True

    def test_wavefunction_uses_bohr_coordinates(self, default_registry):
        """Wavefunction must use Bohr coordinate units."""
        cls = default_registry.get("Wavefunction")
        assert cls.schema.coordinate_units == 'bohr'

    def test_wavefunction_uses_ev_energy(self, default_registry):
        """Wavefunction must use eV energy units."""
        cls = default_registry.get("Wavefunction")
        assert cls.schema.energy_units == 'eV'

    def test_ani1x_uses_coordinate_based(self, default_registry):
        """ANI1x must use coordinate_based strategy."""
        cls = default_registry.get("ANI1x")
        assert cls.get_molecule_creation_strategy() == 'coordinate_based'

    def test_ani1ccx_uses_coordinate_based(self, default_registry):
        """ANI1ccx must use coordinate_based strategy."""
        cls = default_registry.get("ANI1ccx")
        assert cls.get_molecule_creation_strategy() == 'coordinate_based'

    def test_ani1ccx_requires_ccsd_energy(self, default_registry):
        """ANI1ccx must require ccsd_energy."""
        cls = default_registry.get("ANI1ccx")
        assert 'ccsd_energy' in cls.schema.required_properties

    def test_ani2x_uses_coordinate_based(self, default_registry):
        """ANI2x must use coordinate_based strategy."""
        cls = default_registry.get("ANI2x")
        assert cls.get_molecule_creation_strategy() == 'coordinate_based'

    def test_rmd17_uses_coordinate_based(self, default_registry):
        """RMD17 must use coordinate_based strategy."""
        cls = default_registry.get("RMD17")
        assert cls.get_molecule_creation_strategy() == 'coordinate_based'

    def test_xxmd_uses_coordinate_based(self, default_registry):
        """XXMD must use coordinate_based strategy."""
        cls = default_registry.get("XXMD")
        assert cls.get_molecule_creation_strategy() == 'coordinate_based'

    def test_qdpi_uses_coordinate_based(self, default_registry):
        """QDPi must use coordinate_based strategy."""
        cls = default_registry.get("QDPi")
        assert cls.get_molecule_creation_strategy() == 'coordinate_based'

    def test_qdpi_supports_charged_molecules(self, default_registry):
        """QDPi must support charged molecules."""
        cls = default_registry.get("QDPi")
        assert hasattr(cls, 'supports_charged_molecules')
        assert cls.supports_charged_molecules() is True

    def test_rmd17_has_molecules_list(self, default_registry):
        """RMD17 must have MOLECULES class attribute with 10 molecules."""
        cls = default_registry.get("RMD17")
        assert hasattr(cls, 'MOLECULES')
        assert len(cls.MOLECULES) == 10
        assert 'aspirin' in cls.MOLECULES
        assert 'benzene' in cls.MOLECULES

    def test_all_coordinate_based_datasets_have_atomization_energy(
        self, registered_classes
    ):
        """All coordinate_based datasets (except Wavefunction) should support atomization_energy."""
        for name, cls in registered_classes.items():
            strategy = cls.get_molecule_creation_strategy()
            if strategy == 'coordinate_based' and name != 'Wavefunction':
                assert cls.features.atomization_energy is True, (
                    f"{name} is coordinate_based but atomization_energy is False"
                )


# ===========================================================================
# Section P: list_all_classes Test
# ===========================================================================

class TestListAllClasses:
    """Verify list_all_classes() returns the correct dataset classes."""

    def test_list_all_classes_returns_list_of_types(
        self, default_registry: DatasetRegistry
    ):
        """list_all_classes() must return a list of class types."""
        classes = default_registry.list_all_classes()
        assert isinstance(classes, list)
        assert len(classes) > 0
        for cls in classes:
            assert isinstance(cls, type)
            assert issubclass(cls, BaseDataset)

    def test_list_all_classes_count_matches_list_all(
        self, default_registry: DatasetRegistry
    ):
        """list_all_classes() count must match list_all() count."""
        assert len(default_registry.list_all_classes()) == len(default_registry.list_all())
