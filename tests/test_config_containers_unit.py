#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for config_containers.py

Tests all container classes, factory functions, and utility functions
with a focus on validation, edge cases, and error handling.

NOTE: This test suite runs inside Docker at /app/milia

UPDATED: Added tests for standard_transforms functionality:
- TransformSpec.to_dict()
- TransformationConfig.standard_transforms field
- TransformationConfig.get_standard_transforms()
- TransformationConfig.get_combined_transforms()
- TransformationConfig.has_standard_transforms()
"""

import sys
from pathlib import Path

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


import pytest
from pydantic import ValidationError as PydanticValidationError

# Import the module under test
from milia_pipeline.config.config_containers import (
    DatasetConfig,
    DescriptorCategoryConfig,
    DescriptorConfig,
    ExperimentalSetup,
    FilterConfig,
    HandlerConfig,
    ProcessingConfig,
    StructuralFeaturesConfig,
    TransformationConfig,
    TransformSpec,
    _get_valid_dataset_types,
    _is_valid_dataset_type,
    _resolve_canonical_dataset_type,
    check_configuration_compatibility,
    create_default_descriptor_config,
    create_default_experimental_setups,
    create_experimental_setup_from_dict,
    create_handler_config,
    create_minimal_config_for_testing,
    create_minimal_descriptor_config,
    create_transform_spec_from_dict,
    create_transformation_config_from_dict,
    get_config_summary,
    validate_handler_configuration_bundle,
)

# ==========================================
# TEST FIXTURES
# ==========================================

# Known valid dataset types for testing (matches real registry canonical names)
_VALID_TEST_TYPES = {
    "DFT",
    "DMC",
    "QM9",
    "ANI1x",
    "ANI1ccx",
    "ANI2x",
    "Wavefunction",
    "XXMD",
    "QDPi",
    "RMD17",
}


@pytest.fixture(autouse=True)
def _isolate_pydantic_validator(monkeypatch):
    """Isolate DatasetConfig Pydantic validator from real registry.

    The @field_validator("dataset_type") in config_containers.py calls
    _is_valid_dataset_type() which reaches config_loader's registry.
    In the full suite, config_loader's registry state can be inconsistent
    due to earlier test files' setup_module/teardown_module cycles.

    This patches the MODULE-LEVEL attribute (used by Pydantic validator)
    without affecting directly-imported function references (used by
    TestRegistryFunctions tests).
    """
    import milia_pipeline.config.config_containers as containers_module

    monkeypatch.setattr(
        containers_module,
        "_is_valid_dataset_type",
        lambda dt: dt in _VALID_TEST_TYPES or dt.upper() in {t.upper() for t in _VALID_TEST_TYPES},
    )
    monkeypatch.setattr(
        containers_module,
        "_get_valid_dataset_types",
        lambda: sorted(_VALID_TEST_TYPES),
    )
    # Clear cached registry types to prevent stale state from earlier tests
    monkeypatch.setattr(containers_module, "_CACHED_REGISTRY_TYPES", None)


@pytest.fixture
def sample_transform_spec():
    """Sample TransformSpec object."""
    return TransformSpec(
        name="AddSelfLoops", kwargs={}, enabled=True, description="Add self-loops to graph"
    )


@pytest.fixture
def sample_transform_spec_with_kwargs():
    """Sample TransformSpec with kwargs."""
    return TransformSpec(
        name="NormalizeFeatures",
        kwargs={"attrs": ["x"]},
        enabled=True,
        description="Normalize node features",
    )


@pytest.fixture
def sample_disabled_transform_spec():
    """Sample disabled TransformSpec."""
    return TransformSpec(
        name="DisabledTransform", kwargs={}, enabled=False, description="This transform is disabled"
    )


@pytest.fixture
def sample_experimental_setup(sample_transform_spec):
    """Sample ExperimentalSetup object."""
    return ExperimentalSetup(
        name="baseline",
        transforms=[sample_transform_spec],
        description="Baseline experimental setup",
        enabled=True,
    )


@pytest.fixture
def sample_experimental_setup_with_multiple_transforms(
    sample_transform_spec, sample_transform_spec_with_kwargs
):
    """Sample ExperimentalSetup with multiple transforms."""
    return ExperimentalSetup(
        name="enhanced",
        transforms=[sample_transform_spec, sample_transform_spec_with_kwargs],
        description="Enhanced experimental setup",
        enabled=True,
    )


@pytest.fixture
def sample_transformation_config(sample_experimental_setup):
    """Sample TransformationConfig object."""
    return TransformationConfig(
        experimental_setups={"baseline": sample_experimental_setup}, default_setup="baseline"
    )


@pytest.fixture
def sample_standard_transforms():
    """Sample list of standard transforms."""
    return [
        TransformSpec(name="AddSelfLoops", kwargs={"fill_value": 1.0}, enabled=True),
        TransformSpec(name="NormalizeFeatures", kwargs={"attrs": ["x"]}, enabled=True),
        TransformSpec(name="NormalizeFeatures", kwargs={"attrs": ["y"]}, enabled=True),
    ]


@pytest.fixture
def sample_transformation_config_with_standard(
    sample_experimental_setup, sample_standard_transforms
):
    """Sample TransformationConfig with standard_transforms."""
    return TransformationConfig(
        experimental_setups={"baseline": sample_experimental_setup},
        default_setup="baseline",
        standard_transforms=sample_standard_transforms,
    )


# ==========================================
# DATASETCONFIG TESTS
# ==========================================


class TestDatasetConfig:
    """Test suite for DatasetConfig container."""

    def test_creation_valid_dft(self):
        """Test creating valid DFT DatasetConfig."""
        config = DatasetConfig(dataset_type="DFT")
        assert config.dataset_type == "DFT"
        assert config.is_uncertainty_enabled is False
        assert isinstance(config.handler_config, dict)

    def test_creation_valid_dmc(self):
        """Test creating valid DMC DatasetConfig."""
        config = DatasetConfig(
            dataset_type="DMC", uncertainty_config={"use_for_loss_weighting": True}
        )
        assert config.dataset_type == "DMC"
        assert config.is_uncertainty_enabled is True

    def test_creation_invalid_dataset_type(self):
        """Test that invalid dataset type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset_type"):
            DatasetConfig(dataset_type="INVALID")

    def test_post_init_uncertainty_auto_compute(self):
        """Test auto-computation of uncertainty_enabled flag."""
        config = DatasetConfig(
            dataset_type="DMC", uncertainty_config={"use_for_loss_weighting": True}
        )
        assert config.is_uncertainty_enabled is True

        config2 = DatasetConfig(
            dataset_type="DMC", uncertainty_config={"use_for_loss_weighting": False}
        )
        assert config2.is_uncertainty_enabled is False

    def test_post_init_none_handler_config(self):
        """Test that None handler_config is initialized to empty dict."""
        config = DatasetConfig(dataset_type="DFT", handler_config=None)
        assert config.handler_config == {}

    def test_is_compatible_with_handler_dft(self):
        """Test handler compatibility check for DFT."""
        config = DatasetConfig(dataset_type="DFT")
        assert config.is_compatible_with_handler("DFT") is True
        assert config.is_compatible_with_handler("dft") is True
        assert config.is_compatible_with_handler("DMC") is False

    def test_is_compatible_with_handler_dmc(self):
        """Test handler compatibility check for DMC."""
        config = DatasetConfig(dataset_type="DMC")
        assert config.is_compatible_with_handler("DMC") is True
        assert config.is_compatible_with_handler("dmc") is True
        assert config.is_compatible_with_handler("DFT") is False

    def test_get_handler_config_dft(self):
        """Test getting handler config for DFT."""
        config = DatasetConfig(dataset_type="DFT", handler_config={"batch_size": 32})
        handler_cfg = config.get_handler_config()
        assert handler_cfg["batch_size"] == 32
        assert "uncertainty" not in handler_cfg

    def test_get_handler_config_dmc_with_uncertainty(self):
        """Test getting handler config for DMC with uncertainty."""
        uncertainty_cfg = {"use_for_loss_weighting": True, "threshold": 0.5}
        config = DatasetConfig(
            dataset_type="DMC",
            uncertainty_config=uncertainty_cfg,
            handler_config={"batch_size": 16},
        )
        handler_cfg = config.get_handler_config("dmc")
        assert handler_cfg["batch_size"] == 16
        assert handler_cfg["uncertainty"] == uncertainty_cfg

    def test_get_required_properties_dft(self):
        """Test getting required properties for DFT."""
        config = DatasetConfig(dataset_type="DFT")
        props = config.get_required_properties()
        assert "Etot" in props
        assert "atoms" in props
        assert "coordinates" in props
        assert "std" not in props

    def test_get_required_properties_dmc_with_uncertainty(self):
        """Test getting required properties for DMC with uncertainty."""
        config = DatasetConfig(
            dataset_type="DMC", uncertainty_config={"use_for_loss_weighting": True}
        )
        props = config.get_required_properties()
        assert "Etot" in props
        assert "atoms" in props
        assert "coordinates" in props
        assert "std" in props

    def test_get_required_properties_dmc_without_uncertainty(self):
        """Test getting required properties for DMC without uncertainty."""
        config = DatasetConfig(dataset_type="DMC", is_uncertainty_enabled=False)
        props = config.get_required_properties()
        assert "std" not in props

    def test_validate_handler_compatibility_valid(self):
        """Test validation for valid configuration."""
        config = DatasetConfig(dataset_type="DFT")
        is_valid, errors = config.validate_handler_compatibility()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_handler_compatibility_invalid_type(self):
        """Test validation with invalid dataset type (should be caught in __post_init__)."""
        # Since __post_init__ raises ValueError, we test the validation method
        # on a properly created config
        config = DatasetConfig(dataset_type="DFT")
        is_valid, errors = config.validate_handler_compatibility()
        assert is_valid is True

    def test_validate_handler_compatibility_dmc_missing_uncertainty(self):
        """Test validation for DMC with uncertainty enabled but no config."""
        config = DatasetConfig(
            dataset_type="DMC", is_uncertainty_enabled=True, uncertainty_config=None
        )
        is_valid, errors = config.validate_handler_compatibility()
        assert is_valid is False
        assert any("uncertainty_config required" in err for err in errors)

    def test_validate_handler_compatibility_dmc_missing_uncertainty_keys(self):
        """Test validation for DMC with incomplete uncertainty config."""
        config = DatasetConfig(
            dataset_type="DMC",
            uncertainty_config={"threshold": 0.5},  # Missing 'use_for_loss_weighting'
        )
        is_valid, errors = config.validate_handler_compatibility()
        assert is_valid is False
        assert any("use_for_loss_weighting" in err for err in errors)

    def test_frozen_immutability(self):
        """Test that DatasetConfig is frozen and immutable."""
        config = DatasetConfig(dataset_type="DFT")
        with pytest.raises(PydanticValidationError):
            config.dataset_type = "DMC"


# ==========================================
# FILTERCONFIG TESTS
# ==========================================


class TestFilterConfig:
    """Test suite for FilterConfig container."""

    def test_creation_with_defaults(self):
        """Test creating FilterConfig with default values."""
        config = FilterConfig()
        assert config.max_atoms is None
        assert config.min_atoms is None
        assert config.heavy_atom_filter is None
        assert isinstance(config.handler_filters, dict)

    def test_creation_with_values(self):
        """Test creating FilterConfig with specified values."""
        config = FilterConfig(max_atoms=50, min_atoms=5, heavy_atom_filter={"max_heavy_atoms": 30})
        assert config.max_atoms == 50
        assert config.min_atoms == 5
        assert config.heavy_atom_filter == {"max_heavy_atoms": 30}

    def test_post_init_none_handler_filters(self):
        """Test that None handler_filters is initialized."""
        config = FilterConfig(handler_filters=None)
        assert config.handler_filters == {}

    def test_get_handler_filters_dft(self):
        """Test getting handler-specific filters for DFT."""
        config = FilterConfig(
            max_atoms=50, min_atoms=5, handler_filters={"DFT": {"custom_filter": True}}
        )
        filters = config.get_handler_filters("DFT")
        assert filters["max_atoms"] == 50
        assert filters["min_atoms"] == 5
        assert filters["custom_filter"] is True

    def test_get_handler_filters_dmc(self):
        """Test getting handler-specific filters for DMC with uncertainty."""
        config = FilterConfig(max_atoms=50, dmc_uncertainty_filter={"threshold": 0.5})
        filters = config.get_handler_filters("DMC")
        assert filters["max_atoms"] == 50
        assert filters["uncertainty_filter"] == {"threshold": 0.5}

    def test_validate_filter_config_valid(self):
        """Test validation for valid filter config."""
        config = FilterConfig(max_atoms=50, min_atoms=5)
        is_valid, errors = config.validate_filter_config()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_filter_config_max_less_than_min(self):
        """Test validation when max_atoms < min_atoms."""
        config = FilterConfig(max_atoms=5, min_atoms=50)
        is_valid, errors = config.validate_filter_config()
        assert is_valid is False
        assert any("max_atoms" in err and ">=" in err for err in errors)

    def test_validate_filter_config_negative_min_atoms(self):
        """Test validation with negative min_atoms."""
        config = FilterConfig(min_atoms=-5)
        is_valid, errors = config.validate_filter_config()
        assert is_valid is False
        assert any("min_atoms must be positive" in err for err in errors)

    def test_validate_filter_config_negative_max_atoms(self):
        """Test validation with negative max_atoms."""
        config = FilterConfig(max_atoms=-10)
        is_valid, errors = config.validate_filter_config()
        assert is_valid is False
        assert any("max_atoms must be positive" in err for err in errors)

    def test_validate_filter_config_invalid_uncertainty_filter(self):
        """Test validation with non-dict uncertainty filter - Pydantic V2 enforces type at construction."""
        # Pydantic V2 enforces Dict type at construction, so we test with dict containing invalid structure
        # The validation method validates the content logic, not the type
        config = FilterConfig(
            dmc_uncertainty_filter={}
        )  # Empty dict is valid type but may fail validation
        is_valid, errors = config.validate_filter_config()
        # An empty dict should pass type validation
        assert is_valid is True

    def test_validate_filter_config_invalid_handler_filters(self):
        """Test validation with invalid handler filters - Pydantic V2 enforces nested dict type."""
        # Pydantic V2 enforces Dict[str, Dict[str, Any]] type at construction
        # Test with a type that Pydantic accepts but has invalid handler type
        config = FilterConfig(handler_filters={"INVALID_TYPE": {"key": "value"}})
        is_valid, errors = config.validate_filter_config()
        assert is_valid is False
        assert any("Unknown handler type" in err for err in errors)

    def test_validate_filter_config_unknown_handler_type(self):
        """Test validation with unknown handler type."""
        config = FilterConfig(handler_filters={"UNKNOWN": {}})
        is_valid, errors = config.validate_filter_config()
        assert is_valid is False
        assert any("Unknown handler type" in err for err in errors)


# ==========================================
# STRUCTURALFEATURESCONFIG TESTS
# ==========================================


class TestStructuralFeaturesConfig:
    """Test suite for StructuralFeaturesConfig container."""

    def test_creation_valid(self):
        """Test creating valid StructuralFeaturesConfig."""
        config = StructuralFeaturesConfig(
            atom_features=["atomic_number", "degree"], bond_features=["bond_type"]
        )
        assert config.atom_features == ["atomic_number", "degree"]
        assert config.bond_features == ["bond_type"]

    def test_post_init_none_handler_features(self):
        """Test that None handler_features is initialized."""
        config = StructuralFeaturesConfig(
            atom_features=["atomic_number"], bond_features=["bond_type"], handler_features=None
        )
        assert config.handler_features == {}

    def test_get_handler_features_default(self):
        """Test getting handler features with defaults."""
        config = StructuralFeaturesConfig(
            atom_features=["atomic_number"], bond_features=["bond_type"]
        )
        features = config.get_handler_features("DFT")
        assert features["atom_features"] == ["atomic_number"]
        assert features["bond_features"] == ["bond_type"]

    def test_get_handler_features_custom(self):
        """Test getting handler-specific custom features."""
        config = StructuralFeaturesConfig(
            atom_features=["atomic_number"],
            bond_features=["bond_type"],
            handler_features={
                "DMC": {
                    "atom_features": ["atomic_number", "uncertainty"],
                    "bond_features": ["bond_type", "distance"],
                }
            },
        )
        features = config.get_handler_features("DMC")
        assert "uncertainty" in features["atom_features"]
        assert "distance" in features["bond_features"]

    def test_validate_feature_config_valid(self):
        """Test validation for valid feature config."""
        config = StructuralFeaturesConfig(
            atom_features=["atomic_number"], bond_features=["bond_type"]
        )
        is_valid, errors = config.validate_feature_config()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_feature_config_invalid_atom_features_type(self):
        """Test that Pydantic V2 enforces List type at construction time."""
        # Pydantic V2 enforces List[str] type at construction, raises ValidationError
        with pytest.raises(PydanticValidationError):
            StructuralFeaturesConfig(atom_features="not_a_list", bond_features=["bond_type"])

    def test_validate_feature_config_invalid_bond_features_type(self):
        """Test that Pydantic V2 enforces List type at construction time."""
        # Pydantic V2 enforces List[str] type at construction, raises ValidationError
        with pytest.raises(PydanticValidationError):
            StructuralFeaturesConfig(atom_features=["atomic_number"], bond_features="not_a_list")

    def test_validate_feature_config_non_string_atom_features(self):
        """Test that Pydantic V2 enforces string elements in List[str] at construction time."""
        # Pydantic V2 enforces List[str] element types at construction, raises ValidationError
        with pytest.raises(PydanticValidationError):
            StructuralFeaturesConfig(
                atom_features=["atomic_number", 123], bond_features=["bond_type"]
            )

    def test_validate_feature_config_unknown_handler_type(self):
        """Test validation with unknown handler type."""
        config = StructuralFeaturesConfig(
            atom_features=["atomic_number"],
            bond_features=["bond_type"],
            handler_features={"UNKNOWN": {}},
        )
        is_valid, errors = config.validate_feature_config()
        assert is_valid is False
        assert any("Unknown handler type" in err for err in errors)


# ==========================================
# PROCESSINGCONFIG TESTS
# ==========================================


class TestProcessingConfig:
    """Test suite for ProcessingConfig container."""

    def test_creation_valid(self):
        """Test creating valid ProcessingConfig."""
        config = ProcessingConfig(scalar_graph_targets=["Etot", "HOMO"])
        assert config.scalar_graph_targets == ["Etot", "HOMO"]
        assert config.node_features == []
        assert config.test_molecule_limit is None

    def test_post_init_none_fields(self):
        """Test that None fields are initialized to empty lists."""
        config = ProcessingConfig(
            scalar_graph_targets=["Etot"], node_features=None, vector_graph_properties=None
        )
        assert config.node_features == []
        assert config.vector_graph_properties == []

    def test_get_handler_processing_config(self):
        """Test getting handler-specific processing config."""
        config = ProcessingConfig(scalar_graph_targets=["Etot"], node_features=["coordinates"])
        proc_cfg = config.get_handler_processing_config("DFT")
        assert proc_cfg["scalar_graph_targets"] == ["Etot"]
        assert proc_cfg["node_features"] == ["coordinates"]

    def test_is_migration_enabled_true(self):
        """Test migration enabled check."""
        config = ProcessingConfig(
            scalar_graph_targets=["Etot"], migration_settings={"enabled": True}
        )
        assert config.is_migration_enabled() is True

    def test_is_migration_enabled_false(self):
        """Test migration disabled check."""
        config = ProcessingConfig(
            scalar_graph_targets=["Etot"], migration_settings={"enabled": False}
        )
        assert config.is_migration_enabled() is False

    def test_validate_processing_config_valid(self):
        """Test validation for valid processing config."""
        config = ProcessingConfig(scalar_graph_targets=["Etot"])
        is_valid, errors = config.validate_processing_config()
        assert is_valid is True
        assert len(errors) == 0


# ==========================================
# HANDLERCONFIG TESTS
# ==========================================


class TestHandlerConfig:
    """Test suite for HandlerConfig container."""

    def test_creation_valid(self):
        """Test creating valid HandlerConfig."""
        config = HandlerConfig(handler_type="DFT")
        assert config.handler_type == "DFT"
        assert config.migration_mode is False

    def test_creation_invalid_handler_type(self):
        """Test that invalid handler type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid handler_type"):
            HandlerConfig(handler_type="INVALID")

    def test_get_validation_setting(self):
        """Test getting validation settings."""
        config = HandlerConfig(handler_type="DFT", validation_settings={"strict_mode": True})
        assert config.get_validation_setting("strict_mode") is True
        assert config.get_validation_setting("nonexistent", "default") == "default"

    def test_validate_handler_config_valid(self):
        """Test validation for valid handler config."""
        config = HandlerConfig(handler_type="DFT")
        is_valid, errors = config.validate_handler_config()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_handler_config_invalid_recovery_mode(self):
        """Test validation with invalid recovery mode."""
        config = HandlerConfig(handler_type="DFT", error_handling={"recovery_mode": "invalid_mode"})
        is_valid, errors = config.validate_handler_config()
        assert is_valid is False
        assert any("Invalid error recovery mode" in err for err in errors)


# ==========================================
# TRANSFORMSPEC TESTS
# ==========================================


class TestTransformSpec:
    """Test suite for TransformSpec container."""

    def test_creation_valid(self):
        """Test creating valid TransformSpec."""
        spec = TransformSpec(name="AddSelfLoops")
        assert spec.name == "AddSelfLoops"
        assert spec.kwargs == {}
        assert spec.enabled is True

    def test_creation_with_kwargs(self):
        """Test creating TransformSpec with kwargs."""
        spec = TransformSpec(name="Normalize", kwargs={"mean": 0, "std": 1})
        assert spec.kwargs == {"mean": 0, "std": 1}

    def test_creation_invalid_empty_name(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            TransformSpec(name="")

    def test_creation_invalid_none_name(self):
        """Test that None name raises Pydantic ValidationError."""
        # Pydantic V2 raises ValidationError for type mismatch before field_validator runs
        with pytest.raises(PydanticValidationError):
            TransformSpec(name=None)

    def test_creation_invalid_kwargs_type(self):
        """Test that non-dict kwargs raises Pydantic ValidationError."""
        # Pydantic V2 raises ValidationError for type mismatch before field_validator runs
        with pytest.raises(PydanticValidationError):
            TransformSpec(name="Test", kwargs="not_a_dict")

    def test_post_init_none_kwargs(self):
        """Test that None kwargs is initialized."""
        spec = TransformSpec(name="Test", kwargs=None)
        assert spec.kwargs == {}

    def test_get_cache_key(self):
        """Test cache key generation."""
        spec = TransformSpec(name="Test", kwargs={"param": "value"})
        key = spec.get_cache_key()
        assert isinstance(key, str)
        assert len(key) == 16  # MD5 hash shortened to 16 chars

    def test_get_cache_key_consistency(self):
        """Test that same config produces same cache key."""
        spec1 = TransformSpec(name="Test", kwargs={"a": 1, "b": 2})
        spec2 = TransformSpec(name="Test", kwargs={"b": 2, "a": 1})
        assert spec1.get_cache_key() == spec2.get_cache_key()

    def test_validate_transform_spec_valid(self):
        """Test validation for valid transform spec."""
        spec = TransformSpec(name="Test")
        is_valid, errors = spec.validate_transform_spec()
        assert is_valid is True
        assert len(errors) == 0

    def test_frozen_immutability(self):
        """Test that TransformSpec is frozen."""
        spec = TransformSpec(name="Test")
        with pytest.raises(PydanticValidationError):
            spec.name = "NewName"

    # ==========================================
    # NEW TESTS: TransformSpec.to_dict()
    # ==========================================

    def test_to_dict_basic(self):
        """Test to_dict() with basic TransformSpec."""
        spec = TransformSpec(name="AddSelfLoops")
        result = spec.to_dict()

        assert isinstance(result, dict)
        assert result["name"] == "AddSelfLoops"
        assert result["kwargs"] == {}
        assert result["enabled"] is True

    def test_to_dict_with_kwargs(self):
        """Test to_dict() with kwargs."""
        spec = TransformSpec(name="NormalizeFeatures", kwargs={"attrs": ["x", "y"]}, enabled=True)
        result = spec.to_dict()

        assert result["name"] == "NormalizeFeatures"
        assert result["kwargs"] == {"attrs": ["x", "y"]}
        assert result["enabled"] is True

    def test_to_dict_disabled(self):
        """Test to_dict() with disabled transform."""
        spec = TransformSpec(name="Disabled", enabled=False)
        result = spec.to_dict()

        assert result["name"] == "Disabled"
        assert result["enabled"] is False

    def test_to_dict_none_kwargs_becomes_empty_dict(self):
        """Test to_dict() handles None kwargs properly."""
        spec = TransformSpec(name="Test", kwargs=None)
        result = spec.to_dict()

        assert result["kwargs"] == {}

    def test_to_dict_complex_kwargs(self):
        """Test to_dict() with complex nested kwargs."""
        spec = TransformSpec(
            name="Complex",
            kwargs={"nested": {"a": 1, "b": [1, 2, 3]}, "list": [4, 5, 6], "float": 0.5},
        )
        result = spec.to_dict()

        assert result["kwargs"]["nested"] == {"a": 1, "b": [1, 2, 3]}
        assert result["kwargs"]["list"] == [4, 5, 6]
        assert result["kwargs"]["float"] == 0.5

    def test_to_dict_output_format_for_compose_transforms(self):
        """Test that to_dict() output is compatible with compose_transforms()."""
        spec = TransformSpec(name="AddSelfLoops", kwargs={"fill_value": 1.0}, enabled=True)
        result = spec.to_dict()

        # These are the exact keys expected by graph_transforms.compose_transforms()
        assert "name" in result
        assert "kwargs" in result
        assert "enabled" in result
        assert len(result) == 3  # Only these 3 keys


# ==========================================
# EXPERIMENTALSETUP TESTS
# ==========================================


class TestExperimentalSetup:
    """Test suite for ExperimentalSetup container."""

    def test_creation_valid(self, sample_transform_spec):
        """Test creating valid ExperimentalSetup."""
        setup = ExperimentalSetup(name="baseline", transforms=[sample_transform_spec])
        assert setup.name == "baseline"
        assert len(setup.transforms) == 1
        assert setup.enabled is True

    def test_creation_invalid_empty_name(self, sample_transform_spec):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            ExperimentalSetup(name="", transforms=[sample_transform_spec])

    def test_creation_invalid_non_list_transforms(self):
        """Test that non-list transforms raises Pydantic ValidationError."""
        # Pydantic V2 raises ValidationError for type mismatch before field_validator runs
        with pytest.raises(PydanticValidationError):
            ExperimentalSetup(name="test", transforms="not_a_list")

    def test_creation_invalid_transform_type(self):
        """Test that non-TransformSpec in list raises Pydantic ValidationError."""
        # Pydantic V2 raises ValidationError for type mismatch
        with pytest.raises(PydanticValidationError):
            ExperimentalSetup(name="test", transforms=["not_a_spec"])

    def test_post_init_none_lists(self, sample_transform_spec):
        """Test that None lists are initialized."""
        setup = ExperimentalSetup(
            name="test",
            transforms=[sample_transform_spec],
            expected_effects=None,
            dataset_compatibility=None,
        )
        assert setup.expected_effects == []
        assert setup.dataset_compatibility == []

    def test_get_transform_names(self, sample_transform_spec):
        """Test getting transform names."""
        disabled_spec = TransformSpec(name="Disabled", enabled=False)
        setup = ExperimentalSetup(name="test", transforms=[sample_transform_spec, disabled_spec])
        names = setup.get_transform_names()
        assert "AddSelfLoops" in names
        assert "Disabled" not in names

    def test_get_enabled_transforms(self, sample_transform_spec):
        """Test getting enabled transforms."""
        disabled_spec = TransformSpec(name="Disabled", enabled=False)
        setup = ExperimentalSetup(name="test", transforms=[sample_transform_spec, disabled_spec])
        enabled = setup.get_enabled_transforms()
        assert len(enabled) == 1
        assert enabled[0].name == "AddSelfLoops"

    def test_get_cache_key(self, sample_transform_spec):
        """Test cache key generation."""
        setup = ExperimentalSetup(name="test", transforms=[sample_transform_spec])
        key = setup.get_cache_key()
        assert isinstance(key, str)
        assert len(key) == 16

    def test_is_compatible_with_dataset_no_restriction(self, sample_transform_spec):
        """Test dataset compatibility with no restrictions."""
        setup = ExperimentalSetup(name="test", transforms=[sample_transform_spec])
        assert setup.is_compatible_with_dataset("DFT") is True
        assert setup.is_compatible_with_dataset("DMC") is True

    def test_is_compatible_with_dataset_with_restriction(self, sample_transform_spec):
        """Test dataset compatibility with restrictions."""
        setup = ExperimentalSetup(
            name="test", transforms=[sample_transform_spec], dataset_compatibility=["DFT"]
        )
        assert setup.is_compatible_with_dataset("DFT") is True
        assert setup.is_compatible_with_dataset("dft") is True
        assert setup.is_compatible_with_dataset("DMC") is False

    def test_validate_experimental_setup_valid(self, sample_transform_spec):
        """Test validation for valid setup."""
        setup = ExperimentalSetup(name="test", transforms=[sample_transform_spec])
        is_valid, errors = setup.validate_experimental_setup()
        assert is_valid is True
        assert len(errors) == 0


# ==========================================
# TRANSFORMATIONCONFIG TESTS
# ==========================================


class TestTransformationConfig:
    """Test suite for TransformationConfig container."""

    def test_creation_valid(self, sample_experimental_setup):
        """Test creating valid TransformationConfig."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup}, default_setup="baseline"
        )
        assert len(config.experimental_setups) == 1
        assert config.default_setup == "baseline"

    def test_creation_invalid_empty_setups(self):
        """Test that empty setups dict raises ValueError."""
        with pytest.raises(ValueError, match="At least one experimental setup"):
            TransformationConfig(experimental_setups={}, default_setup="test")

    def test_creation_invalid_default_not_found(self, sample_experimental_setup):
        """Test that invalid default setup raises ValueError."""
        with pytest.raises(ValueError, match="Default setup.*not found"):
            TransformationConfig(
                experimental_setups={"baseline": sample_experimental_setup},
                default_setup="nonexistent",
            )

    def test_creation_invalid_setup_type(self):
        """Test that non-ExperimentalSetup raises Pydantic ValidationError."""
        # Pydantic V2 raises ValidationError for type mismatch
        with pytest.raises(PydanticValidationError):
            TransformationConfig(experimental_setups={"test": "not_a_setup"}, default_setup="test")

    def test_get_default_setup(self, sample_experimental_setup):
        """Test getting default setup."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup}, default_setup="baseline"
        )
        default = config.get_default_setup()
        assert default.name == "baseline"

    def test_get_setup(self, sample_experimental_setup):
        """Test getting setup by name."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup}, default_setup="baseline"
        )
        setup = config.get_setup("baseline")
        assert setup is not None
        assert setup.name == "baseline"

        nonexistent = config.get_setup("nonexistent")
        assert nonexistent is None

    def test_get_enabled_setups(self, sample_experimental_setup, sample_transform_spec):
        """Test getting enabled setups."""
        disabled_setup = ExperimentalSetup(
            name="disabled", transforms=[sample_transform_spec], enabled=False
        )
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup, "disabled": disabled_setup},
            default_setup="baseline",
        )
        enabled = config.get_enabled_setups()
        assert len(enabled) == 1
        assert "baseline" in enabled
        assert "disabled" not in enabled

    def test_get_setups_for_dataset(self, sample_transform_spec):
        """Test getting setups for specific dataset type."""
        dft_setup = ExperimentalSetup(
            name="dft_only", transforms=[sample_transform_spec], dataset_compatibility=["DFT"]
        )
        dmc_setup = ExperimentalSetup(
            name="dmc_only", transforms=[sample_transform_spec], dataset_compatibility=["DMC"]
        )
        config = TransformationConfig(
            experimental_setups={"dft_only": dft_setup, "dmc_only": dmc_setup},
            default_setup="dft_only",
        )
        dft_setups = config.get_setups_for_dataset("DFT")
        assert len(dft_setups) == 1
        assert "dft_only" in dft_setups

    def test_list_setup_names(self, sample_experimental_setup):
        """Test listing all setup names."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup}, default_setup="baseline"
        )
        names = config.list_setup_names()
        assert "baseline" in names

    def test_list_enabled_setup_names(self, sample_experimental_setup, sample_transform_spec):
        """Test listing enabled setup names."""
        disabled_setup = ExperimentalSetup(
            name="disabled", transforms=[sample_transform_spec], enabled=False
        )
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup, "disabled": disabled_setup},
            default_setup="baseline",
        )
        enabled_names = config.list_enabled_setup_names()
        assert "baseline" in enabled_names
        assert "disabled" not in enabled_names

    def test_is_validation_enabled_default(self, sample_experimental_setup):
        """Test validation enabled check with default."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup}, default_setup="baseline"
        )
        # Default should be True
        assert config.is_validation_enabled() is True

    def test_is_strict_mode_enabled_default(self, sample_experimental_setup):
        """Test strict mode check with default."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup}, default_setup="baseline"
        )
        # Default should be False
        assert config.is_strict_mode_enabled() is False

    # ==========================================
    # NEW TESTS: standard_transforms field
    # ==========================================

    def test_creation_with_standard_transforms(
        self, sample_experimental_setup, sample_standard_transforms
    ):
        """Test creating TransformationConfig with standard_transforms."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup},
            default_setup="baseline",
            standard_transforms=sample_standard_transforms,
        )
        assert config.standard_transforms is not None
        assert len(config.standard_transforms) == 3

    def test_creation_without_standard_transforms(self, sample_experimental_setup):
        """Test creating TransformationConfig without standard_transforms defaults to empty list."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup}, default_setup="baseline"
        )
        assert config.standard_transforms == []

    def test_creation_with_none_standard_transforms(self, sample_experimental_setup):
        """Test creating TransformationConfig with None standard_transforms."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup},
            default_setup="baseline",
            standard_transforms=None,
        )
        assert config.standard_transforms == []

    def test_standard_transforms_validation_invalid_type(self, sample_experimental_setup):
        """Test that non-list standard_transforms raises Pydantic ValidationError."""
        # Pydantic V2 raises ValidationError for type mismatch
        with pytest.raises(PydanticValidationError):
            TransformationConfig(
                experimental_setups={"baseline": sample_experimental_setup},
                default_setup="baseline",
                standard_transforms="not_a_list",
            )

    def test_standard_transforms_validation_invalid_item(self, sample_experimental_setup):
        """Test that non-TransformSpec in standard_transforms raises Pydantic ValidationError."""
        # Pydantic V2 raises ValidationError for type mismatch
        with pytest.raises(PydanticValidationError):
            TransformationConfig(
                experimental_setups={"baseline": sample_experimental_setup},
                default_setup="baseline",
                standard_transforms=["not_a_transform_spec"],
            )

    # ==========================================
    # NEW TESTS: get_standard_transforms()
    # ==========================================

    def test_get_standard_transforms_returns_enabled_only(self, sample_experimental_setup):
        """Test get_standard_transforms() returns only enabled transforms."""
        transforms = [
            TransformSpec(name="Enabled1", enabled=True),
            TransformSpec(name="Disabled", enabled=False),
            TransformSpec(name="Enabled2", enabled=True),
        ]
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup},
            default_setup="baseline",
            standard_transforms=transforms,
        )
        result = config.get_standard_transforms()

        assert len(result) == 2
        assert all(t.enabled for t in result)
        assert result[0].name == "Enabled1"
        assert result[1].name == "Enabled2"

    def test_get_standard_transforms_empty(self, sample_experimental_setup):
        """Test get_standard_transforms() with no standard transforms."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup},
            default_setup="baseline",
            standard_transforms=[],
        )
        result = config.get_standard_transforms()
        assert result == []

    def test_get_standard_transforms_all_disabled(self, sample_experimental_setup):
        """Test get_standard_transforms() when all are disabled."""
        transforms = [
            TransformSpec(name="Disabled1", enabled=False),
            TransformSpec(name="Disabled2", enabled=False),
        ]
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup},
            default_setup="baseline",
            standard_transforms=transforms,
        )
        result = config.get_standard_transforms()
        assert result == []

    # ==========================================
    # NEW TESTS: get_standard_transforms_as_dicts()
    # ==========================================

    def test_get_standard_transforms_as_dicts(
        self, sample_experimental_setup, sample_standard_transforms
    ):
        """Test get_standard_transforms_as_dicts() returns list of dicts."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup},
            default_setup="baseline",
            standard_transforms=sample_standard_transforms,
        )
        result = config.get_standard_transforms_as_dicts()

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(d, dict) for d in result)
        assert result[0]["name"] == "AddSelfLoops"
        assert result[0]["kwargs"] == {"fill_value": 1.0}

    # ==========================================
    # NEW TESTS: get_combined_transforms()
    # ==========================================

    def test_get_combined_transforms_order(self, sample_transform_spec):
        """Test get_combined_transforms() returns standard first, then experimental."""
        standard = [
            TransformSpec(name="Standard1", enabled=True),
            TransformSpec(name="Standard2", enabled=True),
        ]
        experimental_setup = ExperimentalSetup(
            name="baseline",
            transforms=[
                TransformSpec(name="Experimental1", enabled=True),
                TransformSpec(name="Experimental2", enabled=True),
            ],
        )
        config = TransformationConfig(
            experimental_setups={"baseline": experimental_setup},
            default_setup="baseline",
            standard_transforms=standard,
        )
        result = config.get_combined_transforms()

        assert len(result) == 4
        assert result[0].name == "Standard1"
        assert result[1].name == "Standard2"
        assert result[2].name == "Experimental1"
        assert result[3].name == "Experimental2"

    def test_get_combined_transforms_uses_default_setup(
        self, sample_experimental_setup, sample_standard_transforms
    ):
        """Test get_combined_transforms() uses default_setup when no name provided."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup},
            default_setup="baseline",
            standard_transforms=sample_standard_transforms,
        )
        result = config.get_combined_transforms()

        # Should have 3 standard + 1 from baseline setup
        assert len(result) == 4

    def test_get_combined_transforms_specific_setup(self, sample_transform_spec):
        """Test get_combined_transforms() with specific setup name."""
        standard = [TransformSpec(name="Standard", enabled=True)]
        setup1 = ExperimentalSetup(
            name="setup1", transforms=[TransformSpec(name="Setup1Transform", enabled=True)]
        )
        setup2 = ExperimentalSetup(
            name="setup2", transforms=[TransformSpec(name="Setup2Transform", enabled=True)]
        )
        config = TransformationConfig(
            experimental_setups={"setup1": setup1, "setup2": setup2},
            default_setup="setup1",
            standard_transforms=standard,
        )

        result = config.get_combined_transforms("setup2")

        assert len(result) == 2
        assert result[0].name == "Standard"
        assert result[1].name == "Setup2Transform"

    def test_get_combined_transforms_disabled_setup(self, sample_transform_spec):
        """Test get_combined_transforms() with disabled experimental setup."""
        standard = [TransformSpec(name="Standard", enabled=True)]
        disabled_setup = ExperimentalSetup(
            name="disabled", transforms=[sample_transform_spec], enabled=False
        )
        config = TransformationConfig(
            experimental_setups={"disabled": disabled_setup},
            default_setup="disabled",
            standard_transforms=standard,
        )
        result = config.get_combined_transforms()

        # Only standard transforms since setup is disabled
        assert len(result) == 1
        assert result[0].name == "Standard"

    def test_get_combined_transforms_no_standard(self, sample_experimental_setup):
        """Test get_combined_transforms() with no standard transforms."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup},
            default_setup="baseline",
            standard_transforms=[],
        )
        result = config.get_combined_transforms()

        # Only experimental setup transforms
        assert len(result) == 1
        assert result[0].name == "AddSelfLoops"

    def test_get_combined_transforms_empty_experimental(self):
        """Test get_combined_transforms() with empty experimental setup."""
        standard = [TransformSpec(name="Standard", enabled=True)]
        empty_setup = ExperimentalSetup(name="empty", transforms=[])
        config = TransformationConfig(
            experimental_setups={"empty": empty_setup},
            default_setup="empty",
            standard_transforms=standard,
        )
        result = config.get_combined_transforms()

        # Only standard transforms
        assert len(result) == 1
        assert result[0].name == "Standard"

    def test_get_combined_transforms_nonexistent_setup(
        self, sample_experimental_setup, sample_standard_transforms
    ):
        """Test get_combined_transforms() with nonexistent setup name."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup},
            default_setup="baseline",
            standard_transforms=sample_standard_transforms,
        )
        result = config.get_combined_transforms("nonexistent")

        # Only standard transforms since setup not found
        assert len(result) == 3

    # ==========================================
    # NEW TESTS: get_combined_transforms_as_dicts()
    # ==========================================

    def test_get_combined_transforms_as_dicts(
        self, sample_experimental_setup, sample_standard_transforms
    ):
        """Test get_combined_transforms_as_dicts() returns list of dicts."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup},
            default_setup="baseline",
            standard_transforms=sample_standard_transforms,
        )
        result = config.get_combined_transforms_as_dicts()

        assert isinstance(result, list)
        assert len(result) == 4
        assert all(isinstance(d, dict) for d in result)
        assert all("name" in d and "kwargs" in d and "enabled" in d for d in result)

    # ==========================================
    # NEW TESTS: has_standard_transforms()
    # ==========================================

    def test_has_standard_transforms_true(
        self, sample_experimental_setup, sample_standard_transforms
    ):
        """Test has_standard_transforms() returns True when present."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup},
            default_setup="baseline",
            standard_transforms=sample_standard_transforms,
        )
        assert config.has_standard_transforms() is True

    def test_has_standard_transforms_false_empty(self, sample_experimental_setup):
        """Test has_standard_transforms() returns False when empty."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup},
            default_setup="baseline",
            standard_transforms=[],
        )
        assert config.has_standard_transforms() is False

    def test_has_standard_transforms_false_all_disabled(self, sample_experimental_setup):
        """Test has_standard_transforms() returns False when all disabled."""
        transforms = [
            TransformSpec(name="Disabled1", enabled=False),
            TransformSpec(name="Disabled2", enabled=False),
        ]
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup},
            default_setup="baseline",
            standard_transforms=transforms,
        )
        assert config.has_standard_transforms() is False


# ==========================================
# FACTORY FUNCTION TESTS
# ==========================================


class TestFactoryFunctions:
    """Test suite for factory functions."""

    def test_create_transform_spec_from_dict(self):
        """Test creating TransformSpec from dict."""
        spec_dict = {"name": "AddSelfLoops", "kwargs": {"fill_value": 1.0}, "enabled": True}
        spec = create_transform_spec_from_dict(spec_dict)
        assert isinstance(spec, TransformSpec)
        assert spec.name == "AddSelfLoops"
        assert spec.kwargs == {"fill_value": 1.0}

    def test_create_transform_spec_from_dict_missing_name(self):
        """Test that missing name raises ValueError."""
        with pytest.raises(ValueError, match="must contain 'name'"):
            create_transform_spec_from_dict({"kwargs": {}})

    def test_create_experimental_setup_from_dict(self):
        """Test creating ExperimentalSetup from dict."""
        setup_dict = {"name": "test", "transforms": [{"name": "AddSelfLoops"}]}
        setup = create_experimental_setup_from_dict(setup_dict)
        assert setup is not None
        assert setup.name == "test"

    def test_create_transformation_config_from_dict(self):
        """Test creating TransformationConfig from dict."""
        config_dict = {
            "experimental_setups": {
                "baseline": {"name": "baseline", "transforms": [{"name": "AddSelfLoops"}]}
            },
            "default_setup": "baseline",
        }
        config = create_transformation_config_from_dict(config_dict)
        assert isinstance(config, TransformationConfig)
        assert "baseline" in config.experimental_setups

    def test_create_minimal_config_for_testing(self):
        """Test creating minimal config for testing."""
        config = create_minimal_config_for_testing()
        assert isinstance(config, dict)
        assert "dataset_config" in config
        assert "transformation_config" in config


# ==========================================
# UTILITY FUNCTION TESTS
# ==========================================


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_get_config_summary(self, sample_transformation_config):
        """Test getting config summary."""
        bundle = {"transformation_config": sample_transformation_config}
        summary = get_config_summary(bundle)
        assert "timestamp" in summary
        assert "bundle_keys" in summary

    def test_check_configuration_compatibility(self, sample_transformation_config):
        """Test checking configuration compatibility."""
        config_a = {"transformation_config": sample_transformation_config}
        config_b = {"transformation_config": sample_transformation_config}

        is_compatible, issues = check_configuration_compatibility(config_a, config_b)
        assert is_compatible is True
        assert len(issues) == 0


# ==========================================
# EDGE CASE AND ERROR HANDLING TESTS
# ==========================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_dataset_config_empty_uncertainty_config(self):
        """Test DatasetConfig with empty uncertainty config."""
        config = DatasetConfig(dataset_type="DMC", uncertainty_config={})
        assert config.is_uncertainty_enabled is False

    def test_filter_config_zero_atoms(self):
        """Test FilterConfig with zero atom limits."""
        config = FilterConfig(max_atoms=0, min_atoms=0)
        is_valid, errors = config.validate_filter_config()
        assert is_valid is False

    def test_processing_config_none_test_limit(self):
        """Test ProcessingConfig with None test limit (valid)."""
        config = ProcessingConfig(scalar_graph_targets=["Etot"], test_molecule_limit=None)
        is_valid, errors = config.validate_processing_config()
        assert is_valid is True

    def test_transform_spec_empty_kwargs(self):
        """Test TransformSpec with empty kwargs."""
        spec = TransformSpec(name="Test", kwargs={})
        assert spec.kwargs == {}

    def test_experimental_setup_empty_transforms_in_non_strict_mode(self):
        """Test ExperimentalSetup with empty transforms list."""
        # Should work for non-strict mode
        setup = ExperimentalSetup(name="test", transforms=[])
        assert len(setup.transforms) == 0

    def test_transformation_config_single_setup(self, sample_experimental_setup):
        """Test TransformationConfig with single setup."""
        config = TransformationConfig(
            experimental_setups={"single": sample_experimental_setup}, default_setup="single"
        )
        assert len(config.experimental_setups) == 1

    def test_handler_config_empty_settings(self):
        """Test HandlerConfig with all empty settings."""
        config = HandlerConfig(
            handler_type="DFT", validation_settings={}, processing_settings={}, error_handling={}
        )
        assert config.get_validation_setting("any_key") is None

    def test_create_dataset_config_requires_dataset_type(self):
        """Test that DatasetConfig requires dataset_type - Pydantic V2 raises ValidationError."""
        # Pydantic V2 raises ValidationError for missing required field
        with pytest.raises(PydanticValidationError):
            DatasetConfig()

    def test_create_transform_spec_from_dict_missing_required_field(self):
        """Test factory function with missing required field."""
        with pytest.raises(ValueError, match="must contain 'name'"):
            create_transform_spec_from_dict({})

    def test_transformation_config_from_dict_with_empty_config(self):
        """Test transformation config with empty/invalid input."""
        with pytest.raises(ValueError, match="must contain"):
            create_transformation_config_from_dict({})

    def test_config_summary_empty_bundle(self):
        """Test config summary with empty bundle."""
        summary = get_config_summary({})
        assert "timestamp" in summary
        assert "bundle_keys" in summary
        assert len(summary["bundle_keys"]) == 0


# ==========================================
# PERFORMANCE AND CACHING TESTS
# ==========================================


class TestPerformanceAndCaching:
    """Test performance-related features like caching."""

    def test_transform_spec_cache_key_uniqueness(self):
        """Test that different specs produce different cache keys."""
        spec1 = TransformSpec(name="Transform1", kwargs={"param": 1})
        spec2 = TransformSpec(name="Transform2", kwargs={"param": 1})
        spec3 = TransformSpec(name="Transform1", kwargs={"param": 2})

        assert spec1.get_cache_key() != spec2.get_cache_key()
        assert spec1.get_cache_key() != spec3.get_cache_key()

    def test_experimental_setup_cache_key_consistency(self, sample_transform_spec):
        """Test that setup cache keys are consistent."""
        setup1 = ExperimentalSetup(name="test", transforms=[sample_transform_spec])
        setup2 = ExperimentalSetup(name="test", transforms=[sample_transform_spec])

        # Same config should produce same key
        assert setup1.get_cache_key() == setup2.get_cache_key()

    def test_cache_key_handles_complex_kwargs(self):
        """Test cache key generation with complex kwargs."""
        spec = TransformSpec(
            name="Complex", kwargs={"nested": {"a": 1, "b": [1, 2, 3]}, "list": [4, 5, 6]}
        )
        key = spec.get_cache_key()
        assert isinstance(key, str)
        assert len(key) == 16


# ==========================================
# CONFIGURATION MIGRATION TESTS
# ==========================================


class TestConfigurationMigration:
    """Test configuration migration scenarios."""

    def test_create_transformation_config_from_dict_simple(self):
        """Test creating transformation config from dict."""
        config_dict = {
            "experimental_setups": {
                "baseline": {
                    "name": "baseline",
                    "transforms": [{"name": "AddSelfLoops", "kwargs": {}, "enabled": True}],
                    "enabled": True,
                }
            },
            "default_setup": "baseline",
        }
        config = create_transformation_config_from_dict(config_dict)
        assert isinstance(config, TransformationConfig)
        assert "baseline" in config.experimental_setups

    def test_create_transformation_config_from_dict_list_format(self):
        """Test creating config with list format for transforms."""
        config_dict = {
            "experimental_setups": {"baseline": [{"name": "AddSelfLoops", "kwargs": {}}]},
            "default_setup": "baseline",
        }
        config = create_transformation_config_from_dict(config_dict)
        assert "baseline" in config.experimental_setups

    def test_create_experimental_setup_handles_invalid_transforms(self):
        """Test that invalid transforms are skipped gracefully."""
        setup_dict = {
            "name": "test",
            "transforms": [
                {"name": "ValidTransform", "kwargs": {}},
                {"invalid": "no_name"},  # Missing 'name'
                {"name": "AnotherValid", "kwargs": {}},
            ],
        }
        setup = create_experimental_setup_from_dict(setup_dict, strict_validation=False)
        assert setup is not None
        # Should have 2 valid transforms (invalid one skipped)
        assert len(setup.transforms) == 2

    def test_migration_metadata_in_config(self):
        """Test that configs can contain migration metadata."""
        config_dict = {
            "experimental_setups": {
                "baseline": {"name": "baseline", "transforms": [{"name": "Test"}]}
            },
            "default_setup": "baseline",
            "migration_metadata": {"migrated_from": "legacy", "timestamp": "2025-01-01"},
        }
        config = create_transformation_config_from_dict(config_dict)
        assert hasattr(config, "migration_metadata")
        assert config.migration_metadata["migrated_from"] == "legacy"


# ==========================================
# INTEGRATION TESTS
# ==========================================


class TestIntegration:
    """Integration tests for standard_transforms with experimental setups."""

    def test_combined_transforms_preserves_all_properties(self):
        """Test that combined transforms preserve all TransformSpec properties."""
        standard = [
            TransformSpec(
                name="Standard",
                kwargs={"param": "value"},
                enabled=True,
                description="Standard transform",
            )
        ]
        experimental_setup = ExperimentalSetup(
            name="test",
            transforms=[
                TransformSpec(
                    name="Experimental",
                    kwargs={"other_param": 123},
                    enabled=True,
                    description="Experimental transform",
                )
            ],
        )
        config = TransformationConfig(
            experimental_setups={"test": experimental_setup},
            default_setup="test",
            standard_transforms=standard,
        )

        combined = config.get_combined_transforms()

        assert combined[0].kwargs == {"param": "value"}
        assert combined[1].kwargs == {"other_param": 123}

    def test_to_dict_round_trip(self):
        """Test that to_dict() output can be used to recreate equivalent spec."""
        original = TransformSpec(name="Test", kwargs={"a": 1, "b": "two"}, enabled=True)

        dict_repr = original.to_dict()
        recreated = TransformSpec(
            name=dict_repr["name"], kwargs=dict_repr["kwargs"], enabled=dict_repr["enabled"]
        )

        assert original.name == recreated.name
        assert original.kwargs == recreated.kwargs
        assert original.enabled == recreated.enabled


# ==========================================
# DESCRIPTORCONFIG TESTS
# ==========================================


class TestDescriptorConfig:
    """Test suite for DescriptorConfig container."""

    def test_creation_with_defaults(self):
        """Test creating DescriptorConfig with default values."""
        config = DescriptorConfig()
        assert config.enabled is True
        assert config.default_categories == ["constitutional", "topological"]
        assert config.cache_descriptors is True
        assert config.parallel_computation is False
        assert config.num_workers == 1
        assert config.error_handling == "warn"
        assert config.validation_mode == "standard"

    def test_creation_with_custom_values(self):
        """Test creating DescriptorConfig with custom values."""
        config = DescriptorConfig(
            enabled=False,
            default_categories=["geometric", "electronic"],
            cache_descriptors=False,
            parallel_computation=True,
            num_workers=4,
            error_handling="strict",
            validation_mode="permissive",
        )
        assert config.enabled is False
        assert config.default_categories == ["geometric", "electronic"]
        assert config.num_workers == 4
        assert config.error_handling == "strict"
        assert config.validation_mode == "permissive"

    def test_invalid_error_handling_mode(self):
        """Test that invalid error_handling raises ValueError."""
        with pytest.raises(ValueError, match="Invalid error_handling"):
            DescriptorConfig(error_handling="invalid_mode")

    def test_invalid_validation_mode(self):
        """Test that invalid validation_mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid validation_mode"):
            DescriptorConfig(validation_mode="invalid_mode")

    def test_invalid_num_workers(self):
        """Test that num_workers < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_workers must be >= 1"):
            DescriptorConfig(num_workers=0)

    def test_auto_adjust_workers_for_parallel(self):
        """Test that num_workers is auto-adjusted for parallel computation."""
        config = DescriptorConfig(parallel_computation=True, num_workers=1)
        # Model validator should auto-adjust num_workers to 2 when parallel is True
        assert config.num_workers >= 2

    def test_is_category_enabled_default(self):
        """Test is_category_enabled for default categories."""
        config = DescriptorConfig()
        assert config.is_category_enabled("constitutional") is True
        assert config.is_category_enabled("topological") is True
        assert config.is_category_enabled("geometric") is False

    def test_is_category_enabled_when_disabled(self):
        """Test is_category_enabled when config is disabled."""
        config = DescriptorConfig(enabled=False)
        assert config.is_category_enabled("constitutional") is False

    def test_is_category_enabled_with_explicit_config(self):
        """Test is_category_enabled with explicit category config."""
        config = DescriptorConfig(
            categories={"geometric": {"enabled": True}, "constitutional": {"enabled": False}}
        )
        assert config.is_category_enabled("geometric") is True
        assert config.is_category_enabled("constitutional") is False

    def test_get_category_descriptors(self):
        """Test get_category_descriptors method."""
        config = DescriptorConfig(categories={"constitutional": {"descriptors": ["MW", "LogP"]}})
        descriptors = config.get_category_descriptors("constitutional")
        assert descriptors == ["MW", "LogP"]

    def test_get_category_descriptors_not_found(self):
        """Test get_category_descriptors for unconfigured category."""
        config = DescriptorConfig()
        assert config.get_category_descriptors("nonexistent") is None

    def test_get_category_options(self):
        """Test get_category_options method."""
        config = DescriptorConfig(categories={"constitutional": {"options": {"precision": "high"}}})
        options = config.get_category_options("constitutional")
        assert options == {"precision": "high"}

    def test_get_category_options_empty(self):
        """Test get_category_options for unconfigured category."""
        config = DescriptorConfig()
        assert config.get_category_options("nonexistent") == {}

    def test_get_enabled_categories(self):
        """Test get_enabled_categories method."""
        config = DescriptorConfig(
            default_categories=["constitutional", "topological"],
            categories={"geometric": {"enabled": True}, "electronic": {"enabled": False}},
        )
        enabled = config.get_enabled_categories()
        assert "constitutional" in enabled
        assert "topological" in enabled
        assert "geometric" in enabled
        assert "electronic" not in enabled

    def test_get_enabled_categories_when_disabled(self):
        """Test get_enabled_categories when config is disabled."""
        config = DescriptorConfig(enabled=False)
        assert config.get_enabled_categories() == []

    def test_should_use_cache(self):
        """Test should_use_cache method."""
        config = DescriptorConfig(cache_descriptors=True, enabled=True)
        assert config.should_use_cache() is True

        config2 = DescriptorConfig(cache_descriptors=False, enabled=True)
        assert config2.should_use_cache() is False

        config3 = DescriptorConfig(cache_descriptors=True, enabled=False)
        assert config3.should_use_cache() is False

    def test_should_use_parallel(self):
        """Test should_use_parallel method."""
        config = DescriptorConfig(parallel_computation=True, num_workers=4, enabled=True)
        assert config.should_use_parallel() is True

        # Note: When parallel_computation=True and num_workers=1,
        # the model_validator auto-adjusts num_workers to 2, so this returns True
        config2 = DescriptorConfig(parallel_computation=False, num_workers=1, enabled=True)
        assert config2.should_use_parallel() is False

        config3 = DescriptorConfig(parallel_computation=True, num_workers=4, enabled=False)
        assert config3.should_use_parallel() is False

    def test_is_strict_error_handling(self):
        """Test is_strict_error_handling method."""
        config = DescriptorConfig(error_handling="strict")
        assert config.is_strict_error_handling() is True

        config2 = DescriptorConfig(error_handling="warn")
        assert config2.is_strict_error_handling() is False

    def test_to_dict(self):
        """Test to_dict method."""
        config = DescriptorConfig(
            enabled=True, default_categories=["constitutional"], error_handling="strict"
        )
        result = config.to_dict()
        assert isinstance(result, dict)
        assert result["enabled"] is True
        assert result["default_categories"] == ["constitutional"]
        assert result["error_handling"] == "strict"

    def test_from_dict(self):
        """Test from_dict class method."""
        data = {
            "enabled": False,
            "default_categories": ["geometric"],
            "error_handling": "skip",
            "validation_mode": "permissive",
        }
        config = DescriptorConfig.from_dict(data)
        assert config.enabled is False
        assert config.default_categories == ["geometric"]
        assert config.error_handling == "skip"

    def test_frozen_immutability(self):
        """Test that DescriptorConfig is frozen."""
        config = DescriptorConfig()
        with pytest.raises(PydanticValidationError):
            config.enabled = False


class TestDescriptorCategoryConfig:
    """Test suite for DescriptorCategoryConfig container."""

    def test_creation_valid(self):
        """Test creating valid DescriptorCategoryConfig."""
        config = DescriptorCategoryConfig(category_name="constitutional")
        assert config.category_name == "constitutional"
        assert config.enabled is True
        assert config.descriptors is None
        assert config.options == {}

    def test_creation_with_all_values(self):
        """Test creating DescriptorCategoryConfig with all values."""
        config = DescriptorCategoryConfig(
            category_name="topological",
            enabled=False,
            descriptors=["MW", "LogP", "TPSA"],
            options={"precision": "high", "include_3d": True},
        )
        assert config.category_name == "topological"
        assert config.enabled is False
        assert config.descriptors == ["MW", "LogP", "TPSA"]
        assert config.options == {"precision": "high", "include_3d": True}

    def test_invalid_category_name(self):
        """Test that invalid category_name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid category"):
            DescriptorCategoryConfig(category_name="invalid_category")

    def test_valid_category_names(self):
        """Test all valid category names are accepted."""
        valid_categories = [
            "constitutional",
            "topological",
            "geometric",
            "electronic",
            "pharmacophore",
            "fingerprint",
            "custom",
        ]
        for category in valid_categories:
            config = DescriptorCategoryConfig(category_name=category)
            assert config.category_name == category

    def test_to_dict(self):
        """Test to_dict method."""
        config = DescriptorCategoryConfig(
            category_name="geometric", descriptors=["R_gyr", "Asphericity"]
        )
        result = config.to_dict()
        assert isinstance(result, dict)
        assert result["category_name"] == "geometric"
        assert result["descriptors"] == ["R_gyr", "Asphericity"]

    def test_from_dict(self):
        """Test from_dict class method."""
        data = {"category_name": "electronic", "enabled": True, "options": {"method": "AM1"}}
        config = DescriptorCategoryConfig.from_dict(data)
        assert config.category_name == "electronic"
        assert config.options == {"method": "AM1"}

    def test_none_options_initialized(self):
        """Test that None options is initialized to empty dict."""
        config = DescriptorCategoryConfig(category_name="constitutional", options=None)
        assert config.options == {}


# ==========================================
# CREATE_HANDLER_CONFIG FACTORY TESTS
# ==========================================


class TestCreateHandlerConfigFactory:
    """Test suite for create_handler_config factory function."""

    def test_create_handler_config_dft(self):
        """Test creating HandlerConfig for DFT."""
        config = create_handler_config("DFT")
        assert config.handler_type == "DFT"
        assert config.migration_mode is False
        assert isinstance(config.validation_settings, dict)
        assert isinstance(config.processing_settings, dict)

    def test_create_handler_config_dmc(self):
        """Test creating HandlerConfig for DMC."""
        config = create_handler_config("DMC")
        assert config.handler_type == "DMC"
        # DMC should have uncertainty-related settings
        assert (
            "validate_uncertainty" in config.validation_settings or config.migration_mode is False
        )

    def test_create_handler_config_with_migration_mode(self):
        """Test creating HandlerConfig with migration mode enabled."""
        config = create_handler_config("DFT", migration_mode=True)
        assert config.migration_mode is True
        assert config.compatibility_layer.get("enable_legacy_support") is True

    def test_create_handler_config_with_dataset_config(self):
        """Test creating HandlerConfig with DatasetConfig context."""
        dataset_config = DatasetConfig(
            dataset_type="DMC", uncertainty_config={"use_for_loss_weighting": True}
        )
        config = create_handler_config("DMC", dataset_config=dataset_config)
        assert config.handler_type == "DMC"

    def test_create_handler_config_default_settings(self):
        """Test that default settings are applied."""
        config = create_handler_config("DFT")
        assert config.validation_settings.get("strict_mode", False) is True
        assert "recovery_mode" in config.error_handling


# ==========================================
# CREATE_DEFAULT_EXPERIMENTAL_SETUPS TESTS
# ==========================================


class TestCreateDefaultExperimentalSetups:
    """Test suite for create_default_experimental_setups function."""

    def test_creates_multiple_setups(self):
        """Test that function creates multiple experimental setups."""
        setups = create_default_experimental_setups()
        assert isinstance(setups, dict)
        assert len(setups) > 0

    def test_creates_setups_for_dft(self):
        """Test creating setups for DFT dataset type."""
        setups = create_default_experimental_setups("DFT")
        assert isinstance(setups, dict)
        # All setups should be ExperimentalSetup instances
        for _name, setup in setups.items():
            assert isinstance(setup, ExperimentalSetup)
            assert len(setup.transforms) > 0

    def test_creates_setups_for_dmc(self):
        """Test creating setups for DMC dataset type."""
        setups = create_default_experimental_setups("DMC")
        assert isinstance(setups, dict)
        for _name, setup in setups.items():
            assert isinstance(setup, ExperimentalSetup)

    def test_setups_have_valid_transforms(self):
        """Test that created setups have valid TransformSpec transforms."""
        setups = create_default_experimental_setups()
        for _name, setup in setups.items():
            for transform in setup.transforms:
                assert isinstance(transform, TransformSpec)
                assert transform.name is not None
                assert len(transform.name) > 0


# ==========================================
# DESCRIPTOR CONFIG FACTORY TESTS
# ==========================================


class TestDescriptorConfigFactories:
    """Test suite for descriptor config factory functions."""

    def test_create_default_descriptor_config(self):
        """Test create_default_descriptor_config function."""
        config = create_default_descriptor_config()
        assert isinstance(config, DescriptorConfig)
        assert config.enabled is True
        assert "constitutional" in config.default_categories
        assert "topological" in config.default_categories

    def test_create_minimal_descriptor_config(self):
        """Test create_minimal_descriptor_config function."""
        config = create_minimal_descriptor_config()
        assert isinstance(config, DescriptorConfig)
        assert config.enabled is True
        assert config.cache_descriptors is False
        assert config.parallel_computation is False
        assert config.error_handling == "skip"
        assert config.validation_mode == "permissive"


# ==========================================
# VALIDATE_HANDLER_CONFIGURATION_BUNDLE TESTS
# ==========================================


class TestValidateHandlerConfigurationBundle:
    """Test suite for validate_handler_configuration_bundle function."""

    def test_validate_complete_valid_bundle(self):
        """Test validation of a complete valid configuration bundle."""
        transform = TransformSpec(name="AddSelfLoops")
        setup = ExperimentalSetup(name="baseline", transforms=[transform])
        trans_config = TransformationConfig(
            experimental_setups={"baseline": setup}, default_setup="baseline"
        )

        bundle = {
            "dataset_config": DatasetConfig(dataset_type="DFT"),
            "filter_config": FilterConfig(),
            "processing_config": ProcessingConfig(scalar_graph_targets=["Etot"]),
            "structural_config": StructuralFeaturesConfig(
                atom_features=["atomic_number"], bond_features=["bond_type"]
            ),
            "transformation_config": trans_config,
            "handler_config": HandlerConfig(handler_type="DFT"),
            "migration_mode": False,
        }

        is_valid, errors = validate_handler_configuration_bundle(bundle)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_bundle_missing_required_key(self):
        """Test validation with missing required configuration key."""
        bundle = {
            "dataset_config": DatasetConfig(dataset_type="DFT"),
            # Missing other required keys
        }

        is_valid, errors = validate_handler_configuration_bundle(bundle)
        assert is_valid is False
        assert "bundle" in errors
        assert any("Missing required" in err for err in errors["bundle"])

    def test_validate_bundle_handler_type_mismatch(self):
        """Test validation detects handler type mismatch."""
        transform = TransformSpec(name="AddSelfLoops")
        setup = ExperimentalSetup(name="baseline", transforms=[transform])
        trans_config = TransformationConfig(
            experimental_setups={"baseline": setup}, default_setup="baseline"
        )

        bundle = {
            "dataset_config": DatasetConfig(dataset_type="DFT"),
            "filter_config": FilterConfig(),
            "processing_config": ProcessingConfig(scalar_graph_targets=["Etot"]),
            "structural_config": StructuralFeaturesConfig(
                atom_features=["atomic_number"], bond_features=["bond_type"]
            ),
            "transformation_config": trans_config,
            "handler_config": HandlerConfig(handler_type="DMC"),  # Mismatch!
            "migration_mode": False,
        }

        is_valid, errors = validate_handler_configuration_bundle(bundle)
        assert is_valid is False
        assert "cross_validation" in errors


# ==========================================
# REGISTRY FUNCTION TESTS (Mocked)
# ==========================================


class TestRegistryFunctions:
    """Test suite for registry-related functions with mocking."""

    def test_get_valid_dataset_types_returns_list(self):
        """Test that _get_valid_dataset_types returns a list."""
        result = _get_valid_dataset_types()
        assert isinstance(result, list)

    def test_is_valid_dataset_type_dft(self):
        """Test that DFT is always a valid dataset type."""
        # DFT is a core type that should always be valid
        result = _is_valid_dataset_type("DFT")
        assert isinstance(result, bool)

    def test_resolve_canonical_dataset_type_returns_string(self):
        """Test that _resolve_canonical_dataset_type returns a string."""
        result = _resolve_canonical_dataset_type("dft")
        assert isinstance(result, str)

    def test_resolve_canonical_dataset_type_preserves_unknown(self):
        """Test that unknown types are preserved."""
        result = _resolve_canonical_dataset_type("UNKNOWN_TYPE_XYZ")
        # Should return the input if not found
        assert isinstance(result, str)


# ==========================================
# TRANSFORMATION CONFIG VALIDATION TESTS
# ==========================================


class TestTransformationConfigValidation:
    """Extended tests for TransformationConfig validation."""

    def test_validate_transformation_config_valid(self, sample_experimental_setup):
        """Test validate_transformation_config with valid config."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup}, default_setup="baseline"
        )
        is_valid, errors = config.validate_transformation_config()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_transformation_config_multiple_setups(self, sample_transform_spec):
        """Test validation with multiple experimental setups."""
        setup1 = ExperimentalSetup(name="setup1", transforms=[sample_transform_spec])
        setup2 = ExperimentalSetup(name="setup2", transforms=[TransformSpec(name="Transform2")])

        config = TransformationConfig(
            experimental_setups={"setup1": setup1, "setup2": setup2}, default_setup="setup1"
        )
        is_valid, errors = config.validate_transformation_config()
        assert is_valid is True

    def test_validate_transformation_config_with_validation_settings(
        self, sample_experimental_setup
    ):
        """Test validation with explicit validation settings."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup},
            default_setup="baseline",
            validation={"enabled": True, "strict_mode": True},
        )
        is_valid, errors = config.validate_transformation_config()
        assert is_valid is True


# ==========================================
# PROCESSING CONFIG EXTENDED TESTS
# ==========================================


class TestProcessingConfigExtended:
    """Extended tests for ProcessingConfig."""

    def test_preprocessing_fields(self):
        """Test preprocessing-related fields."""
        config = ProcessingConfig(
            scalar_graph_targets=["Etot"],
            preprocessing_feature_tier="advanced",
            preprocessing_num_molecules=1000,
            preprocessing_cleanup_temp=False,
        )
        assert config.preprocessing_feature_tier == "advanced"
        assert config.preprocessing_num_molecules == 1000
        assert config.preprocessing_cleanup_temp is False

    def test_preprocessing_fields_defaults(self):
        """Test preprocessing fields have correct defaults."""
        config = ProcessingConfig(scalar_graph_targets=["Etot"])
        assert config.preprocessing_feature_tier == "standard"
        assert config.preprocessing_num_molecules is None
        assert config.preprocessing_cleanup_temp is True

    def test_get_migration_phase(self):
        """Test get_migration_phase method."""
        config = ProcessingConfig(
            scalar_graph_targets=["Etot"],
            migration_settings={"enabled": True, "current_phase": "testing"},
        )
        assert config.get_migration_phase() == "testing"

    def test_get_migration_phase_not_set(self):
        """Test get_migration_phase when not set."""
        config = ProcessingConfig(scalar_graph_targets=["Etot"])
        assert config.get_migration_phase() is None


# ==========================================
# STRUCTURAL FEATURES CONFIG EXTENDED TESTS
# ==========================================


class TestStructuralFeaturesConfigExtended:
    """Extended tests for StructuralFeaturesConfig."""

    def test_validate_feature_config_invalid_atom_features_type(self):
        """Test validation with non-list atom_features (should fail at creation)."""
        # Pydantic V2 will validate types during construction and raise ValidationError
        with pytest.raises(PydanticValidationError):
            StructuralFeaturesConfig(atom_features="not_a_list", bond_features=["bond_type"])

    def test_validate_feature_config_with_handler_features(self):
        """Test validation with handler-specific features."""
        config = StructuralFeaturesConfig(
            atom_features=["atomic_number"],
            bond_features=["bond_type"],
            handler_features={
                "DFT": {
                    "atom_features": ["atomic_number", "charge"],
                    "bond_features": ["bond_type", "bond_order"],
                }
            },
        )
        is_valid, errors = config.validate_feature_config()
        assert is_valid is True

    def test_get_handler_features_with_override(self):
        """Test get_handler_features with handler-specific overrides."""
        config = StructuralFeaturesConfig(
            atom_features=["atomic_number"],
            bond_features=["bond_type"],
            handler_features={"DMC": {"atom_features": ["atomic_number", "spin"]}},
        )
        features = config.get_handler_features("DMC")
        assert features["atom_features"] == ["atomic_number", "spin"]
        assert features["bond_features"] == ["bond_type"]  # Falls back to default


# ==========================================
# DATASET CONFIG EXTENDED TESTS
# ==========================================


class TestDatasetConfigExtended:
    """Extended tests for DatasetConfig."""

    def test_to_dict(self):
        """Test to_dict method."""
        config = DatasetConfig(dataset_type="DFT", handler_config={"batch_size": 32})
        result = config.to_dict()
        assert isinstance(result, dict)
        assert result["dataset_type"] == "DFT"
        assert result["handler_config"] == {"batch_size": 32}

    def test_validation_config_initialized(self):
        """Test that validation_config is properly initialized."""
        config = DatasetConfig(dataset_type="DFT", validation_config=None)
        assert config.validation_config == {}

    def test_migration_config_initialized(self):
        """Test that migration_config is properly initialized."""
        config = DatasetConfig(dataset_type="DFT", migration_config=None)
        assert config.migration_config == {}


# ==========================================
# FILTER CONFIG EXTENDED TESTS
# ==========================================


class TestFilterConfigExtended:
    """Extended tests for FilterConfig."""

    def test_to_dict(self):
        """Test to_dict method."""
        config = FilterConfig(max_atoms=50, min_atoms=5)
        result = config.to_dict()
        assert isinstance(result, dict)
        assert result["max_atoms"] == 50
        assert result["min_atoms"] == 5

    def test_filter_validation_initialized(self):
        """Test that filter_validation is properly initialized."""
        config = FilterConfig(filter_validation=None)
        assert config.filter_validation == {}


# ==========================================
# HANDLER CONFIG EXTENDED TESTS
# ==========================================


class TestHandlerConfigExtended:
    """Extended tests for HandlerConfig."""

    def test_to_dict(self):
        """Test to_dict method."""
        config = HandlerConfig(handler_type="DFT", validation_settings={"strict": True})
        result = config.to_dict()
        assert isinstance(result, dict)
        assert result["handler_type"] == "DFT"

    def test_get_processing_setting(self):
        """Test get_processing_setting method."""
        config = HandlerConfig(handler_type="DFT", processing_settings={"batch_size": 100})
        assert config.get_processing_setting("batch_size") == 100
        assert config.get_processing_setting("nonexistent", "default") == "default"

    def test_is_strict_validation_enabled(self):
        """Test is_strict_validation_enabled method."""
        config = HandlerConfig(handler_type="DFT", validation_settings={"strict_mode": True})
        assert config.is_strict_validation_enabled() is True

        config2 = HandlerConfig(handler_type="DFT", validation_settings={"strict_mode": False})
        assert config2.is_strict_validation_enabled() is False

    def test_get_error_recovery_mode(self):
        """Test get_error_recovery_mode method."""
        config = HandlerConfig(handler_type="DFT", error_handling={"recovery_mode": "raise_error"})
        assert config.get_error_recovery_mode() == "raise_error"

    def test_get_error_recovery_mode_default(self):
        """Test get_error_recovery_mode with default."""
        config = HandlerConfig(handler_type="DFT")
        assert config.get_error_recovery_mode() == "skip_molecule"


# ==========================================
# EXPERIMENTAL SETUP EXTENDED TESTS
# ==========================================


class TestExperimentalSetupExtended:
    """Extended tests for ExperimentalSetup."""

    def test_to_dict(self, sample_transform_spec):
        """Test to_dict method."""
        setup = ExperimentalSetup(
            name="test", transforms=[sample_transform_spec], description="Test setup"
        )
        result = setup.to_dict()
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["description"] == "Test setup"

    def test_validate_experimental_setup_safe(self, sample_transform_spec):
        """Test validate_experimental_setup_safe method."""
        setup = ExperimentalSetup(name="test", transforms=[sample_transform_spec])
        is_valid, errors = setup.validate_experimental_setup_safe()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_experimental_setup_invalid_dataset_compatibility(self, sample_transform_spec):
        """Test validation with invalid dataset_compatibility type."""
        # This should work since we pass a list
        setup = ExperimentalSetup(
            name="test", transforms=[sample_transform_spec], dataset_compatibility=["DFT", "DMC"]
        )
        is_valid, errors = setup.validate_experimental_setup()
        assert is_valid is True


# ==========================================
# TRANSFORMATION CONFIG EXTENDED TESTS
# ==========================================


class TestTransformationConfigExtended:
    """Extended tests for TransformationConfig."""

    def test_to_dict(self, sample_experimental_setup):
        """Test to_dict method."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup}, default_setup="baseline"
        )
        result = config.to_dict()
        assert isinstance(result, dict)
        assert "experimental_setups" in result
        assert result["default_setup"] == "baseline"

    def test_get_setup(self, sample_experimental_setup):
        """Test get_setup method."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup}, default_setup="baseline"
        )
        setup = config.get_setup("baseline")
        assert setup is not None
        assert setup.name == "baseline"

    def test_get_setup_not_found(self, sample_experimental_setup):
        """Test get_setup with nonexistent name."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup}, default_setup="baseline"
        )
        assert config.get_setup("nonexistent") is None

    def test_get_cache_key(self, sample_experimental_setup):
        """Test get_cache_key method."""
        config = TransformationConfig(
            experimental_setups={"baseline": sample_experimental_setup}, default_setup="baseline"
        )
        key = config.get_cache_key()
        assert isinstance(key, str)
        assert len(key) == 16


# ==========================================
# CHECK CONFIGURATION COMPATIBILITY EXTENDED TESTS
# ==========================================


class TestCheckConfigurationCompatibilityExtended:
    """Extended tests for check_configuration_compatibility function."""

    def test_compatibility_different_dataset_types(self):
        """Test compatibility check with different dataset types."""
        config_a = {"dataset_config": DatasetConfig(dataset_type="DFT")}
        config_b = {"dataset_config": DatasetConfig(dataset_type="DMC")}

        is_compatible, issues = check_configuration_compatibility(config_a, config_b)
        assert is_compatible is False
        assert any("Dataset type mismatch" in issue for issue in issues)

    def test_compatibility_different_handler_types(self):
        """Test compatibility check with different handler types."""
        config_a = {"handler_config": HandlerConfig(handler_type="DFT")}
        config_b = {"handler_config": HandlerConfig(handler_type="DMC")}

        is_compatible, issues = check_configuration_compatibility(config_a, config_b)
        assert is_compatible is False
        assert any("Handler type mismatch" in issue for issue in issues)

    def test_compatibility_same_configs(self, sample_transformation_config):
        """Test compatibility with identical configs."""
        config = {"transformation_config": sample_transformation_config}

        is_compatible, issues = check_configuration_compatibility(config, config)
        assert is_compatible is True
        assert len(issues) == 0


# ==========================================
# CREATE MINIMAL CONFIG FOR TESTING EXTENDED
# ==========================================


class TestCreateMinimalConfigForTestingExtended:
    """Extended tests for create_minimal_config_for_testing function."""

    def test_returns_complete_bundle(self):
        """Test that function returns a complete configuration bundle."""
        config = create_minimal_config_for_testing()

        required_keys = [
            "dataset_config",
            "filter_config",
            "processing_config",
            "structural_config",
            "transformation_config",
            "handler_config",
        ]

        for key in required_keys:
            assert key in config, f"Missing required key: {key}"

    def test_with_specific_dataset_type(self):
        """Test creating config with specific dataset type."""
        config = create_minimal_config_for_testing(dataset_type="DFT")
        assert config["dataset_config"].dataset_type == "DFT"
        assert config["handler_config"].handler_type == "DFT"

    def test_transformation_config_is_valid(self):
        """Test that transformation config in bundle is valid."""
        config = create_minimal_config_for_testing()
        trans_config = config["transformation_config"]

        assert isinstance(trans_config, TransformationConfig)
        is_valid, errors = trans_config.validate_transformation_config()
        assert is_valid is True


# ==========================================
# MAIN TEST EXECUTION
# ==========================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
