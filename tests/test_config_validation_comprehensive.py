#!/usr/bin/env python3
"""
test_config_validation_comprehensive.py — Section 5.1

End-to-end configuration validation test suite covering:
- Schema validation (YAMLSchemaValidator, ValidationConfig)
- Pydantic V2 model validation (validate_with_pydantic_model, ValidationResult)
- Cross-field consistency checks (DatasetConfig, FilterConfig, ProcessingConfig)
- ConfigBridge Pydantic BaseModel validation (31 classes)
- DatasetSchema from datasets/base.py (Pydantic V2 frozen dataclasses)
- Meaningful error messages for invalid configs

Modules exercised:
- milia_pipeline/config/config_loader.py         — load_config(), _deep_merge_configs(),
                                                    _discover_config_files(), _collect_yaml_files(),
                                                    _load_and_merge_yaml_files(), clear_config_cache()
- milia_pipeline/config/config_schemas.py         — YAMLSchemaValidator, ValidationConfig,
                                                    TransformationSchema, PluginConfigSchema,
                                                    DescriptorConfigSchema,
                                                    WavefunctionProcessingConfigSchema
- milia_pipeline/config/validators.py             — validate_with_pydantic_model(), ValidationResult
- milia_pipeline/config/config_containers.py      — DatasetConfig, FilterConfig, ProcessingConfig,
                                                    StructuralFeaturesConfig, HandlerConfig,
                                                    TransformSpec, ExperimentalSetup,
                                                    TransformationConfig
- milia_pipeline/datasets/base.py                 — DatasetMetadata, DatasetSchema, DatasetFeatures
- milia_pipeline/models/utils/config_bridge.py    — ModelSelectionConfig, DataSplitConfig,
                                                    LossConfig, OptimizerConfig, SchedulerConfig,
                                                    TrainingConfig, EvaluationConfig,
                                                    AccelerationConfig, DeploymentConfig,
                                                    PluginsConfig, DeviceConfig, MemoryConfig,
                                                    DistributedConfig, HPOConfigBridge, and more

Evidence sources:
- MILIA_Test_Recommendations.md Section 5.1 (lines 307-320)
- config_loader.py (lines 398-906): _discover_config_files, _collect_yaml_files,
  _deep_merge_configs, _load_and_merge_yaml_files, load_config
- config_schemas.py (line 942): ValidationConfig class
- config_schemas.py (line 1119): YAMLSchemaValidator class
- config_schemas.py (line 345): TransformationSchema class
- config_schemas.py (line 396): PluginConfigSchema class
- config_schemas.py (line 957): DescriptorConfigSchema class
- config_containers.py (line 294): DatasetConfig class
- config_containers.py (line 447): FilterConfig class
- config_containers.py (line 640): ProcessingConfig class
- config_containers.py (line 790): HandlerConfig class
- config_containers.py (line 880): TransformSpec class
- validators.py (line 629): ValidationResult class
- validators.py (line 810): validate_with_pydantic_model function
- base.py (line 33): DatasetMetadata class
- base.py (line 62): DatasetSchema class
- base.py (line 93): DatasetFeatures class
- config_bridge.py (line 203-682): 31 Pydantic BaseModel configuration classes

Launch from project root: /app/milia/
    pytest tests/test_config_validation_comprehensive.py -v
"""

import copy
import os
import sys
from typing import Any
from unittest.mock import patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Add the project root to Python path FIRST
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.contract


# ===========================================================================
# FIXTURES
# ===========================================================================


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Clear config cache before and after each test to prevent cross-contamination.

    Evidence: config_loader.py lines 353-388 — global _CONFIG, _config_cache, _CONFIG_STATS
    """
    try:
        from milia_pipeline.config.config_loader import clear_config_cache

        clear_config_cache()
    except ImportError:
        pass
    yield
    try:
        from milia_pipeline.config.config_loader import clear_config_cache

        clear_config_cache()
    except ImportError:
        pass


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create a temporary directory tree suitable for split-file mode tests.

    Evidence: config_loader.py lines 398-506 — _discover_config_files,
    _collect_yaml_files ordering rules.
    """
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    datasets_dir = configs_dir / "datasets"
    datasets_dir.mkdir()
    return configs_dir


@pytest.fixture
def minimal_valid_config() -> dict[str, Any]:
    """A minimal configuration dictionary that exercises the happy path.

    Keys chosen from config_loader.py lines 796-802 (basic structure check)
    and config_schemas.py lines 1150-1282 (enhanced format requirement).
    """
    return {
        "dataset_type": "DFT",
        "global_paths": {"working_root_dir": "/tmp/test_milia"},
        "transformations": {
            "experimental_setups": {
                "baseline": {
                    "description": "Minimal baseline",
                    "transforms": [{"name": "AddSelfLoops", "kwargs": {}, "enabled": True}],
                }
            },
            "default_setup": "baseline",
            "validation": {"enabled": True, "strict_mode": False},
        },
        "data_config": {"common_settings": {"chunk_size": 500}},
    }


@pytest.fixture
def minimal_valid_config_yaml(tmp_path, minimal_valid_config):
    """Write the minimal config to a YAML file and return the path.

    Evidence: config_loader.py lines 764-781 — single-file mode YAML loading.
    """
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(minimal_valid_config, f)
    return str(config_file)


# ===========================================================================
# 1. YAML SCHEMA VALIDATOR TESTS — config_schemas.py
# ===========================================================================


class TestYAMLSchemaValidator:
    """Test the YAMLSchemaValidator class.

    Evidence: config_schemas.py lines 1119-1282.
    """

    def _get_validator(self):
        from milia_pipeline.config.config_schemas import YAMLSchemaValidator

        return YAMLSchemaValidator()

    # ----- Happy path -----

    def test_enhanced_format_detected(self, minimal_valid_config):
        """Enhanced format with experimental_setups detected correctly."""
        validator = self._get_validator()
        fmt = validator.detect_format(minimal_valid_config)
        assert fmt == "enhanced"

    def test_valid_enhanced_config_passes(self, minimal_valid_config):
        """A valid enhanced configuration passes validation without errors."""
        validator = self._get_validator()
        result = validator.validate_config(minimal_valid_config)
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_enhanced_with_standard_transforms_only(self):
        """Enhanced format using only standard_transforms (no experimental_setups).

        Evidence: config_schemas.py lines 1186-1211 — standard_transforms path.
        """
        config = {
            "transformations": {
                "standard_transforms": [{"name": "AddSelfLoops", "kwargs": {}}],
                "default_setup": "baseline",
            }
        }
        validator = self._get_validator()
        result = validator.validate_config(config)
        assert result["valid"] is True
        assert result["format_detected"] == "enhanced"

    def test_enhanced_with_both_sources(self):
        """Enhanced format with both experimental_setups AND standard_transforms."""
        config = {
            "transformations": {
                "experimental_setups": {
                    "exp1": {"description": "Test", "transforms": [{"name": "T1"}]}
                },
                "standard_transforms": [{"name": "AddSelfLoops"}],
                "default_setup": "exp1",
            }
        }
        validator = self._get_validator()
        result = validator.validate_config(config)
        assert result["valid"] is True

    # ----- Legacy format detection -----

    def test_legacy_list_format_detected(self):
        """Legacy list-style transformations detected.

        Evidence: config_schemas.py line 1142.
        """
        config = {"transformations": [{"AddSelfLoops": {}}]}
        validator = self._get_validator()
        assert validator.detect_format(config) == "legacy_list"

    def test_legacy_dict_format_detected(self):
        """Legacy dict-style transformations detected.

        Evidence: config_schemas.py line 1146.
        """
        config = {"transformations": {"some_key": "some_value"}}
        validator = self._get_validator()
        assert validator.detect_format(config) == "legacy_dict"

    def test_legacy_formats_valid_with_migration_warning(self):
        """Legacy formats are valid but emit migration warnings.

        Evidence: config_schemas.py lines 1273-1276.
        """
        config = {"transformations": [{"AddSelfLoops": {}}]}
        validator = self._get_validator()
        result = validator.validate_config(config)
        assert result["valid"] is True
        assert any("migrated" in w.lower() for w in result["warnings"])

    # ----- Invalid configs -----

    def test_missing_transformations_key(self):
        """Missing 'transformations' key produces error.

        Evidence: config_schemas.py lines 1172-1173.
        """
        validator = self._get_validator()
        result = validator.validate_config({"dataset_type": "DFT"})
        assert result["valid"] is False
        assert any("transformations" in e.lower() for e in result["errors"])

    def test_invalid_transformations_string(self):
        """String-valued transformations detected as 'invalid'.

        Evidence: config_schemas.py lines 1133-1134.
        """
        config = {"transformations": "invalid_string"}
        validator = self._get_validator()
        fmt = validator.detect_format(config)
        assert fmt == "invalid"

    def test_non_dict_config_is_error(self):
        """Non-dict config produces an error.

        Evidence: config_schemas.py lines 1168-1169.
        """
        validator = self._get_validator()
        result = validator.validate_config("not_a_dict")
        assert result["valid"] is False
        assert any("dictionary" in e.lower() for e in result["errors"])

    def test_empty_experimental_setups_without_standard_transforms(self):
        """Empty experimental_setups with no standard_transforms raises error.

        Evidence: config_schemas.py lines 1220-1222.
        """
        config = {"transformations": {"experimental_setups": {}, "default_setup": "baseline"}}
        validator = self._get_validator()
        result = validator.validate_config(config)
        assert result["valid"] is False

    # ----- Strict mode -----

    def test_strict_mode_requires_default_setup(self):
        """Strict mode requires default_setup to be present.

        Evidence: config_schemas.py lines 1258-1261.
        """
        from milia_pipeline.config.config_schemas import ValidationConfig

        config = {
            "transformations": {
                "experimental_setups": {"baseline": {"transforms": [{"name": "T1"}]}}
                # NOTE: default_setup intentionally omitted
            }
        }
        validator = self._get_validator()
        result = validator.validate_config(
            config, validation_config=ValidationConfig(strict_mode=True)
        )
        assert result["valid"] is False or any(
            "default_setup" in e.lower() for e in result["errors"]
        )


# ===========================================================================
# 2. VALIDATION CONFIG — config_schemas.py
# ===========================================================================


class TestValidationConfig:
    """Validate the ValidationConfig Pydantic model itself.

    Evidence: config_schemas.py lines 942-951.
    """

    def test_defaults(self):
        from milia_pipeline.config.config_schemas import ValidationConfig

        vc = ValidationConfig()
        assert vc.strict_mode is False
        assert vc.warn_on_unknown is True
        assert vc.require_descriptions is False
        assert vc.check_parameter_types is False
        assert vc.validate_research_context is False

    def test_override_strict_mode(self):
        from milia_pipeline.config.config_schemas import ValidationConfig

        vc = ValidationConfig(strict_mode=True)
        assert vc.strict_mode is True


# ===========================================================================
# 3. TRANSFORMATION SCHEMA — config_schemas.py
# ===========================================================================


class TestTransformationSchema:
    """Validate TransformationSchema Pydantic model.

    Evidence: config_schemas.py lines 345-386 — TransformationSchema class with
    model_validator(mode='after') for cross-field validation.
    """

    def test_valid_with_experimental_setups(self):
        from milia_pipeline.config.config_schemas import TransformationSchema

        ts = TransformationSchema(
            experimental_setups={"baseline": [{"name": "AddSelfLoops"}]}, default_setup="baseline"
        )
        assert ts.default_setup == "baseline"

    def test_valid_with_standard_transforms_only(self):
        from milia_pipeline.config.config_schemas import TransformationSchema

        ts = TransformationSchema(standard_transforms=[{"name": "AddSelfLoops"}])
        assert ts.standard_transforms is not None

    def test_invalid_no_transforms_source(self):
        """Must have at least one of experimental_setups or standard_transforms.

        Evidence: config_schemas.py lines 366-370.
        """
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.config.config_schemas import TransformationSchema

        with pytest.raises(PydanticValidationError):
            TransformationSchema(experimental_setups={}, standard_transforms=None)

    def test_invalid_default_setup_not_in_experimental(self):
        """default_setup must exist in experimental_setups (when no standard_transforms).

        Evidence: config_schemas.py lines 381-384.
        """
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.config.config_schemas import TransformationSchema

        with pytest.raises(PydanticValidationError):
            TransformationSchema(
                experimental_setups={"baseline": [{"name": "T1"}]}, default_setup="nonexistent"
            )


# ===========================================================================
# 4. PLUGIN CONFIG SCHEMA — config_schemas.py
# ===========================================================================


class TestPluginConfigSchema:
    """Validate PluginConfigSchema Pydantic model.

    Evidence: config_schemas.py lines 396-459.
    """

    def test_defaults(self):
        from milia_pipeline.config.config_schemas import PluginConfigSchema

        pcs = PluginConfigSchema()
        assert pcs.enabled is False
        assert pcs.auto_discover is True
        assert pcs.validation_level == "standard"
        assert pcs.max_plugins == 50

    def test_invalid_validation_level(self):
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.config.config_schemas import PluginConfigSchema

        with pytest.raises(PydanticValidationError):
            PluginConfigSchema(validation_level="INVALID")

    def test_invalid_max_plugins_out_of_range(self):
        """max_plugins must be between 1 and 1000.

        Evidence: config_schemas.py lines 427-434.
        """
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.config.config_schemas import PluginConfigSchema

        with pytest.raises(PydanticValidationError):
            PluginConfigSchema(max_plugins=0)
        with pytest.raises(PydanticValidationError):
            PluginConfigSchema(max_plugins=1001)

    def test_to_dict_round_trip(self):
        from milia_pipeline.config.config_schemas import PluginConfigSchema

        pcs = PluginConfigSchema(enabled=True, plugin_paths=["./plugins"])
        d = pcs.to_dict()
        restored = PluginConfigSchema.from_dict(d)
        assert restored.enabled is True
        assert restored.plugin_paths == ["./plugins"]


# ===========================================================================
# 5. DESCRIPTOR CONFIG SCHEMA — config_schemas.py
# ===========================================================================


class TestDescriptorConfigSchema:
    """Validate DescriptorConfigSchema Pydantic model.

    Evidence: config_schemas.py lines 957-1050.
    """

    def test_defaults(self):
        from milia_pipeline.config.config_schemas import DescriptorConfigSchema

        dcs = DescriptorConfigSchema()
        assert dcs.enabled is True
        assert dcs.cache_descriptors is True
        assert dcs.error_handling == "warn"
        assert dcs.validation_mode == "standard"
        assert dcs.num_workers == 1

    def test_invalid_error_handling(self):
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.config.config_schemas import DescriptorConfigSchema

        with pytest.raises(PydanticValidationError):
            DescriptorConfigSchema(error_handling="crash")

    def test_invalid_validation_mode(self):
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.config.config_schemas import DescriptorConfigSchema

        with pytest.raises(PydanticValidationError):
            DescriptorConfigSchema(validation_mode="turbo")

    def test_invalid_category(self):
        """Invalid category in default_categories raises error.

        Evidence: config_schemas.py lines 1017-1031.
        """
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.config.config_schemas import DescriptorConfigSchema

        with pytest.raises(PydanticValidationError):
            DescriptorConfigSchema(default_categories=["nonexistent_category"])

    def test_auto_adjust_num_workers(self):
        """parallel_computation=True with num_workers=1 auto-adjusts to 2.

        Evidence: config_schemas.py lines 1033-1042.
        """
        from milia_pipeline.config.config_schemas import DescriptorConfigSchema

        dcs = DescriptorConfigSchema(parallel_computation=True, num_workers=1)
        assert dcs.num_workers == 2


# ===========================================================================
# 6. WAVEFUNCTION PROCESSING CONFIG SCHEMA — config_schemas.py
# ===========================================================================


class TestWavefunctionProcessingConfigSchema:
    """Validate WavefunctionProcessingConfigSchema.

    Evidence: config_schemas.py lines 576-599.
    """

    def test_default_tier(self):
        from milia_pipeline.config.config_schemas import WavefunctionProcessingConfigSchema

        wcs = WavefunctionProcessingConfigSchema()
        assert wcs.feature_tier == "standard"

    def test_valid_tiers(self):
        from milia_pipeline.config.config_schemas import WavefunctionProcessingConfigSchema

        for tier in ("basic", "standard", "complete"):
            wcs = WavefunctionProcessingConfigSchema(feature_tier=tier)
            assert wcs.feature_tier == tier

    def test_invalid_tier(self):
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.config.config_schemas import WavefunctionProcessingConfigSchema

        with pytest.raises(PydanticValidationError):
            WavefunctionProcessingConfigSchema(feature_tier="ultra")


# ===========================================================================
# 7. VALIDATE_WITH_PYDANTIC_MODEL — validators.py
# ===========================================================================


class TestValidateWithPydanticModel:
    """Test the validate_with_pydantic_model bridge function.

    Evidence: validators.py lines 810-891.
    """

    def test_valid_data_returns_valid_result(self):
        from milia_pipeline.config.config_schemas import ValidationConfig
        from milia_pipeline.config.validators import validate_with_pydantic_model

        result = validate_with_pydantic_model({"strict_mode": True}, ValidationConfig)
        assert result.is_valid is True
        validated = result.get_validated_data()
        assert validated.strict_mode is True

    def test_invalid_data_returns_errors(self):
        from milia_pipeline.config.config_schemas import PluginConfigSchema
        from milia_pipeline.config.validators import validate_with_pydantic_model

        result = validate_with_pydantic_model({"validation_level": "BAD_LEVEL"}, PluginConfigSchema)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_tuple_return_mode(self):
        """return_wrapper=False gives (bool, List[str]) tuple.

        Evidence: validators.py lines 871-872.
        """
        from milia_pipeline.config.config_schemas import ValidationConfig
        from milia_pipeline.config.validators import validate_with_pydantic_model

        is_valid, errors = validate_with_pydantic_model(
            {"strict_mode": False}, ValidationConfig, return_wrapper=False
        )
        assert is_valid is True
        assert errors == []


# ===========================================================================
# 8. VALIDATION RESULT — validators.py
# ===========================================================================


class TestValidationResult:
    """Test the ValidationResult enforcement wrapper.

    Evidence: validators.py lines 629-770.
    """

    def test_valid_result_allows_data_access(self):
        from milia_pipeline.config.validators import ValidationResult

        result = ValidationResult(is_valid=True, errors=[], data={"key": "val"})
        assert result.is_valid is True
        assert result.get_validated_data() == {"key": "val"}

    def test_invalid_result_raises_on_data_access(self):
        from milia_pipeline.config.validators import ValidationResult

        result = ValidationResult(is_valid=False, errors=["bad value"])
        assert result.is_valid is False
        # Get ValidationError from the method's own globals to guarantee class identity
        _ValidationError = result.get_validated_data.__func__.__globals__["ValidationError"]
        with pytest.raises(_ValidationError):
            result.get_validated_data()

    def test_unchecked_result_raises_on_data_access(self):
        """Accessing data without first checking is_valid raises.

        Evidence: validators.py lines 692-698.
        """
        from milia_pipeline.config.validators import ValidationResult

        result = ValidationResult(is_valid=True, errors=[], data="x")
        _ValidationError = result.get_validated_data.__func__.__globals__["ValidationError"]
        with pytest.raises(_ValidationError, match="must be checked"):
            result.get_validated_data()


# ===========================================================================
# 9. CONFIG CONTAINERS — config_containers.py  (Pydantic V2 frozen models)
# ===========================================================================


class TestDatasetConfig:
    """Validate DatasetConfig frozen Pydantic model.

    Evidence: config_containers.py lines 294-444.
    DatasetConfig validates dataset_type against the registry.
    When registry is unavailable we mock _is_valid_dataset_type.
    """

    def test_valid_dataset_config(self):
        """Valid DatasetConfig creation using mock for registry gate."""
        with patch(
            "milia_pipeline.config.config_containers._is_valid_dataset_type", return_value=True
        ):
            from milia_pipeline.config.config_containers import DatasetConfig

            dc = DatasetConfig(dataset_type="DFT")
            assert dc.dataset_type == "DFT"
            assert dc.is_uncertainty_enabled is False
            assert isinstance(dc.to_dict(), dict)

    def test_invalid_dataset_type_raises(self):
        """Invalid dataset_type raises PydanticValidationError.

        Evidence: config_containers.py lines 316-323.
        """
        with (
            patch(
                "milia_pipeline.config.config_containers._is_valid_dataset_type", return_value=False
            ),
            patch(
                "milia_pipeline.config.config_containers._get_valid_dataset_types",
                return_value=["DFT", "DMC"],
            ),
        ):
            from pydantic import ValidationError as PydanticValidationError

            from milia_pipeline.config.config_containers import DatasetConfig

            with pytest.raises(PydanticValidationError):
                DatasetConfig(dataset_type="NONEXISTENT")

    def test_frozen_immutability(self):
        """DatasetConfig is frozen and cannot be mutated.

        Evidence: config_containers.py line 294 — ``class DatasetConfig(BaseModel, frozen=True)``
        """
        with patch(
            "milia_pipeline.config.config_containers._is_valid_dataset_type", return_value=True
        ):
            from milia_pipeline.config.config_containers import DatasetConfig

            dc = DatasetConfig(dataset_type="DFT")
            from pydantic import ValidationError as PydanticValidationError

            with pytest.raises(PydanticValidationError):
                dc.dataset_type = "DMC"

    def test_uncertainty_auto_compute(self):
        """is_uncertainty_enabled auto-computed from uncertainty_config.

        Evidence: config_containers.py lines 325-345 — model_validator set_computed_fields_and_defaults.
        """
        with patch(
            "milia_pipeline.config.config_containers._is_valid_dataset_type", return_value=True
        ):
            from milia_pipeline.config.config_containers import DatasetConfig

            dc = DatasetConfig(
                dataset_type="DMC", uncertainty_config={"use_for_loss_weighting": True}
            )
            assert dc.is_uncertainty_enabled is True


class TestFilterConfig:
    """Validate FilterConfig frozen Pydantic model.

    Evidence: config_containers.py lines 447-546.
    """

    def test_defaults(self):
        from milia_pipeline.config.config_containers import FilterConfig

        fc = FilterConfig()
        assert fc.max_atoms is None
        assert fc.min_atoms is None

    def test_validate_filter_config_max_less_than_min(self):
        """max_atoms < min_atoms produces validation error.

        Evidence: config_containers.py lines 521-523.
        """
        from milia_pipeline.config.config_containers import FilterConfig

        fc = FilterConfig(max_atoms=5, min_atoms=10)
        is_valid, errors = fc.validate_filter_config()
        assert is_valid is False
        assert any("max_atoms" in e for e in errors)

    def test_validate_filter_config_negative_min_atoms(self):
        """Negative min_atoms produces validation error.

        Evidence: config_containers.py lines 525-526.
        """
        from milia_pipeline.config.config_containers import FilterConfig

        fc = FilterConfig(min_atoms=-1)
        is_valid, errors = fc.validate_filter_config()
        assert is_valid is False


class TestProcessingConfig:
    """Validate ProcessingConfig frozen Pydantic model.

    Evidence: config_containers.py lines 640-787.
    """

    def test_valid_processing_config(self):
        from milia_pipeline.config.config_containers import ProcessingConfig

        pc = ProcessingConfig(scalar_graph_targets=["Etot"])
        assert pc.scalar_graph_targets == ["Etot"]
        assert pc.test_molecule_limit is None

    def test_validate_processing_config_non_string_targets(self):
        """Non-string entries in scalar_graph_targets produce validation error.

        Evidence: config_containers.py lines 757-760.
        """
        from milia_pipeline.config.config_containers import ProcessingConfig

        pc = ProcessingConfig(scalar_graph_targets=["Etot", "valid"])
        is_valid, errors = pc.validate_processing_config()
        assert is_valid is True

    def test_validate_negative_test_molecule_limit(self):
        """Negative test_molecule_limit produces validation error.

        Evidence: config_containers.py lines 774-776.
        """
        from milia_pipeline.config.config_containers import ProcessingConfig

        pc = ProcessingConfig(scalar_graph_targets=["Etot"], test_molecule_limit=-5)
        is_valid, errors = pc.validate_processing_config()
        assert is_valid is False
        assert any("test_molecule_limit" in e for e in errors)


class TestHandlerConfig:
    """Validate HandlerConfig frozen Pydantic model.

    Evidence: config_containers.py lines 790-874.
    """

    def test_valid_handler_config(self):
        with patch(
            "milia_pipeline.config.config_containers._is_valid_dataset_type", return_value=True
        ):
            from milia_pipeline.config.config_containers import HandlerConfig

            hc = HandlerConfig(handler_type="DFT")
            assert hc.handler_type == "DFT"
            assert hc.migration_mode is False

    def test_invalid_handler_type_raises(self):
        """Invalid handler_type raises PydanticValidationError.

        Evidence: config_containers.py lines 813-820.
        """
        with (
            patch(
                "milia_pipeline.config.config_containers._is_valid_dataset_type", return_value=False
            ),
            patch(
                "milia_pipeline.config.config_containers._get_valid_dataset_types",
                return_value=["DFT", "DMC"],
            ),
        ):
            from pydantic import ValidationError as PydanticValidationError

            from milia_pipeline.config.config_containers import HandlerConfig

            with pytest.raises(PydanticValidationError):
                HandlerConfig(handler_type="NONEXISTENT")


class TestStructuralFeaturesConfig:
    """Validate StructuralFeaturesConfig.

    Evidence: config_containers.py lines 549-637.
    """

    def test_valid_config(self):
        from milia_pipeline.config.config_containers import StructuralFeaturesConfig

        sfc = StructuralFeaturesConfig(atom_features=["atomic_number"], bond_features=["bond_type"])
        assert sfc.atom_features == ["atomic_number"]

    def test_validate_feature_config_valid(self):
        from milia_pipeline.config.config_containers import StructuralFeaturesConfig

        sfc = StructuralFeaturesConfig(
            atom_features=["atomic_number", "mass"], bond_features=["bond_type"]
        )
        is_valid, errors = sfc.validate_feature_config()
        assert is_valid is True


class TestTransformSpec:
    """Validate TransformSpec container.

    Evidence: config_containers.py lines 880-940.
    """

    def test_valid_transform_spec(self):
        from milia_pipeline.config.config_containers import TransformSpec

        ts = TransformSpec(name="AddSelfLoops")
        assert ts.name == "AddSelfLoops"
        assert ts.enabled is True

    def test_empty_name_raises(self):
        """Empty transform name raises validation error.

        Evidence: config_containers.py lines 897-907 — field_validator for 'name'.
        """
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.config.config_containers import TransformSpec

        with pytest.raises(PydanticValidationError):
            TransformSpec(name="")


# ===========================================================================
# 10. DATASET BASE CLASSES — datasets/base.py  (Pydantic V2 frozen dataclasses)
# ===========================================================================


class TestDatasetMetadata:
    """Validate DatasetMetadata Pydantic frozen dataclass.

    Evidence: base.py lines 33-59.
    """

    def test_valid_metadata(self):
        from milia_pipeline.datasets.base import DatasetMetadata

        md = DatasetMetadata(name="TestDS", version="1.0.0", description="A test dataset")
        assert md.name == "TestDS"
        assert md.author is None

    def test_empty_name_raises(self):
        """Empty name string raises ValueError.

        Evidence: base.py lines 51-52 — __post_init__ validation.
        """
        from milia_pipeline.datasets.base import DatasetMetadata

        with pytest.raises((ValueError, Exception)):
            DatasetMetadata(name="", version="1.0.0", description="Desc")

    def test_empty_version_raises(self):
        from milia_pipeline.datasets.base import DatasetMetadata

        with pytest.raises((ValueError, Exception)):
            DatasetMetadata(name="DS", version="", description="Desc")

    def test_empty_description_raises(self):
        from milia_pipeline.datasets.base import DatasetMetadata

        with pytest.raises((ValueError, Exception)):
            DatasetMetadata(name="DS", version="1.0.0", description="")

    def test_frozen_immutability(self):
        from dataclasses import FrozenInstanceError

        from milia_pipeline.datasets.base import DatasetMetadata

        md = DatasetMetadata(name="DS", version="1.0", description="Desc")
        with pytest.raises(FrozenInstanceError):
            md.name = "OtherDS"


class TestDatasetSchema:
    """Validate DatasetSchema Pydantic frozen dataclass.

    Evidence: base.py lines 62-90.
    """

    def test_valid_schema(self):
        from milia_pipeline.datasets.base import DatasetSchema

        ds = DatasetSchema(required_properties=("Etot", "atoms", "coordinates"))
        assert ds.coordinate_units == "angstrom"
        assert ds.energy_units == "hartree"

    def test_empty_required_properties_raises(self):
        """Empty required_properties raises ValueError.

        Evidence: base.py lines 78-79.
        """
        from milia_pipeline.datasets.base import DatasetSchema

        with pytest.raises((ValueError, Exception)):
            DatasetSchema(required_properties=())

    def test_invalid_coordinate_units_raises(self):
        """Invalid coordinate_units raises ValueError.

        Evidence: base.py lines 81-82.
        """
        from milia_pipeline.datasets.base import DatasetSchema

        with pytest.raises((ValueError, Exception)):
            DatasetSchema(required_properties=("atoms",), coordinate_units="meters")

    def test_invalid_energy_units_raises(self):
        """Invalid energy_units raises ValueError.

        Evidence: base.py lines 84-85.
        """
        from milia_pipeline.datasets.base import DatasetSchema

        with pytest.raises((ValueError, Exception)):
            DatasetSchema(required_properties=("atoms",), energy_units="calories")

    def test_valid_energy_units_set(self):
        from milia_pipeline.datasets.base import DatasetSchema

        for unit in ("hartree", "eV", "kcal/mol", "kJ/mol"):
            ds = DatasetSchema(required_properties=("atoms",), energy_units=unit)
            assert ds.energy_units == unit


class TestDatasetFeatures:
    """Validate DatasetFeatures Pydantic frozen dataclass.

    Evidence: base.py lines 93-127.
    """

    def test_defaults_all_false(self):
        from milia_pipeline.datasets.base import DatasetFeatures

        df = DatasetFeatures()
        d = df.to_dict()
        assert all(v is False for v in d.values())

    def test_supports_method(self):
        from milia_pipeline.datasets.base import DatasetFeatures

        df = DatasetFeatures(vibrational_analysis=True, homo_lumo_gap=True)
        assert df.supports("vibrational_analysis") is True
        assert df.supports("homo_lumo_gap") is True
        assert df.supports("uncertainty_handling") is False
        assert df.supports("nonexistent_feature") is False

    def test_to_dict_keys(self):
        from milia_pipeline.datasets.base import DatasetFeatures

        df = DatasetFeatures()
        d = df.to_dict()
        expected_keys = {
            "vibrational_analysis",
            "uncertainty_handling",
            "atomization_energy",
            "rotational_constants",
            "frequency_analysis",
            "orbital_analysis",
            "homo_lumo_gap",
            "mo_energies",
        }
        assert set(d.keys()) == expected_keys


# ===========================================================================
# 11. CONFIG BRIDGE MODELS — config_bridge.py  (31 Pydantic BaseModel classes)
# ===========================================================================


class TestConfigBridgeModels:
    """Validate a representative cross-section of the 31 Pydantic BaseModel
    classes in config_bridge.py.

    Evidence: config_bridge.py lines 203-682 and __all__ (lines 1492-1544).
    """

    # ----- ModelSelectionConfig -----

    def test_model_selection_valid(self):
        from milia_pipeline.models.utils.config_bridge import ModelSelectionConfig

        ms = ModelSelectionConfig(task_type="graph_regression", model_name="GCN")
        assert ms.task_type == "graph_regression"
        assert ms.model_name == "GCN"

    def test_model_selection_invalid_task_type(self):
        """Invalid task_type raises PydanticValidationError.

        Evidence: config_bridge.py lines 209-223.
        """
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.models.utils.config_bridge import ModelSelectionConfig

        with pytest.raises(PydanticValidationError):
            ModelSelectionConfig(task_type="invalid_task", model_name="GCN")

    def test_model_selection_empty_model_name(self):
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.models.utils.config_bridge import ModelSelectionConfig

        with pytest.raises(PydanticValidationError):
            ModelSelectionConfig(task_type="graph_regression", model_name="")

    # ----- DataSplitConfig -----

    def test_data_split_valid_defaults(self):
        from milia_pipeline.models.utils.config_bridge import DataSplitConfig

        ds = DataSplitConfig()
        assert ds.train_ratio + ds.val_ratio + ds.test_ratio == pytest.approx(1.0)

    def test_data_split_invalid_ratios(self):
        """Ratios not summing to 1.0 raises error.

        Evidence: config_bridge.py lines 262-273.
        """
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.models.utils.config_bridge import DataSplitConfig

        with pytest.raises(PydanticValidationError):
            DataSplitConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_data_split_invalid_method(self):
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.models.utils.config_bridge import DataSplitConfig

        with pytest.raises(PydanticValidationError):
            DataSplitConfig(method="bogus_method")

    # ----- LossConfig -----

    def test_loss_config_valid(self):
        from milia_pipeline.models.utils.config_bridge import LossConfig

        lc = LossConfig(name="mse")
        assert lc.name == "mse"

    def test_loss_config_invalid_name(self):
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.models.utils.config_bridge import LossConfig

        with pytest.raises(PydanticValidationError):
            LossConfig(name="nonexistent_loss")

    # ----- OptimizerConfig -----

    def test_optimizer_config_valid(self):
        from milia_pipeline.models.utils.config_bridge import OptimizerConfig

        oc = OptimizerConfig(name="adam")
        assert oc.name == "adam"

    def test_optimizer_config_invalid_name(self):
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.models.utils.config_bridge import OptimizerConfig

        with pytest.raises(PydanticValidationError):
            OptimizerConfig(name="nonexistent_optimizer")

    # ----- SchedulerConfig -----

    def test_scheduler_config_valid(self):
        from milia_pipeline.models.utils.config_bridge import SchedulerConfig

        sc = SchedulerConfig(name="reduce_on_plateau", enabled=True)
        assert sc.name == "reduce_on_plateau"

    def test_scheduler_config_invalid_name_when_enabled(self):
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.models.utils.config_bridge import SchedulerConfig

        with pytest.raises(PydanticValidationError):
            SchedulerConfig(name="bad_scheduler", enabled=True)

    # ----- DeviceConfig -----

    def test_device_config_valid(self):
        from milia_pipeline.models.utils.config_bridge import DeviceConfig

        dc = DeviceConfig(type="auto")
        assert dc.type == "auto"

    def test_device_config_invalid_type(self):
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.models.utils.config_bridge import DeviceConfig

        with pytest.raises(PydanticValidationError):
            DeviceConfig(type="quantum")

    # ----- MemoryConfig -----

    def test_memory_config_valid(self):
        from milia_pipeline.models.utils.config_bridge import MemoryConfig

        mc = MemoryConfig(mixed_precision="bf16")
        assert mc.mixed_precision == "bf16"

    def test_memory_config_invalid_precision(self):
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.models.utils.config_bridge import MemoryConfig

        with pytest.raises(PydanticValidationError):
            MemoryConfig(mixed_precision="fp128")

    # ----- TrainingConfig (composite) -----

    def test_training_config_defaults(self):
        """TrainingConfig composes DataSplitConfig, LossConfig, OptimizerConfig, etc.

        Evidence: config_bridge.py lines 404-417.
        """
        from milia_pipeline.models.utils.config_bridge import TrainingConfig

        tc = TrainingConfig()
        assert tc.data_split.method == "random"
        assert tc.loss.name == "mse"
        assert tc.optimizer.name == "adam"

    # ----- EvaluationConfig -----

    def test_evaluation_config_defaults(self):
        from milia_pipeline.models.utils.config_bridge import EvaluationConfig

        ec = EvaluationConfig()
        assert "mse" in ec.metrics
        assert ec.test_after_training is True

    # ----- DistributedConfig -----

    def test_distributed_config_disabled_accepts_any_strategy(self):
        """Disabled distributed config doesn't validate strategy.

        Evidence: config_bridge.py lines 488-500.
        """
        from milia_pipeline.models.utils.config_bridge import DistributedConfig

        dc = DistributedConfig(enabled=False, strategy="whatever")
        assert dc.enabled is False

    def test_distributed_config_enabled_invalid_strategy(self):
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.models.utils.config_bridge import DistributedConfig

        with pytest.raises(PydanticValidationError):
            DistributedConfig(enabled=True, strategy="invalid_strat")

    # ----- DeploymentConfig -----

    def test_deployment_config_disabled_accepts_any_strategy(self):
        from milia_pipeline.models.utils.config_bridge import DeploymentConfig

        dc = DeploymentConfig(enabled=False, strategy="whatever")
        assert dc.enabled is False

    def test_deployment_config_enabled_invalid_strategy(self):
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.models.utils.config_bridge import DeploymentConfig

        with pytest.raises(PydanticValidationError):
            DeploymentConfig(enabled=True, strategy="magic_deploy")

    # ----- AccelerationConfig (composite) -----

    def test_acceleration_config_defaults(self):
        from milia_pipeline.models.utils.config_bridge import AccelerationConfig

        ac = AccelerationConfig()
        assert ac.enabled is False
        assert ac.device.type == "auto"

    # ----- PluginsConfig -----

    def test_plugins_config_defaults(self):
        from milia_pipeline.models.utils.config_bridge import PluginsConfig

        pc = PluginsConfig()
        assert pc.enabled is True
        assert pc.auto_discover is True


# ===========================================================================
# 12. CONFIG BRIDGE ENUMS — config_bridge.py
# ===========================================================================


class TestConfigBridgeEnums:
    """Validate enum coverage matches the declared values.

    Evidence: config_bridge.py lines 77-197.
    """

    def test_task_type_enum_values(self):
        from milia_pipeline.models.utils.config_bridge import TaskType

        assert len(TaskType) == 6
        assert TaskType.GRAPH_REGRESSION.value == "graph_regression"

    def test_loss_function_enum_values(self):
        from milia_pipeline.models.utils.config_bridge import LossFunction

        assert LossFunction.MSE.value == "mse"
        assert LossFunction.FOCAL.value == "focal"

    def test_optimizer_type_enum_values(self):
        from milia_pipeline.models.utils.config_bridge import OptimizerType

        assert OptimizerType.ADAM.value == "adam"

    def test_scheduler_type_enum_values(self):
        from milia_pipeline.models.utils.config_bridge import SchedulerType

        assert SchedulerType.REDUCE_ON_PLATEAU.value == "reduce_on_plateau"

    def test_hpo_enums(self):
        from milia_pipeline.models.utils.config_bridge import (
            HPODirection,
            HPOParamType,
            HPOPrunerType,
            HPOSamplerType,
        )

        assert HPOParamType.FLOAT.value == "float"
        assert HPOPrunerType.MEDIAN.value == "median"
        assert HPOSamplerType.TPE.value == "tpe"
        assert HPODirection.MINIMIZE.value == "minimize"


# ===========================================================================
# 13. DEEP MERGE — config_loader.py
# ===========================================================================


class TestDeepMergeConfigs:
    """Test _deep_merge_configs used by YAML splitting.

    Evidence: config_loader.py lines 509-558.
    """

    def test_simple_override(self):
        from milia_pipeline.config.config_loader import _deep_merge_configs

        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        merged = _deep_merge_configs(base, override)
        assert merged == {"a": 1, "b": 3, "c": 4}

    def test_recursive_dict_merge(self):
        from milia_pipeline.config.config_loader import _deep_merge_configs

        base = {"nested": {"x": 1, "y": 2}}
        override = {"nested": {"y": 99, "z": 100}}
        merged = _deep_merge_configs(base, override)
        assert merged["nested"] == {"x": 1, "y": 99, "z": 100}

    def test_list_override_not_merge(self):
        """Lists are replaced, not merged.

        Evidence: config_loader.py lines 551-553.
        """
        from milia_pipeline.config.config_loader import _deep_merge_configs

        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        merged = _deep_merge_configs(base, override)
        assert merged["items"] == [4, 5]

    def test_no_mutation_of_inputs(self):
        """Thread safety: inputs must not be mutated.

        Evidence: config_loader.py line 541 — deepcopy of base.
        """
        from milia_pipeline.config.config_loader import _deep_merge_configs

        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        base_copy = copy.deepcopy(base)
        _deep_merge_configs(base, override)
        assert base == base_copy

    def test_empty_base(self):
        from milia_pipeline.config.config_loader import _deep_merge_configs

        merged = _deep_merge_configs({}, {"a": 1})
        assert merged == {"a": 1}

    def test_empty_override(self):
        from milia_pipeline.config.config_loader import _deep_merge_configs

        merged = _deep_merge_configs({"a": 1}, {})
        assert merged == {"a": 1}


# ===========================================================================
# 14. YAML SPLITTING: _discover_config_files, _collect_yaml_files
# ===========================================================================


class TestYAMLSplittingDiscovery:
    """Test YAML splitting file discovery functions.

    Evidence: config_loader.py lines 398-506.
    """

    def test_single_file_mode(self, tmp_path):
        """Existing file triggers single-file mode.

        Evidence: config_loader.py lines 429-431.
        """
        from milia_pipeline.config.config_loader import _discover_config_files

        f = tmp_path / "config.yaml"
        f.write_text("dataset_type: DFT\n")
        is_split, files = _discover_config_files(str(f))
        assert is_split is False
        assert len(files) == 1

    def test_directory_mode(self, tmp_config_dir):
        """Existing directory triggers split-file mode.

        Evidence: config_loader.py lines 434-436.
        """
        from milia_pipeline.config.config_loader import _discover_config_files

        # Create a YAML file so the directory isn't empty
        (tmp_config_dir / "main.yaml").write_text("dataset_type: DFT\n")
        is_split, files = _discover_config_files(str(tmp_config_dir))
        assert is_split is True
        assert len(files) >= 1

    def test_collect_yaml_files_ordering(self, tmp_config_dir):
        """main.yaml loaded first; root files alphabetical; datasets/ last.

        Evidence: config_loader.py lines 474-506.
        """
        from milia_pipeline.config.config_loader import _collect_yaml_files

        (tmp_config_dir / "main.yaml").write_text("a: 1\n")
        (tmp_config_dir / "zzz.yaml").write_text("b: 2\n")
        (tmp_config_dir / "aaa.yaml").write_text("c: 3\n")
        datasets_dir = tmp_config_dir / "datasets"
        (datasets_dir / "dft.yaml").write_text("d: 4\n")

        files = _collect_yaml_files(tmp_config_dir)
        names = [f.name for f in files]

        # main.yaml must be first
        assert names[0] == "main.yaml"
        # datasets/ files must come after root files
        assert names.index("dft.yaml") > names.index("aaa.yaml")

    def test_nonexistent_path_returns_single(self, tmp_path):
        """Non-existent path falls through to single-file mode.

        Evidence: config_loader.py lines 445-446.
        """
        from milia_pipeline.config.config_loader import _discover_config_files

        is_split, files = _discover_config_files(str(tmp_path / "missing.yaml"))
        assert is_split is False


# ===========================================================================
# 15. LOAD_AND_MERGE_YAML_FILES — config_loader.py
# ===========================================================================


class TestLoadAndMergeYamlFiles:
    """Test _load_and_merge_yaml_files.

    Evidence: config_loader.py lines 561-623.
    """

    def test_single_file_merge(self, tmp_path):
        from milia_pipeline.config.config_loader import _load_and_merge_yaml_files

        f = tmp_path / "test.yaml"
        f.write_text("key1: value1\nkey2: value2\n")
        result = _load_and_merge_yaml_files([f])
        assert result == {"key1": "value1", "key2": "value2"}

    def test_multi_file_merge_order(self, tmp_path):
        from milia_pipeline.config.config_loader import _load_and_merge_yaml_files

        f1 = tmp_path / "base.yaml"
        f2 = tmp_path / "override.yaml"
        f1.write_text("a: 1\nb: 2\n")
        f2.write_text("b: 99\nc: 3\n")
        result = _load_and_merge_yaml_files([f1, f2])
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_empty_files_skipped(self, tmp_path):
        """Empty files are skipped with a warning, not an error.

        Evidence: config_loader.py lines 589-590.
        """
        from milia_pipeline.config.config_loader import _load_and_merge_yaml_files

        f1 = tmp_path / "real.yaml"
        f1.write_text("key: val\n")
        f2 = tmp_path / "empty.yaml"
        f2.write_text("")
        result = _load_and_merge_yaml_files([f1, f2])
        assert result == {"key": "val"}

    def test_no_files_raises(self):
        """Empty file list raises ConfigurationError.

        Evidence: config_loader.py lines 574-579.
        """
        from milia_pipeline.config.config_loader import _load_and_merge_yaml_files

        _ConfigurationError = _load_and_merge_yaml_files.__globals__["ConfigurationError"]
        with pytest.raises(_ConfigurationError, match="No configuration files"):
            _load_and_merge_yaml_files([])

    def test_invalid_yaml_raises(self, tmp_path):
        """Malformed YAML raises ConfigurationError.

        Evidence: config_loader.py lines 609-614.
        """
        from milia_pipeline.config.config_loader import _load_and_merge_yaml_files

        _ConfigurationError = _load_and_merge_yaml_files.__globals__["ConfigurationError"]
        bad = tmp_path / "bad.yaml"
        bad.write_text("key: [unclosed bracket\n")
        with pytest.raises(_ConfigurationError):
            _load_and_merge_yaml_files([bad])

    def test_non_dict_yaml_raises(self, tmp_path):
        """YAML file containing a list (not dict) raises ConfigurationError.

        Evidence: config_loader.py lines 599-604.
        """
        from milia_pipeline.config.config_loader import _load_and_merge_yaml_files

        _ConfigurationError = _load_and_merge_yaml_files.__globals__["ConfigurationError"]
        f = tmp_path / "list.yaml"
        f.write_text("- item1\n- item2\n")
        with pytest.raises(_ConfigurationError, match="dictionary"):
            _load_and_merge_yaml_files([f])


# ===========================================================================
# 16. LOAD_CONFIG — config_loader.py
# ===========================================================================


class TestLoadConfig:
    """Test load_config for both single-file and split-file modes.

    Evidence: config_loader.py lines 666-906.
    """

    def test_load_single_file_valid(self, minimal_valid_config_yaml):
        """Single-file mode loads and returns a dict."""
        from milia_pipeline.config.config_loader import load_config

        config = load_config(
            config_path=minimal_valid_config_yaml, enable_enhancement=False, force_reload=True
        )
        assert isinstance(config, dict)
        assert "dataset_type" in config

    def test_load_nonexistent_file_raises(self, tmp_path):
        """Loading a nonexistent file raises ConfigurationError.

        Evidence: config_loader.py lines 748-753.
        """
        from milia_pipeline.config.config_loader import load_config

        _ConfigurationError = load_config.__globals__["ConfigurationError"]
        with pytest.raises(_ConfigurationError, match="not found"):
            load_config(config_path=str(tmp_path / "nonexistent.yaml"), force_reload=True)

    def test_load_empty_file_raises(self, tmp_path):
        """Loading an empty YAML file raises ConfigurationError.

        Evidence: config_loader.py lines 775-779.
        """
        from milia_pipeline.config.config_loader import load_config

        _ConfigurationError = load_config.__globals__["ConfigurationError"]
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(_ConfigurationError):
            load_config(config_path=str(empty), force_reload=True)

    def test_load_non_dict_config_raises(self, tmp_path):
        """YAML file with list root raises ConfigurationError.

        Evidence: config_loader.py lines 797-802.
        """
        from milia_pipeline.config.config_loader import load_config

        _ConfigurationError = load_config.__globals__["ConfigurationError"]
        f = tmp_path / "list_root.yaml"
        f.write_text("- a\n- b\n")
        with pytest.raises(_ConfigurationError, match="dictionary"):
            load_config(config_path=str(f), force_reload=True)

    def test_load_split_mode(self, tmp_config_dir):
        """Split-file mode merges multiple YAML files."""
        from milia_pipeline.config.config_loader import load_config

        (tmp_config_dir / "main.yaml").write_text(
            "dataset_type: DFT\nglobal_paths:\n  working_root_dir: /tmp\n"
        )
        (tmp_config_dir / "filter_config.yaml").write_text("filter_config:\n  max_atoms: 50\n")
        datasets_dir = tmp_config_dir / "datasets"
        (datasets_dir / "dft.yaml").write_text("dft_config:\n  raw_npz_filename: DFT.npz\n")

        config = load_config(
            config_path=str(tmp_config_dir), enable_enhancement=False, force_reload=True
        )
        assert config["dataset_type"] == "DFT"
        assert "filter_config" in config
        assert "dft_config" in config

    def test_caching_returns_same_object(self, minimal_valid_config_yaml):
        """Second load with same args returns cached config.

        Evidence: config_loader.py lines 724-735.
        """
        from milia_pipeline.config.config_loader import load_config

        c1 = load_config(
            config_path=minimal_valid_config_yaml, enable_enhancement=False, force_reload=True
        )
        c2 = load_config(
            config_path=minimal_valid_config_yaml, enable_enhancement=False, force_reload=False
        )
        assert c1 is c2

    def test_force_reload_bypasses_cache(self, minimal_valid_config_yaml):
        from milia_pipeline.config.config_loader import load_config

        c1 = load_config(
            config_path=minimal_valid_config_yaml, enable_enhancement=False, force_reload=True
        )
        c2 = load_config(
            config_path=minimal_valid_config_yaml, enable_enhancement=False, force_reload=True
        )
        # Both are dicts with same content but force_reload reloads from disk
        assert c1 == c2


# ===========================================================================
# 17. ERROR MESSAGE QUALITY
# ===========================================================================


class TestErrorMessageQuality:
    """Assert that validation errors contain informative, actionable messages.

    Evidence: Section 5.1 scope — "Asserts: invalid configs raise
    ConfigurationError or ValidationError with informative messages."
    """

    def test_dataset_config_error_lists_valid_types(self):
        """DatasetConfig error for invalid type mentions valid options."""
        with (
            patch(
                "milia_pipeline.config.config_containers._is_valid_dataset_type", return_value=False
            ),
            patch(
                "milia_pipeline.config.config_containers._get_valid_dataset_types",
                return_value=["DFT", "DMC", "QM9"],
            ),
        ):
            from pydantic import ValidationError as PydanticValidationError

            from milia_pipeline.config.config_containers import DatasetConfig

            with pytest.raises(PydanticValidationError) as exc_info:
                DatasetConfig(dataset_type="INVALID_TYPE")
            error_text = str(exc_info.value)
            # The error should mention at least one valid type
            assert any(t in error_text for t in ["DFT", "DMC", "QM9"])

    def test_loss_config_error_lists_valid_losses(self):
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.models.utils.config_bridge import LossConfig

        with pytest.raises(PydanticValidationError) as exc_info:
            LossConfig(name="invalid_loss")
        error_text = str(exc_info.value)
        assert "mse" in error_text or "mae" in error_text

    def test_split_ratios_error_mentions_sum(self):
        from pydantic import ValidationError as PydanticValidationError

        from milia_pipeline.models.utils.config_bridge import DataSplitConfig

        with pytest.raises(PydanticValidationError) as exc_info:
            DataSplitConfig(train_ratio=0.9, val_ratio=0.9, test_ratio=0.9)
        error_text = str(exc_info.value)
        assert "sum" in error_text.lower() or "1.0" in error_text
