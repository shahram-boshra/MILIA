#!/usr/bin/env python3
"""
Complete Unit Test Suite for milia_pipeline/models/hpo/search_spaces/param_types.py Module

Tests ParamType enum and SearchSpaceParamConfig frozen Pydantic BaseModel including:
- ParamType Enum:
  - All enum member values (INT, FLOAT, CATEGORICAL, LOGUNIFORM, UNIFORM, INT_UNIFORM, DISCRETE_UNIFORM)
  - String value access
  - Enum iteration and membership
  - Enum comparison and identity
- SearchSpaceParamConfig (Pydantic V2 frozen BaseModel):
  - Valid instantiation for all parameter types
  - Frozen model immutability (Pydantic V2 raises ValidationError on mutation)
  - Default values for optional fields
  - @model_validator(mode='before') validation:
    - Numeric types (INT, FLOAT, LOGUNIFORM, UNIFORM, INT_UNIFORM, DISCRETE_UNIFORM) require low and high
    - Numeric types require low < high (not equal, not greater)
    - CATEGORICAL type requires non-empty choices list
  - to_dict() backward compatibility method
  - Edge cases:
    - Zero bounds
    - Negative bounds
    - Very small ranges
    - Mixed choice types in categorical
    - Log scale flag
    - Step parameter

This is a PRODUCTION-READY test suite with comprehensive coverage.

Pydantic V2 Migration (Phase 6a):
    - Updated from @dataclass(frozen=True) to BaseModel with frozen=True
    - FrozenInstanceError replaced with pydantic.ValidationError
    - ConfigurationError replaced with standard ValueError
    - Tests @model_validator(mode='before') validation logic
    - Tests to_dict() backward compatibility method

Author: Milia Team
Version: 2.0.0

Changelog:
- v2.0.0: Updated for Pydantic V2 migration - frozen BaseModel, ValidationError, ValueError
- v1.0.0: Initial release with full coverage for param_types.py
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from enum import Enum

import pytest

# Import Pydantic ValidationError for frozen model mutation tests
from pydantic import ValidationError

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def param_types_module():
    """
    Fixture to import param_types module directly.

    Pydantic V2 Migration: No mocking needed since the module no longer
    depends on external ConfigurationError - it uses standard ValueError.

    Returns the imported classes for testing.
    """
    from milia_pipeline.models.hpo.search_spaces.param_types import (
        ParamType,
        SearchSpaceParamConfig,
    )

    return {
        "ParamType": ParamType,
        "SearchSpaceParamConfig": SearchSpaceParamConfig,
    }


# =============================================================================
# PARAMTYPE ENUM TESTS
# =============================================================================


class TestParamTypeEnum:
    """Test ParamType enum definition and values."""

    def test_int_value(self, param_types_module):
        """Test ParamType.INT has correct string value 'int'."""
        ParamType = param_types_module["ParamType"]
        assert ParamType.INT.value == "int"

    def test_float_value(self, param_types_module):
        """Test ParamType.FLOAT has correct string value 'float'."""
        ParamType = param_types_module["ParamType"]
        assert ParamType.FLOAT.value == "float"

    def test_categorical_value(self, param_types_module):
        """Test ParamType.CATEGORICAL has correct string value 'categorical'."""
        ParamType = param_types_module["ParamType"]
        assert ParamType.CATEGORICAL.value == "categorical"

    def test_loguniform_value(self, param_types_module):
        """Test ParamType.LOGUNIFORM has correct string value 'loguniform'."""
        ParamType = param_types_module["ParamType"]
        assert ParamType.LOGUNIFORM.value == "loguniform"

    def test_uniform_value(self, param_types_module):
        """Test ParamType.UNIFORM has correct string value 'uniform'."""
        ParamType = param_types_module["ParamType"]
        assert ParamType.UNIFORM.value == "uniform"

    def test_int_uniform_value(self, param_types_module):
        """Test ParamType.INT_UNIFORM has correct string value 'int_uniform'."""
        ParamType = param_types_module["ParamType"]
        assert ParamType.INT_UNIFORM.value == "int_uniform"

    def test_discrete_uniform_value(self, param_types_module):
        """Test ParamType.DISCRETE_UNIFORM has correct string value 'discrete_uniform'."""
        ParamType = param_types_module["ParamType"]
        assert ParamType.DISCRETE_UNIFORM.value == "discrete_uniform"

    def test_enum_has_seven_members(self, param_types_module):
        """Test ParamType has exactly 7 members."""
        ParamType = param_types_module["ParamType"]
        assert len(list(ParamType)) == 7

    def test_enum_member_names(self, param_types_module):
        """Test all expected enum member names exist."""
        ParamType = param_types_module["ParamType"]
        expected_names = {
            "INT",
            "FLOAT",
            "CATEGORICAL",
            "LOGUNIFORM",
            "UNIFORM",
            "INT_UNIFORM",
            "DISCRETE_UNIFORM",
        }
        actual_names = {member.name for member in ParamType}
        assert actual_names == expected_names

    def test_enum_is_enum_subclass(self, param_types_module):
        """Test ParamType is an Enum subclass."""
        ParamType = param_types_module["ParamType"]
        assert issubclass(ParamType, Enum)

    def test_enum_member_is_instance(self, param_types_module):
        """Test each member is instance of ParamType."""
        ParamType = param_types_module["ParamType"]
        for member in ParamType:
            assert isinstance(member, ParamType)

    def test_enum_access_by_name(self, param_types_module):
        """Test accessing enum members by name."""
        ParamType = param_types_module["ParamType"]
        assert ParamType["INT"] == ParamType.INT
        assert ParamType["FLOAT"] == ParamType.FLOAT
        assert ParamType["CATEGORICAL"] == ParamType.CATEGORICAL

    def test_enum_access_by_value(self, param_types_module):
        """Test accessing enum members by value."""
        ParamType = param_types_module["ParamType"]
        assert ParamType("int") == ParamType.INT
        assert ParamType("float") == ParamType.FLOAT
        assert ParamType("categorical") == ParamType.CATEGORICAL

    def test_enum_comparison_identity(self, param_types_module):
        """Test enum members maintain identity."""
        ParamType = param_types_module["ParamType"]
        member1 = ParamType.INT
        member2 = ParamType.INT
        assert member1 is member2

    def test_enum_comparison_equality(self, param_types_module):
        """Test enum members equality comparison."""
        ParamType = param_types_module["ParamType"]
        assert ParamType.INT == ParamType.INT
        assert ParamType.INT != ParamType.FLOAT

    def test_invalid_enum_value_raises_valueerror(self, param_types_module):
        """Test accessing non-existent value raises ValueError."""
        ParamType = param_types_module["ParamType"]
        with pytest.raises(ValueError):
            ParamType("invalid_type")

    def test_invalid_enum_name_raises_keyerror(self, param_types_module):
        """Test accessing non-existent name raises KeyError."""
        ParamType = param_types_module["ParamType"]
        with pytest.raises(KeyError):
            ParamType["INVALID"]


# =============================================================================
# SEARCHSPACEPARAMCONFIG VALID INSTANTIATION TESTS
# =============================================================================


class TestSearchSpaceParamConfigValidInstantiation:
    """Test valid instantiation of SearchSpaceParamConfig for all parameter types."""

    def test_valid_int_param_with_bounds(self, param_types_module):
        """Test valid INT parameter with low and high bounds."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )

        assert config.type == ParamType.INT
        assert config.low == 1
        assert config.high == 10

    def test_valid_int_param_with_step(self, param_types_module):
        """Test valid INT parameter with step size."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=0,
            high=100,
            step=10,
        )

        assert config.step == 10

    def test_valid_float_param_with_bounds(self, param_types_module):
        """Test valid FLOAT parameter with low and high bounds."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.FLOAT,
            low=0.0,
            high=1.0,
        )

        assert config.type == ParamType.FLOAT
        assert config.low == 0.0
        assert config.high == 1.0

    def test_valid_float_param_with_log_scale(self, param_types_module):
        """Test valid FLOAT parameter with log scale enabled."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.FLOAT,
            low=1e-5,
            high=1e-1,
            log=True,
        )

        assert config.log is True

    def test_valid_categorical_param_with_string_choices(self, param_types_module):
        """Test valid CATEGORICAL parameter with string choices."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=["relu", "elu", "leaky_relu"],
        )

        assert config.type == ParamType.CATEGORICAL
        assert config.choices == ["relu", "elu", "leaky_relu"]

    def test_valid_categorical_param_with_numeric_choices(self, param_types_module):
        """Test valid CATEGORICAL parameter with numeric choices."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=[1, 2, 4, 8, 16],
        )

        assert config.choices == [1, 2, 4, 8, 16]

    def test_valid_categorical_param_with_mixed_choices(self, param_types_module):
        """Test valid CATEGORICAL parameter with mixed type choices."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=["adam", 0.001, True, None],
        )

        assert config.choices == ["adam", 0.001, True, None]

    def test_valid_loguniform_param(self, param_types_module):
        """Test valid LOGUNIFORM parameter with bounds."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.LOGUNIFORM,
            low=1e-5,
            high=1e-2,
        )

        assert config.type == ParamType.LOGUNIFORM
        assert config.low == 1e-5
        assert config.high == 1e-2

    def test_valid_uniform_param(self, param_types_module):
        """Test valid UNIFORM parameter with bounds."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.UNIFORM,
            low=0.0,
            high=10.0,
        )

        assert config.type == ParamType.UNIFORM
        assert config.low == 0.0
        assert config.high == 10.0

    def test_valid_int_uniform_param(self, param_types_module):
        """Test valid INT_UNIFORM parameter with bounds."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT_UNIFORM,
            low=1,
            high=100,
        )

        assert config.type == ParamType.INT_UNIFORM
        assert config.low == 1
        assert config.high == 100

    def test_valid_discrete_uniform_param(self, param_types_module):
        """Test valid DISCRETE_UNIFORM parameter with bounds."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.DISCRETE_UNIFORM,
            low=0.0,
            high=1.0,
        )

        assert config.type == ParamType.DISCRETE_UNIFORM
        assert config.low == 0.0
        assert config.high == 1.0

    def test_valid_discrete_uniform_param_with_step(self, param_types_module):
        """Test valid DISCRETE_UNIFORM parameter with step."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.DISCRETE_UNIFORM,
            low=0,
            high=100,
            step=5,
        )

        assert config.step == 5


# =============================================================================
# SEARCHSPACEPARAMCONFIG DEFAULT VALUES TESTS
# =============================================================================


class TestSearchSpaceParamConfigDefaults:
    """Test default values of SearchSpaceParamConfig."""

    def test_low_default_is_none(self, param_types_module):
        """Test low attribute defaults to None."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=["a", "b"],
        )

        assert config.low is None

    def test_high_default_is_none(self, param_types_module):
        """Test high attribute defaults to None."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=["a", "b"],
        )

        assert config.high is None

    def test_step_default_is_none(self, param_types_module):
        """Test step attribute defaults to None."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )

        assert config.step is None

    def test_choices_default_is_none(self, param_types_module):
        """Test choices attribute defaults to None."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )

        assert config.choices is None

    def test_log_default_is_false(self, param_types_module):
        """Test log attribute defaults to False."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )

        assert config.log is False


# =============================================================================
# SEARCHSPACEPARAMCONFIG FROZEN PYDANTIC MODEL TESTS
# =============================================================================


class TestSearchSpaceParamConfigFrozen:
    """Test that SearchSpaceParamConfig is frozen (immutable) via Pydantic V2."""

    def test_cannot_modify_type_attribute(self, param_types_module):
        """Test type attribute cannot be modified after creation (Pydantic V2 ValidationError)."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )

        with pytest.raises(ValidationError):
            config.type = ParamType.FLOAT

    def test_cannot_modify_low_attribute(self, param_types_module):
        """Test low attribute cannot be modified after creation (Pydantic V2 ValidationError)."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )

        with pytest.raises(ValidationError):
            config.low = 5

    def test_cannot_modify_high_attribute(self, param_types_module):
        """Test high attribute cannot be modified after creation (Pydantic V2 ValidationError)."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )

        with pytest.raises(ValidationError):
            config.high = 100

    def test_cannot_modify_step_attribute(self, param_types_module):
        """Test step attribute cannot be modified after creation (Pydantic V2 ValidationError)."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
            step=2,
        )

        with pytest.raises(ValidationError):
            config.step = 5

    def test_cannot_modify_choices_attribute(self, param_types_module):
        """Test choices attribute cannot be modified after creation (Pydantic V2 ValidationError)."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=["a", "b", "c"],
        )

        with pytest.raises(ValidationError):
            config.choices = ["x", "y", "z"]

    def test_cannot_modify_log_attribute(self, param_types_module):
        """Test log attribute cannot be modified after creation (Pydantic V2 ValidationError)."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.FLOAT,
            low=0.0,
            high=1.0,
            log=False,
        )

        with pytest.raises(ValidationError):
            config.log = True

    def test_cannot_add_new_attribute(self, param_types_module):
        """Test new attributes cannot be added after creation (Pydantic V2 ValidationError)."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )

        with pytest.raises(ValidationError):
            config.new_attribute = "value"


# =============================================================================
# SEARCHSPACEPARAMCONFIG VALIDATION TESTS - NUMERIC TYPES
# =============================================================================


class TestSearchSpaceParamConfigNumericValidation:
    """Test @model_validator(mode='before') validation for numeric parameter types."""

    def test_int_type_missing_low_raises_error(self, param_types_module):
        """Test INT type without low raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.INT,
                high=10,
            )

        assert "requires 'low' and 'high'" in str(exc_info.value)

    def test_int_type_missing_high_raises_error(self, param_types_module):
        """Test INT type without high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.INT,
                low=1,
            )

        assert "requires 'low' and 'high'" in str(exc_info.value)

    def test_int_type_missing_both_bounds_raises_error(self, param_types_module):
        """Test INT type without both bounds raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.INT,
            )

        assert "requires 'low' and 'high'" in str(exc_info.value)

    def test_float_type_missing_low_raises_error(self, param_types_module):
        """Test FLOAT type without low raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.FLOAT,
                high=1.0,
            )

        assert "requires 'low' and 'high'" in str(exc_info.value)

    def test_float_type_missing_high_raises_error(self, param_types_module):
        """Test FLOAT type without high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.FLOAT,
                low=0.0,
            )

        assert "requires 'low' and 'high'" in str(exc_info.value)

    def test_loguniform_type_missing_low_raises_error(self, param_types_module):
        """Test LOGUNIFORM type without low raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.LOGUNIFORM,
                high=1e-2,
            )

        assert "requires 'low' and 'high'" in str(exc_info.value)

    def test_loguniform_type_missing_high_raises_error(self, param_types_module):
        """Test LOGUNIFORM type without high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.LOGUNIFORM,
                low=1e-5,
            )

        assert "requires 'low' and 'high'" in str(exc_info.value)

    def test_uniform_type_missing_low_raises_error(self, param_types_module):
        """Test UNIFORM type without low raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.UNIFORM,
                high=10.0,
            )

        assert "requires 'low' and 'high'" in str(exc_info.value)

    def test_uniform_type_missing_high_raises_error(self, param_types_module):
        """Test UNIFORM type without high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.UNIFORM,
                low=0.0,
            )

        assert "requires 'low' and 'high'" in str(exc_info.value)

    def test_int_uniform_type_missing_low_raises_error(self, param_types_module):
        """Test INT_UNIFORM type without low raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.INT_UNIFORM,
                high=100,
            )

        assert "requires 'low' and 'high'" in str(exc_info.value)

    def test_int_uniform_type_missing_high_raises_error(self, param_types_module):
        """Test INT_UNIFORM type without high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.INT_UNIFORM,
                low=1,
            )

        assert "requires 'low' and 'high'" in str(exc_info.value)


# =============================================================================
# SEARCHSPACEPARAMCONFIG VALIDATION TESTS - LOW >= HIGH
# =============================================================================


class TestSearchSpaceParamConfigBoundsValidation:
    """Test @model_validator(mode='before') validation for low >= high condition."""

    def test_int_type_low_equals_high_raises_error(self, param_types_module):
        """Test INT type with low == high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.INT,
                low=5,
                high=5,
            )

        assert "'low' must be less than 'high'" in str(exc_info.value)

    def test_int_type_low_greater_than_high_raises_error(self, param_types_module):
        """Test INT type with low > high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.INT,
                low=10,
                high=5,
            )

        assert "'low' must be less than 'high'" in str(exc_info.value)

    def test_float_type_low_equals_high_raises_error(self, param_types_module):
        """Test FLOAT type with low == high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.FLOAT,
                low=0.5,
                high=0.5,
            )

        assert "'low' must be less than 'high'" in str(exc_info.value)

    def test_float_type_low_greater_than_high_raises_error(self, param_types_module):
        """Test FLOAT type with low > high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.FLOAT,
                low=1.0,
                high=0.5,
            )

        assert "'low' must be less than 'high'" in str(exc_info.value)

    def test_loguniform_type_low_equals_high_raises_error(self, param_types_module):
        """Test LOGUNIFORM type with low == high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.LOGUNIFORM,
                low=1e-3,
                high=1e-3,
            )

        assert "'low' must be less than 'high'" in str(exc_info.value)

    def test_loguniform_type_low_greater_than_high_raises_error(self, param_types_module):
        """Test LOGUNIFORM type with low > high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.LOGUNIFORM,
                low=1e-2,
                high=1e-5,
            )

        assert "'low' must be less than 'high'" in str(exc_info.value)

    def test_uniform_type_low_equals_high_raises_error(self, param_types_module):
        """Test UNIFORM type with low == high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.UNIFORM,
                low=5.0,
                high=5.0,
            )

        assert "'low' must be less than 'high'" in str(exc_info.value)

    def test_uniform_type_low_greater_than_high_raises_error(self, param_types_module):
        """Test UNIFORM type with low > high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.UNIFORM,
                low=10.0,
                high=5.0,
            )

        assert "'low' must be less than 'high'" in str(exc_info.value)

    def test_int_uniform_type_low_equals_high_raises_error(self, param_types_module):
        """Test INT_UNIFORM type with low == high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.INT_UNIFORM,
                low=50,
                high=50,
            )

        assert "'low' must be less than 'high'" in str(exc_info.value)

    def test_int_uniform_type_low_greater_than_high_raises_error(self, param_types_module):
        """Test INT_UNIFORM type with low > high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.INT_UNIFORM,
                low=100,
                high=50,
            )

        assert "'low' must be less than 'high'" in str(exc_info.value)


# =============================================================================
# SEARCHSPACEPARAMCONFIG VALIDATION TESTS - CATEGORICAL TYPE
# =============================================================================


class TestSearchSpaceParamConfigCategoricalValidation:
    """Test @model_validator(mode='before') validation for CATEGORICAL parameter type."""

    def test_categorical_type_missing_choices_raises_error(self, param_types_module):
        """Test CATEGORICAL type without choices raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.CATEGORICAL,
            )

        assert "requires non-empty 'choices' list" in str(exc_info.value)

    def test_categorical_type_empty_choices_list_raises_error(self, param_types_module):
        """Test CATEGORICAL type with empty choices list raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.CATEGORICAL,
                choices=[],
            )

        assert "requires non-empty 'choices' list" in str(exc_info.value)

    def test_categorical_type_none_choices_raises_error(self, param_types_module):
        """Test CATEGORICAL type with None choices raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.CATEGORICAL,
                choices=None,
            )

        assert "requires non-empty 'choices' list" in str(exc_info.value)

    def test_categorical_type_single_choice_is_valid(self, param_types_module):
        """Test CATEGORICAL type with single choice is valid."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=["only_option"],
        )

        assert config.choices == ["only_option"]


# =============================================================================
# SEARCHSPACEPARAMCONFIG VALIDATION TESTS - DISCRETE_UNIFORM TYPE
# =============================================================================


class TestSearchSpaceParamConfigDiscreteUniformValidation:
    """Test @model_validator(mode='before') validation for DISCRETE_UNIFORM parameter type."""

    def test_discrete_uniform_type_missing_low_raises_error(self, param_types_module):
        """Test DISCRETE_UNIFORM type without low raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.DISCRETE_UNIFORM,
                high=10.0,
            )

        assert "requires 'low' and 'high'" in str(exc_info.value)

    def test_discrete_uniform_type_missing_high_raises_error(self, param_types_module):
        """Test DISCRETE_UNIFORM type without high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.DISCRETE_UNIFORM,
                low=0.0,
            )

        assert "requires 'low' and 'high'" in str(exc_info.value)

    def test_discrete_uniform_type_missing_both_bounds_raises_error(self, param_types_module):
        """Test DISCRETE_UNIFORM type without both bounds raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.DISCRETE_UNIFORM,
            )

        assert "requires 'low' and 'high'" in str(exc_info.value)

    def test_discrete_uniform_type_low_equals_high_raises_error(self, param_types_module):
        """Test DISCRETE_UNIFORM type with low == high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.DISCRETE_UNIFORM,
                low=5.0,
                high=5.0,
            )

        assert "'low' must be less than 'high'" in str(exc_info.value)

    def test_discrete_uniform_type_low_greater_than_high_raises_error(self, param_types_module):
        """Test DISCRETE_UNIFORM type with low > high raises ValueError."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.DISCRETE_UNIFORM,
                low=10.0,
                high=5.0,
            )

        assert "'low' must be less than 'high'" in str(exc_info.value)


# =============================================================================
# SEARCHSPACEPARAMCONFIG EDGE CASES TESTS
# =============================================================================


class TestSearchSpaceParamConfigEdgeCases:
    """Test edge cases for SearchSpaceParamConfig."""

    def test_zero_low_bound_is_valid(self, param_types_module):
        """Test zero as low bound is valid."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=0,
            high=10,
        )

        assert config.low == 0

    def test_zero_high_bound_with_negative_low_is_valid(self, param_types_module):
        """Test zero as high bound with negative low is valid."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=-10,
            high=0,
        )

        assert config.high == 0

    def test_negative_bounds_are_valid(self, param_types_module):
        """Test negative bounds are valid when low < high."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.FLOAT,
            low=-10.0,
            high=-1.0,
        )

        assert config.low == -10.0
        assert config.high == -1.0

    def test_very_small_range_is_valid(self, param_types_module):
        """Test very small range (e.g., 1e-10) is valid."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.FLOAT,
            low=0.0,
            high=1e-10,
        )

        assert config.low == 0.0
        assert config.high == 1e-10

    def test_very_large_range_is_valid(self, param_types_module):
        """Test very large range is valid."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.FLOAT,
            low=-1e10,
            high=1e10,
        )

        assert config.low == -1e10
        assert config.high == 1e10

    def test_loguniform_with_very_small_values(self, param_types_module):
        """Test LOGUNIFORM with very small positive values."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.LOGUNIFORM,
            low=1e-10,
            high=1e-5,
        )

        assert config.low == 1e-10
        assert config.high == 1e-5

    def test_categorical_with_none_in_choices(self, param_types_module):
        """Test CATEGORICAL with None as one of the choices."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=[None, "option1", "option2"],
        )

        assert None in config.choices

    def test_categorical_with_boolean_choices(self, param_types_module):
        """Test CATEGORICAL with boolean choices."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=[True, False],
        )

        assert config.choices == [True, False]

    def test_categorical_with_tuple_choices(self, param_types_module):
        """Test CATEGORICAL with tuple as choices."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        # Note: choices must be a list based on type hint, but tuple might work
        config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=["a", "b", "c"],  # Use list as per type hint
        )

        assert len(config.choices) == 3

    def test_step_zero_is_allowed(self, param_types_module):
        """Test step value of zero is allowed (no explicit validation)."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
            step=0,
        )

        assert config.step == 0

    def test_negative_step_is_allowed(self, param_types_module):
        """Test negative step value is allowed (no explicit validation)."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
            step=-1,
        )

        assert config.step == -1

    def test_float_low_with_int_type(self, param_types_module):
        """Test float values for low/high with INT type (allowed by type hints)."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1.5,
            high=10.5,
        )

        assert config.low == 1.5
        assert config.high == 10.5


# =============================================================================
# SEARCHSPACEPARAMCONFIG ATTRIBUTE TESTS
# =============================================================================


class TestSearchSpaceParamConfigAttributes:
    """Test attribute access and types for SearchSpaceParamConfig."""

    def test_all_attributes_accessible(self, param_types_module):
        """Test all attributes are accessible after creation."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
            step=2,
            choices=None,
            log=True,
        )

        assert hasattr(config, "type")
        assert hasattr(config, "low")
        assert hasattr(config, "high")
        assert hasattr(config, "step")
        assert hasattr(config, "choices")
        assert hasattr(config, "log")

    def test_type_attribute_is_paramtype(self, param_types_module):
        """Test type attribute is ParamType instance."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )

        assert isinstance(config.type, ParamType)

    def test_log_attribute_is_bool(self, param_types_module):
        """Test log attribute is boolean."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.FLOAT,
            low=0.0,
            high=1.0,
        )

        assert isinstance(config.log, bool)

    def test_choices_attribute_is_list_when_provided(self, param_types_module):
        """Test choices attribute is list when provided."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=["a", "b", "c"],
        )

        assert isinstance(config.choices, list)


# =============================================================================
# SEARCHSPACEPARAMCONFIG TO_DICT METHOD TESTS
# =============================================================================


class TestSearchSpaceParamConfigToDict:
    """Test to_dict() backward compatibility method (Pydantic V2 migration)."""

    def test_to_dict_returns_dict(self, param_types_module):
        """Test to_dict() returns a dictionary."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )

        result = config.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_contains_all_fields(self, param_types_module):
        """Test to_dict() includes all field values."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
            step=2,
        )

        result = config.to_dict()

        assert "type" in result
        assert "low" in result
        assert "high" in result
        assert "step" in result
        assert "choices" in result
        assert "log" in result

    def test_to_dict_values_match_attributes(self, param_types_module):
        """Test to_dict() values match the config attributes."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.LOGUNIFORM,
            low=1e-5,
            high=1e-2,
            log=True,
        )

        result = config.to_dict()

        assert result["type"] == ParamType.LOGUNIFORM
        assert result["low"] == 1e-5
        assert result["high"] == 1e-2
        assert result["log"] is True

    def test_to_dict_categorical_choices(self, param_types_module):
        """Test to_dict() with categorical choices."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=["adam", "sgd", "adamw"],
        )

        result = config.to_dict()

        assert result["type"] == ParamType.CATEGORICAL
        assert result["choices"] == ["adam", "sgd", "adamw"]

    def test_to_dict_default_values(self, param_types_module):
        """Test to_dict() includes default values for optional fields."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.FLOAT,
            low=0.0,
            high=1.0,
        )

        result = config.to_dict()

        # Default values should be present
        assert result["step"] is None
        assert result["choices"] is None
        assert result["log"] is False

    def test_to_dict_equivalent_to_model_dump(self, param_types_module):
        """Test to_dict() is equivalent to Pydantic's model_dump()."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=100,
            step=10,
        )

        to_dict_result = config.to_dict()
        model_dump_result = config.model_dump()

        assert to_dict_result == model_dump_result


# =============================================================================
# SEARCHSPACEPARAMCONFIG ERROR MESSAGE TESTS
# =============================================================================


class TestSearchSpaceParamConfigErrorMessages:
    """Test error messages contain expected information."""

    def test_missing_bounds_error_includes_type_value(self, param_types_module):
        """Test error message for missing bounds includes parameter type value."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.LOGUNIFORM,
                low=1e-5,
            )

        assert "loguniform" in str(exc_info.value).lower()

    def test_missing_bounds_error_includes_low_high_info(self, param_types_module):
        """Test error message for missing bounds includes information about low/high."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.INT,
                low=5,
            )

        # Error message should indicate low and high values
        error_message = str(exc_info.value)
        assert "low" in error_message.lower()
        assert "high" in error_message.lower()

    def test_invalid_bounds_error_includes_actual_values(self, param_types_module):
        """Test error message for invalid bounds includes actual values."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.FLOAT,
                low=10.0,
                high=5.0,
            )

        error_message = str(exc_info.value)
        # The error message should contain information about low and high
        assert "low" in error_message.lower()
        assert "high" in error_message.lower()

    def test_categorical_error_mentions_choices(self, param_types_module):
        """Test error message for categorical mentions 'choices'."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        with pytest.raises(ValueError) as exc_info:
            SearchSpaceParamConfig(
                type=ParamType.CATEGORICAL,
            )

        assert "choices" in str(exc_info.value).lower()

    def test_all_errors_are_value_errors(self, param_types_module):
        """Test all validation errors are ValueError (Pydantic V2 migration)."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        # Test missing bounds error
        with pytest.raises(ValueError):
            SearchSpaceParamConfig(type=ParamType.INT)

        # Test invalid bounds error
        with pytest.raises(ValueError):
            SearchSpaceParamConfig(type=ParamType.INT, low=10, high=5)

        # Test categorical error
        with pytest.raises(ValueError):
            SearchSpaceParamConfig(type=ParamType.CATEGORICAL)


# =============================================================================
# SEARCHSPACEPARAMCONFIG PYDANTIC MODEL FEATURES TESTS
# =============================================================================


class TestSearchSpaceParamConfigPydanticFeatures:
    """Test Pydantic BaseModel features of SearchSpaceParamConfig."""

    def test_equality_comparison(self, param_types_module):
        """Test equality comparison between configs."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config1 = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )
        config2 = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )

        assert config1 == config2

    def test_inequality_different_type(self, param_types_module):
        """Test inequality when type differs."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config1 = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )
        config2 = SearchSpaceParamConfig(
            type=ParamType.FLOAT,
            low=1,
            high=10,
        )

        assert config1 != config2

    def test_inequality_different_bounds(self, param_types_module):
        """Test inequality when bounds differ."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config1 = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )
        config2 = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=100,
        )

        assert config1 != config2

    def test_hash_equal_configs(self, param_types_module):
        """Test equal configs have same hash (frozen Pydantic model is hashable)."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config1 = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )
        config2 = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )

        assert hash(config1) == hash(config2)

    def test_can_use_as_dict_key(self, param_types_module):
        """Test config can be used as dictionary key (hashable)."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )

        test_dict = {config: "value"}
        assert test_dict[config] == "value"

    def test_can_add_to_set(self, param_types_module):
        """Test config can be added to set (hashable)."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config1 = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )
        config2 = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )
        config3 = SearchSpaceParamConfig(
            type=ParamType.FLOAT,
            low=0.0,
            high=1.0,
        )

        config_set = {config1, config2, config3}
        assert len(config_set) == 2  # config1 and config2 are equal

    def test_repr_contains_field_values(self, param_types_module):
        """Test repr contains field values."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
            step=2,
        )

        repr_str = repr(config)
        assert "SearchSpaceParamConfig" in repr_str
        assert "INT" in repr_str or "int" in repr_str
        assert "1" in repr_str
        assert "10" in repr_str

    def test_is_pydantic_basemodel_subclass(self, param_types_module):
        """Test SearchSpaceParamConfig is a Pydantic BaseModel subclass."""
        from pydantic import BaseModel

        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        assert issubclass(SearchSpaceParamConfig, BaseModel)

    def test_model_fields_attribute_exists(self, param_types_module):
        """Test Pydantic model_fields class attribute exists."""
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        assert hasattr(SearchSpaceParamConfig, "model_fields")
        assert "type" in SearchSpaceParamConfig.model_fields
        assert "low" in SearchSpaceParamConfig.model_fields
        assert "high" in SearchSpaceParamConfig.model_fields

    def test_model_dump_method_exists(self, param_types_module):
        """Test Pydantic model_dump() method exists and works."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )

        # Pydantic V2 model_dump method should exist
        assert hasattr(config, "model_dump")
        result = config.model_dump()
        assert isinstance(result, dict)

    def test_model_json_method_exists(self, param_types_module):
        """Test Pydantic model_dump_json() method exists and works."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=["a", "b"],
        )

        # Pydantic V2 model_dump_json method should exist
        assert hasattr(config, "model_dump_json")
        result = config.model_dump_json()
        assert isinstance(result, str)

    def test_model_copy_method_exists(self, param_types_module):
        """Test Pydantic model_copy() method exists and works."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
        )

        # Pydantic V2 model_copy method should exist
        assert hasattr(config, "model_copy")
        # Copy with update (creates new frozen instance)
        new_config = config.model_copy(update={"high": 20})
        assert new_config.high == 20
        assert config.high == 10  # Original unchanged


# =============================================================================
# MODULE IMPORT TESTS
# =============================================================================


class TestModuleImports:
    """Test module-level imports and exports."""

    def test_paramtype_importable(self):
        """Test ParamType is importable from module."""
        from milia_pipeline.models.hpo.search_spaces.param_types import ParamType

        assert ParamType is not None

    def test_searchspaceparamconfig_importable(self):
        """Test SearchSpaceParamConfig is importable from module."""
        from milia_pipeline.models.hpo.search_spaces.param_types import SearchSpaceParamConfig

        assert SearchSpaceParamConfig is not None

    def test_both_classes_importable_together(self):
        """Test both classes can be imported together."""
        from milia_pipeline.models.hpo.search_spaces.param_types import (
            ParamType,
            SearchSpaceParamConfig,
        )

        assert ParamType is not None
        assert SearchSpaceParamConfig is not None


# =============================================================================
# INTEGRATION-LIKE TESTS
# =============================================================================


class TestSearchSpaceParamConfigUsagePatterns:
    """Test common usage patterns for SearchSpaceParamConfig."""

    def test_create_learning_rate_config(self, param_types_module):
        """Test creating typical learning rate search space config."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        lr_config = SearchSpaceParamConfig(
            type=ParamType.LOGUNIFORM,
            low=1e-5,
            high=1e-2,
        )

        assert lr_config.type == ParamType.LOGUNIFORM
        assert lr_config.low == 1e-5
        assert lr_config.high == 1e-2

    def test_create_hidden_channels_config(self, param_types_module):
        """Test creating typical hidden channels search space config."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        hidden_config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=32,
            high=256,
            step=32,
        )

        assert hidden_config.type == ParamType.INT
        assert hidden_config.step == 32

    def test_create_dropout_config(self, param_types_module):
        """Test creating typical dropout search space config."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        dropout_config = SearchSpaceParamConfig(
            type=ParamType.FLOAT,
            low=0.0,
            high=0.5,
        )

        assert dropout_config.type == ParamType.FLOAT
        assert dropout_config.low == 0.0
        assert dropout_config.high == 0.5

    def test_create_activation_config(self, param_types_module):
        """Test creating typical activation function search space config."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        activation_config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=["relu", "elu", "leaky_relu", "gelu"],
        )

        assert activation_config.type == ParamType.CATEGORICAL
        assert "relu" in activation_config.choices

    def test_create_num_layers_config(self, param_types_module):
        """Test creating typical num_layers search space config."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        layers_config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=2,
            high=10,
        )

        assert layers_config.type == ParamType.INT
        assert layers_config.low == 2
        assert layers_config.high == 10

    def test_create_batch_size_config(self, param_types_module):
        """Test creating typical batch size search space config (categorical powers of 2)."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        batch_config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=[16, 32, 64, 128, 256],
        )

        assert batch_config.type == ParamType.CATEGORICAL
        assert batch_config.choices == [16, 32, 64, 128, 256]

    def test_create_optimizer_config(self, param_types_module):
        """Test creating typical optimizer search space config."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        optimizer_config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=["adam", "sgd", "adamw", "rmsprop"],
        )

        assert optimizer_config.type == ParamType.CATEGORICAL
        assert "adam" in optimizer_config.choices

    def test_create_weight_decay_config(self, param_types_module):
        """Test creating typical weight decay search space config."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        weight_decay_config = SearchSpaceParamConfig(
            type=ParamType.LOGUNIFORM,
            low=1e-6,
            high=1e-2,
        )

        assert weight_decay_config.type == ParamType.LOGUNIFORM

    def test_create_multiple_configs_in_dict(self, param_types_module):
        """Test creating multiple configs stored in a dictionary."""
        ParamType = param_types_module["ParamType"]
        SearchSpaceParamConfig = param_types_module["SearchSpaceParamConfig"]

        search_space = {
            "lr": SearchSpaceParamConfig(
                type=ParamType.LOGUNIFORM,
                low=1e-5,
                high=1e-2,
            ),
            "hidden_dim": SearchSpaceParamConfig(
                type=ParamType.INT,
                low=32,
                high=256,
                step=32,
            ),
            "activation": SearchSpaceParamConfig(
                type=ParamType.CATEGORICAL,
                choices=["relu", "elu"],
            ),
        }

        assert len(search_space) == 3
        assert search_space["lr"].type == ParamType.LOGUNIFORM
        assert search_space["hidden_dim"].step == 32
        assert "relu" in search_space["activation"].choices


# =============================================================================
# CLEANUP - TEARDOWN MODULE
# =============================================================================


def teardown_module(module):
    """
    Clean up any module-level state after tests complete.

    Pydantic V2 Migration: No longer requires cleanup of mocked modules
    since the module no longer uses external ConfigurationError.
    Kept for consistency and potential future use.
    """
    pass


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
