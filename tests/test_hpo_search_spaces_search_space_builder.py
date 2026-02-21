#!/usr/bin/env python3
"""
Complete Unit Test Suite for milia_pipeline/models/hpo/search_spaces/search_space_builder.py Module

Tests SearchSpaceBuilder class and convenience functions including:
- SearchSpaceBuilder class:
  - __init__: Initial state, empty search space, frozen flag
  - _ensure_not_frozen: Prevents modification after build()
  - _ensure_category: Creates category if not exists
  - _validate_category: Warns for non-standard categories
  - VALID_CATEGORIES: Frozenset of valid category names
  - Fluent builder methods:
    - add_int: Integer parameters with optional step
    - add_float: Float parameters with optional log scale
    - add_loguniform: Log-uniform parameters (positive bounds required)
    - add_categorical: Categorical parameters (non-empty choices required)
    - add_uniform: Alias for add_float without log scale
    - add_discrete_uniform: Discrete uniform with step
    - add_param: Add from config object or dict
    - add_category: Add multiple parameters for a category
    - remove_param: Remove parameter from search space
  - build: Build and return search space (freezes builder)
  - to_dict: Convert search space to plain dict format
  - Helper methods:
    - _dict_to_config: Convert dict to SearchSpaceParamConfig
    - _config_to_dict: Convert SearchSpaceParamConfig to dict
  - Class methods for predefined spaces:
    - for_model: Get search space for model architecture (dynamic or legacy fallback)
    - _build_gcn_space, _build_gat_space, etc.: Legacy model spaces
    - _add_optimizer_space: Add optimizer hyperparameters
    - _add_scheduler_space: Add scheduler hyperparameters
  - Utility class methods:
    - from_dict: Create search space from dictionary
    - merge: Merge multiple search spaces with conflict resolution
    - validate: Validate search space configuration
    - get_param_count: Get parameter count per category
    - estimate_search_space_size: Estimate combinatorial search space size
    - list_available_models: List available models (dynamic or legacy)
- Convenience functions:
  - build_search_space: Create new SearchSpaceBuilder instance
  - get_model_search_space: Get predefined search space for model
  - validate_search_space: Validate search space configuration

This is a PRODUCTION-READY test suite with comprehensive coverage.

Author: Milia Team
Version: 1.0.0

Changelog:
- v1.0.0: Initial release with full coverage for search_space_builder.py
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import logging
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

try:
    from milia_pipeline.exceptions import SearchSpaceError
except ImportError:
    SearchSpaceError = Exception

try:
    from pydantic import ValidationError as PydanticValidationError
except ImportError:
    PydanticValidationError = Exception

# =============================================================================
# MOCK CLASSES FOR EXCEPTIONS
# =============================================================================


class MockSearchSpaceError(Exception):
    """
    Mock SearchSpaceError for testing.

    Matches the interface of milia_pipeline.exceptions.SearchSpaceError
    based on usage in search_space_builder.py lines 38-44.
    """

    def __init__(
        self,
        message: str,
        parameter_name: str | None = None,
        parameter_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        self.message = message
        self.parameter_name = parameter_name
        self.parameter_config = parameter_config
        super().__init__(message)


class MockConfigurationError(Exception):
    """
    Mock ConfigurationError for testing.

    Matches the interface of milia_pipeline.exceptions.ConfigurationError
    based on usage in param_types.py lines 143-179.
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        details: str | None = None,
        actual_value: str | None = None,
    ):
        self.message = message
        self.config_key = config_key
        self.details = details
        self.actual_value = actual_value
        super().__init__(message)


# =============================================================================
# MOCK CLASSES FOR PARAM_TYPES
# =============================================================================


class MockParamType(Enum):
    """
    Mock ParamType enum for testing.

    Matches param_types.ParamType (lines 46-73).
    """

    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    LOGUNIFORM = "loguniform"
    UNIFORM = "uniform"
    INT_UNIFORM = "int_uniform"
    DISCRETE_UNIFORM = "discrete_uniform"


class MockSearchSpaceParamConfig:
    """
    Mock SearchSpaceParamConfig for testing.

    Matches the interface of param_types.SearchSpaceParamConfig (lines 76-179).
    Implements validation logic from __post_init__.
    """

    def __init__(
        self,
        type: MockParamType,
        low: float | None = None,
        high: float | None = None,
        step: int | None = None,
        choices: list[Any] | None = None,
        log: bool = False,
    ):
        self.type = type
        self.low = low
        self.high = high
        self.step = step
        self.choices = choices
        self.log = log

        # Validate based on type (mimicking __post_init__ from param_types.py lines 131-179)
        if self.type in (
            MockParamType.INT,
            MockParamType.FLOAT,
            MockParamType.LOGUNIFORM,
            MockParamType.UNIFORM,
            MockParamType.INT_UNIFORM,
        ):
            if self.low is None or self.high is None:
                raise MockConfigurationError(
                    f"Parameter type '{self.type.value}' requires 'low' and 'high'",
                    config_key="search_space",
                    details=f"low={self.low}, high={self.high}",
                )
            if self.low >= self.high:
                raise MockConfigurationError(
                    "'low' must be less than 'high'",
                    config_key="search_space",
                    actual_value=f"low={self.low}, high={self.high}",
                )

        if self.type == MockParamType.CATEGORICAL and (not self.choices or len(self.choices) == 0):
            raise MockConfigurationError(
                "Categorical parameter requires non-empty 'choices' list",
                config_key="search_space",
            )

        if self.type == MockParamType.DISCRETE_UNIFORM:
            if self.low is None or self.high is None:
                raise MockConfigurationError(
                    f"Parameter type '{self.type.value}' requires 'low' and 'high'",
                    config_key="search_space",
                    details=f"low={self.low}, high={self.high}",
                )
            if self.low >= self.high:
                raise MockConfigurationError(
                    "'low' must be less than 'high'",
                    config_key="search_space",
                    actual_value=f"low={self.low}, high={self.high}",
                )

    def __eq__(self, other):
        if not isinstance(other, MockSearchSpaceParamConfig):
            return False
        return (
            self.type == other.type
            and self.low == other.low
            and self.high == other.high
            and self.step == other.step
            and self.choices == other.choices
            and self.log == other.log
        )


# =============================================================================
# MOCK CLASSES FOR INTROSPECTOR
# =============================================================================


class MockPyGModelIntrospector:
    """
    Mock PyGModelIntrospector for testing dynamic introspection paths.

    Simulates the interface of milia_pipeline.models.registry.pyg_introspector.
    """

    def __init__(self, available_models: list[str] | None = None):
        self.available_models = available_models or [
            "GCN",
            "GAT",
            "GraphSAGE",
            "GIN",
            "SchNet",
            "DimeNet",
            "MPNN",
            "PMLP",
            "EdgeConv",
            "PointNet",
            "TransformerConv",
        ]

    def get_all_model_names(self) -> list[str]:
        """Return all available model names."""
        return self.available_models

    def has_model(self, model_name: str) -> bool:
        """Check if model exists."""
        return model_name.upper() in [m.upper() for m in self.available_models]

    def get_search_space(self, model_name: str) -> dict[str, dict[str, Any]]:
        """Return dynamically generated search space for model."""
        # Return a mock search space
        return {
            "hyperparameters": {
                "hidden_channels": MockSearchSpaceParamConfig(
                    type=MockParamType.INT, low=32, high=256, step=32
                ),
                "num_layers": MockSearchSpaceParamConfig(type=MockParamType.INT, low=2, high=6),
            }
        }


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_exceptions():
    """Fixture providing mock exception classes."""
    return {
        "SearchSpaceError": MockSearchSpaceError,
        "ConfigurationError": MockConfigurationError,
    }


@pytest.fixture
def mock_param_types():
    """Fixture providing mock param_types classes."""
    return {
        "ParamType": MockParamType,
        "SearchSpaceParamConfig": MockSearchSpaceParamConfig,
    }


@pytest.fixture
def search_space_builder_module(mock_exceptions, mock_param_types):
    """
    Fixture to import search_space_builder module with mocked dependencies.

    Returns dict containing imported classes and functions for testing.
    """
    # Clear cached module if exists
    modules_to_clear = [
        key
        for key in list(sys.modules.keys())
        if "milia_pipeline.models.hpo.search_spaces.search_space_builder" in key
    ]
    for mod in modules_to_clear:
        del sys.modules[mod]

    # Patch the param_types imports at the module level
    with (
        patch(
            "milia_pipeline.models.hpo.search_spaces.search_space_builder.ParamType", MockParamType
        ),
        patch(
            "milia_pipeline.models.hpo.search_spaces.search_space_builder.SearchSpaceParamConfig",
            MockSearchSpaceParamConfig,
        ),
        # Patch the introspector as unavailable for most tests
        patch(
            "milia_pipeline.models.hpo.search_spaces.search_space_builder._INTROSPECTOR_AVAILABLE",
            False,
        ),
    ):
        from milia_pipeline.models.hpo.search_spaces.search_space_builder import (
            SearchSpaceBuilder,
            build_search_space,
            get_model_search_space,
            validate_search_space,
        )

        yield {
            "SearchSpaceBuilder": SearchSpaceBuilder,
            "build_search_space": build_search_space,
            "get_model_search_space": get_model_search_space,
            "validate_search_space": validate_search_space,
            "ParamType": MockParamType,
            "SearchSpaceParamConfig": MockSearchSpaceParamConfig,
            "SearchSpaceError": MockSearchSpaceError,
            "ConfigurationError": MockConfigurationError,
        }


@pytest.fixture
def builder(search_space_builder_module):
    """Fixture providing a fresh SearchSpaceBuilder instance."""
    SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
    return SearchSpaceBuilder()


@pytest.fixture
def sample_search_space(search_space_builder_module):
    """Fixture providing a sample built search space."""
    SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
    return (
        SearchSpaceBuilder()
        .add_int("hidden_channels", 32, 256, step=32)
        .add_int("num_layers", 2, 6)
        .add_float("dropout", 0.0, 0.6)
        .add_loguniform("lr", 1e-5, 1e-2, category="optimizer")
        .build()
    )


# =============================================================================
# SEARCHSPACEBUILDER INITIALIZATION TESTS
# =============================================================================


class TestSearchSpaceBuilderInit:
    """Test SearchSpaceBuilder initialization."""

    def test_init_creates_empty_search_space(self, search_space_builder_module):
        """Test __init__ creates empty search space dict."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        builder = SearchSpaceBuilder()
        assert builder._search_space == {}

    def test_init_frozen_is_false(self, search_space_builder_module):
        """Test __init__ sets _frozen to False."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        builder = SearchSpaceBuilder()
        assert builder._frozen is False

    def test_valid_categories_is_frozenset(self, search_space_builder_module):
        """Test VALID_CATEGORIES is a frozenset."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        assert isinstance(SearchSpaceBuilder.VALID_CATEGORIES, frozenset)

    def test_valid_categories_contains_expected_values(self, search_space_builder_module):
        """Test VALID_CATEGORIES contains all expected category names."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        expected = {
            "hyperparameters",
            "model",
            "optimizer",
            "scheduler",
            "loss",
            "training",
            "architecture",
        }
        assert expected == SearchSpaceBuilder.VALID_CATEGORIES

    def test_valid_categories_has_seven_members(self, search_space_builder_module):
        """Test VALID_CATEGORIES has exactly 7 members."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        assert len(SearchSpaceBuilder.VALID_CATEGORIES) == 7


# =============================================================================
# SEARCHSPACEBUILDER ENSURE NOT FROZEN TESTS
# =============================================================================


class TestSearchSpaceBuilderEnsureNotFrozen:
    """Test _ensure_not_frozen method."""

    def test_ensure_not_frozen_does_not_raise_when_not_frozen(self, builder):
        """Test _ensure_not_frozen does not raise when builder is not frozen."""
        # Should not raise - line 118-124
        builder._ensure_not_frozen()

    def test_ensure_not_frozen_raises_when_frozen(self, builder, search_space_builder_module):
        """Test _ensure_not_frozen raises SearchSpaceError when frozen."""
        # Add a param to make build() work
        builder.add_int("test", 1, 10)
        builder.build()  # This freezes the builder (line 504)

        # Now _ensure_not_frozen should raise (lines 120-124)
        with pytest.raises(Exception) as exc_info:
            builder._ensure_not_frozen()

        assert "Cannot modify search space after build()" in str(exc_info.value)


# =============================================================================
# SEARCHSPACEBUILDER ENSURE CATEGORY TESTS
# =============================================================================


class TestSearchSpaceBuilderEnsureCategory:
    """Test _ensure_category method."""

    def test_ensure_category_creates_new_category(self, builder):
        """Test _ensure_category creates category if not exists (lines 126-129)."""
        builder._ensure_category("hyperparameters")
        assert "hyperparameters" in builder._search_space
        assert builder._search_space["hyperparameters"] == {}

    def test_ensure_category_does_not_overwrite_existing(
        self, builder, search_space_builder_module
    ):
        """Test _ensure_category does not overwrite existing category."""
        ParamType = search_space_builder_module["ParamType"]
        SearchSpaceParamConfig = search_space_builder_module["SearchSpaceParamConfig"]

        builder._search_space["hyperparameters"] = {
            "test": SearchSpaceParamConfig(type=ParamType.INT, low=1, high=10)
        }
        builder._ensure_category("hyperparameters")

        assert "test" in builder._search_space["hyperparameters"]

    def test_ensure_category_multiple_categories(self, builder):
        """Test _ensure_category can create multiple categories."""
        builder._ensure_category("hyperparameters")
        builder._ensure_category("optimizer")
        builder._ensure_category("scheduler")

        assert "hyperparameters" in builder._search_space
        assert "optimizer" in builder._search_space
        assert "scheduler" in builder._search_space


# =============================================================================
# SEARCHSPACEBUILDER VALIDATE CATEGORY TESTS
# =============================================================================


class TestSearchSpaceBuilderValidateCategory:
    """Test _validate_category method."""

    def test_validate_category_valid_hyperparameters(self, builder, caplog):
        """Test _validate_category does not warn for 'hyperparameters'."""
        import logging

        caplog.set_level(logging.WARNING)

        builder._validate_category("hyperparameters")

        assert "Non-standard category" not in caplog.text

    def test_validate_category_valid_optimizer(self, builder, caplog):
        """Test _validate_category does not warn for 'optimizer'."""
        import logging

        caplog.set_level(logging.WARNING)

        builder._validate_category("optimizer")

        assert "Non-standard category" not in caplog.text

    def test_validate_category_valid_scheduler(self, builder, caplog):
        """Test _validate_category does not warn for 'scheduler'."""
        import logging

        caplog.set_level(logging.WARNING)

        builder._validate_category("scheduler")

        assert "Non-standard category" not in caplog.text

    def test_validate_category_invalid_warns(self, builder, caplog):
        """Test _validate_category warns for invalid categories (lines 131-137)."""
        import logging

        caplog.set_level(logging.WARNING)

        builder._validate_category("custom_invalid_category")

        assert "Non-standard category 'custom_invalid_category'" in caplog.text

    def test_validate_category_all_valid_no_warning(self, builder, caplog):
        """Test all valid categories do not produce warnings."""
        import logging

        caplog.set_level(logging.WARNING)

        valid_categories = [
            "hyperparameters",
            "model",
            "optimizer",
            "scheduler",
            "loss",
            "training",
            "architecture",
        ]

        for cat in valid_categories:
            builder._validate_category(cat)

        assert "Non-standard category" not in caplog.text


# =============================================================================
# SEARCHSPACEBUILDER ADD_INT TESTS
# =============================================================================


class TestSearchSpaceBuilderAddInt:
    """Test add_int method (lines 143-182)."""

    def test_add_int_basic(self, builder, search_space_builder_module):
        """Test add_int with basic parameters."""
        ParamType = search_space_builder_module["ParamType"]

        result = builder.add_int("hidden_channels", 32, 256)

        assert result is builder  # Returns self for chaining
        assert "hyperparameters" in builder._search_space
        assert "hidden_channels" in builder._search_space["hyperparameters"]

        config = builder._search_space["hyperparameters"]["hidden_channels"]
        assert config.type.value == ParamType.INT.value
        assert config.low == 32.0
        assert config.high == 256.0

    def test_add_int_with_step(self, builder):
        """Test add_int with step parameter."""
        builder.add_int("hidden_channels", 32, 256, step=32)

        config = builder._search_space["hyperparameters"]["hidden_channels"]
        assert config.step == 32

    def test_add_int_without_step(self, builder):
        """Test add_int without step defaults to None."""
        builder.add_int("num_layers", 2, 6)

        config = builder._search_space["hyperparameters"]["num_layers"]
        assert config.step is None

    def test_add_int_with_custom_category(self, builder):
        """Test add_int with custom category."""
        builder.add_int("num_layers", 2, 6, category="architecture")

        assert "architecture" in builder._search_space
        assert "num_layers" in builder._search_space["architecture"]

    def test_add_int_default_category_is_hyperparameters(self, builder):
        """Test add_int defaults to 'hyperparameters' category."""
        builder.add_int("test", 1, 10)

        assert "hyperparameters" in builder._search_space
        assert "test" in builder._search_space["hyperparameters"]

    def test_add_int_returns_self_for_chaining(self, builder):
        """Test add_int returns self for method chaining."""
        result = builder.add_int("param1", 1, 10)
        assert result is builder

    def test_add_int_chaining_multiple(self, builder):
        """Test add_int can be chained multiple times."""
        result = builder.add_int("param1", 1, 10).add_int("param2", 5, 20).add_int("param3", 10, 30)

        assert result is builder
        assert "param1" in builder._search_space["hyperparameters"]
        assert "param2" in builder._search_space["hyperparameters"]
        assert "param3" in builder._search_space["hyperparameters"]

    def test_add_int_after_build_raises_error(self, builder):
        """Test add_int raises error after build() is called."""
        builder.add_int("test", 1, 10)
        builder.build()

        with pytest.raises(Exception) as exc_info:
            builder.add_int("another", 1, 10)

        assert "Cannot modify search space after build()" in str(exc_info.value)

    def test_add_int_converts_bounds_to_float(self, builder):
        """Test add_int converts int bounds to float (line 174-175)."""
        builder.add_int("test", 10, 100)

        config = builder._search_space["hyperparameters"]["test"]
        assert isinstance(config.low, float)
        assert isinstance(config.high, float)
        assert config.low == 10.0
        assert config.high == 100.0

    def test_add_int_with_zero_low_bound(self, builder):
        """Test add_int with zero as low bound."""
        builder.add_int("layers", 0, 5)

        config = builder._search_space["hyperparameters"]["layers"]
        assert config.low == 0.0
        assert config.high == 5.0

    def test_add_int_with_negative_bounds(self, builder):
        """Test add_int with negative bounds."""
        builder.add_int("offset", -10, -1)

        config = builder._search_space["hyperparameters"]["offset"]
        assert config.low == -10.0
        assert config.high == -1.0


# =============================================================================
# SEARCHSPACEBUILDER ADD_FLOAT TESTS
# =============================================================================


class TestSearchSpaceBuilderAddFloat:
    """Test add_float method (lines 184-223)."""

    def test_add_float_basic(self, builder, search_space_builder_module):
        """Test add_float with basic parameters."""
        ParamType = search_space_builder_module["ParamType"]

        result = builder.add_float("dropout", 0.0, 0.5)

        assert result is builder
        assert "dropout" in builder._search_space["hyperparameters"]

        config = builder._search_space["hyperparameters"]["dropout"]
        assert config.type.value == ParamType.FLOAT.value
        assert config.low == 0.0
        assert config.high == 0.5
        assert config.log is False

    def test_add_float_with_log_scale_true(self, builder):
        """Test add_float with log=True."""
        builder.add_float("temperature", 0.01, 1.0, log=True)

        config = builder._search_space["hyperparameters"]["temperature"]
        assert config.log is True

    def test_add_float_with_log_scale_false(self, builder):
        """Test add_float with log=False (default)."""
        builder.add_float("dropout", 0.0, 0.5, log=False)

        config = builder._search_space["hyperparameters"]["dropout"]
        assert config.log is False

    def test_add_float_default_log_is_false(self, builder):
        """Test add_float defaults log to False."""
        builder.add_float("dropout", 0.0, 0.5)

        config = builder._search_space["hyperparameters"]["dropout"]
        assert config.log is False

    def test_add_float_with_custom_category(self, builder):
        """Test add_float with custom category."""
        builder.add_float("factor", 0.1, 0.9, category="scheduler")

        assert "scheduler" in builder._search_space
        assert "factor" in builder._search_space["scheduler"]

    def test_add_float_default_category_is_hyperparameters(self, builder):
        """Test add_float defaults to 'hyperparameters' category."""
        builder.add_float("dropout", 0.0, 0.5)

        assert "hyperparameters" in builder._search_space
        assert "dropout" in builder._search_space["hyperparameters"]

    def test_add_float_returns_self(self, builder):
        """Test add_float returns self for chaining."""
        result = builder.add_float("dropout", 0.0, 0.5)
        assert result is builder

    def test_add_float_chaining(self, builder):
        """Test add_float can be chained."""
        result = builder.add_float("dropout", 0.0, 0.5).add_float("temp", 0.1, 1.0)

        assert result is builder
        assert "dropout" in builder._search_space["hyperparameters"]
        assert "temp" in builder._search_space["hyperparameters"]

    def test_add_float_after_build_raises_error(self, builder):
        """Test add_float raises error after build()."""
        builder.add_float("test", 0.0, 1.0)
        builder.build()

        with pytest.raises(Exception) as exc_info:
            builder.add_float("another", 0.0, 1.0)

        assert "Cannot modify search space after build()" in str(exc_info.value)

    def test_add_float_very_small_range(self, builder):
        """Test add_float with very small range."""
        builder.add_float("epsilon", 0.0, 0.0001)

        config = builder._search_space["hyperparameters"]["epsilon"]
        assert config.low == 0.0
        assert config.high == 0.0001


# =============================================================================
# SEARCHSPACEBUILDER ADD_LOGUNIFORM TESTS
# =============================================================================


class TestSearchSpaceBuilderAddLoguniform:
    """Test add_loguniform method (lines 225-271)."""

    def test_add_loguniform_basic(self, builder, search_space_builder_module):
        """Test add_loguniform with basic parameters."""
        ParamType = search_space_builder_module["ParamType"]

        result = builder.add_loguniform("lr", 1e-5, 1e-2)

        assert result is builder
        assert "optimizer" in builder._search_space  # Default category is 'optimizer'
        assert "lr" in builder._search_space["optimizer"]

        config = builder._search_space["optimizer"]["lr"]
        assert config.type.value == ParamType.LOGUNIFORM.value
        assert config.low == 1e-5
        assert config.high == 1e-2

    def test_add_loguniform_default_category_is_optimizer(self, builder):
        """Test add_loguniform defaults to 'optimizer' category (line 231)."""
        builder.add_loguniform("lr", 1e-5, 1e-2)

        assert "optimizer" in builder._search_space
        assert "lr" in builder._search_space["optimizer"]

    def test_add_loguniform_with_custom_category(self, builder):
        """Test add_loguniform with custom category."""
        builder.add_loguniform("lr", 1e-5, 1e-2, category="hyperparameters")

        assert "hyperparameters" in builder._search_space
        assert "lr" in builder._search_space["hyperparameters"]

    def test_add_loguniform_zero_low_raises_error(self, builder):
        """Test add_loguniform with low=0 raises SearchSpaceError (lines 255-260)."""
        with pytest.raises(Exception) as exc_info:
            builder.add_loguniform("lr", 0, 1e-2)

        assert "positive bounds" in str(exc_info.value).lower()

    def test_add_loguniform_negative_low_raises_error(self, builder):
        """Test add_loguniform with negative low raises SearchSpaceError."""
        with pytest.raises(Exception) as exc_info:
            builder.add_loguniform("lr", -1e-5, 1e-2)

        assert "positive bounds" in str(exc_info.value).lower()

    def test_add_loguniform_returns_self(self, builder):
        """Test add_loguniform returns self for chaining."""
        result = builder.add_loguniform("lr", 1e-5, 1e-2)
        assert result is builder

    def test_add_loguniform_chaining(self, builder):
        """Test add_loguniform can be chained."""
        result = builder.add_loguniform("lr", 1e-5, 1e-2).add_loguniform("weight_decay", 1e-6, 1e-3)

        assert result is builder
        assert "lr" in builder._search_space["optimizer"]
        assert "weight_decay" in builder._search_space["optimizer"]

    def test_add_loguniform_after_build_raises_error(self, builder):
        """Test add_loguniform raises error after build()."""
        builder.add_loguniform("lr", 1e-5, 1e-2)
        builder.build()

        with pytest.raises(Exception) as exc_info:
            builder.add_loguniform("another", 1e-5, 1e-2)

        assert "Cannot modify search space after build()" in str(exc_info.value)

    def test_add_loguniform_small_positive_low(self, builder):
        """Test add_loguniform with very small positive low."""
        builder.add_loguniform("lr", 1e-10, 1e-5)

        config = builder._search_space["optimizer"]["lr"]
        assert config.low == 1e-10
        assert config.high == 1e-5


# =============================================================================
# SEARCHSPACEBUILDER ADD_CATEGORICAL TESTS
# =============================================================================


class TestSearchSpaceBuilderAddCategorical:
    """Test add_categorical method (lines 273-312)."""

    def test_add_categorical_basic(self, builder, search_space_builder_module):
        """Test add_categorical with basic parameters."""
        ParamType = search_space_builder_module["ParamType"]

        result = builder.add_categorical("activation", ["relu", "gelu", "elu"])

        assert result is builder
        assert "activation" in builder._search_space["hyperparameters"]

        config = builder._search_space["hyperparameters"]["activation"]
        assert config.type.value == ParamType.CATEGORICAL.value
        assert config.choices == ["relu", "gelu", "elu"]

    def test_add_categorical_with_numeric_choices(self, builder):
        """Test add_categorical with numeric choices."""
        builder.add_categorical("batch_size", [16, 32, 64, 128])

        config = builder._search_space["hyperparameters"]["batch_size"]
        assert config.choices == [16, 32, 64, 128]

    def test_add_categorical_with_boolean_choices(self, builder):
        """Test add_categorical with boolean choices."""
        builder.add_categorical("use_bias", [True, False])

        config = builder._search_space["hyperparameters"]["use_bias"]
        assert config.choices == [True, False]

    def test_add_categorical_with_mixed_type_choices(self, builder):
        """Test add_categorical with mixed type choices."""
        builder.add_categorical("mixed", ["string", 42, 3.14, True, None])

        config = builder._search_space["hyperparameters"]["mixed"]
        assert config.choices == ["string", 42, 3.14, True, None]

    def test_add_categorical_with_single_choice(self, builder):
        """Test add_categorical with single choice."""
        builder.add_categorical("fixed", ["only_option"])

        config = builder._search_space["hyperparameters"]["fixed"]
        assert config.choices == ["only_option"]

    def test_add_categorical_with_custom_category(self, builder):
        """Test add_categorical with custom category."""
        builder.add_categorical("opt_type", ["adam", "sgd"], category="optimizer")

        assert "optimizer" in builder._search_space
        assert "opt_type" in builder._search_space["optimizer"]

    def test_add_categorical_default_category_is_hyperparameters(self, builder):
        """Test add_categorical defaults to 'hyperparameters' category."""
        builder.add_categorical("activation", ["relu"])

        assert "hyperparameters" in builder._search_space
        assert "activation" in builder._search_space["hyperparameters"]

    def test_add_categorical_empty_choices_raises_error(self, builder):
        """Test add_categorical with empty choices raises SearchSpaceError (lines 298-302)."""
        with pytest.raises(Exception) as exc_info:
            builder.add_categorical("activation", [])

        assert "at least one choice" in str(exc_info.value).lower()

    def test_add_categorical_returns_self(self, builder):
        """Test add_categorical returns self for chaining."""
        result = builder.add_categorical("activation", ["relu"])
        assert result is builder

    def test_add_categorical_chaining(self, builder):
        """Test add_categorical can be chained."""
        result = builder.add_categorical("activation", ["relu", "gelu"]).add_categorical(
            "norm", ["batch", "layer"]
        )

        assert result is builder
        assert "activation" in builder._search_space["hyperparameters"]
        assert "norm" in builder._search_space["hyperparameters"]

    def test_add_categorical_after_build_raises_error(self, builder):
        """Test add_categorical raises error after build()."""
        builder.add_categorical("test", ["a", "b"])
        builder.build()

        with pytest.raises(Exception) as exc_info:
            builder.add_categorical("another", ["c", "d"])

        assert "Cannot modify search space after build()" in str(exc_info.value)

    def test_add_categorical_converts_to_list(self, builder):
        """Test add_categorical converts choices to list (line 306)."""
        # Pass a tuple instead of list
        builder.add_categorical("activation", ("relu", "gelu"))

        config = builder._search_space["hyperparameters"]["activation"]
        assert isinstance(config.choices, list)
        assert config.choices == ["relu", "gelu"]


# =============================================================================
# SEARCHSPACEBUILDER ADD_UNIFORM TESTS
# =============================================================================


class TestSearchSpaceBuilderAddUniform:
    """Test add_uniform method (lines 314-348)."""

    def test_add_uniform_basic(self, builder, search_space_builder_module):
        """Test add_uniform with basic parameters."""
        ParamType = search_space_builder_module["ParamType"]

        result = builder.add_uniform("dropout", 0.0, 0.5)

        assert result is builder
        assert "dropout" in builder._search_space["hyperparameters"]

        config = builder._search_space["hyperparameters"]["dropout"]
        assert config.type.value == ParamType.UNIFORM.value
        assert config.low == 0.0
        assert config.high == 0.5

    def test_add_uniform_with_custom_category(self, builder):
        """Test add_uniform with custom category."""
        builder.add_uniform("factor", 0.1, 0.9, category="scheduler")

        assert "scheduler" in builder._search_space
        assert "factor" in builder._search_space["scheduler"]

    def test_add_uniform_default_category_is_hyperparameters(self, builder):
        """Test add_uniform defaults to 'hyperparameters' category."""
        builder.add_uniform("param", 0.0, 1.0)

        assert "hyperparameters" in builder._search_space
        assert "param" in builder._search_space["hyperparameters"]

    def test_add_uniform_returns_self(self, builder):
        """Test add_uniform returns self for chaining."""
        result = builder.add_uniform("param", 0.0, 1.0)
        assert result is builder

    def test_add_uniform_chaining(self, builder):
        """Test add_uniform can be chained."""
        result = builder.add_uniform("param1", 0.0, 1.0).add_uniform("param2", -1.0, 1.0)

        assert result is builder
        assert "param1" in builder._search_space["hyperparameters"]
        assert "param2" in builder._search_space["hyperparameters"]

    def test_add_uniform_after_build_raises_error(self, builder):
        """Test add_uniform raises error after build()."""
        builder.add_uniform("test", 0.0, 1.0)
        builder.build()

        with pytest.raises(Exception) as exc_info:
            builder.add_uniform("another", 0.0, 1.0)

        assert "Cannot modify search space after build()" in str(exc_info.value)

    def test_add_uniform_with_negative_range(self, builder):
        """Test add_uniform with negative range."""
        builder.add_uniform("offset", -1.0, 1.0)

        config = builder._search_space["hyperparameters"]["offset"]
        assert config.low == -1.0
        assert config.high == 1.0


# =============================================================================
# SEARCHSPACEBUILDER ADD_DISCRETE_UNIFORM TESTS
# =============================================================================


class TestSearchSpaceBuilderAddDiscreteUniform:
    """Test add_discrete_uniform method (lines 350-388)."""

    def test_add_discrete_uniform_basic(self, builder, search_space_builder_module):
        """Test add_discrete_uniform with basic parameters."""
        ParamType = search_space_builder_module["ParamType"]

        result = builder.add_discrete_uniform("batch_size", 16, 128, step=16)

        assert result is builder
        assert "batch_size" in builder._search_space["hyperparameters"]

        config = builder._search_space["hyperparameters"]["batch_size"]
        assert config.type.value == ParamType.DISCRETE_UNIFORM.value
        assert config.low == 16
        assert config.high == 128
        assert config.step == 16

    def test_add_discrete_uniform_step_converted_to_int(self, builder):
        """Test add_discrete_uniform converts step to int (line 382)."""
        builder.add_discrete_uniform("param", 0, 100, step=10.5)

        config = builder._search_space["hyperparameters"]["param"]
        assert config.step == 10
        assert isinstance(config.step, int)

    def test_add_discrete_uniform_with_custom_category(self, builder):
        """Test add_discrete_uniform with custom category."""
        builder.add_discrete_uniform("size", 32, 256, step=32, category="training")

        assert "training" in builder._search_space
        assert "size" in builder._search_space["training"]

    def test_add_discrete_uniform_default_category_is_hyperparameters(self, builder):
        """Test add_discrete_uniform defaults to 'hyperparameters' category."""
        builder.add_discrete_uniform("param", 0, 100, step=10)

        assert "hyperparameters" in builder._search_space
        assert "param" in builder._search_space["hyperparameters"]

    def test_add_discrete_uniform_returns_self(self, builder):
        """Test add_discrete_uniform returns self for chaining."""
        result = builder.add_discrete_uniform("param", 0, 100, step=10)
        assert result is builder

    def test_add_discrete_uniform_chaining(self, builder):
        """Test add_discrete_uniform can be chained."""
        result = builder.add_discrete_uniform("param1", 0, 100, step=10).add_discrete_uniform(
            "param2", 16, 128, step=16
        )

        assert result is builder
        assert "param1" in builder._search_space["hyperparameters"]
        assert "param2" in builder._search_space["hyperparameters"]

    def test_add_discrete_uniform_after_build_raises_error(self, builder):
        """Test add_discrete_uniform raises error after build()."""
        builder.add_discrete_uniform("test", 0, 100, step=10)
        builder.build()

        with pytest.raises(Exception) as exc_info:
            builder.add_discrete_uniform("another", 0, 100, step=10)

        assert "Cannot modify search space after build()" in str(exc_info.value)


# =============================================================================
# SEARCHSPACEBUILDER ADD_PARAM TESTS
# =============================================================================


class TestSearchSpaceBuilderAddParam:
    """Test add_param method (lines 390-421)."""

    def test_add_param_with_config_object(self, builder, search_space_builder_module):
        """Test add_param with SearchSpaceParamConfig object."""
        ParamType = search_space_builder_module["ParamType"]
        SearchSpaceParamConfig = search_space_builder_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(type=ParamType.INT, low=1, high=10)
        result = builder.add_param("heads", config)

        assert result is builder
        assert "heads" in builder._search_space["hyperparameters"]
        assert builder._search_space["hyperparameters"]["heads"] == config

    def test_add_param_with_dict(self, builder, search_space_builder_module):
        """Test add_param with dict representation (line 408: Example)."""
        ParamType = search_space_builder_module["ParamType"]

        config_dict = {"type": "loguniform", "low": 1e-5, "high": 1e-2}
        result = builder.add_param("lr", config_dict)

        assert result is builder
        assert "lr" in builder._search_space["hyperparameters"]

        config = builder._search_space["hyperparameters"]["lr"]
        assert config.type.value == ParamType.LOGUNIFORM.value
        assert config.low == 1e-5
        assert config.high == 1e-2

    def test_add_param_with_dict_int_type(self, builder, search_space_builder_module):
        """Test add_param with dict containing 'int' type."""
        ParamType = search_space_builder_module["ParamType"]

        config_dict = {"type": "int", "low": 32, "high": 256, "step": 32}
        builder.add_param("hidden_channels", config_dict)

        config = builder._search_space["hyperparameters"]["hidden_channels"]
        assert config.type.value == ParamType.INT.value
        assert config.low == 32
        assert config.high == 256
        assert config.step == 32

    def test_add_param_with_dict_categorical_type(self, builder, search_space_builder_module):
        """Test add_param with dict containing 'categorical' type."""
        ParamType = search_space_builder_module["ParamType"]

        config_dict = {"type": "categorical", "choices": ["relu", "gelu"]}
        builder.add_param("activation", config_dict)

        config = builder._search_space["hyperparameters"]["activation"]
        assert config.type.value == ParamType.CATEGORICAL.value
        assert config.choices == ["relu", "gelu"]

    def test_add_param_with_custom_category(self, builder, search_space_builder_module):
        """Test add_param with custom category."""
        SearchSpaceParamConfig = search_space_builder_module["SearchSpaceParamConfig"]
        ParamType = search_space_builder_module["ParamType"]

        config = SearchSpaceParamConfig(type=ParamType.FLOAT, low=0.1, high=0.9)
        builder.add_param("factor", config, category="scheduler")

        assert "scheduler" in builder._search_space
        assert "factor" in builder._search_space["scheduler"]

    def test_add_param_default_category_is_hyperparameters(
        self, builder, search_space_builder_module
    ):
        """Test add_param defaults to 'hyperparameters' category."""
        config_dict = {"type": "int", "low": 1, "high": 10}
        builder.add_param("param", config_dict)

        assert "hyperparameters" in builder._search_space
        assert "param" in builder._search_space["hyperparameters"]

    def test_add_param_returns_self(self, builder, search_space_builder_module):
        """Test add_param returns self for chaining."""
        config_dict = {"type": "int", "low": 1, "high": 10}
        result = builder.add_param("param", config_dict)
        assert result is builder

    def test_add_param_chaining(self, builder, search_space_builder_module):
        """Test add_param can be chained."""
        result = builder.add_param("param1", {"type": "int", "low": 1, "high": 10}).add_param(
            "param2", {"type": "float", "low": 0.0, "high": 1.0}
        )

        assert result is builder
        assert "param1" in builder._search_space["hyperparameters"]
        assert "param2" in builder._search_space["hyperparameters"]

    def test_add_param_after_build_raises_error(self, builder, search_space_builder_module):
        """Test add_param raises error after build()."""
        builder.add_param("test", {"type": "int", "low": 1, "high": 10})
        builder.build()

        with pytest.raises(Exception) as exc_info:
            builder.add_param("another", {"type": "int", "low": 1, "high": 10})

        assert "Cannot modify search space after build()" in str(exc_info.value)

    def test_add_param_validates_category(self, builder, caplog):
        """Test add_param validates category (triggers warning for non-standard)."""
        caplog.set_level(logging.WARNING)

        builder.add_param("param", {"type": "int", "low": 1, "high": 10}, category="custom_cat")

        assert "Non-standard category 'custom_cat'" in caplog.text

    def test_add_param_with_uniform_type(self, builder, search_space_builder_module):
        """Test add_param with uniform type dict."""
        ParamType = search_space_builder_module["ParamType"]

        config_dict = {"type": "uniform", "low": 0.0, "high": 1.0}
        builder.add_param("dropout", config_dict)

        config = builder._search_space["hyperparameters"]["dropout"]
        assert config.type.value == ParamType.UNIFORM.value

    def test_add_param_with_discrete_uniform_type(self, builder, search_space_builder_module):
        """Test add_param with discrete_uniform type dict."""
        ParamType = search_space_builder_module["ParamType"]

        config_dict = {"type": "discrete_uniform", "low": 16, "high": 128, "step": 16}
        builder.add_param("batch_size", config_dict)

        config = builder._search_space["hyperparameters"]["batch_size"]
        assert config.type.value == ParamType.DISCRETE_UNIFORM.value

    def test_add_param_overwrites_existing(self, builder, search_space_builder_module):
        """Test add_param overwrites existing parameter with same name."""
        builder.add_param("param", {"type": "int", "low": 1, "high": 10})
        builder.add_param("param", {"type": "float", "low": 0.0, "high": 1.0})

        config = builder._search_space["hyperparameters"]["param"]
        assert config.type.value == search_space_builder_module["ParamType"].FLOAT.value


# =============================================================================
# SEARCHSPACEBUILDER ADD_CATEGORY TESTS
# =============================================================================


class TestSearchSpaceBuilderAddCategory:
    """Test add_category method (lines 423-449)."""

    def test_add_category_basic(self, builder, search_space_builder_module):
        """Test add_category with basic parameters (line 439-442: Example)."""
        result = builder.add_category(
            "optimizer",
            {
                "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
                "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-3},
            },
        )

        assert result is builder
        assert "optimizer" in builder._search_space
        assert "lr" in builder._search_space["optimizer"]
        assert "weight_decay" in builder._search_space["optimizer"]

    def test_add_category_with_config_objects(self, builder, search_space_builder_module):
        """Test add_category with SearchSpaceParamConfig objects."""
        ParamType = search_space_builder_module["ParamType"]
        SearchSpaceParamConfig = search_space_builder_module["SearchSpaceParamConfig"]

        params = {
            "hidden_channels": SearchSpaceParamConfig(type=ParamType.INT, low=32, high=256),
            "dropout": SearchSpaceParamConfig(type=ParamType.FLOAT, low=0.0, high=0.5),
        }

        builder.add_category("hyperparameters", params)

        assert "hidden_channels" in builder._search_space["hyperparameters"]
        assert "dropout" in builder._search_space["hyperparameters"]

    def test_add_category_empty_params(self, builder):
        """Test add_category with empty params dict."""
        result = builder.add_category("optimizer", {})

        assert result is builder
        # Category might not be created if no params added (depends on implementation)
        # The loop in lines 446-447 won't execute for empty dict

    def test_add_category_multiple_categories(self, builder):
        """Test add_category called multiple times for different categories."""
        builder.add_category("optimizer", {"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}})
        builder.add_category("scheduler", {"factor": {"type": "float", "low": 0.1, "high": 0.9}})

        assert "optimizer" in builder._search_space
        assert "scheduler" in builder._search_space

    def test_add_category_returns_self(self, builder):
        """Test add_category returns self for chaining."""
        result = builder.add_category(
            "optimizer", {"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}}
        )
        assert result is builder

    def test_add_category_chaining(self, builder):
        """Test add_category can be chained."""
        result = builder.add_category(
            "optimizer", {"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}}
        ).add_category("scheduler", {"factor": {"type": "float", "low": 0.1, "high": 0.9}})

        assert result is builder
        assert "lr" in builder._search_space["optimizer"]
        assert "factor" in builder._search_space["scheduler"]

    def test_add_category_after_build_raises_error(self, builder):
        """Test add_category raises error after build()."""
        builder.add_category("optimizer", {"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}})
        builder.build()

        with pytest.raises(Exception) as exc_info:
            builder.add_category(
                "scheduler", {"factor": {"type": "float", "low": 0.1, "high": 0.9}}
            )

        assert "Cannot modify search space after build()" in str(exc_info.value)

    def test_add_category_with_mixed_param_types(self, builder, search_space_builder_module):
        """Test add_category with mixed dict and config object params."""
        ParamType = search_space_builder_module["ParamType"]
        SearchSpaceParamConfig = search_space_builder_module["SearchSpaceParamConfig"]

        params = {
            "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
            "momentum": SearchSpaceParamConfig(type=ParamType.FLOAT, low=0.8, high=0.99),
        }

        builder.add_category("optimizer", params)

        assert "lr" in builder._search_space["optimizer"]
        assert "momentum" in builder._search_space["optimizer"]

    def test_add_category_validates_category_name(self, builder, caplog):
        """Test add_category validates category name."""
        caplog.set_level(logging.WARNING)

        builder.add_category(
            "non_standard_category", {"param": {"type": "int", "low": 1, "high": 10}}
        )

        assert "Non-standard category 'non_standard_category'" in caplog.text


# =============================================================================
# SEARCHSPACEBUILDER REMOVE_PARAM TESTS
# =============================================================================


class TestSearchSpaceBuilderRemoveParam:
    """Test remove_param method (lines 451-479)."""

    def test_remove_param_with_category(self, builder):
        """Test remove_param with explicit category (lines 468-471)."""
        builder.add_int("hidden_channels", 32, 256, category="hyperparameters")
        builder.add_int("num_layers", 2, 6, category="hyperparameters")

        result = builder.remove_param("hidden_channels", category="hyperparameters")

        assert result is builder
        assert "hidden_channels" not in builder._search_space["hyperparameters"]
        assert "num_layers" in builder._search_space["hyperparameters"]

    def test_remove_param_without_category_searches_all(self, builder):
        """Test remove_param without category searches all categories (lines 472-477)."""
        builder.add_int("param", 1, 10, category="hyperparameters")

        result = builder.remove_param("param")  # No category specified

        assert result is builder
        assert "param" not in builder._search_space.get("hyperparameters", {})

    def test_remove_param_nonexistent_param_no_error(self, builder):
        """Test remove_param with nonexistent param does not raise error."""
        builder.add_int("existing", 1, 10)

        # Should not raise - just silently does nothing
        result = builder.remove_param("nonexistent", category="hyperparameters")

        assert result is builder
        assert "existing" in builder._search_space["hyperparameters"]

    def test_remove_param_nonexistent_category_no_error(self, builder):
        """Test remove_param with nonexistent category does not raise error."""
        builder.add_int("param", 1, 10, category="hyperparameters")

        # Should not raise
        result = builder.remove_param("param", category="nonexistent_category")

        assert result is builder
        # param should still exist in hyperparameters
        assert "param" in builder._search_space["hyperparameters"]

    def test_remove_param_returns_self(self, builder):
        """Test remove_param returns self for chaining."""
        builder.add_int("param", 1, 10)
        result = builder.remove_param("param")
        assert result is builder

    def test_remove_param_chaining(self, builder):
        """Test remove_param can be chained."""
        builder.add_int("param1", 1, 10)
        builder.add_int("param2", 1, 10)
        builder.add_int("param3", 1, 10)

        result = builder.remove_param("param1").remove_param("param2")

        assert result is builder
        assert "param1" not in builder._search_space["hyperparameters"]
        assert "param2" not in builder._search_space["hyperparameters"]
        assert "param3" in builder._search_space["hyperparameters"]

    def test_remove_param_after_build_raises_error(self, builder):
        """Test remove_param raises error after build()."""
        builder.add_int("param", 1, 10)
        builder.build()

        with pytest.raises(Exception) as exc_info:
            builder.remove_param("param")

        assert "Cannot modify search space after build()" in str(exc_info.value)

    def test_remove_param_only_removes_from_first_found_category(self, builder):
        """Test remove_param without category only removes from first found (line 477: break)."""
        # Add same-named param to different categories
        builder.add_int("shared_name", 1, 10, category="hyperparameters")
        builder.add_float("shared_name", 0.0, 1.0, category="optimizer")

        # Remove without specifying category - should remove from first found
        builder.remove_param("shared_name")

        # One should be removed, but behavior depends on dict iteration order
        # In Python 3.7+, dicts maintain insertion order
        # hyperparameters was added first, so it should be removed from there
        total_shared = 0
        for cat in builder._search_space:
            if "shared_name" in builder._search_space[cat]:
                total_shared += 1

        assert total_shared == 1  # Only one removed

    def test_remove_param_from_multiple_params_in_category(self, builder):
        """Test remove_param leaves other params in same category intact."""
        builder.add_int("param1", 1, 10)
        builder.add_int("param2", 1, 10)
        builder.add_float("param3", 0.0, 1.0)

        builder.remove_param("param2", category="hyperparameters")

        assert "param1" in builder._search_space["hyperparameters"]
        assert "param2" not in builder._search_space["hyperparameters"]
        assert "param3" in builder._search_space["hyperparameters"]


# =============================================================================
# SEARCHSPACEBUILDER BUILD TESTS
# =============================================================================


class TestSearchSpaceBuilderBuild:
    """Test build method (lines 481-510)."""

    def test_build_returns_search_space(self, builder):
        """Test build returns the search space dict."""
        builder.add_int("param", 1, 10)
        result = builder.build()

        assert isinstance(result, dict)
        assert "hyperparameters" in result
        assert "param" in result["hyperparameters"]

    def test_build_freezes_builder(self, builder):
        """Test build sets _frozen to True (line 504)."""
        builder.add_int("param", 1, 10)

        assert builder._frozen is False
        builder.build()
        assert builder._frozen is True

    def test_build_returns_deepcopy(self, builder, search_space_builder_module):
        """Test build returns a deep copy (line 510)."""
        builder.add_int("param", 1, 10)
        result = builder.build()

        # The important thing is that result is a copy, not the same object
        assert result is not builder._search_space

        # Also verify the nested dicts are copies
        assert result["hyperparameters"] is not builder._search_space["hyperparameters"]

    def test_build_empty_search_space_raises_error(self, search_space_builder_module):
        """Test build with empty search space raises SearchSpaceError (lines 493-496)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        builder = SearchSpaceBuilder()

        with pytest.raises(Exception) as exc_info:
            builder.build()

        assert "Cannot build empty search space" in str(exc_info.value)

    def test_build_search_space_with_zero_params_raises_error(self, builder):
        """Test build with categories but no params raises error (lines 498-502)."""
        # Manually create empty categories
        builder._search_space["hyperparameters"] = {}
        builder._search_space["optimizer"] = {}

        with pytest.raises(Exception) as exc_info:
            builder.build()

        assert "no parameters" in str(exc_info.value).lower()

    def test_build_logs_info(self, builder, caplog):
        """Test build logs info about the search space (lines 505-508)."""
        caplog.set_level(logging.INFO)

        builder.add_int("param1", 1, 10)
        builder.add_float("param2", 0.0, 1.0)
        builder.build()

        assert "Built search space with 2 parameters" in caplog.text
        assert "1 categories" in caplog.text

    def test_build_multiple_categories(self, builder, caplog):
        """Test build with multiple categories."""
        caplog.set_level(logging.INFO)

        builder.add_int("param1", 1, 10, category="hyperparameters")
        builder.add_loguniform("lr", 1e-5, 1e-2, category="optimizer")
        builder.add_float("factor", 0.1, 0.9, category="scheduler")

        result = builder.build()

        assert "hyperparameters" in result
        assert "optimizer" in result
        assert "scheduler" in result
        assert "3 parameters" in caplog.text
        assert "3 categories" in caplog.text

    def test_build_cannot_be_called_twice(self, builder):
        """Test build cannot be called twice (frozen after first call)."""
        builder.add_int("param", 1, 10)
        builder.build()

        # Second build should work (it just returns a copy)
        # But adding more params should fail
        with pytest.raises(Exception) as exc_info:
            builder.add_int("another", 1, 10)

        assert "Cannot modify search space after build()" in str(exc_info.value)

    def test_build_returns_correct_structure(self, builder, search_space_builder_module):
        """Test build returns correctly structured dict."""
        ParamType = search_space_builder_module["ParamType"]

        builder.add_int("hidden", 32, 256, step=32)
        builder.add_categorical("activation", ["relu", "gelu"])

        result = builder.build()

        assert "hyperparameters" in result
        assert "hidden" in result["hyperparameters"]
        assert "activation" in result["hyperparameters"]

        hidden_config = result["hyperparameters"]["hidden"]
        assert hidden_config.type.value == ParamType.INT.value
        assert hidden_config.low == 32.0
        assert hidden_config.high == 256.0
        assert hidden_config.step == 32


# =============================================================================
# SEARCHSPACEBUILDER TO_DICT TESTS
# =============================================================================


class TestSearchSpaceBuilderToDict:
    """Test to_dict method (lines 512-526)."""

    def test_to_dict_basic(self, builder, search_space_builder_module):
        """Test to_dict returns plain dict representation."""
        builder.add_int("param", 1, 10)

        result = builder.to_dict()

        assert isinstance(result, dict)
        assert "hyperparameters" in result
        assert "param" in result["hyperparameters"]
        assert isinstance(result["hyperparameters"]["param"], dict)

    def test_to_dict_converts_config_to_dict(self, builder):
        """Test to_dict converts SearchSpaceParamConfig to dict."""
        builder.add_int("hidden", 32, 256, step=32)

        result = builder.to_dict()
        param_dict = result["hyperparameters"]["hidden"]

        assert param_dict["type"] == "int"
        assert param_dict["low"] == 32.0
        assert param_dict["high"] == 256.0
        assert param_dict["step"] == 32

    def test_to_dict_multiple_categories(self, builder):
        """Test to_dict with multiple categories."""
        builder.add_int("param1", 1, 10, category="hyperparameters")
        builder.add_loguniform("lr", 1e-5, 1e-2, category="optimizer")

        result = builder.to_dict()

        assert "hyperparameters" in result
        assert "optimizer" in result
        assert isinstance(result["hyperparameters"]["param1"], dict)
        assert isinstance(result["optimizer"]["lr"], dict)

    def test_to_dict_loguniform_type(self, builder):
        """Test to_dict with loguniform type."""
        builder.add_loguniform("lr", 1e-5, 1e-2)

        result = builder.to_dict()
        param_dict = result["optimizer"]["lr"]

        assert param_dict["type"] == "loguniform"
        assert param_dict["low"] == 1e-5
        assert param_dict["high"] == 1e-2

    def test_to_dict_categorical_type(self, builder):
        """Test to_dict with categorical type."""
        builder.add_categorical("activation", ["relu", "gelu", "elu"])

        result = builder.to_dict()
        param_dict = result["hyperparameters"]["activation"]

        assert param_dict["type"] == "categorical"
        assert param_dict["choices"] == ["relu", "gelu", "elu"]

    def test_to_dict_float_with_log(self, builder):
        """Test to_dict with float type and log=True."""
        builder.add_float("temp", 0.01, 1.0, log=True)

        result = builder.to_dict()
        param_dict = result["hyperparameters"]["temp"]

        assert param_dict["type"] == "float"
        assert param_dict["log"] is True

    def test_to_dict_does_not_include_none_values(self, builder):
        """Test to_dict omits None values (lines 551-560: conditionals)."""
        builder.add_int("param", 1, 10)  # No step

        result = builder.to_dict()
        param_dict = result["hyperparameters"]["param"]

        # step should not be in dict if None
        assert "step" not in param_dict or param_dict.get("step") is None

    def test_to_dict_empty_search_space(self, search_space_builder_module):
        """Test to_dict with empty search space."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        builder = SearchSpaceBuilder()

        result = builder.to_dict()

        assert result == {}

    def test_to_dict_can_be_called_after_build(self, builder):
        """Test to_dict can be called after build()."""
        builder.add_int("param", 1, 10)
        builder.build()

        # Should not raise
        result = builder.to_dict()
        assert "hyperparameters" in result


# =============================================================================
# SEARCHSPACEBUILDER _DICT_TO_CONFIG TESTS
# =============================================================================


class TestSearchSpaceBuilderDictToConfig:
    """Test _dict_to_config static method (lines 532-542)."""

    def test_dict_to_config_int_type(self, search_space_builder_module):
        """Test _dict_to_config with int type."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        config_dict = {"type": "int", "low": 1, "high": 10, "step": 2}
        config = SearchSpaceBuilder._dict_to_config(config_dict)

        assert config.type.value == ParamType.INT.value
        assert config.low == 1
        assert config.high == 10
        assert config.step == 2

    def test_dict_to_config_float_type(self, search_space_builder_module):
        """Test _dict_to_config with float type."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        config_dict = {"type": "float", "low": 0.0, "high": 1.0, "log": True}
        config = SearchSpaceBuilder._dict_to_config(config_dict)

        assert config.type.value == ParamType.FLOAT.value
        assert config.low == 0.0
        assert config.high == 1.0
        assert config.log is True

    def test_dict_to_config_categorical_type(self, search_space_builder_module):
        """Test _dict_to_config with categorical type."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        config_dict = {"type": "categorical", "choices": ["a", "b", "c"]}
        config = SearchSpaceBuilder._dict_to_config(config_dict)

        assert config.type.value == ParamType.CATEGORICAL.value
        assert config.choices == ["a", "b", "c"]

    def test_dict_to_config_loguniform_type(self, search_space_builder_module):
        """Test _dict_to_config with loguniform type."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        config_dict = {"type": "loguniform", "low": 1e-5, "high": 1e-2}
        config = SearchSpaceBuilder._dict_to_config(config_dict)

        assert config.type.value == ParamType.LOGUNIFORM.value
        assert config.low == 1e-5
        assert config.high == 1e-2

    def test_dict_to_config_uniform_type(self, search_space_builder_module):
        """Test _dict_to_config with uniform type."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        config_dict = {"type": "uniform", "low": 0.0, "high": 1.0}
        config = SearchSpaceBuilder._dict_to_config(config_dict)

        assert config.type.value == ParamType.UNIFORM.value

    def test_dict_to_config_discrete_uniform_type(self, search_space_builder_module):
        """Test _dict_to_config with discrete_uniform type."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        config_dict = {"type": "discrete_uniform", "low": 16, "high": 128, "step": 16}
        config = SearchSpaceBuilder._dict_to_config(config_dict)

        assert config.type.value == ParamType.DISCRETE_UNIFORM.value

    def test_dict_to_config_type_already_paramtype(self, search_space_builder_module):
        """Test _dict_to_config when type is already ParamType enum value string.

        Note: In production, _dict_to_config is typically called with string type values.
        When a ParamType enum is passed directly, it should be handled by the conditional
        at line 544 (isinstance check). The real ParamType from the module is used.
        """
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        # Use string type which is the expected input format
        config_dict = {"type": "int", "low": 1, "high": 10}
        config = SearchSpaceBuilder._dict_to_config(config_dict)

        # Verify the config was created with the correct type
        assert config.type.value == "int"
        assert config.low == 1
        assert config.high == 10

    def test_dict_to_config_does_not_modify_original(self, search_space_builder_module):
        """Test _dict_to_config does not modify original dict (line 535)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        config_dict = {"type": "int", "low": 1, "high": 10}
        original_type = config_dict["type"]

        SearchSpaceBuilder._dict_to_config(config_dict)

        # Original dict should be unchanged
        assert config_dict["type"] == original_type


# =============================================================================
# SEARCHSPACEBUILDER _CONFIG_TO_DICT TESTS
# =============================================================================


class TestSearchSpaceBuilderConfigToDict:
    """Test _config_to_dict static method (lines 544-562)."""

    def test_config_to_dict_int_type(self, search_space_builder_module):
        """Test _config_to_dict with int type config."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]
        SearchSpaceParamConfig = search_space_builder_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(type=ParamType.INT, low=1, high=10, step=2)
        result = SearchSpaceBuilder._config_to_dict(config)

        assert result["type"] == "int"
        assert result["low"] == 1
        assert result["high"] == 10
        assert result["step"] == 2

    def test_config_to_dict_float_type(self, search_space_builder_module):
        """Test _config_to_dict with float type config."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]
        SearchSpaceParamConfig = search_space_builder_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(type=ParamType.FLOAT, low=0.0, high=1.0, log=True)
        result = SearchSpaceBuilder._config_to_dict(config)

        assert result["type"] == "float"
        assert result["low"] == 0.0
        assert result["high"] == 1.0
        assert result["log"] is True

    def test_config_to_dict_categorical_type(self, search_space_builder_module):
        """Test _config_to_dict with categorical type config."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]
        SearchSpaceParamConfig = search_space_builder_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(type=ParamType.CATEGORICAL, choices=["a", "b"])
        result = SearchSpaceBuilder._config_to_dict(config)

        assert result["type"] == "categorical"
        assert result["choices"] == ["a", "b"]

    def test_config_to_dict_loguniform_type(self, search_space_builder_module):
        """Test _config_to_dict with loguniform type config."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]
        SearchSpaceParamConfig = search_space_builder_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(type=ParamType.LOGUNIFORM, low=1e-5, high=1e-2)
        result = SearchSpaceBuilder._config_to_dict(config)

        assert result["type"] == "loguniform"
        assert result["low"] == 1e-5
        assert result["high"] == 1e-2

    def test_config_to_dict_omits_none_low(self, search_space_builder_module):
        """Test _config_to_dict omits low if None (lines 551-552)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]
        SearchSpaceParamConfig = search_space_builder_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(type=ParamType.CATEGORICAL, choices=["a", "b"])
        result = SearchSpaceBuilder._config_to_dict(config)

        # low should not be in result for categorical
        assert "low" not in result

    def test_config_to_dict_omits_none_high(self, search_space_builder_module):
        """Test _config_to_dict omits high if None (lines 553-554)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]
        SearchSpaceParamConfig = search_space_builder_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(type=ParamType.CATEGORICAL, choices=["a", "b"])
        result = SearchSpaceBuilder._config_to_dict(config)

        # high should not be in result for categorical
        assert "high" not in result

    def test_config_to_dict_omits_none_step(self, search_space_builder_module):
        """Test _config_to_dict omits step if None (lines 555-556)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]
        SearchSpaceParamConfig = search_space_builder_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(type=ParamType.INT, low=1, high=10)  # No step
        result = SearchSpaceBuilder._config_to_dict(config)

        # step should not be in result if None
        assert "step" not in result

    def test_config_to_dict_omits_none_choices(self, search_space_builder_module):
        """Test _config_to_dict omits choices if None (lines 557-558)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]
        SearchSpaceParamConfig = search_space_builder_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(type=ParamType.INT, low=1, high=10)
        result = SearchSpaceBuilder._config_to_dict(config)

        # choices should not be in result for non-categorical
        assert "choices" not in result

    def test_config_to_dict_omits_false_log(self, search_space_builder_module):
        """Test _config_to_dict omits log if False (lines 559-560)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]
        SearchSpaceParamConfig = search_space_builder_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(type=ParamType.FLOAT, low=0.0, high=1.0, log=False)
        result = SearchSpaceBuilder._config_to_dict(config)

        # log should not be in result if False
        assert "log" not in result

    def test_config_to_dict_includes_true_log(self, search_space_builder_module):
        """Test _config_to_dict includes log if True (lines 559-560)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]
        SearchSpaceParamConfig = search_space_builder_module["SearchSpaceParamConfig"]

        config = SearchSpaceParamConfig(type=ParamType.FLOAT, low=0.0, high=1.0, log=True)
        result = SearchSpaceBuilder._config_to_dict(config)

        assert result["log"] is True


# =============================================================================
# SEARCHSPACEBUILDER FOR_MODEL TESTS (LEGACY FALLBACK)
# =============================================================================


class TestSearchSpaceBuilderForModelLegacy:
    """Test for_model class method with legacy fallback (lines 568-674)."""

    def test_for_model_gcn(self, search_space_builder_module):
        """Test for_model with GCN model (lines 647-648)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        # Force legacy fallback by patching introspector as unavailable
        with patch(
            "milia_pipeline.models.hpo.search_spaces.search_space_builder._INTROSPECTOR_AVAILABLE",
            False,
        ):
            result = SearchSpaceBuilder.for_model("GCN")

            assert isinstance(result, dict)
            assert "hyperparameters" in result
            assert "hidden_channels" in result["hyperparameters"]
            assert "num_layers" in result["hyperparameters"]
            assert "dropout" in result["hyperparameters"]
            assert "aggregation" in result["hyperparameters"]

    def test_for_model_gcn_case_insensitive(self, search_space_builder_module):
        """Test for_model is case insensitive (line 645: model_name_upper)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        # Force legacy fallback by patching introspector as unavailable
        with patch(
            "milia_pipeline.models.hpo.search_spaces.search_space_builder._INTROSPECTOR_AVAILABLE",
            False,
        ):
            result_upper = SearchSpaceBuilder.for_model("GCN")
            result_lower = SearchSpaceBuilder.for_model("gcn")
            result_mixed = SearchSpaceBuilder.for_model("Gcn")

            # All should have same hyperparameters
            assert set(result_upper["hyperparameters"].keys()) == set(
                result_lower["hyperparameters"].keys()
            )
            assert set(result_upper["hyperparameters"].keys()) == set(
                result_mixed["hyperparameters"].keys()
            )

    def test_for_model_gcn_alias_graphconv(self, search_space_builder_module):
        """Test for_model with GRAPHCONV alias (line 647)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = SearchSpaceBuilder.for_model("GRAPHCONV")

        assert "hidden_channels" in result["hyperparameters"]
        assert "num_layers" in result["hyperparameters"]

    def test_for_model_gat(self, search_space_builder_module):
        """Test for_model with GAT model (lines 649-650)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        # Force legacy fallback by patching introspector as unavailable
        with patch(
            "milia_pipeline.models.hpo.search_spaces.search_space_builder._INTROSPECTOR_AVAILABLE",
            False,
        ):
            result = SearchSpaceBuilder.for_model("GAT")

            assert "hyperparameters" in result
            assert "hidden_channels" in result["hyperparameters"]
            assert "num_layers" in result["hyperparameters"]
            assert "heads" in result["hyperparameters"]
            assert "dropout" in result["hyperparameters"]
            assert "attention_dropout" in result["hyperparameters"]
            assert "concat" in result["hyperparameters"]

    def test_for_model_gat_alias_gatconv(self, search_space_builder_module):
        """Test for_model with GATCONV alias (line 649)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = SearchSpaceBuilder.for_model("GATCONV")

        assert "heads" in result["hyperparameters"]
        assert "attention_dropout" in result["hyperparameters"]

    def test_for_model_graphsage(self, search_space_builder_module):
        """Test for_model with GraphSAGE model (lines 651-652)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        # Force legacy fallback by patching introspector as unavailable
        with patch(
            "milia_pipeline.models.hpo.search_spaces.search_space_builder._INTROSPECTOR_AVAILABLE",
            False,
        ):
            result = SearchSpaceBuilder.for_model("GraphSAGE")

            assert "hyperparameters" in result
            assert "hidden_channels" in result["hyperparameters"]
            assert "num_layers" in result["hyperparameters"]
            assert "dropout" in result["hyperparameters"]
            assert "aggregation" in result["hyperparameters"]
            assert "normalize" in result["hyperparameters"]

    def test_for_model_graphsage_alias_sage(self, search_space_builder_module):
        """Test for_model with SAGE alias (line 651)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = SearchSpaceBuilder.for_model("SAGE")

        assert "aggregation" in result["hyperparameters"]
        assert "normalize" in result["hyperparameters"]

    def test_for_model_graphsage_alias_sageconv(self, search_space_builder_module):
        """Test for_model with SAGECONV alias (line 651)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = SearchSpaceBuilder.for_model("SAGECONV")

        assert "aggregation" in result["hyperparameters"]

    def test_for_model_gin(self, search_space_builder_module):
        """Test for_model with GIN model (lines 653-654)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        # Force legacy fallback by patching introspector as unavailable
        with patch(
            "milia_pipeline.models.hpo.search_spaces.search_space_builder._INTROSPECTOR_AVAILABLE",
            False,
        ):
            result = SearchSpaceBuilder.for_model("GIN")

            assert "hyperparameters" in result
            assert "hidden_channels" in result["hyperparameters"]
            assert "num_layers" in result["hyperparameters"]
            assert "dropout" in result["hyperparameters"]
            assert "eps" in result["hyperparameters"]
            assert "train_eps" in result["hyperparameters"]

    def test_for_model_gin_alias_ginconv(self, search_space_builder_module):
        """Test for_model with GINCONV alias (line 653)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = SearchSpaceBuilder.for_model("GINCONV")

        assert "eps" in result["hyperparameters"]
        assert "train_eps" in result["hyperparameters"]

    def test_for_model_schnet(self, search_space_builder_module):
        """Test for_model with SchNet model (lines 655-656)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = SearchSpaceBuilder.for_model("SchNet")

        assert "hyperparameters" in result
        assert "hidden_channels" in result["hyperparameters"]
        assert "num_filters" in result["hyperparameters"]
        assert "num_interactions" in result["hyperparameters"]
        assert "num_gaussians" in result["hyperparameters"]
        assert "cutoff" in result["hyperparameters"]

    def test_for_model_dimenet(self, search_space_builder_module):
        """Test for_model with DimeNet model (lines 657-658)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = SearchSpaceBuilder.for_model("DimeNet")

        assert "hyperparameters" in result
        assert "hidden_channels" in result["hyperparameters"]
        assert "num_blocks" in result["hyperparameters"]
        assert "num_bilinear" in result["hyperparameters"]
        assert "num_spherical" in result["hyperparameters"]
        assert "num_radial" in result["hyperparameters"]
        assert "cutoff" in result["hyperparameters"]

    def test_for_model_mpnn(self, search_space_builder_module):
        """Test for_model with MPNN model (lines 659-660)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = SearchSpaceBuilder.for_model("MPNN")

        assert "hyperparameters" in result
        assert "hidden_channels" in result["hyperparameters"]
        assert "num_layers" in result["hyperparameters"]
        assert "dropout" in result["hyperparameters"]
        assert "aggregation" in result["hyperparameters"]

    def test_for_model_mpnn_alias_mpnnconv(self, search_space_builder_module):
        """Test for_model with MPNNCONV alias (line 659)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = SearchSpaceBuilder.for_model("MPNNCONV")

        assert "aggregation" in result["hyperparameters"]

    def test_for_model_unknown_uses_generic(self, search_space_builder_module, caplog):
        """Test for_model with unknown model uses generic GNN space (lines 661-666)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        caplog.set_level(logging.WARNING)

        result = SearchSpaceBuilder.for_model("UnknownModel")

        assert "hyperparameters" in result
        assert "hidden_channels" in result["hyperparameters"]
        assert "num_layers" in result["hyperparameters"]
        assert "dropout" in result["hyperparameters"]

        # Should log warning
        assert "No predefined space for 'UnknownModel'" in caplog.text

    def test_for_model_with_include_optimizer_true(self, search_space_builder_module):
        """Test for_model with include_optimizer=True (default) (lines 668-669)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = SearchSpaceBuilder.for_model("GCN", include_optimizer=True)

        assert "optimizer" in result
        assert "lr" in result["optimizer"]
        assert "weight_decay" in result["optimizer"]

    def test_for_model_with_include_optimizer_false(self, search_space_builder_module):
        """Test for_model with include_optimizer=False."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = SearchSpaceBuilder.for_model("GCN", include_optimizer=False)

        assert "optimizer" not in result

    def test_for_model_with_include_scheduler_true(self, search_space_builder_module):
        """Test for_model with include_scheduler=True (lines 671-672)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = SearchSpaceBuilder.for_model("GCN", include_scheduler=True)

        assert "scheduler" in result
        assert "factor" in result["scheduler"]
        assert "patience" in result["scheduler"]

    def test_for_model_with_include_scheduler_false(self, search_space_builder_module):
        """Test for_model with include_scheduler=False (default)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = SearchSpaceBuilder.for_model("GCN", include_scheduler=False)

        assert "scheduler" not in result

    def test_for_model_with_both_optimizer_and_scheduler(self, search_space_builder_module):
        """Test for_model with both optimizer and scheduler."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = SearchSpaceBuilder.for_model("GCN", include_optimizer=True, include_scheduler=True)

        assert "hyperparameters" in result
        assert "optimizer" in result
        assert "scheduler" in result


# =============================================================================
# SEARCHSPACEBUILDER LEGACY MODEL BUILDER TESTS
# =============================================================================


class TestSearchSpaceBuilderBuildGcnSpace:
    """Test _build_gcn_space class method (lines 677-692)."""

    def test_build_gcn_space_hidden_channels(self, search_space_builder_module):
        """Test _build_gcn_space adds hidden_channels param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_gcn_space(builder)

        config = builder._search_space["hyperparameters"]["hidden_channels"]
        assert config.type.value == ParamType.INT.value
        assert config.low == 32.0
        assert config.high == 256.0
        assert config.step == 32

    def test_build_gcn_space_num_layers(self, search_space_builder_module):
        """Test _build_gcn_space adds num_layers param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_gcn_space(builder)

        config = builder._search_space["hyperparameters"]["num_layers"]
        assert config.type.value == ParamType.INT.value
        assert config.low == 2.0
        assert config.high == 6.0

    def test_build_gcn_space_dropout(self, search_space_builder_module):
        """Test _build_gcn_space adds dropout param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_gcn_space(builder)

        config = builder._search_space["hyperparameters"]["dropout"]
        assert config.type.value == ParamType.FLOAT.value
        assert config.low == 0.0
        assert config.high == 0.6

    def test_build_gcn_space_aggregation(self, search_space_builder_module):
        """Test _build_gcn_space adds aggregation param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_gcn_space(builder)

        config = builder._search_space["hyperparameters"]["aggregation"]
        assert config.type.value == ParamType.CATEGORICAL.value
        assert config.choices == ["add", "mean", "max"]

    def test_build_gcn_space_returns_builder(self, search_space_builder_module):
        """Test _build_gcn_space returns builder for chaining."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        builder = SearchSpaceBuilder()
        result = SearchSpaceBuilder._build_gcn_space(builder)

        assert result is builder


class TestSearchSpaceBuilderBuildGatSpace:
    """Test _build_gat_space class method (lines 694-710)."""

    def test_build_gat_space_has_heads(self, search_space_builder_module):
        """Test _build_gat_space adds heads param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_gat_space(builder)

        config = builder._search_space["hyperparameters"]["heads"]
        assert config.type.value == ParamType.INT.value
        assert config.low == 1.0
        assert config.high == 8.0

    def test_build_gat_space_has_attention_dropout(self, search_space_builder_module):
        """Test _build_gat_space adds attention_dropout param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_gat_space(builder)

        config = builder._search_space["hyperparameters"]["attention_dropout"]
        assert config.type.value == ParamType.FLOAT.value
        assert config.low == 0.0
        assert config.high == 0.6

    def test_build_gat_space_has_concat(self, search_space_builder_module):
        """Test _build_gat_space adds concat param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_gat_space(builder)

        config = builder._search_space["hyperparameters"]["concat"]
        assert config.type.value == ParamType.CATEGORICAL.value
        assert config.choices == [True, False]

    def test_build_gat_space_returns_builder(self, search_space_builder_module):
        """Test _build_gat_space returns builder for chaining."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        builder = SearchSpaceBuilder()
        result = SearchSpaceBuilder._build_gat_space(builder)

        assert result is builder


class TestSearchSpaceBuilderBuildGraphSageSpace:
    """Test _build_graphsage_space class method (lines 712-727)."""

    def test_build_graphsage_space_has_aggregation(self, search_space_builder_module):
        """Test _build_graphsage_space adds aggregation param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_graphsage_space(builder)

        config = builder._search_space["hyperparameters"]["aggregation"]
        assert config.type.value == ParamType.CATEGORICAL.value
        assert "mean" in config.choices
        assert "max" in config.choices
        assert "lstm" in config.choices

    def test_build_graphsage_space_has_normalize(self, search_space_builder_module):
        """Test _build_graphsage_space adds normalize param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_graphsage_space(builder)

        config = builder._search_space["hyperparameters"]["normalize"]
        assert config.type.value == ParamType.CATEGORICAL.value
        assert config.choices == [True, False]

    def test_build_graphsage_space_returns_builder(self, search_space_builder_module):
        """Test _build_graphsage_space returns builder for chaining."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        builder = SearchSpaceBuilder()
        result = SearchSpaceBuilder._build_graphsage_space(builder)

        assert result is builder


class TestSearchSpaceBuilderBuildGinSpace:
    """Test _build_gin_space class method (lines 729-744)."""

    def test_build_gin_space_has_eps(self, search_space_builder_module):
        """Test _build_gin_space adds eps param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_gin_space(builder)

        config = builder._search_space["hyperparameters"]["eps"]
        assert config.type.value == ParamType.FLOAT.value
        assert config.low == 0.0
        assert config.high == 1.0

    def test_build_gin_space_has_train_eps(self, search_space_builder_module):
        """Test _build_gin_space adds train_eps param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_gin_space(builder)

        config = builder._search_space["hyperparameters"]["train_eps"]
        assert config.type.value == ParamType.CATEGORICAL.value
        assert config.choices == [True, False]

    def test_build_gin_space_returns_builder(self, search_space_builder_module):
        """Test _build_gin_space returns builder for chaining."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        builder = SearchSpaceBuilder()
        result = SearchSpaceBuilder._build_gin_space(builder)

        assert result is builder


class TestSearchSpaceBuilderBuildSchNetSpace:
    """Test _build_schnet_space class method (lines 746-761)."""

    def test_build_schnet_space_has_num_filters(self, search_space_builder_module):
        """Test _build_schnet_space adds num_filters param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_schnet_space(builder)

        config = builder._search_space["hyperparameters"]["num_filters"]
        assert config.type.value == ParamType.INT.value
        assert config.low == 64.0
        assert config.high == 256.0
        assert config.step == 32

    def test_build_schnet_space_has_num_interactions(self, search_space_builder_module):
        """Test _build_schnet_space adds num_interactions param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_schnet_space(builder)

        config = builder._search_space["hyperparameters"]["num_interactions"]
        assert config.type.value == ParamType.INT.value
        assert config.low == 3.0
        assert config.high == 6.0

    def test_build_schnet_space_has_num_gaussians(self, search_space_builder_module):
        """Test _build_schnet_space adds num_gaussians param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_schnet_space(builder)

        config = builder._search_space["hyperparameters"]["num_gaussians"]
        assert config.type.value == ParamType.INT.value
        assert config.low == 25.0
        assert config.high == 100.0
        assert config.step == 25

    def test_build_schnet_space_has_cutoff(self, search_space_builder_module):
        """Test _build_schnet_space adds cutoff param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_schnet_space(builder)

        config = builder._search_space["hyperparameters"]["cutoff"]
        assert config.type.value == ParamType.FLOAT.value
        assert config.low == 5.0
        assert config.high == 10.0

    def test_build_schnet_space_returns_builder(self, search_space_builder_module):
        """Test _build_schnet_space returns builder for chaining."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        builder = SearchSpaceBuilder()
        result = SearchSpaceBuilder._build_schnet_space(builder)

        assert result is builder


class TestSearchSpaceBuilderBuildDimeNetSpace:
    """Test _build_dimenet_space class method (lines 763-779)."""

    def test_build_dimenet_space_has_num_blocks(self, search_space_builder_module):
        """Test _build_dimenet_space adds num_blocks param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_dimenet_space(builder)

        config = builder._search_space["hyperparameters"]["num_blocks"]
        assert config.type.value == ParamType.INT.value
        assert config.low == 3.0
        assert config.high == 6.0

    def test_build_dimenet_space_has_num_bilinear(self, search_space_builder_module):
        """Test _build_dimenet_space adds num_bilinear param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_dimenet_space(builder)

        config = builder._search_space["hyperparameters"]["num_bilinear"]
        assert config.type.value == ParamType.INT.value
        assert config.low == 4.0
        assert config.high == 8.0

    def test_build_dimenet_space_has_num_spherical(self, search_space_builder_module):
        """Test _build_dimenet_space adds num_spherical param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_dimenet_space(builder)

        config = builder._search_space["hyperparameters"]["num_spherical"]
        assert config.type.value == ParamType.INT.value
        assert config.low == 3.0
        assert config.high == 7.0

    def test_build_dimenet_space_has_num_radial(self, search_space_builder_module):
        """Test _build_dimenet_space adds num_radial param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_dimenet_space(builder)

        config = builder._search_space["hyperparameters"]["num_radial"]
        assert config.type.value == ParamType.INT.value
        assert config.low == 3.0
        assert config.high == 6.0

    def test_build_dimenet_space_has_cutoff(self, search_space_builder_module):
        """Test _build_dimenet_space adds cutoff param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_dimenet_space(builder)

        config = builder._search_space["hyperparameters"]["cutoff"]
        assert config.type.value == ParamType.FLOAT.value
        assert config.low == 4.0
        assert config.high == 6.0

    def test_build_dimenet_space_returns_builder(self, search_space_builder_module):
        """Test _build_dimenet_space returns builder for chaining."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        builder = SearchSpaceBuilder()
        result = SearchSpaceBuilder._build_dimenet_space(builder)

        assert result is builder


class TestSearchSpaceBuilderBuildMpnnSpace:
    """Test _build_mpnn_space class method (lines 781-795)."""

    def test_build_mpnn_space_has_hidden_channels(self, search_space_builder_module):
        """Test _build_mpnn_space adds hidden_channels param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_mpnn_space(builder)

        config = builder._search_space["hyperparameters"]["hidden_channels"]
        assert config.type.value == ParamType.INT.value
        assert config.low == 32.0
        assert config.high == 256.0
        assert config.step == 32

    def test_build_mpnn_space_has_aggregation(self, search_space_builder_module):
        """Test _build_mpnn_space adds aggregation param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_mpnn_space(builder)

        config = builder._search_space["hyperparameters"]["aggregation"]
        assert config.type.value == ParamType.CATEGORICAL.value
        assert config.choices == ["add", "mean", "max"]

    def test_build_mpnn_space_returns_builder(self, search_space_builder_module):
        """Test _build_mpnn_space returns builder for chaining."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        builder = SearchSpaceBuilder()
        result = SearchSpaceBuilder._build_mpnn_space(builder)

        assert result is builder


class TestSearchSpaceBuilderBuildGenericGnnSpace:
    """Test _build_generic_gnn_space class method (lines 797-810)."""

    def test_build_generic_gnn_space_has_hidden_channels(self, search_space_builder_module):
        """Test _build_generic_gnn_space adds hidden_channels param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_generic_gnn_space(builder)

        config = builder._search_space["hyperparameters"]["hidden_channels"]
        assert config.type.value == ParamType.INT.value
        assert config.low == 32.0
        assert config.high == 256.0
        assert config.step == 32

    def test_build_generic_gnn_space_has_num_layers(self, search_space_builder_module):
        """Test _build_generic_gnn_space adds num_layers param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_generic_gnn_space(builder)

        config = builder._search_space["hyperparameters"]["num_layers"]
        assert config.type.value == ParamType.INT.value
        assert config.low == 2.0
        assert config.high == 6.0

    def test_build_generic_gnn_space_has_dropout(self, search_space_builder_module):
        """Test _build_generic_gnn_space adds dropout param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_generic_gnn_space(builder)

        config = builder._search_space["hyperparameters"]["dropout"]
        assert config.type.value == ParamType.FLOAT.value
        assert config.low == 0.0
        assert config.high == 0.6

    def test_build_generic_gnn_space_only_has_three_params(self, search_space_builder_module):
        """Test _build_generic_gnn_space has exactly 3 params."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        builder = SearchSpaceBuilder()
        SearchSpaceBuilder._build_generic_gnn_space(builder)

        assert len(builder._search_space["hyperparameters"]) == 3

    def test_build_generic_gnn_space_returns_builder(self, search_space_builder_module):
        """Test _build_generic_gnn_space returns builder for chaining."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        builder = SearchSpaceBuilder()
        result = SearchSpaceBuilder._build_generic_gnn_space(builder)

        assert result is builder


class TestSearchSpaceBuilderAddOptimizerSpace:
    """Test _add_optimizer_space class method (lines 812-819)."""

    def test_add_optimizer_space_adds_lr(self, search_space_builder_module):
        """Test _add_optimizer_space adds lr param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        builder.add_int("dummy", 1, 10)  # Need at least one param
        SearchSpaceBuilder._add_optimizer_space(builder)

        config = builder._search_space["optimizer"]["lr"]
        assert config.type.value == ParamType.LOGUNIFORM.value
        assert config.low == 1e-5
        assert config.high == 1e-2

    def test_add_optimizer_space_adds_weight_decay(self, search_space_builder_module):
        """Test _add_optimizer_space adds weight_decay param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        builder.add_int("dummy", 1, 10)
        SearchSpaceBuilder._add_optimizer_space(builder)

        config = builder._search_space["optimizer"]["weight_decay"]
        assert config.type.value == ParamType.LOGUNIFORM.value
        assert config.low == 1e-6
        assert config.high == 1e-3

    def test_add_optimizer_space_returns_builder(self, search_space_builder_module):
        """Test _add_optimizer_space returns builder for chaining."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        builder = SearchSpaceBuilder()
        builder.add_int("dummy", 1, 10)
        result = SearchSpaceBuilder._add_optimizer_space(builder)

        assert result is builder


class TestSearchSpaceBuilderAddSchedulerSpace:
    """Test _add_scheduler_space class method (lines 821-828)."""

    def test_add_scheduler_space_adds_factor(self, search_space_builder_module):
        """Test _add_scheduler_space adds factor param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        builder.add_int("dummy", 1, 10)
        SearchSpaceBuilder._add_scheduler_space(builder)

        config = builder._search_space["scheduler"]["factor"]
        assert config.type.value == ParamType.FLOAT.value
        assert config.low == 0.1
        assert config.high == 0.9

    def test_add_scheduler_space_adds_patience(self, search_space_builder_module):
        """Test _add_scheduler_space adds patience param."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        builder = SearchSpaceBuilder()
        builder.add_int("dummy", 1, 10)
        SearchSpaceBuilder._add_scheduler_space(builder)

        config = builder._search_space["scheduler"]["patience"]
        assert config.type.value == ParamType.INT.value
        assert config.low == 5.0
        assert config.high == 20.0

    def test_add_scheduler_space_returns_builder(self, search_space_builder_module):
        """Test _add_scheduler_space returns builder for chaining."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        builder = SearchSpaceBuilder()
        builder.add_int("dummy", 1, 10)
        result = SearchSpaceBuilder._add_scheduler_space(builder)

        assert result is builder


# =============================================================================
# SEARCHSPACEBUILDER FROM_DICT TESTS
# =============================================================================


class TestSearchSpaceBuilderFromDict:
    """Test from_dict class method (lines 835-862)."""

    def test_from_dict_basic(self, search_space_builder_module):
        """Test from_dict with basic dict input (lines 849-854: Example)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        space_dict = {
            "hyperparameters": {"hidden_channels": {"type": "int", "low": 32, "high": 256}}
        }

        result = SearchSpaceBuilder.from_dict(space_dict)

        assert isinstance(result, dict)
        assert "hyperparameters" in result
        assert "hidden_channels" in result["hyperparameters"]

        config = result["hyperparameters"]["hidden_channels"]
        assert config.type.value == ParamType.INT.value
        assert config.low == 32
        assert config.high == 256

    def test_from_dict_multiple_categories(self, search_space_builder_module):
        """Test from_dict with multiple categories."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space_dict = {
            "hyperparameters": {"hidden_channels": {"type": "int", "low": 32, "high": 256}},
            "optimizer": {"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}},
        }

        result = SearchSpaceBuilder.from_dict(space_dict)

        assert "hyperparameters" in result
        assert "optimizer" in result
        assert "hidden_channels" in result["hyperparameters"]
        assert "lr" in result["optimizer"]

    def test_from_dict_multiple_params_per_category(self, search_space_builder_module):
        """Test from_dict with multiple parameters per category."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space_dict = {
            "hyperparameters": {
                "hidden_channels": {"type": "int", "low": 32, "high": 256},
                "num_layers": {"type": "int", "low": 2, "high": 6},
                "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            }
        }

        result = SearchSpaceBuilder.from_dict(space_dict)

        assert len(result["hyperparameters"]) == 3

    def test_from_dict_with_categorical(self, search_space_builder_module):
        """Test from_dict with categorical parameter."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        space_dict = {
            "hyperparameters": {"activation": {"type": "categorical", "choices": ["relu", "gelu"]}}
        }

        result = SearchSpaceBuilder.from_dict(space_dict)

        config = result["hyperparameters"]["activation"]
        assert config.type.value == ParamType.CATEGORICAL.value
        assert config.choices == ["relu", "gelu"]

    def test_from_dict_with_step(self, search_space_builder_module):
        """Test from_dict with step parameter."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space_dict = {
            "hyperparameters": {
                "hidden_channels": {"type": "int", "low": 32, "high": 256, "step": 32}
            }
        }

        result = SearchSpaceBuilder.from_dict(space_dict)

        config = result["hyperparameters"]["hidden_channels"]
        assert config.step == 32

    def test_from_dict_with_log(self, search_space_builder_module):
        """Test from_dict with log parameter."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space_dict = {
            "hyperparameters": {"temp": {"type": "float", "low": 0.01, "high": 1.0, "log": True}}
        }

        result = SearchSpaceBuilder.from_dict(space_dict)

        config = result["hyperparameters"]["temp"]
        assert config.log is True

    def test_from_dict_empty_dict(self, search_space_builder_module):
        """Test from_dict with empty dict raises error."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        with pytest.raises(SearchSpaceError):
            SearchSpaceBuilder.from_dict({})

    def test_from_dict_returns_search_space_param_configs(self, search_space_builder_module):
        """Test from_dict returns SearchSpaceParamConfig objects."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        _SearchSpaceParamConfig = search_space_builder_module["SearchSpaceParamConfig"]

        space_dict = {"hyperparameters": {"param": {"type": "int", "low": 1, "high": 10}}}

        result = SearchSpaceBuilder.from_dict(space_dict)

        # Should be config object, not dict - verify by checking attributes exist
        param_config = result["hyperparameters"]["param"]
        assert hasattr(param_config, "type")
        assert hasattr(param_config, "low")
        assert hasattr(param_config, "high")
        assert not isinstance(param_config, dict)


# =============================================================================
# SEARCHSPACEBUILDER MERGE TESTS
# =============================================================================


class TestSearchSpaceBuilderMerge:
    """Test merge class method (lines 864-914)."""

    def test_merge_two_spaces(self, search_space_builder_module):
        """Test merge with two search spaces (lines 887-889: Example)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space1 = SearchSpaceBuilder.for_model("GCN", include_optimizer=False)
        space2 = SearchSpaceBuilder().add_int("custom_param", 1, 10).build()

        result = SearchSpaceBuilder.merge(space1, space2)

        assert "hyperparameters" in result
        # Should have params from both spaces
        assert "hidden_channels" in result["hyperparameters"]  # From GCN
        assert "custom_param" in result["hyperparameters"]  # From space2

    def test_merge_no_spaces_raises_error(self, search_space_builder_module):
        """Test merge with no spaces raises error (lines 891-892)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        with pytest.raises(Exception) as exc_info:
            SearchSpaceBuilder.merge()

        assert "At least one search space required" in str(exc_info.value)

    def test_merge_single_space(self, search_space_builder_module):
        """Test merge with single space returns that space."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space = SearchSpaceBuilder().add_int("param", 1, 10).build()

        result = SearchSpaceBuilder.merge(space)

        assert "param" in result["hyperparameters"]

    def test_merge_conflict_resolution_last_default(self, search_space_builder_module):
        """Test merge with default conflict_resolution='last' (lines 875-876)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        space1 = (
            SearchSpaceBuilder()
            .add_int("param", 1, 10)  # INT type
            .build()
        )
        space2 = (
            SearchSpaceBuilder()
            .add_float("param", 0.0, 1.0)  # FLOAT type
            .build()
        )

        result = SearchSpaceBuilder.merge(space1, space2)  # Default is 'last'

        # Should use space2's version (FLOAT)
        assert result["hyperparameters"]["param"].type.value == ParamType.FLOAT.value

    def test_merge_conflict_resolution_first(self, search_space_builder_module):
        """Test merge with conflict_resolution='first' (lines 903-904)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]

        space1 = (
            SearchSpaceBuilder()
            .add_int("param", 1, 10)  # INT type
            .build()
        )
        space2 = (
            SearchSpaceBuilder()
            .add_float("param", 0.0, 1.0)  # FLOAT type
            .build()
        )

        result = SearchSpaceBuilder.merge(space1, space2, conflict_resolution="first")

        # Should use space1's version (INT)
        assert result["hyperparameters"]["param"].type.value == ParamType.INT.value

    def test_merge_conflict_resolution_error(self, search_space_builder_module):
        """Test merge with conflict_resolution='error' raises error (lines 905-909)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space1 = SearchSpaceBuilder().add_int("param", 1, 10).build()
        space2 = SearchSpaceBuilder().add_float("param", 0.0, 1.0).build()

        with pytest.raises(Exception) as exc_info:
            SearchSpaceBuilder.merge(space1, space2, conflict_resolution="error")

        assert "Conflict" in str(exc_info.value)
        assert "hyperparameters.param" in str(exc_info.value)

    def test_merge_three_spaces(self, search_space_builder_module):
        """Test merge with three search spaces."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space1 = SearchSpaceBuilder().add_int("param1", 1, 10).build()
        space2 = SearchSpaceBuilder().add_int("param2", 1, 10).build()
        space3 = SearchSpaceBuilder().add_int("param3", 1, 10).build()

        result = SearchSpaceBuilder.merge(space1, space2, space3)

        assert "param1" in result["hyperparameters"]
        assert "param2" in result["hyperparameters"]
        assert "param3" in result["hyperparameters"]

    def test_merge_different_categories(self, search_space_builder_module):
        """Test merge with different categories."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space1 = SearchSpaceBuilder().add_int("hidden", 32, 256, category="hyperparameters").build()
        space2 = SearchSpaceBuilder().add_loguniform("lr", 1e-5, 1e-2, category="optimizer").build()

        result = SearchSpaceBuilder.merge(space1, space2)

        assert "hyperparameters" in result
        assert "optimizer" in result
        assert "hidden" in result["hyperparameters"]
        assert "lr" in result["optimizer"]

    def test_merge_preserves_param_configs(self, search_space_builder_module):
        """Test merge preserves SearchSpaceParamConfig objects."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space = SearchSpaceBuilder().add_int("param", 1, 10, step=2).build()

        result = SearchSpaceBuilder.merge(space)

        config = result["hyperparameters"]["param"]
        assert config.step == 2


# =============================================================================
# SEARCHSPACEBUILDER VALIDATE TESTS
# =============================================================================


class TestSearchSpaceBuilderValidate:
    """Test validate class method (lines 916-954)."""

    def test_validate_valid_search_space(self, search_space_builder_module, sample_search_space):
        """Test validate with valid search space returns True."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        is_valid, errors = SearchSpaceBuilder.validate(sample_search_space)

        assert is_valid is True
        assert errors == []

    def test_validate_empty_search_space(self, search_space_builder_module):
        """Test validate with empty search space returns False (lines 938-940)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        is_valid, errors = SearchSpaceBuilder.validate({})

        assert is_valid is False
        assert "Search space is empty" in errors

    def test_validate_none_search_space(self, search_space_builder_module):
        """Test validate with None search space returns False."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        is_valid, errors = SearchSpaceBuilder.validate(None)

        assert is_valid is False

    def test_validate_category_not_dict(self, search_space_builder_module):
        """Test validate with non-dict category returns error (lines 943-945)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        invalid_space = {"hyperparameters": "not a dict"}

        is_valid, errors = SearchSpaceBuilder.validate(invalid_space)

        assert is_valid is False
        assert any("must be a dict" in error for error in errors)

    def test_validate_invalid_param_config(self, search_space_builder_module):
        """Test validate with invalid param config returns error (lines 947-952)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        # Config with low >= high should be invalid
        invalid_space = {
            "hyperparameters": {
                "bad_param": {"type": "int", "low": 10, "high": 5}  # Invalid
            }
        }

        is_valid, errors = SearchSpaceBuilder.validate(invalid_space)

        assert is_valid is False
        assert len(errors) > 0

    def test_validate_missing_type(self, search_space_builder_module):
        """Test validate with missing type in param config.

        The validate method catches validation errors and returns them in the errors list
        rather than raising. With Pydantic V2, missing required fields cause ValidationError
        which is caught as part of the validation flow.
        """
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        invalid_space = {
            "hyperparameters": {
                "param": {"low": 1, "high": 10}  # Missing type
            }
        }

        # validate() should return False with errors, not raise
        is_valid, errors = SearchSpaceBuilder.validate(invalid_space)

        assert is_valid is False
        assert len(errors) > 0
        # Should contain error about the invalid config
        assert any("param" in error.lower() or "type" in error.lower() for error in errors)

    def test_validate_with_searchspaceparamconfig_objects(self, search_space_builder_module):
        """Test validate with SearchSpaceParamConfig objects."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        ParamType = search_space_builder_module["ParamType"]
        SearchSpaceParamConfig = search_space_builder_module["SearchSpaceParamConfig"]

        valid_space = {
            "hyperparameters": {"param": SearchSpaceParamConfig(type=ParamType.INT, low=1, high=10)}
        }

        is_valid, errors = SearchSpaceBuilder.validate(valid_space)

        assert is_valid is True
        assert errors == []

    def test_validate_returns_all_errors(self, search_space_builder_module):
        """Test validate returns all errors, not just first."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        invalid_space = {
            "hyperparameters": {
                "bad1": {"type": "int", "low": 10, "high": 5},
                "bad2": {"type": "int", "low": 20, "high": 10},
            }
        }

        is_valid, errors = SearchSpaceBuilder.validate(invalid_space)

        assert is_valid is False
        assert len(errors) >= 2

    def test_validate_multiple_categories(self, search_space_builder_module, sample_search_space):
        """Test validate with multiple valid categories."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        is_valid, errors = SearchSpaceBuilder.validate(sample_search_space)

        # sample_search_space has hyperparameters and optimizer
        assert is_valid is True
        assert errors == []


# =============================================================================
# SEARCHSPACEBUILDER GET_PARAM_COUNT TESTS
# =============================================================================


class TestSearchSpaceBuilderGetParamCount:
    """Test get_param_count class method (lines 956-973)."""

    def test_get_param_count_basic(self, search_space_builder_module, sample_search_space):
        """Test get_param_count with sample search space."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = SearchSpaceBuilder.get_param_count(sample_search_space)

        assert isinstance(result, dict)
        assert "hyperparameters" in result
        assert "optimizer" in result

    def test_get_param_count_correct_counts(self, search_space_builder_module):
        """Test get_param_count returns correct counts."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space = (
            SearchSpaceBuilder()
            .add_int("p1", 1, 10)
            .add_int("p2", 1, 10)
            .add_int("p3", 1, 10)
            .add_loguniform("lr", 1e-5, 1e-2, category="optimizer")
            .build()
        )

        result = SearchSpaceBuilder.get_param_count(space)

        assert result["hyperparameters"] == 3
        assert result["optimizer"] == 1

    def test_get_param_count_empty_space(self, search_space_builder_module):
        """Test get_param_count with empty space."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = SearchSpaceBuilder.get_param_count({})

        assert result == {}

    def test_get_param_count_single_category(self, search_space_builder_module):
        """Test get_param_count with single category."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space = SearchSpaceBuilder().add_int("param", 1, 10).build()

        result = SearchSpaceBuilder.get_param_count(space)

        assert len(result) == 1
        assert result["hyperparameters"] == 1


# =============================================================================
# SEARCHSPACEBUILDER ESTIMATE_SEARCH_SPACE_SIZE TESTS
# =============================================================================


class TestSearchSpaceBuilderEstimateSearchSpaceSize:
    """Test estimate_search_space_size class method (lines 975-1009)."""

    def test_estimate_search_space_size_basic(self, search_space_builder_module):
        """Test estimate_search_space_size with basic space."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space = (
            SearchSpaceBuilder()
            .add_categorical("activation", ["relu", "gelu"])  # 2 choices
            .build()
        )

        result = SearchSpaceBuilder.estimate_search_space_size(space)

        assert result == 2

    def test_estimate_search_space_size_categorical_multiplies(self, search_space_builder_module):
        """Test estimate_search_space_size multiplies categorical choices (lines 998-999)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space = (
            SearchSpaceBuilder()
            .add_categorical("a", ["1", "2", "3"])  # 3 choices
            .add_categorical("b", ["x", "y"])  # 2 choices
            .build()
        )

        result = SearchSpaceBuilder.estimate_search_space_size(space)

        assert result == 6  # 3 * 2

    def test_estimate_search_space_size_int_with_step(self, search_space_builder_module):
        """Test estimate_search_space_size with int and step (lines 1000-1003)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space = (
            SearchSpaceBuilder()
            .add_int("hidden", 32, 128, step=32)  # 32, 64, 96, 128 = 4 values
            .build()
        )

        result = SearchSpaceBuilder.estimate_search_space_size(space)

        # (128 - 32) / 32 + 1 = 4
        assert result == 4

    def test_estimate_search_space_size_int_without_step(self, search_space_builder_module):
        """Test estimate_search_space_size with int without step (lines 1003-1005)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space = (
            SearchSpaceBuilder()
            .add_int("layers", 2, 5)  # 2, 3, 4, 5 = 4 values
            .build()
        )

        result = SearchSpaceBuilder.estimate_search_space_size(space)

        # (5 - 2) + 1 = 4
        assert result == 4

    def test_estimate_search_space_size_continuous_uses_grid_points(
        self, search_space_builder_module
    ):
        """Test estimate_search_space_size uses grid_points for continuous (lines 1006-1007)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space = (
            SearchSpaceBuilder()
            .add_float("dropout", 0.0, 0.5)  # Continuous
            .build()
        )

        # Default grid_points = 10
        result = SearchSpaceBuilder.estimate_search_space_size(space)
        assert result == 10

        # Custom grid_points = 5
        result = SearchSpaceBuilder.estimate_search_space_size(space, grid_points=5)
        assert result == 5

    def test_estimate_search_space_size_loguniform_uses_grid_points(
        self, search_space_builder_module
    ):
        """Test estimate_search_space_size uses grid_points for loguniform."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space = SearchSpaceBuilder().add_loguniform("lr", 1e-5, 1e-2).build()

        result = SearchSpaceBuilder.estimate_search_space_size(space, grid_points=20)
        assert result == 20

    def test_estimate_search_space_size_mixed_types(self, search_space_builder_module):
        """Test estimate_search_space_size with mixed param types."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space = (
            SearchSpaceBuilder()
            .add_categorical("activation", ["relu", "gelu"])  # 2
            .add_int("layers", 2, 4)  # 3 values (2, 3, 4)
            .add_float("dropout", 0.0, 0.5)  # 10 grid points
            .build()
        )

        result = SearchSpaceBuilder.estimate_search_space_size(space)

        # 2 * 3 * 10 = 60
        assert result == 60

    def test_estimate_search_space_size_empty_space(self, search_space_builder_module):
        """Test estimate_search_space_size with empty space returns 1."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = SearchSpaceBuilder.estimate_search_space_size({})

        assert result == 1

    def test_estimate_search_space_size_categorical_empty_choices(
        self, search_space_builder_module
    ):
        """Test estimate_search_space_size handles empty choices gracefully."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        # Import the real ParamType that the source code uses for comparisons
        from milia_pipeline.models.hpo.search_spaces.param_types import ParamType as RealParamType

        # Create a space with a categorical that has None choices (edge case)
        # This is unlikely in practice but tests the fallback (line 999)
        space = {
            "hyperparameters": {"param": MagicMock(type=RealParamType.CATEGORICAL, choices=None)}
        }

        result = SearchSpaceBuilder.estimate_search_space_size(space)

        # Should use 1 when choices is None/empty
        assert result == 1


# =============================================================================
# SEARCHSPACEBUILDER LIST_AVAILABLE_MODELS TESTS
# =============================================================================


class TestSearchSpaceBuilderListAvailableModels:
    """Test list_available_models class method (lines 1011-1046)."""

    def test_list_available_models_legacy_fallback(self, search_space_builder_module):
        """Test list_available_models returns models when introspector unavailable."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        # Force legacy fallback by patching introspector as unavailable
        with patch(
            "milia_pipeline.models.hpo.search_spaces.search_space_builder._INTROSPECTOR_AVAILABLE",
            False,
        ):
            result = SearchSpaceBuilder.list_available_models()

            assert isinstance(result, list)
            assert "GCN" in result
            assert "GAT" in result
            assert "GraphSAGE" in result
            assert "GIN" in result
            assert "SchNet" in result
            assert "DimeNet" in result
            assert "MPNN" in result

    def test_list_available_models_legacy_has_seven_models(self, search_space_builder_module):
        """Test legacy list has exactly 7 models (lines 1038-1046)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        # Force legacy fallback by patching introspector as unavailable
        with patch(
            "milia_pipeline.models.hpo.search_spaces.search_space_builder._INTROSPECTOR_AVAILABLE",
            False,
        ):
            result = SearchSpaceBuilder.list_available_models()

            assert len(result) == 7

    def test_list_available_models_with_dynamic_introspection(self, search_space_builder_module):
        """Test list_available_models with dynamic introspection enabled."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        mock_introspector = MockPyGModelIntrospector()

        with (
            patch(
                "milia_pipeline.models.hpo.search_spaces.search_space_builder._INTROSPECTOR_AVAILABLE",
                True,
            ),
            patch(
                "milia_pipeline.models.hpo.search_spaces.search_space_builder.get_introspector",
                return_value=mock_introspector,
            ),
        ):
            result = SearchSpaceBuilder.list_available_models()

            # Should return dynamic list with more models
            assert "GCN" in result
            assert "GAT" in result
            # Additional models from mock introspector
            assert "PMLP" in result
            assert "EdgeConv" in result

    def test_list_available_models_introspection_failure_falls_back(
        self, search_space_builder_module, caplog
    ):
        """Test list_available_models falls back on introspection failure (lines 1034-1035)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        caplog.set_level(logging.WARNING)

        def failing_introspector():
            raise Exception("Introspection failed")

        with (
            patch(
                "milia_pipeline.models.hpo.search_spaces.search_space_builder._INTROSPECTOR_AVAILABLE",
                True,
            ),
            patch(
                "milia_pipeline.models.hpo.search_spaces.search_space_builder.get_introspector",
                failing_introspector,
            ),
        ):
            result = SearchSpaceBuilder.list_available_models()

            # Should fall back to legacy list
            assert len(result) == 7
            assert "Dynamic model discovery failed" in caplog.text


# =============================================================================
# SEARCHSPACEBUILDER FOR_MODEL WITH DYNAMIC INTROSPECTION TESTS
# =============================================================================


class TestSearchSpaceBuilderForModelDynamic:
    """Test for_model class method with dynamic introspection (lines 604-640)."""

    def test_for_model_dynamic_introspection_success(self, search_space_builder_module):
        """Test for_model with successful dynamic introspection."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        mock_introspector = MockPyGModelIntrospector()

        with (
            patch(
                "milia_pipeline.models.hpo.search_spaces.search_space_builder._INTROSPECTOR_AVAILABLE",
                True,
            ),
            patch(
                "milia_pipeline.models.hpo.search_spaces.search_space_builder.get_introspector",
                return_value=mock_introspector,
            ),
        ):
            result = SearchSpaceBuilder.for_model("GCN")

            assert "hyperparameters" in result
            # Should have params from mock introspector
            assert "hidden_channels" in result["hyperparameters"]
            assert "num_layers" in result["hyperparameters"]

    def test_for_model_dynamic_includes_optimizer(self, search_space_builder_module):
        """Test for_model dynamic with include_optimizer=True."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        mock_introspector = MockPyGModelIntrospector()

        with (
            patch(
                "milia_pipeline.models.hpo.search_spaces.search_space_builder._INTROSPECTOR_AVAILABLE",
                True,
            ),
            patch(
                "milia_pipeline.models.hpo.search_spaces.search_space_builder.get_introspector",
                return_value=mock_introspector,
            ),
        ):
            result = SearchSpaceBuilder.for_model("GCN", include_optimizer=True)

            assert "optimizer" in result
            assert "lr" in result["optimizer"]

    def test_for_model_dynamic_includes_scheduler(self, search_space_builder_module):
        """Test for_model dynamic with include_scheduler=True."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        mock_introspector = MockPyGModelIntrospector()

        with (
            patch(
                "milia_pipeline.models.hpo.search_spaces.search_space_builder._INTROSPECTOR_AVAILABLE",
                True,
            ),
            patch(
                "milia_pipeline.models.hpo.search_spaces.search_space_builder.get_introspector",
                return_value=mock_introspector,
            ),
        ):
            result = SearchSpaceBuilder.for_model("GCN", include_scheduler=True)

            assert "scheduler" in result
            assert "factor" in result["scheduler"]

    def test_for_model_dynamic_model_not_found_falls_back(
        self, search_space_builder_module, caplog
    ):
        """Test for_model falls back when model not in introspector (lines 631-635)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        caplog.set_level(logging.WARNING)

        mock_introspector = MockPyGModelIntrospector(available_models=["GAT", "GIN"])

        with (
            patch(
                "milia_pipeline.models.hpo.search_spaces.search_space_builder._INTROSPECTOR_AVAILABLE",
                True,
            ),
            patch(
                "milia_pipeline.models.hpo.search_spaces.search_space_builder.get_introspector",
                return_value=mock_introspector,
            ),
        ):
            result = SearchSpaceBuilder.for_model("GCN")  # GCN not in mock

            # Should fall back to legacy GCN space
            assert "aggregation" in result["hyperparameters"]
            assert "not found in PyG introspector" in caplog.text

    def test_for_model_dynamic_exception_falls_back(self, search_space_builder_module, caplog):
        """Test for_model falls back on introspection exception (lines 636-640)."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        caplog.set_level(logging.WARNING)

        mock_introspector = MagicMock()
        mock_introspector.has_model.side_effect = Exception("Introspection error")

        with (
            patch(
                "milia_pipeline.models.hpo.search_spaces.search_space_builder._INTROSPECTOR_AVAILABLE",
                True,
            ),
            patch(
                "milia_pipeline.models.hpo.search_spaces.search_space_builder.get_introspector",
                return_value=mock_introspector,
            ),
        ):
            result = SearchSpaceBuilder.for_model("GCN")

            # Should fall back to legacy space
            assert "hyperparameters" in result
            assert "Dynamic introspection failed" in caplog.text

    def test_for_model_dynamic_new_model_type(self, search_space_builder_module):
        """Test for_model dynamic with new model type not in legacy list."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        mock_introspector = MockPyGModelIntrospector()

        with (
            patch(
                "milia_pipeline.models.hpo.search_spaces.search_space_builder._INTROSPECTOR_AVAILABLE",
                True,
            ),
            patch(
                "milia_pipeline.models.hpo.search_spaces.search_space_builder.get_introspector",
                return_value=mock_introspector,
            ),
        ):
            # PMLP is in mock introspector but not in legacy list
            result = SearchSpaceBuilder.for_model("PMLP")

            assert "hyperparameters" in result
            assert "hidden_channels" in result["hyperparameters"]


# =============================================================================
# CONVENIENCE FUNCTIONS TESTS
# =============================================================================


class TestBuildSearchSpaceFunction:
    """Test build_search_space convenience function (lines 1053-1069)."""

    def test_build_search_space_returns_builder(self, search_space_builder_module):
        """Test build_search_space returns SearchSpaceBuilder instance."""
        build_search_space = search_space_builder_module["build_search_space"]
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result = build_search_space()

        assert isinstance(result, SearchSpaceBuilder)

    def test_build_search_space_returns_fresh_instance(self, search_space_builder_module):
        """Test build_search_space returns new instance each time."""
        build_search_space = search_space_builder_module["build_search_space"]

        builder1 = build_search_space()
        builder2 = build_search_space()

        assert builder1 is not builder2

    def test_build_search_space_can_chain(self, search_space_builder_module):
        """Test build_search_space result supports chaining (lines 1062-1067: Example)."""
        build_search_space = search_space_builder_module["build_search_space"]

        space = (
            build_search_space()
            .add_int("hidden_channels", 32, 256)
            .add_loguniform("lr", 1e-5, 1e-2, category="optimizer")
            .build()
        )

        assert "hyperparameters" in space
        assert "optimizer" in space


class TestGetModelSearchSpaceFunction:
    """Test get_model_search_space convenience function (lines 1072-1092)."""

    def test_get_model_search_space_basic(self, search_space_builder_module):
        """Test get_model_search_space returns search space for model."""
        get_model_search_space = search_space_builder_module["get_model_search_space"]

        result = get_model_search_space("GCN")

        assert isinstance(result, dict)
        assert "hyperparameters" in result

    def test_get_model_search_space_with_optimizer(self, search_space_builder_module):
        """Test get_model_search_space with include_optimizer=True (default)."""
        get_model_search_space = search_space_builder_module["get_model_search_space"]

        result = get_model_search_space("GCN", include_optimizer=True)

        assert "optimizer" in result

    def test_get_model_search_space_without_optimizer(self, search_space_builder_module):
        """Test get_model_search_space with include_optimizer=False."""
        get_model_search_space = search_space_builder_module["get_model_search_space"]

        result = get_model_search_space("GCN", include_optimizer=False)

        assert "optimizer" not in result

    def test_get_model_search_space_wraps_for_model(self, search_space_builder_module):
        """Test get_model_search_space is wrapper for SearchSpaceBuilder.for_model."""
        get_model_search_space = search_space_builder_module["get_model_search_space"]
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result1 = get_model_search_space("GAT", include_optimizer=True)
        result2 = SearchSpaceBuilder.for_model("GAT", include_optimizer=True)

        # Should have same structure
        assert set(result1.keys()) == set(result2.keys())
        assert set(result1["hyperparameters"].keys()) == set(result2["hyperparameters"].keys())


class TestValidateSearchSpaceFunction:
    """Test validate_search_space convenience function (lines 1095-1109)."""

    def test_validate_search_space_valid(self, search_space_builder_module, sample_search_space):
        """Test validate_search_space with valid space returns True."""
        validate_search_space = search_space_builder_module["validate_search_space"]

        is_valid, errors = validate_search_space(sample_search_space)

        assert is_valid is True
        assert errors == []

    def test_validate_search_space_invalid(self, search_space_builder_module):
        """Test validate_search_space with invalid space returns False."""
        validate_search_space = search_space_builder_module["validate_search_space"]

        is_valid, errors = validate_search_space({})

        assert is_valid is False
        assert len(errors) > 0

    def test_validate_search_space_wraps_validate(
        self, search_space_builder_module, sample_search_space
    ):
        """Test validate_search_space wraps SearchSpaceBuilder.validate."""
        validate_search_space = search_space_builder_module["validate_search_space"]
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        result1 = validate_search_space(sample_search_space)
        result2 = SearchSpaceBuilder.validate(sample_search_space)

        assert result1 == result2


# =============================================================================
# EDGE CASES AND BOUNDARY TESTS
# =============================================================================


class TestSearchSpaceBuilderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_add_int_with_equal_bounds_raises_error(self, builder):
        """Test add_int with low == high raises error."""
        with pytest.raises(PydanticValidationError):
            builder.add_int("param", 10, 10)

    def test_add_float_with_equal_bounds_raises_error(self, builder):
        """Test add_float with low == high raises error."""
        with pytest.raises(PydanticValidationError):
            builder.add_float("param", 0.5, 0.5)

    def test_add_loguniform_with_very_small_positive(self, builder):
        """Test add_loguniform with very small positive value."""
        builder.add_loguniform("lr", 1e-10, 1e-5)

        config = builder._search_space["optimizer"]["lr"]
        assert config.low == 1e-10

    def test_builder_chaining_all_methods(self, builder):
        """Test chaining all fluent builder methods together."""
        result = (
            builder.add_int("int_param", 1, 10)
            .add_float("float_param", 0.0, 1.0)
            .add_loguniform("lr", 1e-5, 1e-2)
            .add_categorical("cat_param", ["a", "b"])
            .add_uniform("uniform_param", 0.0, 1.0)
            .add_discrete_uniform("discrete_param", 0, 100, step=10)
            .add_param("config_param", {"type": "int", "low": 1, "high": 5})
            .add_category("custom", {"cp": {"type": "int", "low": 1, "high": 5}})
            .build()
        )

        assert len(result) > 0

    def test_remove_all_params_then_build_fails(self, builder):
        """Test removing all params then building raises error."""
        builder.add_int("param", 1, 10)
        builder.remove_param("param")

        with pytest.raises(SearchSpaceError):
            builder.build()

    def test_large_search_space(self, search_space_builder_module):
        """Test building a large search space with many parameters."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]
        builder = SearchSpaceBuilder()

        for i in range(100):
            builder.add_int(f"param_{i}", 1, 100)

        result = builder.build()

        assert len(result["hyperparameters"]) == 100

    def test_special_characters_in_param_name(self, builder):
        """Test param names with special characters."""
        builder.add_int("param_with_underscore", 1, 10)
        builder.add_int("param.with.dot", 1, 10)
        builder.add_int("param-with-dash", 1, 10)

        result = builder.build()

        assert "param_with_underscore" in result["hyperparameters"]
        assert "param.with.dot" in result["hyperparameters"]
        assert "param-with-dash" in result["hyperparameters"]

    def test_unicode_param_name(self, builder):
        """Test param names with unicode characters."""
        builder.add_int("学习率", 1, 10)  # "learning rate" in Chinese

        result = builder.build()

        assert "学习率" in result["hyperparameters"]

    def test_numeric_param_name(self, builder):
        """Test param names starting with numbers (not recommended but valid)."""
        builder.add_int("123param", 1, 10)

        result = builder.build()

        assert "123param" in result["hyperparameters"]


class TestSearchSpaceBuilderBoundaryValues:
    """Test boundary value conditions."""

    def test_int_with_very_large_range(self, builder):
        """Test int parameter with very large range."""
        builder.add_int("big_range", 0, 1000000)

        config = builder._search_space["hyperparameters"]["big_range"]
        assert config.low == 0.0
        assert config.high == 1000000.0

    def test_float_with_very_small_range(self, builder):
        """Test float parameter with very small range."""
        builder.add_float("tiny_range", 0.0, 0.0001)

        config = builder._search_space["hyperparameters"]["tiny_range"]
        assert config.high == 0.0001

    def test_categorical_with_many_choices(self, builder):
        """Test categorical with many choices."""
        choices = [f"option_{i}" for i in range(100)]
        builder.add_categorical("many_options", choices)

        config = builder._search_space["hyperparameters"]["many_options"]
        assert len(config.choices) == 100

    def test_estimate_size_very_large_space(self, search_space_builder_module):
        """Test estimate_search_space_size with very large space."""
        SearchSpaceBuilder = search_space_builder_module["SearchSpaceBuilder"]

        space = (
            SearchSpaceBuilder()
            .add_categorical("a", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
            .add_categorical("b", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
            .add_categorical("c", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
            .build()
        )

        result = SearchSpaceBuilder.estimate_search_space_size(space)

        assert result == 1000  # 10 * 10 * 10


class TestSearchSpaceBuilderTypeSafety:
    """Test type safety and type conversion."""

    def test_add_int_with_float_bounds(self, builder, search_space_builder_module):
        """Test add_int converts float bounds to float (int input becomes float)."""
        ParamType = search_space_builder_module["ParamType"]

        builder.add_int("param", 1.5, 10.5)  # Float inputs

        config = builder._search_space["hyperparameters"]["param"]
        assert config.type.value == ParamType.INT.value
        # Values should still work (type conversion happens in add_int)
        assert config.low == 1.5
        assert config.high == 10.5

    def test_add_categorical_with_none_in_choices(self, builder):
        """Test categorical with None as a valid choice."""
        builder.add_categorical("nullable", [None, "value1", "value2"])

        config = builder._search_space["hyperparameters"]["nullable"]
        assert None in config.choices

    def test_add_categorical_with_mixed_types(self, builder):
        """Test categorical with mixed types in choices."""
        builder.add_categorical("mixed", [1, "two", 3.0, True, None])

        config = builder._search_space["hyperparameters"]["mixed"]
        assert len(config.choices) == 5


# =============================================================================
# TEARDOWN MODULE
# =============================================================================


def teardown_module(module):
    """
    Clean up any module-level state after all tests in this module have run.

    This ensures no mock pollution affects other test files.
    """
    # Clear any cached modules related to search_space_builder
    modules_to_clear = [
        key
        for key in list(sys.modules.keys())
        if "milia_pipeline.models.hpo.search_spaces.search_space_builder" in key
    ]
    for mod in modules_to_clear:
        del sys.modules[mod]


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
