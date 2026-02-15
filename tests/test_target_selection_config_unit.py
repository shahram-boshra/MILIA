#!/usr/bin/env python3
"""
Complete Unit Test Suite for milia_pipeline/models/factory/target_selection_config.py Module

Tests target selection configuration including:
- SelectionMode enum
- TargetLevel enum (NEW)
- TargetSource enum (NEW)
- TargetSelectionConfig dataclass
- from_config() class method (with target_level, target_source parsing)
- _normalize_list() static method
- infer_level_from_task_type() class method (NEW)
- infer_source_from_level() class method (NEW)
- resolve_for_task() method (NEW)
- _resolve_source_auto() method (NEW)
- resolve() method
- _resolve_properties(), _resolve_indices(), _resolve_range() methods
- to_dict() method
- __repr__() method

This is a PRODUCTION-READY test suite with comprehensive coverage.

Aligned with:
- target_selection_config.py: Target selection configuration container (v1.1.0, Pydantic V2)
- config.yaml: target_selection section with target_level, target_source

Author: Milia Team
Version: 2.0.0
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from unittest.mock import MagicMock

import pytest

# =============================================================================
# IMPORTS WITH GRACEFUL FALLBACK
# =============================================================================

try:
    from milia_pipeline.models.factory.target_selection_config import (
        SelectionMode,
        TargetLevel,
        TargetSelectionConfig,
        TargetSource,
    )

    TARGET_SELECTION_AVAILABLE = True
except ImportError:
    TARGET_SELECTION_AVAILABLE = False
    SelectionMode = None
    TargetLevel = None
    TargetSource = None
    TargetSelectionConfig = None

try:
    from milia_pipeline.exceptions import ConfigurationError

    EXCEPTIONS_AVAILABLE = True
except ImportError:
    EXCEPTIONS_AVAILABLE = False
    ConfigurationError = Exception


# =============================================================================
# SKIP MARKERS
# =============================================================================

skip_if_not_available = pytest.mark.skipif(
    not TARGET_SELECTION_AVAILABLE, reason="target_selection_config module not available"
)


# =============================================================================
# TEST: SelectionMode Enum
# =============================================================================


@skip_if_not_available
class TestSelectionModeEnum:
    """Test SelectionMode enum values and behavior."""

    def test_properties_mode_exists(self):
        """Test PROPERTIES mode exists."""
        assert hasattr(SelectionMode, "PROPERTIES")

    def test_indices_mode_exists(self):
        """Test INDICES mode exists."""
        assert hasattr(SelectionMode, "INDICES")

    def test_range_mode_exists(self):
        """Test RANGE mode exists."""
        assert hasattr(SelectionMode, "RANGE")

    def test_all_mode_exists(self):
        """Test ALL mode exists."""
        assert hasattr(SelectionMode, "ALL")

    def test_modes_are_unique(self):
        """Test all modes have unique values."""
        modes = [
            SelectionMode.PROPERTIES,
            SelectionMode.INDICES,
            SelectionMode.RANGE,
            SelectionMode.ALL,
        ]
        values = [m.value for m in modes]
        assert len(values) == len(set(values))


# =============================================================================
# TEST: TargetLevel Enum (NEW)
# =============================================================================


@skip_if_not_available
class TestTargetLevelEnum:
    """Test TargetLevel enum values and behavior."""

    def test_graph_level_exists(self):
        """Test GRAPH level exists."""
        assert hasattr(TargetLevel, "GRAPH")
        assert TargetLevel.GRAPH.value == "graph"

    def test_node_level_exists(self):
        """Test NODE level exists."""
        assert hasattr(TargetLevel, "NODE")
        assert TargetLevel.NODE.value == "node"

    def test_edge_level_exists(self):
        """Test EDGE level exists."""
        assert hasattr(TargetLevel, "EDGE")
        assert TargetLevel.EDGE.value == "edge"

    def test_levels_are_unique(self):
        """Test all levels have unique values."""
        levels = [TargetLevel.GRAPH, TargetLevel.NODE, TargetLevel.EDGE]
        values = [l.value for l in levels]
        assert len(values) == len(set(values))

    def test_level_value_types(self):
        """Test level values are strings."""
        for level in [TargetLevel.GRAPH, TargetLevel.NODE, TargetLevel.EDGE]:
            assert isinstance(level.value, str)


# =============================================================================
# TEST: TargetSource Enum (NEW)
# =============================================================================


@skip_if_not_available
class TestTargetSourceEnum:
    """Test TargetSource enum values and behavior."""

    def test_y_source_exists(self):
        """Test Y source exists."""
        assert hasattr(TargetSource, "Y")
        assert TargetSource.Y.value == "y"

    def test_x_source_exists(self):
        """Test X source exists."""
        assert hasattr(TargetSource, "X")
        assert TargetSource.X.value == "x"

    def test_edge_attr_source_exists(self):
        """Test EDGE_ATTR source exists."""
        assert hasattr(TargetSource, "EDGE_ATTR")
        assert TargetSource.EDGE_ATTR.value == "edge_attr"

    def test_edge_label_source_exists(self):
        """Test EDGE_LABEL source exists."""
        assert hasattr(TargetSource, "EDGE_LABEL")
        assert TargetSource.EDGE_LABEL.value == "edge_label"

    def test_edge_y_source_exists(self):
        """Test EDGE_Y source exists."""
        assert hasattr(TargetSource, "EDGE_Y")
        assert TargetSource.EDGE_Y.value == "edge_y"

    def test_custom_source_exists(self):
        """Test CUSTOM source exists."""
        assert hasattr(TargetSource, "CUSTOM")
        assert TargetSource.CUSTOM.value == "custom"

    def test_sources_are_unique(self):
        """Test all sources have unique values."""
        sources = [
            TargetSource.Y,
            TargetSource.X,
            TargetSource.EDGE_ATTR,
            TargetSource.EDGE_LABEL,
            TargetSource.EDGE_Y,
            TargetSource.CUSTOM,
        ]
        values = [s.value for s in sources]
        assert len(values) == len(set(values))


# =============================================================================
# TEST: TargetSelectionConfig Dataclass
# =============================================================================


@skip_if_not_available
class TestTargetSelectionConfigDataclass:
    """Test TargetSelectionConfig dataclass fields and defaults."""

    def test_default_mode_is_all(self):
        """Test default mode is ALL."""
        config = TargetSelectionConfig()
        assert config.mode == SelectionMode.ALL

    def test_default_properties_is_none(self):
        """Test default properties is None."""
        config = TargetSelectionConfig()
        assert config.properties is None

    def test_default_indices_is_none(self):
        """Test default indices is None."""
        config = TargetSelectionConfig()
        assert config.indices is None

    def test_default_range_spec_is_none(self):
        """Test default range_spec is None."""
        config = TargetSelectionConfig()
        assert config.range_spec is None

    def test_default_strict_is_true(self):
        """Test default strict is True."""
        config = TargetSelectionConfig()
        assert config.strict is True

    def test_default_config_level_is_auto(self):
        """Test default config_level is 'auto'."""
        config = TargetSelectionConfig()
        assert config.config_level == "auto"

    def test_default_config_source_is_auto(self):
        """Test default config_source is 'auto'."""
        config = TargetSelectionConfig()
        assert config.config_source == "auto"

    def test_default_resolved_level_is_none(self):
        """Test default resolved_level is None."""
        config = TargetSelectionConfig()
        assert config.resolved_level is None

    def test_default_resolved_source_is_none(self):
        """Test default resolved_source is None."""
        config = TargetSelectionConfig()
        assert config.resolved_source is None

    def test_default_resolved_source_attr_is_none(self):
        """Test default resolved_source_attr is None."""
        config = TargetSelectionConfig()
        assert config.resolved_source_attr is None

    def test_raw_config_default_is_empty_dict(self):
        """Test raw_config default is empty dict."""
        config = TargetSelectionConfig()
        assert config.raw_config == {}


# =============================================================================
# TEST: from_config() Class Method
# =============================================================================


@skip_if_not_available
class TestFromConfigMethod:
    """Test TargetSelectionConfig.from_config() class method."""

    def test_none_config_returns_defaults(self):
        """Test None config returns default instance."""
        config = TargetSelectionConfig.from_config(None)
        assert config.mode == SelectionMode.ALL
        assert config.config_level == "auto"
        assert config.config_source == "auto"

    def test_empty_config_returns_defaults(self):
        """Test empty config returns default instance."""
        config = TargetSelectionConfig.from_config({})
        assert config.mode == SelectionMode.ALL

    def test_parses_properties_mode(self):
        """Test parses properties mode correctly."""
        config = TargetSelectionConfig.from_config({"properties": ["gap", "Etot"]})
        assert config.mode == SelectionMode.PROPERTIES
        assert config.properties == ["gap", "Etot"]

    def test_parses_single_property(self):
        """Test parses single property as list."""
        config = TargetSelectionConfig.from_config({"properties": "gap"})
        assert config.mode == SelectionMode.PROPERTIES
        assert config.properties == ["gap"]

    def test_parses_indices_mode(self):
        """Test parses indices mode correctly."""
        config = TargetSelectionConfig.from_config({"indices": [0, 1, 2]})
        assert config.mode == SelectionMode.INDICES
        assert config.indices == [0, 1, 2]

    def test_parses_single_index(self):
        """Test parses single index as list."""
        config = TargetSelectionConfig.from_config({"indices": 5})
        assert config.mode == SelectionMode.INDICES
        assert config.indices == [5]

    def test_parses_range_mode(self):
        """Test parses range string mode correctly."""
        config = TargetSelectionConfig.from_config({"indices": "0:3"})
        assert config.mode == SelectionMode.RANGE
        assert config.range_spec == "0:3"

    def test_parses_range_with_step(self):
        """Test parses range with step."""
        config = TargetSelectionConfig.from_config({"indices": "0:10:2"})
        assert config.mode == SelectionMode.RANGE
        assert config.range_spec == "0:10:2"

    def test_parses_strict_true(self):
        """Test parses strict=True."""
        config = TargetSelectionConfig.from_config({"strict": True})
        assert config.strict is True

    def test_parses_strict_false(self):
        """Test parses strict=False."""
        config = TargetSelectionConfig.from_config({"strict": False})
        assert config.strict is False

    def test_parses_target_level_auto(self):
        """Test parses target_level='auto'."""
        config = TargetSelectionConfig.from_config({"target_level": "auto"})
        assert config.config_level == "auto"

    def test_parses_target_level_graph(self):
        """Test parses target_level='graph'."""
        config = TargetSelectionConfig.from_config({"target_level": "graph"})
        assert config.config_level == "graph"

    def test_parses_target_level_node(self):
        """Test parses target_level='node'."""
        config = TargetSelectionConfig.from_config({"target_level": "node"})
        assert config.config_level == "node"

    def test_parses_target_level_edge(self):
        """Test parses target_level='edge'."""
        config = TargetSelectionConfig.from_config({"target_level": "edge"})
        assert config.config_level == "edge"

    def test_parses_target_level_case_insensitive(self):
        """Test parses target_level case-insensitively."""
        config = TargetSelectionConfig.from_config({"target_level": "NODE"})
        assert config.config_level == "node"

    def test_invalid_target_level_defaults_to_auto(self, caplog):
        """Test invalid target_level defaults to 'auto' with warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            config = TargetSelectionConfig.from_config({"target_level": "invalid"})
        assert config.config_level == "auto"
        assert "Invalid target_level" in caplog.text

    def test_parses_target_source_auto(self):
        """Test parses target_source='auto'."""
        config = TargetSelectionConfig.from_config({"target_source": "auto"})
        assert config.config_source == "auto"

    def test_parses_target_source_y(self):
        """Test parses target_source='y'."""
        config = TargetSelectionConfig.from_config({"target_source": "y"})
        assert config.config_source == "y"

    def test_parses_target_source_x(self):
        """Test parses target_source='x'."""
        config = TargetSelectionConfig.from_config({"target_source": "x"})
        assert config.config_source == "x"

    def test_parses_target_source_edge_attr(self):
        """Test parses target_source='edge_attr'."""
        config = TargetSelectionConfig.from_config({"target_source": "edge_attr"})
        assert config.config_source == "edge_attr"

    def test_parses_target_source_custom(self, caplog):
        """Test parses custom target_source with info log."""
        import logging

        with caplog.at_level(logging.INFO):
            config = TargetSelectionConfig.from_config({"target_source": "atomic_charges"})
        assert config.config_source == "atomic_charges"
        assert "not a standard PyG attribute" in caplog.text

    def test_stores_raw_config(self):
        """Test stores raw config for debugging."""
        raw = {"properties": ["gap"], "strict": True}
        config = TargetSelectionConfig.from_config(raw)
        assert config.raw_config == raw

    def test_raises_on_multiple_modes(self):
        """Test raises ConfigurationError when multiple modes specified."""
        with pytest.raises(Exception) as exc_info:  # ConfigurationError
            TargetSelectionConfig.from_config({"properties": ["gap"], "indices": [0, 1]})
        assert "Multiple selection modes" in str(exc_info.value)

    def test_returns_same_instance_if_already_config(self):
        """Test from_config returns same instance if already a TargetSelectionConfig.

        This is critical for integration with model_factory.py which may receive
        a TargetSelectionConfig object from hpo_manager.py instead of a raw dict.
        """
        original = TargetSelectionConfig.from_config(
            {"target_level": "node", "target_source": "x", "indices": [0, 1]}
        )

        # Calling from_config again with the instance should return the same object
        result = TargetSelectionConfig.from_config(original)

        assert result is original
        assert result.config_level == "node"
        assert result.config_source == "x"
        assert result.indices == [0, 1]

    def test_parses_all_true_mode(self):
        """Test parses all=True into ALL mode (line 170, 200-202)."""
        config = TargetSelectionConfig.from_config({"all": True})
        assert config.mode == SelectionMode.ALL

    def test_raises_on_properties_plus_all(self):
        """Test raises ConfigurationError when properties AND all specified."""
        with pytest.raises(Exception) as exc_info:
            TargetSelectionConfig.from_config({"properties": ["gap"], "all": True})
        assert "Multiple selection modes" in str(exc_info.value)

    def test_raises_on_indices_plus_all(self):
        """Test raises ConfigurationError when indices AND all specified."""
        with pytest.raises(Exception) as exc_info:
            TargetSelectionConfig.from_config({"indices": [0], "all": True})
        assert "Multiple selection modes" in str(exc_info.value)

    def test_explicit_none_target_level_defaults_to_auto(self):
        """Test explicit target_level=None defaults to 'auto' (line 217-218)."""
        config = TargetSelectionConfig.from_config({"target_level": None})
        assert config.config_level == "auto"

    def test_explicit_none_target_source_defaults_to_auto(self):
        """Test explicit target_source=None defaults to 'auto' (line 235-236)."""
        config = TargetSelectionConfig.from_config({"target_source": None})
        assert config.config_source == "auto"

    def test_parses_target_source_edge_label(self):
        """Test parses target_source='edge_label' as known source."""
        config = TargetSelectionConfig.from_config({"target_source": "edge_label"})
        assert config.config_source == "edge_label"

    def test_parses_target_source_edge_y(self):
        """Test parses target_source='edge_y' as known source."""
        config = TargetSelectionConfig.from_config({"target_source": "edge_y"})
        assert config.config_source == "edge_y"

    def test_default_strict_when_not_specified(self):
        """Test strict defaults to True when not in config dict."""
        config = TargetSelectionConfig.from_config({"indices": [0]})
        assert config.strict is True


# =============================================================================
# TEST: _normalize_list() Static Method
# =============================================================================


@skip_if_not_available
class TestNormalizeListMethod:
    """Test TargetSelectionConfig._normalize_list() static method."""

    def test_none_returns_empty_list(self):
        """Test None returns empty list."""
        result = TargetSelectionConfig._normalize_list(None, str)
        assert result == []

    def test_single_value_returns_list(self):
        """Test single value returns list with one element."""
        result = TargetSelectionConfig._normalize_list("gap", str)
        assert result == ["gap"]

    def test_list_returns_list(self):
        """Test list returns list."""
        result = TargetSelectionConfig._normalize_list(["a", "b"], str)
        assert result == ["a", "b"]

    def test_tuple_returns_list(self):
        """Test tuple returns list."""
        result = TargetSelectionConfig._normalize_list(("a", "b"), str)
        assert result == ["a", "b"]

    def test_converts_to_int(self):
        """Test converts elements to int."""
        result = TargetSelectionConfig._normalize_list(["1", "2", "3"], int)
        assert result == [1, 2, 3]

    def test_converts_single_to_int(self):
        """Test converts single value to int."""
        result = TargetSelectionConfig._normalize_list(5, int)
        assert result == [5]


# =============================================================================
# TEST: infer_level_from_task_type() Class Method (NEW)
# =============================================================================


@skip_if_not_available
class TestInferLevelFromTaskType:
    """Test TargetSelectionConfig.infer_level_from_task_type() class method."""

    def test_node_classification_returns_node(self):
        """Test 'node_classification' returns NODE level."""
        result = TargetSelectionConfig.infer_level_from_task_type("node_classification")
        assert result == TargetLevel.NODE

    def test_node_regression_returns_node(self):
        """Test 'node_regression' returns NODE level."""
        result = TargetSelectionConfig.infer_level_from_task_type("node_regression")
        assert result == TargetLevel.NODE

    def test_graph_classification_returns_graph(self):
        """Test 'graph_classification' returns GRAPH level."""
        result = TargetSelectionConfig.infer_level_from_task_type("graph_classification")
        assert result == TargetLevel.GRAPH

    def test_graph_regression_returns_graph(self):
        """Test 'graph_regression' returns GRAPH level."""
        result = TargetSelectionConfig.infer_level_from_task_type("graph_regression")
        assert result == TargetLevel.GRAPH

    def test_edge_classification_returns_edge(self):
        """Test 'edge_classification' returns EDGE level."""
        result = TargetSelectionConfig.infer_level_from_task_type("edge_classification")
        assert result == TargetLevel.EDGE

    def test_edge_regression_returns_edge(self):
        """Test 'edge_regression' returns EDGE level."""
        result = TargetSelectionConfig.infer_level_from_task_type("edge_regression")
        assert result == TargetLevel.EDGE

    def test_link_prediction_returns_edge(self):
        """Test 'link_prediction' returns EDGE level."""
        result = TargetSelectionConfig.infer_level_from_task_type("link_prediction")
        assert result == TargetLevel.EDGE

    def test_none_returns_graph(self):
        """Test None task_type returns GRAPH (default)."""
        result = TargetSelectionConfig.infer_level_from_task_type(None)
        assert result == TargetLevel.GRAPH

    def test_unknown_task_returns_graph(self, caplog):
        """Test unknown task_type returns GRAPH with warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            result = TargetSelectionConfig.infer_level_from_task_type("unknown_task")
        assert result == TargetLevel.GRAPH
        assert "Unknown task_type pattern" in caplog.text

    def test_case_insensitive(self):
        """Test task_type is case-insensitive."""
        result = TargetSelectionConfig.infer_level_from_task_type("NODE_CLASSIFICATION")
        assert result == TargetLevel.NODE


# =============================================================================
# TEST: infer_source_from_level() Class Method (NEW)
# =============================================================================


@skip_if_not_available
class TestInferSourceFromLevel:
    """Test TargetSelectionConfig.infer_source_from_level() class method."""

    def test_graph_level_returns_y(self):
        """Test GRAPH level returns Y source."""
        result = TargetSelectionConfig.infer_source_from_level(TargetLevel.GRAPH)
        assert result == TargetSource.Y

    def test_node_level_returns_x(self):
        """Test NODE level returns X source (fallback)."""
        result = TargetSelectionConfig.infer_source_from_level(TargetLevel.NODE)
        assert result == TargetSource.X

    def test_edge_level_returns_edge_attr(self):
        """Test EDGE level returns EDGE_ATTR source (fallback)."""
        result = TargetSelectionConfig.infer_source_from_level(TargetLevel.EDGE)
        assert result == TargetSource.EDGE_ATTR

    def test_link_prediction_returns_edge_label(self):
        """Test link_prediction task returns EDGE_LABEL."""
        result = TargetSelectionConfig.infer_source_from_level(TargetLevel.EDGE, "link_prediction")
        assert result == TargetSource.EDGE_LABEL

    def test_edge_regression_returns_edge_y(self):
        """Test edge_regression task returns EDGE_Y."""
        result = TargetSelectionConfig.infer_source_from_level(TargetLevel.EDGE, "edge_regression")
        assert result == TargetSource.EDGE_Y

    def test_none_task_type_uses_level_fallback(self):
        """Test None task_type uses level-based fallback."""
        result = TargetSelectionConfig.infer_source_from_level(TargetLevel.NODE, None)
        assert result == TargetSource.X


# =============================================================================
# TEST: resolve_for_task() Method (NEW)
# =============================================================================


@skip_if_not_available
class TestResolveForTask:
    """Test TargetSelectionConfig.resolve_for_task() method."""

    def test_auto_level_infers_from_task_type(self):
        """Test 'auto' level infers from task_type."""
        config = TargetSelectionConfig(config_level="auto")
        config.resolve_for_task("node_classification")
        assert config.resolved_level == TargetLevel.NODE

    def test_explicit_level_overrides_inference(self):
        """Test explicit level overrides task_type inference."""
        config = TargetSelectionConfig(config_level="graph")
        config.resolve_for_task("node_classification")  # Would normally be NODE
        assert config.resolved_level == TargetLevel.GRAPH

    def test_auto_source_defaults_to_y(self):
        """Test 'auto' source defaults to Y without data sample."""
        config = TargetSelectionConfig(config_source="auto")
        config.resolve_for_task("graph_regression")
        assert config.resolved_source == TargetSource.Y

    def test_explicit_source_y(self):
        """Test explicit source 'y'."""
        config = TargetSelectionConfig(config_source="y")
        config.resolve_for_task("node_regression")
        assert config.resolved_source == TargetSource.Y
        assert config.resolved_source_attr == "y"

    def test_explicit_source_x(self):
        """Test explicit source 'x'."""
        config = TargetSelectionConfig(config_source="x")
        config.resolve_for_task("node_regression")
        assert config.resolved_source == TargetSource.X
        assert config.resolved_source_attr == "x"

    def test_explicit_source_edge_attr(self):
        """Test explicit source 'edge_attr'."""
        config = TargetSelectionConfig(config_source="edge_attr")
        config.resolve_for_task("edge_regression")
        assert config.resolved_source == TargetSource.EDGE_ATTR
        assert config.resolved_source_attr == "edge_attr"

    def test_custom_source_sets_custom_enum(self):
        """Test custom source sets CUSTOM enum."""
        config = TargetSelectionConfig(config_source="atomic_charges")
        config.resolve_for_task("node_regression")
        assert config.resolved_source == TargetSource.CUSTOM
        assert config.resolved_source_attr == "atomic_charges"

    def test_returns_self(self):
        """Test returns self for chaining."""
        config = TargetSelectionConfig()
        result = config.resolve_for_task("graph_regression")
        assert result is config

    def test_link_prediction_special_case(self):
        """Test link_prediction uses EDGE_LABEL."""
        config = TargetSelectionConfig(config_source="auto")
        config.resolve_for_task("link_prediction")
        assert config.resolved_source == TargetSource.EDGE_LABEL

    def test_idempotent_multiple_calls(self, caplog):
        """Test resolve_for_task is idempotent - multiple calls don't re-resolve.

        This is critical for train_final_model where the same config is passed
        to _prepare_data_for_task_hpo for both train/val and test data.
        """
        import logging

        config = TargetSelectionConfig(config_level="auto", config_source="auto")

        # First call - should resolve
        with caplog.at_level(logging.INFO):
            config.resolve_for_task("node_regression")

        first_call_info_count = len([r for r in caplog.records if r.levelno == logging.INFO])
        assert config.resolved_level == TargetLevel.NODE
        assert config.resolved_source == TargetSource.Y

        # Clear log
        caplog.clear()

        # Second call - should skip (idempotent)
        with caplog.at_level(logging.DEBUG):
            config.resolve_for_task("node_regression")

        # Should have debug message about already resolved, not INFO
        assert "already resolved" in caplog.text

        # Values should be unchanged
        assert config.resolved_level == TargetLevel.NODE
        assert config.resolved_source == TargetSource.Y

    def test_explicit_source_edge_label(self):
        """Test explicit source 'edge_label' (line 409 source_map)."""
        config = TargetSelectionConfig(config_source="edge_label")
        config.resolve_for_task("link_prediction")
        assert config.resolved_source == TargetSource.EDGE_LABEL
        assert config.resolved_source_attr == "edge_label"

    def test_explicit_source_edge_y(self):
        """Test explicit source 'edge_y' (line 409 source_map)."""
        config = TargetSelectionConfig(config_source="edge_y")
        config.resolve_for_task("edge_regression")
        assert config.resolved_source == TargetSource.EDGE_Y
        assert config.resolved_source_attr == "edge_y"

    def test_invalid_config_level_falls_back_to_graph(self):
        """Test unknown config_level value falls back to GRAPH (line 390 level_map.get default)."""
        config = TargetSelectionConfig(config_level="subgraph")
        config.resolve_for_task("graph_regression")
        assert config.resolved_level == TargetLevel.GRAPH


# =============================================================================
# TEST: _resolve_source_auto() Method (NEW)
# =============================================================================


@skip_if_not_available
class TestResolveSourceAuto:
    """Test TargetSelectionConfig._resolve_source_auto() method."""

    def test_no_data_sample_returns_y(self):
        """Test no data sample returns Y (PyG convention)."""
        config = TargetSelectionConfig()
        result = config._resolve_source_auto(TargetLevel.NODE, "node_regression", None)
        assert result == TargetSource.Y

    def test_graph_level_returns_y(self):
        """Test GRAPH level always returns Y."""
        import torch

        mock_data = MagicMock()
        mock_data.y = torch.tensor([1.0])

        config = TargetSelectionConfig()
        result = config._resolve_source_auto(TargetLevel.GRAPH, "graph_regression", mock_data)
        assert result == TargetSource.Y

    def test_node_level_y_correct_shape_returns_y(self):
        """Test NODE level with correct y shape returns Y."""
        import torch

        mock_data = MagicMock()
        mock_data.y = torch.tensor([1.0, 2.0, 3.0])
        mock_data.num_nodes = 3

        config = TargetSelectionConfig()
        result = config._resolve_source_auto(TargetLevel.NODE, "node_regression", mock_data)
        assert result == TargetSource.Y

    def test_node_level_y_wrong_shape_returns_x(self):
        """Test NODE level with wrong y shape returns X."""
        import torch

        mock_data = MagicMock()
        mock_data.y = torch.tensor([1.0])  # Graph-level
        mock_data.num_nodes = 5

        config = TargetSelectionConfig()
        result = config._resolve_source_auto(TargetLevel.NODE, "node_regression", mock_data)
        assert result == TargetSource.X

    def test_edge_level_y_correct_shape_returns_y(self):
        """Test EDGE level with correct y shape returns Y."""
        import torch

        mock_data = MagicMock()
        mock_data.y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mock_data.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])  # 4 edges

        config = TargetSelectionConfig()
        result = config._resolve_source_auto(TargetLevel.EDGE, "edge_regression", mock_data)
        # Note: edge_regression is a special case
        assert result == TargetSource.EDGE_Y

    def test_link_prediction_returns_edge_label(self):
        """Test link_prediction returns EDGE_LABEL."""
        config = TargetSelectionConfig()
        result = config._resolve_source_auto(TargetLevel.EDGE, "link_prediction", None)
        assert result == TargetSource.EDGE_LABEL

    def test_y_none_uses_fallback(self):
        """Test y=None uses level-based fallback."""
        mock_data = MagicMock()
        mock_data.y = None

        config = TargetSelectionConfig()
        result = config._resolve_source_auto(TargetLevel.NODE, "node_regression", mock_data)
        assert result == TargetSource.X

    def test_edge_level_y_wrong_shape_returns_edge_attr(self):
        """Test EDGE level with y shape mismatch falls back to EDGE_ATTR (lines 491-502)."""
        import torch

        mock_data = MagicMock()
        mock_data.y = torch.tensor([1.0])  # Graph-level shape, not edge-level
        mock_data.edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # 3 edges

        config = TargetSelectionConfig()
        result = config._resolve_source_auto(TargetLevel.EDGE, "edge_classification", mock_data)
        assert result == TargetSource.EDGE_ATTR

    def test_node_level_infers_num_nodes_from_x(self):
        """Test NODE level infers num_nodes from x when num_nodes is None (lines 478-479)."""
        import torch

        mock_data = MagicMock()
        mock_data.y = torch.tensor([1.0, 2.0, 3.0])
        mock_data.num_nodes = None
        mock_data.x = torch.randn(3, 10)  # 3 nodes, 10 features

        config = TargetSelectionConfig()
        result = config._resolve_source_auto(TargetLevel.NODE, "node_regression", mock_data)
        assert result == TargetSource.Y

    def test_no_y_attribute_uses_fallback(self):
        """Test data_sample without y attribute uses level-based fallback (line 504-507)."""
        mock_data = MagicMock(spec=[])  # No attributes at all

        config = TargetSelectionConfig()
        result = config._resolve_source_auto(TargetLevel.NODE, "node_regression", mock_data)
        assert result == TargetSource.X

    def test_edge_regression_special_case_with_data(self):
        """Test edge_regression returns EDGE_Y even when data_sample provided (lines 458-459)."""
        import torch

        mock_data = MagicMock()
        mock_data.y = torch.tensor([1.0, 2.0])

        config = TargetSelectionConfig()
        result = config._resolve_source_auto(TargetLevel.EDGE, "edge_regression", mock_data)
        assert result == TargetSource.EDGE_Y

    def test_node_level_no_num_nodes_no_x_falls_back(self):
        """Test NODE level with no num_nodes and no x attribute falls back to X (lines 477-489)."""
        import torch

        mock_data = MagicMock()
        mock_data.y = torch.tensor([1.0, 2.0, 3.0])
        mock_data.num_nodes = None
        mock_data.x = None  # No x attribute to infer num_nodes

        config = TargetSelectionConfig()
        result = config._resolve_source_auto(TargetLevel.NODE, "node_regression", mock_data)
        # num_nodes is None, so y_first_dim != None check fails, falls back to X
        assert result == TargetSource.X

    def test_edge_level_no_edge_index_falls_back(self):
        """Test EDGE level with no edge_index attribute falls back to EDGE_ATTR."""
        import torch

        mock_data = MagicMock(spec=[])
        mock_data.y = torch.tensor([1.0, 2.0])

        config = TargetSelectionConfig()
        # Neither link_prediction nor edge_regression, so it checks y shape
        result = config._resolve_source_auto(TargetLevel.EDGE, "edge_classification", mock_data)
        assert result == TargetSource.EDGE_ATTR


# =============================================================================
# TEST: resolve() Method
# =============================================================================


@skip_if_not_available
class TestResolveMethod:
    """Test TargetSelectionConfig.resolve() method."""

    def test_all_mode_resolves_all_indices(self):
        """Test ALL mode resolves all indices."""
        config = TargetSelectionConfig(mode=SelectionMode.ALL)
        config.resolve(None, 5)
        assert config.resolved_indices == [0, 1, 2, 3, 4]
        assert config.total_available == 5

    def test_all_mode_copies_names(self):
        """Test ALL mode copies available names."""
        config = TargetSelectionConfig(mode=SelectionMode.ALL)
        config.resolve(["a", "b", "c"], 3)
        assert config.resolved_names == ["a", "b", "c"]

    def test_indices_mode_resolves(self):
        """Test INDICES mode resolves correctly."""
        config = TargetSelectionConfig(mode=SelectionMode.INDICES, indices=[0, 2, 4])
        config.resolve(None, 5)
        assert config.resolved_indices == [0, 2, 4]

    def test_indices_mode_negative_indices(self):
        """Test INDICES mode handles negative indices."""
        config = TargetSelectionConfig(mode=SelectionMode.INDICES, indices=[-1, -2])
        config.resolve(None, 5)
        assert config.resolved_indices == [4, 3]

    def test_range_mode_resolves(self):
        """Test RANGE mode resolves correctly."""
        config = TargetSelectionConfig(mode=SelectionMode.RANGE, range_spec="0:3")
        config.resolve(None, 10)
        assert config.resolved_indices == [0, 1, 2]

    def test_range_mode_with_step(self):
        """Test RANGE mode with step."""
        config = TargetSelectionConfig(mode=SelectionMode.RANGE, range_spec="0:10:2")
        config.resolve(None, 10)
        assert config.resolved_indices == [0, 2, 4, 6, 8]

    def test_properties_mode_resolves(self):
        """Test PROPERTIES mode resolves to indices."""
        config = TargetSelectionConfig(mode=SelectionMode.PROPERTIES, properties=["b", "d"])
        config.resolve(["a", "b", "c", "d"], 4)
        assert config.resolved_indices == [1, 3]
        assert config.resolved_names == ["b", "d"]

    def test_returns_self(self):
        """Test returns self for chaining."""
        config = TargetSelectionConfig()
        result = config.resolve(None, 5)
        assert result is config

    def test_properties_non_strict_falls_back_when_no_names(self):
        """Test PROPERTIES mode non-strict falls back to ALL when no names (lines 568-575)."""
        config = TargetSelectionConfig(
            mode=SelectionMode.PROPERTIES, properties=["gap"], strict=False
        )
        config.resolve(None, 5)
        assert config.resolved_indices == [0, 1, 2, 3, 4]
        assert config.resolved_names is None

    def test_properties_non_strict_skips_invalid(self):
        """Test PROPERTIES mode non-strict skips invalid names (lines 599-601)."""
        config = TargetSelectionConfig(
            mode=SelectionMode.PROPERTIES, properties=["a", "nonexistent", "c"], strict=False
        )
        config.resolve(["a", "b", "c"], 3)
        assert config.resolved_indices == [0, 2]
        assert config.resolved_names == ["a", "c"]

    def test_indices_resolves_with_available_names(self):
        """Test INDICES mode resolves names from available_names (lines 650-653)."""
        config = TargetSelectionConfig(mode=SelectionMode.INDICES, indices=[0, 2])
        config.resolve(["Etot", "gap", "zpves"], 3)
        assert config.resolved_indices == [0, 2]
        assert config.resolved_names == ["Etot", "zpves"]

    def test_indices_without_names_resolved_names_none(self):
        """Test INDICES mode without available_names has resolved_names=None."""
        config = TargetSelectionConfig(mode=SelectionMode.INDICES, indices=[0, 1])
        config.resolve(None, 5)
        assert config.resolved_names is None

    def test_all_invalid_indices_strict_raises(self):
        """Test all indices invalid in strict mode raises (lines 641-647)."""
        config = TargetSelectionConfig(mode=SelectionMode.INDICES, indices=[100, 200], strict=True)
        with pytest.raises(Exception):
            config.resolve(None, 5)

    def test_all_invalid_indices_non_strict_still_raises_when_empty(self):
        """Test all indices invalid in non-strict mode still raises when none resolved (lines 641-647)."""
        config = TargetSelectionConfig(mode=SelectionMode.INDICES, indices=[100, 200], strict=False)
        with pytest.raises(Exception):
            config.resolve(None, 5)

    def test_range_resolves_names_when_available(self):
        """Test RANGE mode resolves names when available_names provided (lines 685-688)."""
        config = TargetSelectionConfig(mode=SelectionMode.RANGE, range_spec="0:2")
        config.resolve(["Etot", "gap", "zpves"], 3)
        assert config.resolved_indices == [0, 1]
        assert config.resolved_names == ["Etot", "gap"]

    def test_range_omitted_start_stop(self):
        """Test RANGE mode with ::2 (omitted start and stop) (lines 662-669)."""
        config = TargetSelectionConfig(mode=SelectionMode.RANGE, range_spec="::2")
        config.resolve(None, 6)
        assert config.resolved_indices == [0, 2, 4]

    def test_range_four_colons_raises(self):
        """Test RANGE mode with 4+ colon parts raises error (line 671)."""
        config = TargetSelectionConfig(mode=SelectionMode.RANGE, range_spec="0:3:1:2")
        with pytest.raises(Exception):
            config.resolve(None, 10)


# =============================================================================
# TEST: to_dict() Method
# =============================================================================


@skip_if_not_available
class TestToDictMethod:
    """Test TargetSelectionConfig.to_dict() method."""

    def test_includes_mode(self):
        """Test includes mode name."""
        config = TargetSelectionConfig()
        result = config.to_dict()
        assert "mode" in result
        assert result["mode"] == "ALL"

    def test_includes_config_level(self):
        """Test includes config_level."""
        config = TargetSelectionConfig(config_level="node")
        result = config.to_dict()
        assert "config_level" in result
        assert result["config_level"] == "node"

    def test_includes_config_source(self):
        """Test includes config_source."""
        config = TargetSelectionConfig(config_source="x")
        result = config.to_dict()
        assert "config_source" in result
        assert result["config_source"] == "x"

    def test_includes_resolved_level(self):
        """Test includes resolved_level name."""
        config = TargetSelectionConfig()
        config.resolved_level = TargetLevel.NODE
        result = config.to_dict()
        assert "resolved_level" in result
        assert result["resolved_level"] == "NODE"

    def test_includes_resolved_source(self):
        """Test includes resolved_source name."""
        config = TargetSelectionConfig()
        config.resolved_source = TargetSource.X
        result = config.to_dict()
        assert "resolved_source" in result
        assert result["resolved_source"] == "X"

    def test_includes_resolved_source_attr(self):
        """Test includes resolved_source_attr."""
        config = TargetSelectionConfig()
        config.resolved_source_attr = "custom_attr"
        result = config.to_dict()
        assert "resolved_source_attr" in result
        assert result["resolved_source_attr"] == "custom_attr"

    def test_includes_resolved_indices(self):
        """Test includes resolved_indices."""
        config = TargetSelectionConfig()
        config.resolved_indices = [0, 1, 2]
        result = config.to_dict()
        assert "resolved_indices" in result
        assert result["resolved_indices"] == [0, 1, 2]

    def test_includes_strict(self):
        """Test includes strict."""
        config = TargetSelectionConfig(strict=False)
        result = config.to_dict()
        assert "strict" in result
        assert result["strict"] is False

    def test_specified_shows_properties(self):
        """Test 'specified' field shows properties list (line 710)."""
        config = TargetSelectionConfig(mode=SelectionMode.PROPERTIES, properties=["gap", "Etot"])
        result = config.to_dict()
        assert result["specified"] == ["gap", "Etot"]

    def test_specified_shows_indices(self):
        """Test 'specified' field shows indices list (line 710)."""
        config = TargetSelectionConfig(mode=SelectionMode.INDICES, indices=[0, 1, 2])
        result = config.to_dict()
        assert result["specified"] == [0, 1, 2]

    def test_specified_shows_range_spec(self):
        """Test 'specified' field shows range_spec string (line 710)."""
        config = TargetSelectionConfig(mode=SelectionMode.RANGE, range_spec="0:5")
        result = config.to_dict()
        assert result["specified"] == "0:5"

    def test_specified_shows_all_string(self):
        """Test 'specified' field shows 'ALL' when mode is ALL (line 710)."""
        config = TargetSelectionConfig(mode=SelectionMode.ALL)
        result = config.to_dict()
        assert result["specified"] == "ALL"

    def test_resolved_level_none_is_none(self):
        """Test resolved_level=None serializes to None (line 706)."""
        config = TargetSelectionConfig()
        result = config.to_dict()
        assert result["resolved_level"] is None

    def test_resolved_source_none_is_none(self):
        """Test resolved_source=None serializes to None (line 707)."""
        config = TargetSelectionConfig()
        result = config.to_dict()
        assert result["resolved_source"] is None

    def test_includes_total_available(self):
        """Test includes total_available (line 713)."""
        config = TargetSelectionConfig()
        config.total_available = 10
        result = config.to_dict()
        assert "total_available" in result
        assert result["total_available"] == 10

    def test_includes_resolved_names(self):
        """Test includes resolved_names (line 712)."""
        config = TargetSelectionConfig()
        config.resolved_names = ["gap", "Etot"]
        result = config.to_dict()
        assert "resolved_names" in result
        assert result["resolved_names"] == ["gap", "Etot"]


# =============================================================================
# TEST: __repr__() Method
# =============================================================================


@skip_if_not_available
class TestReprMethod:
    """Test TargetSelectionConfig.__repr__() method."""

    def test_repr_includes_mode(self):
        """Test __repr__ includes mode."""
        config = TargetSelectionConfig()
        result = repr(config)
        assert "mode=ALL" in result

    def test_repr_includes_level(self):
        """Test __repr__ includes level."""
        config = TargetSelectionConfig(config_level="node")
        config.resolved_level = TargetLevel.NODE
        result = repr(config)
        assert "level=NODE" in result

    def test_repr_includes_source(self):
        """Test __repr__ includes source."""
        config = TargetSelectionConfig()
        config.resolved_source_attr = "x"
        result = repr(config)
        assert "source=x" in result

    def test_repr_uses_config_level_if_not_resolved(self):
        """Test __repr__ uses config_level if not resolved."""
        config = TargetSelectionConfig(config_level="edge")
        result = repr(config)
        assert "level=edge" in result

    def test_repr_uses_config_source_if_no_resolved_attr(self):
        """Test __repr__ uses config_source when resolved_source_attr is None (line 719)."""
        config = TargetSelectionConfig(config_source="y")
        result = repr(config)
        assert "source=y" in result

    def test_repr_shows_resolved_indices(self):
        """Test __repr__ includes resolved indices (line 723)."""
        config = TargetSelectionConfig()
        config.resolved_indices = [0, 2, 4]
        result = repr(config)
        assert "resolved=[0, 2, 4]" in result


# =============================================================================
# TEST: Edge Cases and Error Handling
# =============================================================================


@skip_if_not_available
class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_invalid_indices_strict_mode_raises(self):
        """Test invalid indices in strict mode raises error."""
        config = TargetSelectionConfig(mode=SelectionMode.INDICES, indices=[10, 20], strict=True)
        with pytest.raises(Exception):  # ConfigurationError
            config.resolve(None, 5)

    def test_invalid_indices_non_strict_mode_skips(self, caplog):
        """Test invalid indices in non-strict mode skips."""
        import logging

        config = TargetSelectionConfig(
            mode=SelectionMode.INDICES,
            indices=[0, 10],  # 10 is invalid
            strict=False,
        )
        with caplog.at_level(logging.WARNING):
            config.resolve(None, 5)
        assert config.resolved_indices == [0]

    def test_invalid_properties_strict_mode_raises(self):
        """Test invalid properties in strict mode raises error."""
        config = TargetSelectionConfig(
            mode=SelectionMode.PROPERTIES, properties=["nonexistent"], strict=True
        )
        with pytest.raises(Exception):  # ConfigurationError
            config.resolve(["a", "b", "c"], 3)

    def test_properties_without_names_strict_raises(self):
        """Test properties mode without available names raises error."""
        config = TargetSelectionConfig(
            mode=SelectionMode.PROPERTIES, properties=["gap"], strict=True
        )
        with pytest.raises(Exception):  # ConfigurationError
            config.resolve(None, 5)

    def test_empty_range_raises(self):
        """Test empty range raises error."""
        config = TargetSelectionConfig(
            mode=SelectionMode.RANGE,
            range_spec="5:3",  # Empty range
        )
        with pytest.raises(Exception):  # ConfigurationError
            config.resolve(None, 10)

    def test_invalid_range_format_raises(self):
        """Test invalid range format raises error."""
        config = TargetSelectionConfig(mode=SelectionMode.RANGE, range_spec="invalid")
        with pytest.raises(Exception):  # ConfigurationError
            config.resolve(None, 10)

    def test_no_valid_properties_raises_even_non_strict(self):
        """Test that zero resolved properties always raises (lines 603-609)."""
        config = TargetSelectionConfig(
            mode=SelectionMode.PROPERTIES, properties=["nonexistent"], strict=False
        )
        with pytest.raises(Exception):
            config.resolve(["a", "b", "c"], 3)


# =============================================================================
# TEST: Pydantic BaseModel Behavior
# =============================================================================


@skip_if_not_available
class TestPydanticBaseModelBehavior:
    """Test Pydantic V2 BaseModel-specific behavior of TargetSelectionConfig."""

    def test_model_is_mutable(self):
        """Test TargetSelectionConfig is mutable (no frozen=True, line 79)."""
        config = TargetSelectionConfig()
        config.mode = SelectionMode.PROPERTIES
        assert config.mode == SelectionMode.PROPERTIES

    def test_raw_config_default_factory_creates_independent_dicts(self):
        """Test Field(default_factory=dict) creates independent instances (line 117)."""
        config1 = TargetSelectionConfig()
        config2 = TargetSelectionConfig()
        config1.raw_config["key"] = "value"
        assert "key" not in config2.raw_config

    def test_inherits_from_pydantic_base_model(self):
        """Test TargetSelectionConfig inherits from Pydantic BaseModel."""
        from pydantic import BaseModel

        assert issubclass(TargetSelectionConfig, BaseModel)

    def test_model_dump_available(self):
        """Test Pydantic V2 model_dump() is available."""
        config = TargetSelectionConfig()
        dump = config.model_dump()
        assert isinstance(dump, dict)
        assert "mode" in dump

    def test_direct_attribute_assignment(self):
        """Test direct attribute assignment works on resolved fields."""
        config = TargetSelectionConfig()
        config.resolved_indices = [0, 1]
        config.resolved_names = ["a", "b"]
        config.total_available = 5
        config.resolved_level = TargetLevel.NODE
        config.resolved_source = TargetSource.X
        config.resolved_source_attr = "x"

        assert config.resolved_indices == [0, 1]
        assert config.resolved_names == ["a", "b"]
        assert config.total_available == 5
        assert config.resolved_level == TargetLevel.NODE
        assert config.resolved_source == TargetSource.X
        assert config.resolved_source_attr == "x"


# =============================================================================
# TEST: Integration Scenarios
# =============================================================================


@skip_if_not_available
class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    def test_graph_regression_workflow(self):
        """Test complete graph regression workflow."""
        config = TargetSelectionConfig.from_config(
            {"target_level": "auto", "target_source": "auto", "indices": [0, 1, 2], "strict": True}
        )

        # Resolve for task
        config.resolve_for_task("graph_regression")
        assert config.resolved_level == TargetLevel.GRAPH
        assert config.resolved_source == TargetSource.Y

        # Resolve against dataset
        config.resolve(["Etot", "gap", "zpves", "U0"], 4)
        assert config.resolved_indices == [0, 1, 2]
        assert config.resolved_names == ["Etot", "gap", "zpves"]

    def test_node_classification_from_x_workflow(self):
        """Test node classification extracting from x."""
        config = TargetSelectionConfig.from_config(
            {"target_level": "auto", "target_source": "x", "indices": [5, 6], "strict": True}
        )

        config.resolve_for_task("node_classification")
        assert config.resolved_level == TargetLevel.NODE
        assert config.resolved_source == TargetSource.X
        assert config.resolved_source_attr == "x"

    def test_edge_regression_workflow(self):
        """Test edge regression workflow."""
        config = TargetSelectionConfig.from_config(
            {"target_level": "auto", "target_source": "edge_attr", "indices": [0], "strict": True}
        )

        config.resolve_for_task("edge_regression")
        assert config.resolved_level == TargetLevel.EDGE
        assert config.resolved_source == TargetSource.EDGE_ATTR

    def test_custom_attribute_workflow(self):
        """Test custom attribute workflow."""
        config = TargetSelectionConfig.from_config(
            {
                "target_level": "node",
                "target_source": "atomic_charges",
                "indices": None,
                "strict": True,
            }
        )

        config.resolve_for_task("node_regression")
        assert config.resolved_level == TargetLevel.NODE
        assert config.resolved_source == TargetSource.CUSTOM
        assert config.resolved_source_attr == "atomic_charges"

    def test_link_prediction_auto_workflow(self):
        """Test link_prediction with auto level and source resolves correctly."""
        config = TargetSelectionConfig.from_config(
            {
                "target_level": "auto",
                "target_source": "auto",
            }
        )

        config.resolve_for_task("link_prediction")
        assert config.resolved_level == TargetLevel.EDGE
        assert config.resolved_source == TargetSource.EDGE_LABEL

    def test_properties_mode_with_task_resolution(self):
        """Test complete properties-based selection with task resolution."""
        config = TargetSelectionConfig.from_config(
            {
                "properties": ["gap", "zpves"],
                "target_level": "auto",
                "target_source": "auto",
            }
        )

        config.resolve_for_task("graph_regression")
        config.resolve(["Etot", "gap", "zpves", "U0"], 4)

        assert config.mode == SelectionMode.PROPERTIES
        assert config.resolved_level == TargetLevel.GRAPH
        assert config.resolved_source == TargetSource.Y
        assert config.resolved_indices == [1, 2]
        assert config.resolved_names == ["gap", "zpves"]

    def test_range_with_task_resolution_workflow(self):
        """Test range-based selection with task resolution."""
        config = TargetSelectionConfig.from_config(
            {
                "indices": "::2",
                "target_level": "auto",
                "target_source": "y",
            }
        )

        config.resolve_for_task("graph_regression")
        config.resolve(["Etot", "gap", "zpves", "U0"], 4)

        assert config.mode == SelectionMode.RANGE
        assert config.resolved_level == TargetLevel.GRAPH
        assert config.resolved_indices == [0, 2]
        assert config.resolved_names == ["Etot", "zpves"]

    def test_to_dict_after_full_resolution(self):
        """Test to_dict() produces complete dict after full resolution workflow."""
        config = TargetSelectionConfig.from_config(
            {
                "indices": [0, 1],
                "target_level": "auto",
                "target_source": "auto",
                "strict": True,
            }
        )

        config.resolve_for_task("graph_regression")
        config.resolve(["Etot", "gap", "zpves"], 3)

        result = config.to_dict()
        assert result["mode"] == "INDICES"
        assert result["config_level"] == "auto"
        assert result["config_source"] == "auto"
        assert result["resolved_level"] == "GRAPH"
        assert result["resolved_source"] == "Y"
        assert result["resolved_indices"] == [0, 1]
        assert result["resolved_names"] == ["Etot", "gap"]
        assert result["total_available"] == 3
        assert result["strict"] is True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
