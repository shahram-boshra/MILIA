#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for research_api.py Module

This test suite provides extensive unit test coverage for the research_api.py module,
testing all data classes, fluent builders, experiment runner, convenience functions,
and configuration loading APIs.

Test Coverage:
- ExperimentConfiguration Pydantic model
  - Creation, validation, serialization (to_dict, from_dict)
  - YAML save/load
  - @model_validator validation (empty name, invalid num_runs, non-list base_transforms)
  - get_total_runs(), get_variant_names()
- AblationStudyBuilder (Fluent API)
  - with_baseline(), remove_transform(), keep_only(), add_variant()
  - replace_transform(), with_metadata(), build()
  - Error: build without baseline
  - Method chaining
- ParameterSweepBuilder (Fluent API)
  - for_transform(), sweep_parameter(), with_baseline_transforms()
  - with_metadata(), build()
  - _generate_combinations() Cartesian product
  - Error: no target transform, no parameters
  - Multi-parameter sweeps
- ComparativeStudyBuilder (Fluent API)
  - add_approach(), with_evaluation_metric(), with_metadata(), build()
  - Error: fewer than 2 approaches
- ExperimentRunner
  - Initialization and output directory creation
  - run_experiment() with mocked callables
  - _analyze_results() statistical analysis
  - _save_results() multi-format output
  - _save_results_csv() with pandas
  - _generate_markdown_report()
  - Error handling in run_experiment (failed runs)
- Convenience functions
  - create_ablation_study()
  - create_parameter_sweep()
  - create_comparative_study()
- Configuration loading
  - load_experiments_from_config()
  - get_experiment()
  - list_available_experiments()
- Module exports (__all__)
- Edge cases and boundary conditions

NOTE: This test suite runs inside Docker at /app/milia

Author: milia Project Team
Created: February 2026
Updated: Production-ready test coverage
"""

import sys
from pathlib import Path

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import json
from collections import defaultdict
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from milia_pipeline.config.config_containers import TransformSpec
from milia_pipeline.exceptions import ConfigurationError

# Import the module under test
from milia_pipeline.transformations.research_api import (
    AblationStudyBuilder,
    ComparativeStudyBuilder,
    # Core classes
    ExperimentConfiguration,
    ExperimentRunner,
    ParameterSweepBuilder,
    # Convenience functions
    create_ablation_study,
    create_comparative_study,
    create_parameter_sweep,
    get_experiment,
    list_available_experiments,
    # Configuration loaders
    load_experiments_from_config,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_transform_specs():
    """Sample TransformSpec objects for testing."""
    return [
        TransformSpec(name="AddSelfLoops", kwargs={}, enabled=True),
        TransformSpec(name="GCNNorm", kwargs={}, enabled=True),
        TransformSpec(name="RandomRotate", kwargs={"degrees": 180}, enabled=True),
    ]


@pytest.fixture
def minimal_transform_specs():
    """Minimal list with one TransformSpec."""
    return [
        TransformSpec(name="AddSelfLoops", kwargs={}, enabled=True),
    ]


@pytest.fixture
def sample_ablations():
    """Sample ablation variants."""
    return [
        {"name": "baseline", "transforms": ["AddSelfLoops", "GCNNorm", "RandomRotate"]},
        {"name": "without_GCNNorm", "transforms": ["AddSelfLoops", "RandomRotate"]},
        {"name": "without_RandomRotate", "transforms": ["AddSelfLoops", "GCNNorm"]},
    ]


@pytest.fixture
def sample_parameter_sweeps():
    """Sample parameter sweep variants."""
    return [
        {
            "name": "sweep_0_p=0.1",
            "transforms": ["AddSelfLoops", {"name": "DropEdge", "p": 0.1}],
            "parameters": {"p": 0.1},
        },
        {
            "name": "sweep_1_p=0.3",
            "transforms": ["AddSelfLoops", {"name": "DropEdge", "p": 0.3}],
            "parameters": {"p": 0.3},
        },
    ]


@pytest.fixture
def sample_experiment_config(sample_transform_specs, sample_ablations):
    """Full ExperimentConfiguration for testing."""
    return ExperimentConfiguration(
        name="test_ablation",
        description="Test ablation study",
        base_transforms=sample_transform_specs,
        ablations=sample_ablations,
        parameter_sweeps=[],
        hypothesis="Self-loops are critical",
        expected_outcome="baseline > no_self_loops",
        num_runs=3,
        random_seed=42,
    )


@pytest.fixture
def experiment_config_with_sweeps(sample_transform_specs, sample_parameter_sweeps):
    """ExperimentConfiguration with parameter sweeps."""
    return ExperimentConfiguration(
        name="test_sweep",
        description="Test parameter sweep",
        base_transforms=sample_transform_specs,
        ablations=[],
        parameter_sweeps=sample_parameter_sweeps,
        hypothesis="Moderate dropout improves generalization",
        num_runs=2,
    )


@pytest.fixture
def experiment_config_no_variants(minimal_transform_specs):
    """ExperimentConfiguration with no variants (triggers warning)."""
    return ExperimentConfiguration(
        name="empty_variants",
        description="No variants configured",
        base_transforms=minimal_transform_specs,
        ablations=[],
        parameter_sweeps=[],
    )


@pytest.fixture
def ablation_builder():
    """Pre-configured AblationStudyBuilder."""
    builder = AblationStudyBuilder("test_ablation")
    builder.with_baseline(["AddSelfLoops", "GCNNorm", "RandomRotate"])
    return builder


@pytest.fixture
def parameter_sweep_builder():
    """Pre-configured ParameterSweepBuilder."""
    builder = ParameterSweepBuilder("test_sweep")
    builder.for_transform("DropEdge")
    builder.sweep_parameter("p", [0.1, 0.2, 0.3])
    builder.with_baseline_transforms(["AddSelfLoops", "GCNNorm"])
    return builder


@pytest.fixture
def comparative_builder():
    """Pre-configured ComparativeStudyBuilder."""
    builder = ComparativeStudyBuilder("test_comparison")
    builder.add_approach("gcn_norm", ["AddSelfLoops", "GCNNorm"])
    builder.add_approach("no_norm", ["AddSelfLoops"])
    return builder


@pytest.fixture
def mock_dataset_loader():
    """Mock dataset loader callable."""
    mock = Mock()
    mock.return_value = Mock(name="mock_dataset")
    return mock


@pytest.fixture
def mock_model_trainer():
    """Mock model trainer callable."""
    mock = Mock()
    mock.return_value = (Mock(name="mock_model"), {"loss": 0.5, "lr": 0.001})
    return mock


@pytest.fixture
def mock_evaluator():
    """Mock evaluator callable."""
    mock = Mock()
    mock.return_value = {"mae": 0.1, "rmse": 0.15}
    return mock


@pytest.fixture
def experiment_runner(sample_experiment_config, tmp_path):
    """ExperimentRunner instance with temp directory."""
    return ExperimentRunner(
        config=sample_experiment_config,
        output_dir=tmp_path / "experiment_results",
        seed=42,
    )


@pytest.fixture
def sample_experiment_dict():
    """Raw dictionary for ExperimentConfiguration.from_dict()."""
    return {
        "name": "dict_experiment",
        "description": "Created from dict",
        "base_transforms": [
            {"name": "AddSelfLoops", "kwargs": {}, "enabled": True},
            {"name": "GCNNorm", "kwargs": {}, "enabled": True},
        ],
        "ablations": [
            {"name": "variant_1", "transforms": ["AddSelfLoops"]},
        ],
        "parameter_sweeps": [],
        "hypothesis": "Testing from dict",
        "expected_outcome": "Works correctly",
        "num_runs": 5,
        "random_seed": 123,
        "metadata": {"author": "test"},
    }


# =============================================================================
# EXPERIMENT CONFIGURATION TESTS
# =============================================================================


class TestExperimentConfiguration:
    """Test suite for ExperimentConfiguration Pydantic model."""

    def test_creation_full(self, sample_experiment_config):
        """Test creating ExperimentConfiguration with all fields."""
        config = sample_experiment_config
        assert config.name == "test_ablation"
        assert config.description == "Test ablation study"
        assert len(config.base_transforms) == 3
        assert len(config.ablations) == 3
        assert len(config.parameter_sweeps) == 0
        assert config.hypothesis == "Self-loops are critical"
        assert config.expected_outcome == "baseline > no_self_loops"
        assert config.num_runs == 3
        assert config.random_seed == 42

    def test_creation_minimal(self, minimal_transform_specs):
        """Test creating ExperimentConfiguration with minimal fields."""
        config = ExperimentConfiguration(
            name="minimal",
            description="Minimal config",
            base_transforms=minimal_transform_specs,
        )
        assert config.name == "minimal"
        assert config.ablations == []
        assert config.parameter_sweeps == []
        assert config.paper_reference is None
        assert config.hypothesis is None
        assert config.expected_outcome is None
        assert config.num_runs == 3  # default
        assert config.random_seed == 42  # default
        assert config.results == {}
        assert config.metadata == {}

    def test_base_transforms_are_transform_spec(self, sample_experiment_config):
        """Test that base_transforms contains TransformSpec objects."""
        for spec in sample_experiment_config.base_transforms:
            assert isinstance(spec, TransformSpec)

    def test_validation_empty_name(self, minimal_transform_specs):
        """Test that empty name raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            ExperimentConfiguration(
                name="",
                description="Should fail",
                base_transforms=minimal_transform_specs,
            )

    def test_validation_num_runs_less_than_one(self, minimal_transform_specs):
        """Test that num_runs < 1 raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            ExperimentConfiguration(
                name="invalid_runs",
                description="Should fail",
                base_transforms=minimal_transform_specs,
                num_runs=0,
            )

    def test_validation_negative_num_runs(self, minimal_transform_specs):
        """Test that negative num_runs raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            ExperimentConfiguration(
                name="invalid_runs",
                description="Should fail",
                base_transforms=minimal_transform_specs,
                num_runs=-5,
            )

    def test_validation_no_variants_warning(self, minimal_transform_specs):
        """Test that no variants logs a warning but does not raise."""
        # Should not raise, just warn
        config = ExperimentConfiguration(
            name="no_variants",
            description="No variants",
            base_transforms=minimal_transform_specs,
        )
        assert config.name == "no_variants"
        assert len(config.ablations) == 0
        assert len(config.parameter_sweeps) == 0

    def test_to_dict(self, sample_experiment_config):
        """Test serialization to dictionary."""
        d = sample_experiment_config.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test_ablation"
        assert d["description"] == "Test ablation study"
        assert isinstance(d["base_transforms"], list)
        assert len(d["base_transforms"]) == 3
        # Each transform should be serialized via TransformSpec.to_dict()
        for t in d["base_transforms"]:
            assert isinstance(t, dict)
            assert "name" in t
        assert d["ablations"] == sample_experiment_config.ablations
        assert d["parameter_sweeps"] == sample_experiment_config.parameter_sweeps
        assert d["hypothesis"] == "Self-loops are critical"
        assert d["expected_outcome"] == "baseline > no_self_loops"
        assert d["num_runs"] == 3
        assert d["random_seed"] == 42

    def test_to_dict_optional_fields_none(self, minimal_transform_specs):
        """Test to_dict with None optional fields."""
        config = ExperimentConfiguration(
            name="test",
            description="test",
            base_transforms=minimal_transform_specs,
        )
        d = config.to_dict()
        assert d["paper_reference"] is None
        assert d["hypothesis"] is None
        assert d["expected_outcome"] is None

    def test_from_dict_with_dict_transforms(self, sample_experiment_dict):
        """Test deserialization from dict with transform dicts."""
        config = ExperimentConfiguration.from_dict(sample_experiment_dict)
        assert config.name == "dict_experiment"
        assert config.description == "Created from dict"
        assert len(config.base_transforms) == 2
        for spec in config.base_transforms:
            assert isinstance(spec, TransformSpec)
        assert config.num_runs == 5
        assert config.random_seed == 123
        assert config.hypothesis == "Testing from dict"

    def test_from_dict_with_transform_spec_objects(self, sample_transform_specs):
        """Test from_dict when transforms are already TransformSpec objects."""
        data = {
            "name": "spec_objects",
            "description": "test",
            "base_transforms": sample_transform_specs,
        }
        config = ExperimentConfiguration.from_dict(data)
        assert len(config.base_transforms) == 3
        for spec in config.base_transforms:
            assert isinstance(spec, TransformSpec)

    def test_from_dict_with_string_transforms(self):
        """Test from_dict when transforms are plain strings."""
        data = {
            "name": "string_transforms",
            "description": "test",
            "base_transforms": ["AddSelfLoops", "GCNNorm"],
        }
        config = ExperimentConfiguration.from_dict(data)
        assert len(config.base_transforms) == 2
        for spec in config.base_transforms:
            assert isinstance(spec, TransformSpec)
            assert spec.kwargs == {}
            assert spec.enabled is True

    def test_from_dict_defaults(self):
        """Test from_dict fills defaults for missing optional fields."""
        data = {"name": "defaults_test", "base_transforms": []}
        config = ExperimentConfiguration.from_dict(data)
        assert config.description == ""
        assert config.ablations == []
        assert config.parameter_sweeps == []
        assert config.num_runs == 3
        assert config.random_seed == 42
        assert config.metadata == {}

    def test_roundtrip_to_dict_from_dict(self, sample_experiment_config):
        """Test that to_dict -> from_dict roundtrip preserves data."""
        d = sample_experiment_config.to_dict()
        restored = ExperimentConfiguration.from_dict(d)
        assert restored.name == sample_experiment_config.name
        assert restored.description == sample_experiment_config.description
        assert len(restored.base_transforms) == len(sample_experiment_config.base_transforms)
        assert restored.ablations == sample_experiment_config.ablations
        assert restored.hypothesis == sample_experiment_config.hypothesis
        assert restored.num_runs == sample_experiment_config.num_runs
        assert restored.random_seed == sample_experiment_config.random_seed

    def test_save_to_yaml(self, sample_experiment_config, tmp_path):
        """Test saving configuration to YAML file."""
        yaml_path = tmp_path / "test_config.yaml"
        sample_experiment_config.save_to_yaml(yaml_path)
        assert yaml_path.exists()

        with open(yaml_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["name"] == "test_ablation"
        assert loaded["num_runs"] == 3
        assert isinstance(loaded["base_transforms"], list)

    def test_load_from_yaml(self, sample_experiment_config, tmp_path):
        """Test loading configuration from YAML file."""
        yaml_path = tmp_path / "test_config.yaml"
        sample_experiment_config.save_to_yaml(yaml_path)

        loaded = ExperimentConfiguration.load_from_yaml(yaml_path)
        assert loaded.name == sample_experiment_config.name
        assert loaded.description == sample_experiment_config.description
        assert len(loaded.base_transforms) == len(sample_experiment_config.base_transforms)
        assert loaded.num_runs == sample_experiment_config.num_runs

    def test_yaml_roundtrip(self, sample_experiment_config, tmp_path):
        """Test full YAML save -> load roundtrip."""
        yaml_path = tmp_path / "roundtrip.yaml"
        sample_experiment_config.save_to_yaml(yaml_path)
        loaded = ExperimentConfiguration.load_from_yaml(yaml_path)

        assert loaded.name == sample_experiment_config.name
        assert loaded.hypothesis == sample_experiment_config.hypothesis
        assert loaded.num_runs == sample_experiment_config.num_runs
        assert loaded.random_seed == sample_experiment_config.random_seed
        assert loaded.ablations == sample_experiment_config.ablations

    def test_get_total_runs_ablations_only(self, sample_experiment_config):
        """Test get_total_runs with ablations only."""
        total = sample_experiment_config.get_total_runs()
        expected = len(sample_experiment_config.ablations) * sample_experiment_config.num_runs
        assert total == expected

    def test_get_total_runs_sweeps_only(self, experiment_config_with_sweeps):
        """Test get_total_runs with parameter sweeps only."""
        config = experiment_config_with_sweeps
        total = config.get_total_runs()
        expected = len(config.parameter_sweeps) * config.num_runs
        assert total == expected

    def test_get_total_runs_mixed(self, sample_transform_specs):
        """Test get_total_runs with both ablations and sweeps."""
        config = ExperimentConfiguration(
            name="mixed",
            description="mixed",
            base_transforms=sample_transform_specs,
            ablations=[{"name": "a1", "transforms": []}],
            parameter_sweeps=[{"name": "s1", "transforms": []}, {"name": "s2", "transforms": []}],
            num_runs=5,
        )
        total = config.get_total_runs()
        assert total == (1 + 2) * 5

    def test_get_total_runs_no_variants(self, experiment_config_no_variants):
        """Test get_total_runs returns 0 when no variants."""
        assert experiment_config_no_variants.get_total_runs() == 0

    def test_get_variant_names(self, sample_experiment_config):
        """Test get_variant_names returns all variant names."""
        names = sample_experiment_config.get_variant_names()
        assert isinstance(names, list)
        assert "baseline" in names
        assert "without_GCNNorm" in names
        assert "without_RandomRotate" in names

    def test_get_variant_names_mixed(self, sample_transform_specs):
        """Test get_variant_names with both ablations and sweeps."""
        config = ExperimentConfiguration(
            name="mixed",
            description="mixed",
            base_transforms=sample_transform_specs,
            ablations=[{"name": "abl_1", "transforms": []}],
            parameter_sweeps=[{"name": "sweep_1", "transforms": []}],
        )
        names = config.get_variant_names()
        assert "abl_1" in names
        assert "sweep_1" in names

    def test_get_variant_names_empty(self, experiment_config_no_variants):
        """Test get_variant_names with no variants."""
        names = experiment_config_no_variants.get_variant_names()
        assert names == []

    def test_results_mutable(self, sample_experiment_config):
        """Test that results dict is mutable."""
        sample_experiment_config.results["test_key"] = "test_value"
        assert sample_experiment_config.results["test_key"] == "test_value"

    def test_metadata_mutable(self, sample_experiment_config):
        """Test that metadata dict is mutable."""
        sample_experiment_config.metadata["extra"] = 42
        assert sample_experiment_config.metadata["extra"] == 42


# =============================================================================
# ABLATION STUDY BUILDER TESTS
# =============================================================================


class TestAblationStudyBuilder:
    """Test suite for AblationStudyBuilder fluent API."""

    def test_initialization(self):
        """Test builder initialization."""
        builder = AblationStudyBuilder("my_study")
        assert builder.study_name == "my_study"
        assert builder.baseline_transforms == []
        assert builder.variants == {}
        assert builder.metadata == {}

    def test_with_baseline(self):
        """Test setting baseline transforms."""
        builder = AblationStudyBuilder("test")
        result = builder.with_baseline(["A", "B", "C"])
        assert result is builder  # Returns self for chaining
        assert builder.baseline_transforms == ["A", "B", "C"]
        assert builder.variants["baseline"] == ["A", "B", "C"]

    def test_with_baseline_copies_list(self):
        """Test that with_baseline copies the list to prevent mutation."""
        original = ["A", "B", "C"]
        builder = AblationStudyBuilder("test")
        builder.with_baseline(original)
        original.append("D")
        assert builder.baseline_transforms == ["A", "B", "C"]

    def test_remove_transform_default_name(self, ablation_builder):
        """Test remove_transform with auto-generated variant name."""
        result = ablation_builder.remove_transform("GCNNorm")
        assert result is ablation_builder
        assert "without_GCNNorm" in ablation_builder.variants
        assert "GCNNorm" not in ablation_builder.variants["without_GCNNorm"]
        assert "AddSelfLoops" in ablation_builder.variants["without_GCNNorm"]
        assert "RandomRotate" in ablation_builder.variants["without_GCNNorm"]

    def test_remove_transform_custom_name(self, ablation_builder):
        """Test remove_transform with custom variant name."""
        ablation_builder.remove_transform("GCNNorm", variant_name="no_normalization")
        assert "no_normalization" in ablation_builder.variants
        assert "GCNNorm" not in ablation_builder.variants["no_normalization"]

    def test_remove_transform_nonexistent(self, ablation_builder):
        """Test removing a transform that is not in baseline."""
        ablation_builder.remove_transform("NonExistent")
        variant = ablation_builder.variants["without_NonExistent"]
        # Should produce same list as baseline since nothing was removed
        assert variant == ablation_builder.baseline_transforms

    def test_keep_only(self, ablation_builder):
        """Test keep_only creates variant with specified transforms."""
        result = ablation_builder.keep_only(["AddSelfLoops"], variant_name="minimal")
        assert result is ablation_builder
        assert "minimal" in ablation_builder.variants
        assert ablation_builder.variants["minimal"] == ["AddSelfLoops"]

    def test_keep_only_preserves_order(self, ablation_builder):
        """Test keep_only preserves baseline ordering."""
        ablation_builder.keep_only(["RandomRotate", "AddSelfLoops"], variant_name="two")
        # Should be in baseline order: AddSelfLoops, RandomRotate
        assert ablation_builder.variants["two"] == ["AddSelfLoops", "RandomRotate"]

    def test_keep_only_default_name(self, ablation_builder):
        """Test keep_only default variant name."""
        ablation_builder.keep_only(["AddSelfLoops"])
        assert "minimal" in ablation_builder.variants

    def test_add_variant_append(self, ablation_builder):
        """Test add_variant appending transforms at end."""
        result = ablation_builder.add_variant("with_dropout", ["DropEdge"])
        assert result is ablation_builder
        variant = ablation_builder.variants["with_dropout"]
        assert variant[-1] == "DropEdge"
        assert len(variant) == 4  # 3 baseline + 1 added

    def test_add_variant_at_position(self, ablation_builder):
        """Test add_variant inserting transforms at specific position."""
        ablation_builder.add_variant("inserted", ["NewTransform"], position=1)
        variant = ablation_builder.variants["inserted"]
        assert variant[1] == "NewTransform"
        assert len(variant) == 4

    def test_add_variant_at_start(self, ablation_builder):
        """Test add_variant inserting at position 0."""
        ablation_builder.add_variant("prepended", ["First"], position=0)
        variant = ablation_builder.variants["prepended"]
        assert variant[0] == "First"
        assert variant[1] == "AddSelfLoops"

    def test_add_variant_multiple_transforms(self, ablation_builder):
        """Test add_variant with multiple additional transforms."""
        ablation_builder.add_variant("multi", ["T1", "T2", "T3"], position=1)
        variant = ablation_builder.variants["multi"]
        assert variant[1:4] == ["T1", "T2", "T3"]
        assert len(variant) == 6

    def test_replace_transform_default_name(self, ablation_builder):
        """Test replace_transform with auto-generated name."""
        result = ablation_builder.replace_transform("GCNNorm", "PairNorm")
        assert result is ablation_builder
        expected_name = "replace_GCNNorm_with_PairNorm"
        assert expected_name in ablation_builder.variants
        variant = ablation_builder.variants[expected_name]
        assert "PairNorm" in variant
        assert "GCNNorm" not in variant

    def test_replace_transform_custom_name(self, ablation_builder):
        """Test replace_transform with custom name."""
        ablation_builder.replace_transform("GCNNorm", "PairNorm", variant_name="pair_norm_variant")
        assert "pair_norm_variant" in ablation_builder.variants

    def test_replace_transform_preserves_position(self, ablation_builder):
        """Test that replace preserves position in sequence."""
        ablation_builder.replace_transform("GCNNorm", "PairNorm")
        variant = ablation_builder.variants["replace_GCNNorm_with_PairNorm"]
        assert variant.index("PairNorm") == 1  # GCNNorm was at index 1

    def test_with_metadata(self, ablation_builder):
        """Test adding metadata."""
        result = ablation_builder.with_metadata(
            hypothesis="Self-loops critical",
            expected_outcome="baseline > without_self_loops",
            paper_section="Section 4.2",
        )
        assert result is ablation_builder
        assert ablation_builder.metadata["hypothesis"] == "Self-loops critical"
        assert ablation_builder.metadata["expected_outcome"] == "baseline > without_self_loops"
        assert ablation_builder.metadata["paper_section"] == "Section 4.2"

    def test_build_basic(self, ablation_builder):
        """Test building configuration from builder."""
        ablation_builder.remove_transform("GCNNorm")
        config = ablation_builder.build()
        assert isinstance(config, ExperimentConfiguration)
        assert config.name == "test_ablation"
        assert "Ablation study" in config.description
        assert len(config.base_transforms) == 3
        for spec in config.base_transforms:
            assert isinstance(spec, TransformSpec)
        # ablations should include baseline + without_GCNNorm
        assert len(config.ablations) >= 2

    def test_build_without_baseline_raises(self):
        """Test that build() without baseline raises ConfigurationError."""
        builder = AblationStudyBuilder("no_baseline")
        with pytest.raises(ConfigurationError):
            builder.build()

    def test_build_metadata_separation(self, ablation_builder):
        """Test that build separates direct fields from extra metadata."""
        ablation_builder.with_metadata(
            hypothesis="test hypothesis",
            expected_outcome="test outcome",
            paper_section="Section 1",
        )
        config = ablation_builder.build()
        assert config.hypothesis == "test hypothesis"
        assert config.expected_outcome == "test outcome"
        # paper_section is extra metadata, not a direct ExperimentConfiguration field
        assert "paper_section" in config.metadata

    def test_method_chaining(self):
        """Test full fluent method chaining."""
        config = (
            AblationStudyBuilder("chained_study")
            .with_baseline(["A", "B", "C"])
            .remove_transform("A")
            .remove_transform("B")
            .keep_only(["C"], variant_name="only_c")
            .add_variant("with_d", ["D"])
            .replace_transform("C", "E", variant_name="replaced")
            .with_metadata(
                hypothesis="Testing chaining",
                expected_outcome="All works",
            )
            .build()
        )
        assert isinstance(config, ExperimentConfiguration)
        assert config.name == "chained_study"
        assert len(config.ablations) > 0


# =============================================================================
# PARAMETER SWEEP BUILDER TESTS
# =============================================================================


class TestParameterSweepBuilder:
    """Test suite for ParameterSweepBuilder fluent API."""

    def test_initialization(self):
        """Test builder initialization."""
        builder = ParameterSweepBuilder("my_sweep")
        assert builder.sweep_name == "my_sweep"
        assert builder.target_transform is None
        assert builder.parameter_sweeps == {}
        assert builder.baseline_transforms == []
        assert builder.metadata == {}

    def test_for_transform(self):
        """Test specifying target transform."""
        builder = ParameterSweepBuilder("test")
        result = builder.for_transform("DropEdge")
        assert result is builder
        assert builder.target_transform == "DropEdge"

    def test_sweep_parameter(self):
        """Test defining parameter sweep values."""
        builder = ParameterSweepBuilder("test")
        result = builder.sweep_parameter("p", [0.1, 0.2, 0.3])
        assert result is builder
        assert builder.parameter_sweeps["p"] == [0.1, 0.2, 0.3]

    def test_sweep_multiple_parameters(self):
        """Test defining multiple parameter sweeps."""
        builder = ParameterSweepBuilder("test")
        builder.sweep_parameter("p", [0.1, 0.2])
        builder.sweep_parameter("training", [True, False])
        assert "p" in builder.parameter_sweeps
        assert "training" in builder.parameter_sweeps

    def test_with_baseline_transforms(self):
        """Test setting baseline transforms."""
        builder = ParameterSweepBuilder("test")
        result = builder.with_baseline_transforms(["A", "B"])
        assert result is builder
        assert builder.baseline_transforms == ["A", "B"]

    def test_with_metadata(self):
        """Test adding metadata."""
        builder = ParameterSweepBuilder("test")
        result = builder.with_metadata(
            hypothesis="Moderate dropout helps",
            expected_outcome="p=0.2 optimal",
        )
        assert result is builder
        assert builder.metadata["hypothesis"] == "Moderate dropout helps"
        assert builder.metadata["expected_outcome"] == "p=0.2 optimal"

    def test_build_basic(self, parameter_sweep_builder):
        """Test building parameter sweep configuration."""
        config = parameter_sweep_builder.build()
        assert isinstance(config, ExperimentConfiguration)
        assert config.name == "test_sweep"
        assert "Parameter sweep" in config.description
        assert "DropEdge" in config.description
        assert len(config.parameter_sweeps) == 3  # 3 values for p

    def test_build_variant_naming(self, parameter_sweep_builder):
        """Test that variants are named correctly."""
        config = parameter_sweep_builder.build()
        for variant in config.parameter_sweeps:
            assert variant["name"].startswith("sweep_")
            assert "p=" in variant["name"]

    def test_build_variant_transforms(self, parameter_sweep_builder):
        """Test that each variant has correct transform sequence."""
        config = parameter_sweep_builder.build()
        for variant in config.parameter_sweeps:
            transforms = variant["transforms"]
            # Baseline transforms + target transform
            assert len(transforms) == 3  # AddSelfLoops, GCNNorm, {DropEdge with param}
            assert transforms[0] == "AddSelfLoops"
            assert transforms[1] == "GCNNorm"
            assert isinstance(transforms[2], dict)
            assert transforms[2]["name"] == "DropEdge"

    def test_build_without_target_raises(self):
        """Test that build without target transform raises ConfigurationError."""
        builder = ParameterSweepBuilder("no_target")
        builder.sweep_parameter("p", [0.1])
        with pytest.raises(ConfigurationError):
            builder.build()

    def test_build_without_parameters_raises(self):
        """Test that build without parameters raises ConfigurationError."""
        builder = ParameterSweepBuilder("no_params")
        builder.for_transform("DropEdge")
        with pytest.raises(ConfigurationError):
            builder.build()

    def test_build_multi_parameter_cartesian_product(self):
        """Test that multi-parameter sweep generates Cartesian product."""
        builder = ParameterSweepBuilder("multi_param")
        builder.for_transform("DropEdge")
        builder.sweep_parameter("p", [0.1, 0.2])
        builder.sweep_parameter("force_undirected", [True, False])
        builder.with_baseline_transforms(["A"])
        config = builder.build()
        # 2 * 2 = 4 combinations
        assert len(config.parameter_sweeps) == 4

    def test_build_metadata_separation(self):
        """Test that build separates direct fields from extra metadata."""
        builder = ParameterSweepBuilder("meta_test")
        builder.for_transform("DropEdge")
        builder.sweep_parameter("p", [0.1])
        builder.metadata["hypothesis"] = "test"
        builder.metadata["custom_field"] = "custom"
        config = builder.build()
        assert config.hypothesis == "test"
        assert "custom_field" in config.metadata

    def test_generate_combinations_single_param(self):
        """Test _generate_combinations with single parameter."""
        combos = ParameterSweepBuilder._generate_combinations({"p": [0.1, 0.2, 0.3]})
        assert len(combos) == 3
        assert combos[0] == {"p": 0.1}
        assert combos[1] == {"p": 0.2}
        assert combos[2] == {"p": 0.3}

    def test_generate_combinations_multi_param(self):
        """Test _generate_combinations with multiple parameters (Cartesian product)."""
        combos = ParameterSweepBuilder._generate_combinations(
            {
                "p": [0.1, 0.2],
                "k": [1, 2, 3],
            }
        )
        assert len(combos) == 6  # 2 * 3
        # Check all combinations are present
        expected = [
            {"p": 0.1, "k": 1},
            {"p": 0.1, "k": 2},
            {"p": 0.1, "k": 3},
            {"p": 0.2, "k": 1},
            {"p": 0.2, "k": 2},
            {"p": 0.2, "k": 3},
        ]
        for exp in expected:
            assert exp in combos

    def test_generate_combinations_single_value(self):
        """Test _generate_combinations with single value per parameter."""
        combos = ParameterSweepBuilder._generate_combinations({"p": [0.5]})
        assert len(combos) == 1
        assert combos[0] == {"p": 0.5}

    def test_generate_combinations_empty(self):
        """Test _generate_combinations with empty dict."""
        combos = ParameterSweepBuilder._generate_combinations({})
        assert len(combos) == 1  # itertools.product of empty = one empty dict
        assert combos[0] == {}

    def test_method_chaining(self):
        """Test full fluent chaining for parameter sweep builder."""
        config = (
            ParameterSweepBuilder("chained")
            .for_transform("DropEdge")
            .sweep_parameter("p", [0.1, 0.2])
            .with_baseline_transforms(["A", "B"])
            .with_metadata(hypothesis="test", expected_outcome="ok")
            .build()
        )
        assert isinstance(config, ExperimentConfiguration)
        assert len(config.parameter_sweeps) == 2


# =============================================================================
# COMPARATIVE STUDY BUILDER TESTS
# =============================================================================


class TestComparativeStudyBuilder:
    """Test suite for ComparativeStudyBuilder fluent API."""

    def test_initialization(self):
        """Test builder initialization."""
        builder = ComparativeStudyBuilder("my_comparison")
        assert builder.study_name == "my_comparison"
        assert builder.approaches == {}
        assert builder.evaluation_metrics == []
        assert builder.metadata == {}

    def test_add_approach(self):
        """Test adding an approach."""
        builder = ComparativeStudyBuilder("test")
        result = builder.add_approach("gcn_norm", ["AddSelfLoops", "GCNNorm"])
        assert result is builder
        assert "gcn_norm" in builder.approaches
        assert builder.approaches["gcn_norm"] == ["AddSelfLoops", "GCNNorm"]

    def test_add_multiple_approaches(self, comparative_builder):
        """Test multiple approaches are stored."""
        assert len(comparative_builder.approaches) == 2
        assert "gcn_norm" in comparative_builder.approaches
        assert "no_norm" in comparative_builder.approaches

    def test_with_evaluation_metric(self, comparative_builder):
        """Test adding evaluation metrics."""
        result = comparative_builder.with_evaluation_metric("validation_mae")
        assert result is comparative_builder
        assert "validation_mae" in comparative_builder.evaluation_metrics

    def test_with_multiple_metrics(self, comparative_builder):
        """Test adding multiple evaluation metrics."""
        comparative_builder.with_evaluation_metric("mae")
        comparative_builder.with_evaluation_metric("rmse")
        assert len(comparative_builder.evaluation_metrics) == 2

    def test_with_metadata(self, comparative_builder):
        """Test adding metadata."""
        result = comparative_builder.with_metadata(
            research_question="Best normalization?",
            expected_best="gcn_norm",
        )
        assert result is comparative_builder
        assert comparative_builder.metadata["research_question"] == "Best normalization?"
        assert comparative_builder.metadata["expected_best"] == "gcn_norm"

    def test_build_basic(self, comparative_builder):
        """Test building comparative study configuration."""
        comparative_builder.with_evaluation_metric("mae")
        config = comparative_builder.build()
        assert isinstance(config, ExperimentConfiguration)
        assert config.name == "test_comparison"
        assert "Comparative study" in config.description
        assert len(config.ablations) == 2  # Comparative uses ablations list
        assert config.base_transforms == []  # No base for comparative

    def test_build_stores_evaluation_metrics_in_results(self, comparative_builder):
        """Test that evaluation metrics are stored in results."""
        comparative_builder.with_evaluation_metric("mae")
        comparative_builder.with_evaluation_metric("rmse")
        config = comparative_builder.build()
        assert "evaluation_metrics" in config.results
        assert config.results["evaluation_metrics"] == ["mae", "rmse"]

    def test_build_fewer_than_two_approaches_raises(self):
        """Test that build with < 2 approaches raises ConfigurationError."""
        builder = ComparativeStudyBuilder("one_approach")
        builder.add_approach("only_one", ["A"])
        with pytest.raises(ConfigurationError):
            builder.build()

    def test_build_zero_approaches_raises(self):
        """Test that build with 0 approaches raises ConfigurationError."""
        builder = ComparativeStudyBuilder("none")
        with pytest.raises(ConfigurationError):
            builder.build()

    def test_build_metadata_separation(self, comparative_builder):
        """Test metadata separation between direct fields and extras."""
        comparative_builder.metadata["hypothesis"] = "test hyp"
        comparative_builder.metadata["research_question"] = "test question"
        config = comparative_builder.build()
        assert config.hypothesis == "test hyp"
        assert "research_question" in config.metadata

    def test_method_chaining(self):
        """Test full fluent method chaining."""
        config = (
            ComparativeStudyBuilder("chained")
            .add_approach("a1", ["T1", "T2"])
            .add_approach("a2", ["T3", "T4"])
            .add_approach("a3", ["T5"])
            .with_evaluation_metric("mae")
            .with_evaluation_metric("rmse")
            .with_metadata(
                research_question="Which is best?",
                expected_best="a1",
            )
            .build()
        )
        assert isinstance(config, ExperimentConfiguration)
        assert len(config.ablations) == 3


# =============================================================================
# EXPERIMENT RUNNER TESTS
# =============================================================================


class TestExperimentRunner:
    """Test suite for ExperimentRunner."""

    def test_initialization(self, sample_experiment_config, tmp_path):
        """Test ExperimentRunner initialization."""
        output_dir = tmp_path / "results"
        runner = ExperimentRunner(sample_experiment_config, output_dir, seed=99)
        assert runner.config is sample_experiment_config
        assert runner.output_dir == output_dir
        assert output_dir.exists()
        assert runner.seed == 99
        assert isinstance(runner.results, defaultdict)

    def test_initialization_creates_output_directory(self, sample_experiment_config, tmp_path):
        """Test that initialization creates output directory."""
        nested_dir = tmp_path / "deep" / "nested" / "dir"
        runner = ExperimentRunner(sample_experiment_config, nested_dir)
        assert nested_dir.exists()

    def test_initialization_default_seed(self, sample_experiment_config, tmp_path):
        """Test default seed value."""
        runner = ExperimentRunner(sample_experiment_config, tmp_path / "out")
        assert runner.seed == 42

    def test_run_experiment_basic(
        self,
        experiment_runner,
        mock_dataset_loader,
        mock_model_trainer,
        mock_evaluator,
    ):
        """Test basic experiment execution."""
        summary = experiment_runner.run_experiment(
            dataset_loader=mock_dataset_loader,
            model_trainer=mock_model_trainer,
            evaluator=mock_evaluator,
        )
        assert isinstance(summary, dict)
        assert "experiment" in summary
        assert "variants" in summary
        assert "best_variant" in summary
        assert summary["experiment"] == "test_ablation"

    def test_run_experiment_calls_dataset_loader(
        self,
        experiment_runner,
        mock_dataset_loader,
        mock_model_trainer,
        mock_evaluator,
    ):
        """Test that dataset_loader is called correctly for each variant/run."""
        config = experiment_runner.config
        num_variants = len(config.ablations) + len(config.parameter_sweeps)
        num_runs = config.num_runs

        experiment_runner.run_experiment(
            dataset_loader=mock_dataset_loader,
            model_trainer=mock_model_trainer,
            evaluator=mock_evaluator,
        )
        assert mock_dataset_loader.call_count == num_variants * num_runs

    def test_run_experiment_custom_num_runs(
        self,
        experiment_runner,
        mock_dataset_loader,
        mock_model_trainer,
        mock_evaluator,
    ):
        """Test run_experiment with custom num_runs override."""
        experiment_runner.run_experiment(
            dataset_loader=mock_dataset_loader,
            model_trainer=mock_model_trainer,
            evaluator=mock_evaluator,
            num_runs=1,
        )
        num_variants = len(experiment_runner.config.ablations) + len(
            experiment_runner.config.parameter_sweeps
        )
        assert mock_dataset_loader.call_count == num_variants * 1

    def test_run_experiment_seed_incremented(
        self,
        experiment_runner,
        mock_dataset_loader,
        mock_model_trainer,
        mock_evaluator,
    ):
        """Test that seed is incremented per run."""
        experiment_runner.run_experiment(
            dataset_loader=mock_dataset_loader,
            model_trainer=mock_model_trainer,
            evaluator=mock_evaluator,
            num_runs=3,
        )
        # Check seeds passed to dataset_loader
        base_seed = experiment_runner.seed
        for c in mock_dataset_loader.call_args_list:
            _, kwargs = c
            seed = kwargs.get("seed")
            assert seed is not None
            assert seed >= base_seed
            assert seed < base_seed + 3

    def test_run_experiment_handles_failures(
        self,
        experiment_runner,
        mock_dataset_loader,
        mock_model_trainer,
        mock_evaluator,
    ):
        """Test that run_experiment handles failed runs gracefully."""
        mock_dataset_loader.side_effect = [
            RuntimeError("Dataset load failed"),
            Mock(name="good_dataset"),
            Mock(name="good_dataset"),
        ] * 3  # Repeat for multiple variants

        summary = experiment_runner.run_experiment(
            dataset_loader=mock_dataset_loader,
            model_trainer=mock_model_trainer,
            evaluator=mock_evaluator,
            num_runs=3,
        )
        # Should complete without raising
        assert isinstance(summary, dict)

    def test_run_experiment_stores_results(
        self,
        experiment_runner,
        mock_dataset_loader,
        mock_model_trainer,
        mock_evaluator,
    ):
        """Test that results are stored per variant."""
        experiment_runner.run_experiment(
            dataset_loader=mock_dataset_loader,
            model_trainer=mock_model_trainer,
            evaluator=mock_evaluator,
        )
        assert len(experiment_runner.results) > 0
        for variant_name, runs in experiment_runner.results.items():
            assert isinstance(runs, list)
            assert len(runs) > 0

    def test_run_experiment_saves_files(
        self,
        experiment_runner,
        mock_dataset_loader,
        mock_model_trainer,
        mock_evaluator,
    ):
        """Test that result files are created."""
        experiment_runner.run_experiment(
            dataset_loader=mock_dataset_loader,
            model_trainer=mock_model_trainer,
            evaluator=mock_evaluator,
        )
        output_dir = experiment_runner.output_dir
        config_name = experiment_runner.config.name
        assert (output_dir / f"{config_name}_summary.json").exists()
        assert (output_dir / f"{config_name}_detailed.json").exists()
        assert (output_dir / f"{config_name}_results.csv").exists()
        assert (output_dir / f"{config_name}_report.md").exists()

    def test_analyze_results_successful_runs(self, experiment_runner):
        """Test _analyze_results with successful runs."""
        experiment_runner.results["variant_a"] = [
            {"run": 0, "variant": "variant_a", "eval_metrics": {"mae": 0.1}, "train_metrics": {}},
            {"run": 1, "variant": "variant_a", "eval_metrics": {"mae": 0.2}, "train_metrics": {}},
        ]
        experiment_runner.results["variant_b"] = [
            {"run": 0, "variant": "variant_b", "eval_metrics": {"mae": 0.3}, "train_metrics": {}},
        ]

        summary = experiment_runner._analyze_results()
        assert "variant_a" in summary["variants"]
        assert "variant_b" in summary["variants"]

        va = summary["variants"]["variant_a"]
        assert va["status"] == "success"
        assert va["num_runs"] == 2
        assert va["failed_runs"] == 0
        assert np.isclose(va["mean"], 0.15)
        assert np.isclose(va["min"], 0.1)
        assert np.isclose(va["max"], 0.2)

    def test_analyze_results_best_variant(self, experiment_runner):
        """Test _analyze_results identifies best variant."""
        experiment_runner.results["good"] = [
            {"run": 0, "variant": "good", "eval_metrics": {"mae": 0.05}, "train_metrics": {}},
        ]
        experiment_runner.results["bad"] = [
            {"run": 0, "variant": "bad", "eval_metrics": {"mae": 0.5}, "train_metrics": {}},
        ]

        summary = experiment_runner._analyze_results()
        assert summary["best_variant"]["name"] == "good"
        assert np.isclose(summary["best_variant"]["score"], 0.05)

    def test_analyze_results_all_failed(self, experiment_runner):
        """Test _analyze_results when all runs failed."""
        experiment_runner.results["failed_variant"] = [
            {"run": 0, "variant": "failed_variant", "error": "boom"},
            {"run": 1, "variant": "failed_variant", "error": "crash"},
        ]

        summary = experiment_runner._analyze_results()
        assert summary["variants"]["failed_variant"]["status"] == "all_failed"

    def test_analyze_results_no_valid_variants(self, experiment_runner):
        """Test _analyze_results when no valid variants exist."""
        experiment_runner.results["all_fail"] = [
            {"run": 0, "variant": "all_fail", "error": "error"},
        ]

        summary = experiment_runner._analyze_results()
        assert summary["best_variant"] is None

    def test_analyze_results_mixed_success_failure(self, experiment_runner):
        """Test _analyze_results with mix of successful and failed runs."""
        experiment_runner.results["mixed"] = [
            {"run": 0, "variant": "mixed", "eval_metrics": {"mae": 0.2}, "train_metrics": {}},
            {"run": 1, "variant": "mixed", "error": "failed"},
            {"run": 2, "variant": "mixed", "eval_metrics": {"mae": 0.4}, "train_metrics": {}},
        ]

        summary = experiment_runner._analyze_results()
        stats = summary["variants"]["mixed"]
        assert stats["status"] == "success"
        assert stats["num_runs"] == 2
        assert stats["failed_runs"] == 1

    def test_save_results_json(self, experiment_runner):
        """Test _save_results creates JSON files."""
        experiment_runner.results["test"] = [
            {
                "run": 0,
                "variant": "test",
                "eval_metrics": {"mae": 0.1},
                "train_metrics": {"loss": 0.5},
            },
        ]
        summary = experiment_runner._analyze_results()
        experiment_runner._save_results(summary)

        json_path = experiment_runner.output_dir / f"{experiment_runner.config.name}_summary.json"
        assert json_path.exists()
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["experiment"] == experiment_runner.config.name

    def test_save_results_csv(self, experiment_runner):
        """Test _save_results_csv creates CSV with correct columns."""
        experiment_runner.results["v1"] = [
            {
                "run": 0,
                "variant": "v1",
                "eval_metrics": {"mae": 0.1, "rmse": 0.2},
                "train_metrics": {"loss": 0.5},
            },
            {
                "run": 1,
                "variant": "v1",
                "eval_metrics": {"mae": 0.15, "rmse": 0.25},
                "train_metrics": {"loss": 0.4},
            },
        ]

        csv_path = experiment_runner.output_dir / "test.csv"
        experiment_runner._save_results_csv(csv_path)
        assert csv_path.exists()

        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert "variant" in df.columns
        assert "run" in df.columns
        assert "eval_mae" in df.columns
        assert "eval_rmse" in df.columns
        assert "train_loss" in df.columns

    def test_save_results_csv_no_successful_runs(self, experiment_runner):
        """Test _save_results_csv with only failed runs."""
        experiment_runner.results["fail"] = [
            {"run": 0, "variant": "fail", "error": "boom"},
        ]
        csv_path = experiment_runner.output_dir / "empty.csv"
        experiment_runner._save_results_csv(csv_path)
        # Should not create file or create empty
        # The method logs a warning for no successful runs
        assert not csv_path.exists()

    def test_generate_markdown_report(self, experiment_runner):
        """Test _generate_markdown_report creates proper markdown."""
        experiment_runner.results["v1"] = [
            {"run": 0, "variant": "v1", "eval_metrics": {"mae": 0.1}, "train_metrics": {}},
        ]
        summary = experiment_runner._analyze_results()

        md_path = experiment_runner.output_dir / "report.md"
        experiment_runner._generate_markdown_report(md_path, summary)
        assert md_path.exists()

        content = md_path.read_text()
        assert f"# Experiment Report: {experiment_runner.config.name}" in content
        assert "## Results Summary" in content
        assert "## Detailed Results" in content
        assert "| Variant |" in content
        assert "## Conclusion" in content

    def test_generate_markdown_report_with_metadata(self, experiment_runner):
        """Test markdown report includes hypothesis and expected outcome."""
        summary = {
            "experiment": "test",
            "timestamp": datetime.now().isoformat(),
            "variants": {},
            "hypothesis": experiment_runner.config.hypothesis,
            "expected_outcome": experiment_runner.config.expected_outcome,
            "best_variant": None,
        }

        md_path = experiment_runner.output_dir / "meta_report.md"
        experiment_runner._generate_markdown_report(md_path, summary)
        content = md_path.read_text()

        if experiment_runner.config.hypothesis:
            assert "## Hypothesis" in content
        if experiment_runner.config.expected_outcome:
            assert "## Expected Outcome" in content

    def test_generate_markdown_report_with_best_variant(self, experiment_runner):
        """Test markdown report shows best variant when available."""
        summary = {
            "experiment": "test",
            "timestamp": datetime.now().isoformat(),
            "variants": {
                "v1": {
                    "status": "success",
                    "num_runs": 1,
                    "failed_runs": 0,
                    "mean": 0.1,
                    "std": 0.0,
                    "min": 0.1,
                    "max": 0.1,
                    "median": 0.1,
                },
            },
            "hypothesis": None,
            "expected_outcome": None,
            "best_variant": {"name": "v1", "score": 0.1},
        }

        md_path = experiment_runner.output_dir / "best_report.md"
        experiment_runner._generate_markdown_report(md_path, summary)
        content = md_path.read_text()
        assert "**Best Variant**: v1" in content
        assert "**Score**: 0.1000" in content


# =============================================================================
# CONVENIENCE FUNCTIONS TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_create_ablation_study_basic(self):
        """Test create_ablation_study with basic inputs."""
        config = create_ablation_study(
            "importance_study",
            ["AddSelfLoops", "GCNNorm", "RandomRotate"],
            ["GCNNorm", "RandomRotate"],
        )
        assert isinstance(config, ExperimentConfiguration)
        assert config.name == "importance_study"
        assert len(config.ablations) > 0

    def test_create_ablation_study_with_metadata(self):
        """Test create_ablation_study with metadata kwargs."""
        config = create_ablation_study(
            "meta_study",
            ["A", "B"],
            ["B"],
            hypothesis="B is important",
            expected_outcome="baseline > without_B",
        )
        assert isinstance(config, ExperimentConfiguration)
        # Metadata is stored in builder.metadata, then passed through build()
        # hypothesis/expected_outcome are direct fields if recognized

    def test_create_ablation_study_empty_ablations(self):
        """Test create_ablation_study with empty transforms_to_ablate."""
        config = create_ablation_study(
            "no_ablate",
            ["A", "B"],
            [],
        )
        assert isinstance(config, ExperimentConfiguration)
        # Should have baseline variant only
        assert len(config.ablations) >= 1

    def test_create_parameter_sweep_basic(self):
        """Test create_parameter_sweep with basic inputs."""
        config = create_parameter_sweep(
            "dropout_sweep",
            "DropEdge",
            {"p": [0.1, 0.2, 0.3, 0.5]},
            ["AddSelfLoops", "GCNNorm"],
        )
        assert isinstance(config, ExperimentConfiguration)
        assert config.name == "dropout_sweep"
        assert len(config.parameter_sweeps) == 4

    def test_create_parameter_sweep_with_metadata(self):
        """Test create_parameter_sweep with metadata kwargs."""
        config = create_parameter_sweep(
            "meta_sweep",
            "DropEdge",
            {"p": [0.1, 0.2]},
            ["A"],
            hypothesis="Moderate dropout improves generalization",
        )
        assert isinstance(config, ExperimentConfiguration)

    def test_create_parameter_sweep_multi_param(self):
        """Test create_parameter_sweep with multiple parameters."""
        config = create_parameter_sweep(
            "multi_sweep",
            "SomeTransform",
            {"alpha": [0.1, 0.5], "beta": [1, 2]},
            ["A"],
        )
        assert len(config.parameter_sweeps) == 4  # 2 * 2

    def test_create_comparative_study_basic(self):
        """Test create_comparative_study with basic inputs."""
        config = create_comparative_study(
            "norm_comparison",
            {
                "gcn": ["AddSelfLoops", "GCNNorm"],
                "no_norm": ["AddSelfLoops"],
            },
            ["validation_mae", "test_mae"],
        )
        assert isinstance(config, ExperimentConfiguration)
        assert config.name == "norm_comparison"
        assert len(config.ablations) == 2

    def test_create_comparative_study_with_metadata(self):
        """Test create_comparative_study with metadata kwargs."""
        config = create_comparative_study(
            "meta_compare",
            {"a": ["T1"], "b": ["T2"]},
            ["mae"],
            research_question="Which is best?",
        )
        assert isinstance(config, ExperimentConfiguration)

    def test_create_comparative_study_evaluation_metrics_stored(self):
        """Test that evaluation metrics are stored in results."""
        config = create_comparative_study(
            "metrics_test",
            {"a1": ["T1"], "a2": ["T2"]},
            ["mae", "rmse", "r2"],
        )
        assert config.results["evaluation_metrics"] == ["mae", "rmse", "r2"]


# =============================================================================
# CONFIGURATION LOADING TESTS
# =============================================================================


class TestConfigurationLoading:
    """Test suite for configuration loading functions."""

    def test_load_experiments_from_yaml_file(self, tmp_path):
        """Test loading experiments from YAML config file."""
        config_data = {
            "experiments": {
                "exp1": {
                    "name": "exp1",
                    "description": "First experiment",
                    "base_transforms": [{"name": "AddSelfLoops", "kwargs": {}, "enabled": True}],
                    "ablations": [{"name": "v1", "transforms": ["AddSelfLoops"]}],
                },
                "exp2": {
                    "name": "exp2",
                    "description": "Second experiment",
                    "base_transforms": [],
                    "ablations": [],
                },
            }
        }
        config_path = tmp_path / "research_experiments.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        experiments = load_experiments_from_config(config_path)
        assert isinstance(experiments, dict)
        assert "exp1" in experiments
        assert "exp2" in experiments
        assert isinstance(experiments["exp1"], ExperimentConfiguration)

    def test_load_experiments_from_yaml_no_experiments_key(self, tmp_path):
        """Test loading from YAML with no experiments key."""
        config_path = tmp_path / "empty.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"other_key": "value"}, f)

        experiments = load_experiments_from_config(config_path)
        assert experiments == {}

    @patch("milia_pipeline.transformations.research_api.load_config")
    def test_load_experiments_fallback_to_main_config(self, mock_load_config):
        """Test fallback to main config.yaml when file not found."""
        mock_load_config.return_value = {
            "experiments": {
                "fallback_exp": {
                    "name": "fallback_exp",
                    "description": "Loaded from main config",
                    "base_transforms": [],
                }
            }
        }
        # Use a non-existent path to trigger fallback
        experiments = load_experiments_from_config(Path("/nonexistent/path.yaml"))
        assert "fallback_exp" in experiments

    @patch("milia_pipeline.transformations.research_api.load_config")
    def test_load_experiments_no_experiments_in_main_config(self, mock_load_config):
        """Test when main config has no experiments section."""
        mock_load_config.return_value = {"some_other_key": "value"}
        experiments = load_experiments_from_config(Path("/nonexistent/path.yaml"))
        assert experiments == {}

    def test_load_experiments_handles_invalid_experiment(self, tmp_path):
        """Test that invalid experiment configs are logged and skipped."""
        config_data = {
            "experiments": {
                "good_exp": {
                    "name": "good_exp",
                    "description": "Valid",
                    "base_transforms": [],
                },
                "bad_exp": {
                    "name": "",  # Invalid: empty name
                    "description": "Invalid",
                    "base_transforms": [],
                },
            }
        }
        config_path = tmp_path / "mixed.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        experiments = load_experiments_from_config(config_path)
        assert "good_exp" in experiments
        assert "bad_exp" not in experiments

    def test_get_experiment_found(self, tmp_path):
        """Test get_experiment returns correct experiment."""
        config_data = {
            "experiments": {
                "target": {
                    "name": "target",
                    "description": "Target experiment",
                    "base_transforms": [],
                }
            }
        }
        config_path = tmp_path / "experiments.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        exp = get_experiment("target", config_path)
        assert isinstance(exp, ExperimentConfiguration)
        assert exp.name == "target"

    def test_get_experiment_not_found_raises(self, tmp_path):
        """Test get_experiment raises ConfigurationError for missing experiment."""
        config_data = {
            "experiments": {
                "existing": {
                    "name": "existing",
                    "description": "exists",
                    "base_transforms": [],
                }
            }
        }
        config_path = tmp_path / "experiments.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ConfigurationError):
            get_experiment("nonexistent", config_path)

    def test_list_available_experiments(self, tmp_path):
        """Test list_available_experiments returns names."""
        config_data = {
            "experiments": {
                "exp_a": {"name": "exp_a", "description": "A", "base_transforms": []},
                "exp_b": {"name": "exp_b", "description": "B", "base_transforms": []},
                "exp_c": {"name": "exp_c", "description": "C", "base_transforms": []},
            }
        }
        config_path = tmp_path / "experiments.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        names = list_available_experiments(config_path)
        assert isinstance(names, list)
        assert "exp_a" in names
        assert "exp_b" in names
        assert "exp_c" in names

    def test_list_available_experiments_empty(self, tmp_path):
        """Test list_available_experiments with no experiments."""
        config_data = {"experiments": {}}
        config_path = tmp_path / "empty.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        names = list_available_experiments(config_path)
        assert names == []


# =============================================================================
# MODULE EXPORTS TESTS
# =============================================================================


class TestModuleExports:
    """Test that all expected classes and functions are exported."""

    def test_core_classes_exported(self):
        """Test that core classes are in __all__."""
        from milia_pipeline.transformations.research_api import __all__

        expected_exports = [
            "ExperimentConfiguration",
            "AblationStudyBuilder",
            "ParameterSweepBuilder",
            "ComparativeStudyBuilder",
            "ExperimentRunner",
            "create_ablation_study",
            "create_parameter_sweep",
            "create_comparative_study",
            "load_experiments_from_config",
            "get_experiment",
            "list_available_experiments",
        ]

        for export in expected_exports:
            assert export in __all__, f"{export} not in __all__"

    def test_all_exports_importable(self):
        """Test that all __all__ entries are importable."""
        from milia_pipeline.transformations import research_api

        for name in research_api.__all__:
            assert hasattr(research_api, name), f"{name} not importable from research_api"


# =============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_experiment_config_large_num_runs(self, minimal_transform_specs):
        """Test configuration with large num_runs."""
        config = ExperimentConfiguration(
            name="large_runs",
            description="Many runs",
            base_transforms=minimal_transform_specs,
            ablations=[{"name": "v1", "transforms": []}],
            num_runs=1000,
        )
        assert config.num_runs == 1000
        assert config.get_total_runs() == 1000

    def test_experiment_config_single_run(self, minimal_transform_specs):
        """Test configuration with num_runs=1."""
        config = ExperimentConfiguration(
            name="single_run",
            description="Single run",
            base_transforms=minimal_transform_specs,
            ablations=[{"name": "v1", "transforms": []}],
            num_runs=1,
        )
        assert config.num_runs == 1
        assert config.get_total_runs() == 1

    def test_ablation_builder_remove_all_transforms(self):
        """Test removing all transforms from baseline."""
        builder = AblationStudyBuilder("empty_ablation")
        builder.with_baseline(["A", "B"])
        builder.keep_only([], variant_name="empty")
        assert builder.variants["empty"] == []

    def test_ablation_builder_replace_nonexistent_transform(self):
        """Test replacing a transform that isn't in baseline."""
        builder = AblationStudyBuilder("test")
        builder.with_baseline(["A", "B"])
        builder.replace_transform("Z", "Y")
        # Since Z is not in baseline, all transforms stay as-is
        assert builder.variants["replace_Z_with_Y"] == ["A", "B"]

    def test_parameter_sweep_single_value_parameter(self):
        """Test parameter sweep with only one value."""
        config = create_parameter_sweep(
            "single_val",
            "Transform",
            {"p": [0.5]},
            ["A"],
        )
        assert len(config.parameter_sweeps) == 1

    def test_parameter_sweep_many_parameters(self):
        """Test parameter sweep with many parameters (combinatorial explosion)."""
        config = create_parameter_sweep(
            "explosion",
            "Transform",
            {"a": [1, 2], "b": [3, 4], "c": [5, 6]},
            ["A"],
        )
        assert len(config.parameter_sweeps) == 8  # 2^3

    def test_comparative_study_many_approaches(self):
        """Test comparative study with many approaches."""
        approaches = {f"approach_{i}": [f"T{i}"] for i in range(10)}
        config = create_comparative_study(
            "many_approaches",
            approaches,
            ["mae"],
        )
        assert len(config.ablations) == 10

    def test_experiment_runner_all_runs_fail(self, sample_experiment_config, tmp_path):
        """Test ExperimentRunner when every run fails."""
        runner = ExperimentRunner(sample_experiment_config, tmp_path / "fail_test")
        failing_loader = Mock(side_effect=RuntimeError("Always fails"))

        summary = runner.run_experiment(
            dataset_loader=failing_loader,
            model_trainer=Mock(),
            evaluator=Mock(),
            num_runs=2,
        )
        # Should complete without crashing
        assert isinstance(summary, dict)
        assert summary["best_variant"] is None

    def test_experiment_runner_output_dir_as_string(self, sample_experiment_config, tmp_path):
        """Test ExperimentRunner accepts string path."""
        str_path = str(tmp_path / "string_path")
        runner = ExperimentRunner(sample_experiment_config, str_path)
        assert isinstance(runner.output_dir, Path)
        assert runner.output_dir.exists()

    def test_from_dict_with_empty_base_transforms(self):
        """Test from_dict with empty base_transforms list."""
        data = {
            "name": "empty_transforms",
            "base_transforms": [],
        }
        config = ExperimentConfiguration.from_dict(data)
        assert config.base_transforms == []

    def test_experiment_config_special_characters_in_name(self, minimal_transform_specs):
        """Test experiment name with special characters."""
        config = ExperimentConfiguration(
            name="test-experiment_v2.0 (updated)",
            description="Special chars",
            base_transforms=minimal_transform_specs,
        )
        assert config.name == "test-experiment_v2.0 (updated)"

    def test_ablation_builder_baseline_overwrite(self):
        """Test that calling with_baseline twice overwrites the first."""
        builder = AblationStudyBuilder("test")
        builder.with_baseline(["A", "B"])
        builder.with_baseline(["C", "D", "E"])
        assert builder.baseline_transforms == ["C", "D", "E"]

    def test_parameter_sweep_builder_overwrite_transform(self):
        """Test that calling for_transform twice overwrites."""
        builder = ParameterSweepBuilder("test")
        builder.for_transform("First")
        builder.for_transform("Second")
        assert builder.target_transform == "Second"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
