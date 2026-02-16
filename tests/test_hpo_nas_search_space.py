#!/usr/bin/env python3
"""
Complete Unit Test Suite for milia_pipeline/models/hpo/nas/search_space.py Module

Tests the search_space.py module including:
- LayerType, PoolingType, AggregationType, ActivationType enum tests
- LayerConfig Pydantic BaseModel tests (frozen=True)
- GNNArchitectureSpace Pydantic BaseModel tests (mutable)
- Validation, serialization, and factory function tests
- to_optuna_search_space(), get_search_dimensions(), estimate_search_space_size()
- create_gnn_search_space(), get_default_gnn_search_space()
- Integration tests and edge cases

Location of module under test: milia_pipeline/models/hpo/nas/search_space.py
Location of test file: tests/test_hpo_nas_search_space.py

Note: The module under test uses Pydantic V2 BaseModel:
    - LayerConfig is a frozen BaseModel (immutable)
    - GNNArchitectureSpace is a mutable BaseModel
    Attempting to modify frozen model attributes raises pydantic.ValidationError.

Author: Milia Team
Version: 2.0.0
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from enum import Enum
from typing import Any
from unittest.mock import patch

import pytest

# Import pydantic for type checking and validation error handling
from pydantic import BaseModel, ValidationError

# =============================================================================
# HELPER FUNCTIONS FOR PYDANTIC MODEL CHECKS
# =============================================================================


def is_pydantic_model(cls: type) -> bool:
    """
    Check if a class is a Pydantic BaseModel subclass.

    This replaces the dataclasses.is_dataclass() check for Pydantic V2 models.

    Args:
        cls: The class to check.

    Returns:
        True if cls is a subclass of pydantic.BaseModel, False otherwise.
    """
    try:
        return issubclass(cls, BaseModel)
    except TypeError:
        return False


def is_frozen_pydantic_model(cls: type) -> bool:
    """
    Check if a Pydantic BaseModel is configured as frozen (immutable).

    Args:
        cls: The class to check.

    Returns:
        True if cls is a frozen Pydantic BaseModel, False otherwise.
    """
    if not is_pydantic_model(cls):
        return False
    # Check model_config for frozen setting
    model_config = getattr(cls, "model_config", {})
    if isinstance(model_config, dict):
        return model_config.get("frozen", False)
    # For ConfigDict style
    return getattr(model_config, "frozen", False)


# =============================================================================
# MOCK CLASSES FOR EXCEPTIONS (Consolidated - Single Definition)
# =============================================================================


class MockSearchSpaceError(Exception):
    """
    Mock SearchSpaceError for testing.

    Mirrors the actual SearchSpaceError from milia_pipeline.exceptions
    with all required attributes for validation testing.
    """

    def __init__(
        self,
        message: str,
        parameter_name: str | None = None,
        parameter_config: dict[str, Any] | None = None,
        study_name: str | None = None,
        trial_number: int | None = None,
        details: str | None = None,
        **kwargs,
    ):
        super().__init__(message)
        self.message = message
        self.parameter_name = parameter_name
        self.parameter_config = parameter_config
        self.study_name = study_name
        self.trial_number = trial_number
        self.details = details
        self.extra_info = kwargs

    def __str__(self) -> str:
        msg = self.message
        if self.parameter_name:
            msg += f". Parameter: '{self.parameter_name}'"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class MockConfigurationError(Exception):
    """
    Mock ConfigurationError for testing.

    Mirrors the actual ConfigurationError from milia_pipeline.exceptions
    with all required attributes for validation testing.
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        actual_value: Any = None,
        expected_value: Any = None,
        details: str | None = None,
        **kwargs,
    ):
        super().__init__(message)
        self.message = message
        self.config_key = config_key
        self.actual_value = actual_value
        self.expected_value = expected_value
        self.details = details
        self.extra_info = kwargs

    def __str__(self) -> str:
        parts = [self.message]
        if self.config_key:
            parts.append(f"Key: '{self.config_key}'")
        if self.expected_value is not None:
            parts.append(f"Expected Type: {self.expected_value}")
        if self.actual_value is not None:
            parts.append(f"Actual Value: '{self.actual_value}'")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " ".join(parts)


# =============================================================================
# TEST FIXTURES (Consolidated - Single Definition)
# =============================================================================


@pytest.fixture
def mock_search_space_error():
    """Provide MockSearchSpaceError class for patching."""
    return MockSearchSpaceError


@pytest.fixture
def mock_configuration_error():
    """Provide MockConfigurationError class for patching."""
    return MockConfigurationError


# =============================================================================
# LAYERTYPE ENUM TESTS - VALUE VERIFICATION
# =============================================================================


class TestLayerTypeEnumValues:
    """Test LayerType enum value definitions."""

    def test_layer_type_gcn_value(self):
        """Test LayerType.GCN has correct string value 'gcn'."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        assert LayerType.GCN.value == "gcn"

    def test_layer_type_gat_value(self):
        """Test LayerType.GAT has correct string value 'gat'."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        assert LayerType.GAT.value == "gat"

    def test_layer_type_sage_value(self):
        """Test LayerType.SAGE has correct string value 'sage'."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        assert LayerType.SAGE.value == "sage"

    def test_layer_type_gin_value(self):
        """Test LayerType.GIN has correct string value 'gin'."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        assert LayerType.GIN.value == "gin"

    def test_layer_type_gatv2_value(self):
        """Test LayerType.GATV2 has correct string value 'gatv2'."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        assert LayerType.GATV2.value == "gatv2"

    def test_layer_type_transformer_value(self):
        """Test LayerType.TRANSFORMER has correct string value 'transformer'."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        assert LayerType.TRANSFORMER.value == "transformer"

    def test_layer_type_pna_value(self):
        """Test LayerType.PNA has correct string value 'pna'."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        assert LayerType.PNA.value == "pna"


# =============================================================================
# LAYERTYPE ENUM TESTS - ENUM BEHAVIOR
# =============================================================================


class TestLayerTypeEnumBehavior:
    """Test LayerType enum behaviors and properties."""

    def test_layer_type_is_enum_subclass(self):
        """Test LayerType is a subclass of Enum."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        assert issubclass(LayerType, Enum)

    def test_layer_type_has_seven_members(self):
        """Test LayerType has exactly 7 enum members."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        assert len(LayerType) == 7

    def test_layer_type_member_is_instance(self):
        """Test each LayerType member is instance of LayerType."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        for layer_type in LayerType:
            assert isinstance(layer_type, LayerType)

    def test_layer_type_can_iterate(self):
        """Test LayerType can be iterated."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        members = list(LayerType)
        assert len(members) == 7
        assert LayerType.GCN in members
        assert LayerType.GAT in members
        assert LayerType.SAGE in members

    def test_layer_type_membership_check(self):
        """Test membership check for LayerType."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        assert LayerType.GCN in LayerType
        assert LayerType.GAT in LayerType
        assert LayerType.TRANSFORMER in LayerType

    def test_layer_type_value_to_member(self):
        """Test converting string value to LayerType member."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        assert LayerType("gcn") == LayerType.GCN
        assert LayerType("gat") == LayerType.GAT
        assert LayerType("sage") == LayerType.SAGE
        assert LayerType("gin") == LayerType.GIN
        assert LayerType("gatv2") == LayerType.GATV2
        assert LayerType("transformer") == LayerType.TRANSFORMER
        assert LayerType("pna") == LayerType.PNA

    def test_layer_type_invalid_value_raises(self):
        """Test invalid value raises ValueError."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        with pytest.raises(ValueError):
            LayerType("invalid_layer_type")

    def test_layer_type_name_attribute(self):
        """Test LayerType name attribute."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        assert LayerType.GCN.name == "GCN"
        assert LayerType.GAT.name == "GAT"
        assert LayerType.GATV2.name == "GATV2"

    def test_layer_type_equality(self):
        """Test LayerType equality comparison."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        assert LayerType.GCN == LayerType.GCN
        assert LayerType.GAT == LayerType.GAT
        assert LayerType.GCN != LayerType.GAT

    def test_layer_type_identity(self):
        """Test LayerType identity (same object)."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        gcn1 = LayerType.GCN
        gcn2 = LayerType.GCN
        assert gcn1 is gcn2

    def test_layer_type_hashable(self):
        """Test LayerType members are hashable."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        layer_set = {LayerType.GCN, LayerType.GAT, LayerType.SAGE}
        assert len(layer_set) == 3
        assert LayerType.GCN in layer_set

    def test_layer_type_in_dict_key(self):
        """Test LayerType can be used as dict key."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        layer_dict = {
            LayerType.GCN: "Graph Convolutional",
            LayerType.GAT: "Graph Attention",
        }
        assert layer_dict[LayerType.GCN] == "Graph Convolutional"


# =============================================================================
# LAYERTYPE ENUM TESTS - ATTENTION LAYER IDENTIFICATION
# =============================================================================


class TestLayerTypeAttentionLayers:
    """Test identifying attention-based layer types."""

    def test_gat_is_attention_layer(self):
        """Test GAT is identified as attention layer."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        attention_layers = [LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER]
        assert LayerType.GAT in attention_layers

    def test_gatv2_is_attention_layer(self):
        """Test GATV2 is identified as attention layer."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        attention_layers = [LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER]
        assert LayerType.GATV2 in attention_layers

    def test_transformer_is_attention_layer(self):
        """Test TRANSFORMER is identified as attention layer."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        attention_layers = [LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER]
        assert LayerType.TRANSFORMER in attention_layers

    def test_gcn_is_not_attention_layer(self):
        """Test GCN is not an attention layer."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        attention_layers = [LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER]
        assert LayerType.GCN not in attention_layers

    def test_sage_is_not_attention_layer(self):
        """Test SAGE is not an attention layer."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        attention_layers = [LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER]
        assert LayerType.SAGE not in attention_layers

    def test_gin_is_not_attention_layer(self):
        """Test GIN is not an attention layer."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        attention_layers = [LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER]
        assert LayerType.GIN not in attention_layers

    def test_pna_is_not_attention_layer(self):
        """Test PNA is not an attention layer."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        attention_layers = [LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER]
        assert LayerType.PNA not in attention_layers


# =============================================================================
# POOLINGTYPE ENUM TESTS - VALUE VERIFICATION
# =============================================================================


class TestPoolingTypeEnumValues:
    """Test PoolingType enum value definitions."""

    def test_pooling_type_mean_value(self):
        """Test PoolingType.MEAN has correct string value 'mean'."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        assert PoolingType.MEAN.value == "mean"

    def test_pooling_type_max_value(self):
        """Test PoolingType.MAX has correct string value 'max'."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        assert PoolingType.MAX.value == "max"

    def test_pooling_type_sum_value(self):
        """Test PoolingType.SUM has correct string value 'sum'."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        assert PoolingType.SUM.value == "sum"

    def test_pooling_type_attention_value(self):
        """Test PoolingType.ATTENTION has correct string value 'attention'."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        assert PoolingType.ATTENTION.value == "attention"

    def test_pooling_type_set2set_value(self):
        """Test PoolingType.SET2SET has correct string value 'set2set'."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        assert PoolingType.SET2SET.value == "set2set"

    def test_pooling_type_topk_value(self):
        """Test PoolingType.TOPK has correct string value 'topk'."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        assert PoolingType.TOPK.value == "topk"


# =============================================================================
# POOLINGTYPE ENUM TESTS - ENUM BEHAVIOR
# =============================================================================


class TestPoolingTypeEnumBehavior:
    """Test PoolingType enum behaviors and properties."""

    def test_pooling_type_is_enum_subclass(self):
        """Test PoolingType is a subclass of Enum."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        assert issubclass(PoolingType, Enum)

    def test_pooling_type_has_six_members(self):
        """Test PoolingType has exactly 6 enum members."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        assert len(PoolingType) == 6

    def test_pooling_type_member_is_instance(self):
        """Test each PoolingType member is instance of PoolingType."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        for pooling_type in PoolingType:
            assert isinstance(pooling_type, PoolingType)

    def test_pooling_type_can_iterate(self):
        """Test PoolingType can be iterated."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        members = list(PoolingType)
        assert len(members) == 6
        assert PoolingType.MEAN in members
        assert PoolingType.ATTENTION in members

    def test_pooling_type_membership_check(self):
        """Test membership check for PoolingType."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        assert PoolingType.MEAN in PoolingType
        assert PoolingType.MAX in PoolingType
        assert PoolingType.TOPK in PoolingType

    def test_pooling_type_value_to_member(self):
        """Test converting string value to PoolingType member."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        assert PoolingType("mean") == PoolingType.MEAN
        assert PoolingType("max") == PoolingType.MAX
        assert PoolingType("sum") == PoolingType.SUM
        assert PoolingType("attention") == PoolingType.ATTENTION
        assert PoolingType("set2set") == PoolingType.SET2SET
        assert PoolingType("topk") == PoolingType.TOPK

    def test_pooling_type_invalid_value_raises(self):
        """Test invalid value raises ValueError."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        with pytest.raises(ValueError):
            PoolingType("invalid_pooling_type")

    def test_pooling_type_name_attribute(self):
        """Test PoolingType name attribute."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        assert PoolingType.MEAN.name == "MEAN"
        assert PoolingType.ATTENTION.name == "ATTENTION"
        assert PoolingType.SET2SET.name == "SET2SET"

    def test_pooling_type_equality(self):
        """Test PoolingType equality comparison."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        assert PoolingType.MEAN == PoolingType.MEAN
        assert PoolingType.MAX == PoolingType.MAX
        assert PoolingType.MEAN != PoolingType.MAX

    def test_pooling_type_hashable(self):
        """Test PoolingType members are hashable."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        pooling_set = {PoolingType.MEAN, PoolingType.MAX, PoolingType.SUM}
        assert len(pooling_set) == 3
        assert PoolingType.MEAN in pooling_set


# =============================================================================
# AGGREGATIONTYPE ENUM TESTS - VALUE VERIFICATION
# =============================================================================


class TestAggregationTypeEnumValues:
    """Test AggregationType enum value definitions."""

    def test_aggregation_type_mean_value(self):
        """Test AggregationType.MEAN has correct string value 'mean'."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType

        assert AggregationType.MEAN.value == "mean"

    def test_aggregation_type_max_value(self):
        """Test AggregationType.MAX has correct string value 'max'."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType

        assert AggregationType.MAX.value == "max"

    def test_aggregation_type_sum_value(self):
        """Test AggregationType.SUM has correct string value 'sum'."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType

        assert AggregationType.SUM.value == "sum"

    def test_aggregation_type_lstm_value(self):
        """Test AggregationType.LSTM has correct string value 'lstm'."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType

        assert AggregationType.LSTM.value == "lstm"

    def test_aggregation_type_multi_value(self):
        """Test AggregationType.MULTI has correct string value 'multi'."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType

        assert AggregationType.MULTI.value == "multi"


# =============================================================================
# AGGREGATIONTYPE ENUM TESTS - ENUM BEHAVIOR
# =============================================================================


class TestAggregationTypeEnumBehavior:
    """Test AggregationType enum behaviors and properties."""

    def test_aggregation_type_is_enum_subclass(self):
        """Test AggregationType is a subclass of Enum."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType

        assert issubclass(AggregationType, Enum)

    def test_aggregation_type_has_five_members(self):
        """Test AggregationType has exactly 5 enum members."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType

        assert len(AggregationType) == 5

    def test_aggregation_type_member_is_instance(self):
        """Test each AggregationType member is instance of AggregationType."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType

        for agg_type in AggregationType:
            assert isinstance(agg_type, AggregationType)

    def test_aggregation_type_can_iterate(self):
        """Test AggregationType can be iterated."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType

        members = list(AggregationType)
        assert len(members) == 5
        assert AggregationType.MEAN in members
        assert AggregationType.MULTI in members

    def test_aggregation_type_membership_check(self):
        """Test membership check for AggregationType."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType

        assert AggregationType.MEAN in AggregationType
        assert AggregationType.LSTM in AggregationType
        assert AggregationType.MULTI in AggregationType

    def test_aggregation_type_value_to_member(self):
        """Test converting string value to AggregationType member."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType

        assert AggregationType("mean") == AggregationType.MEAN
        assert AggregationType("max") == AggregationType.MAX
        assert AggregationType("sum") == AggregationType.SUM
        assert AggregationType("lstm") == AggregationType.LSTM
        assert AggregationType("multi") == AggregationType.MULTI

    def test_aggregation_type_invalid_value_raises(self):
        """Test invalid value raises ValueError."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType

        with pytest.raises(ValueError):
            AggregationType("invalid_aggregation_type")

    def test_aggregation_type_name_attribute(self):
        """Test AggregationType name attribute."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType

        assert AggregationType.MEAN.name == "MEAN"
        assert AggregationType.LSTM.name == "LSTM"
        assert AggregationType.MULTI.name == "MULTI"

    def test_aggregation_type_equality(self):
        """Test AggregationType equality comparison."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType

        assert AggregationType.MEAN == AggregationType.MEAN
        assert AggregationType.SUM == AggregationType.SUM
        assert AggregationType.MEAN != AggregationType.SUM

    def test_aggregation_type_hashable(self):
        """Test AggregationType members are hashable."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType

        agg_set = {AggregationType.MEAN, AggregationType.MAX, AggregationType.SUM}
        assert len(agg_set) == 3
        assert AggregationType.MEAN in agg_set


# =============================================================================
# ACTIVATIONTYPE ENUM TESTS - VALUE VERIFICATION
# =============================================================================


class TestActivationTypeEnumValues:
    """Test ActivationType enum value definitions."""

    def test_activation_type_relu_value(self):
        """Test ActivationType.RELU has correct string value 'relu'."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        assert ActivationType.RELU.value == "relu"

    def test_activation_type_gelu_value(self):
        """Test ActivationType.GELU has correct string value 'gelu'."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        assert ActivationType.GELU.value == "gelu"

    def test_activation_type_elu_value(self):
        """Test ActivationType.ELU has correct string value 'elu'."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        assert ActivationType.ELU.value == "elu"

    def test_activation_type_leaky_relu_value(self):
        """Test ActivationType.LEAKY_RELU has correct string value 'leaky_relu'."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        assert ActivationType.LEAKY_RELU.value == "leaky_relu"

    def test_activation_type_silu_value(self):
        """Test ActivationType.SILU has correct string value 'silu'."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        assert ActivationType.SILU.value == "silu"

    def test_activation_type_tanh_value(self):
        """Test ActivationType.TANH has correct string value 'tanh'."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        assert ActivationType.TANH.value == "tanh"

    def test_activation_type_prelu_value(self):
        """Test ActivationType.PRELU has correct string value 'prelu'."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        assert ActivationType.PRELU.value == "prelu"


# =============================================================================
# ACTIVATIONTYPE ENUM TESTS - ENUM BEHAVIOR
# =============================================================================


class TestActivationTypeEnumBehavior:
    """Test ActivationType enum behaviors and properties."""

    def test_activation_type_is_enum_subclass(self):
        """Test ActivationType is a subclass of Enum."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        assert issubclass(ActivationType, Enum)

    def test_activation_type_has_seven_members(self):
        """Test ActivationType has exactly 7 enum members."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        assert len(ActivationType) == 7

    def test_activation_type_member_is_instance(self):
        """Test each ActivationType member is instance of ActivationType."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        for act_type in ActivationType:
            assert isinstance(act_type, ActivationType)

    def test_activation_type_can_iterate(self):
        """Test ActivationType can be iterated."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        members = list(ActivationType)
        assert len(members) == 7
        assert ActivationType.RELU in members
        assert ActivationType.GELU in members

    def test_activation_type_membership_check(self):
        """Test membership check for ActivationType."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        assert ActivationType.RELU in ActivationType
        assert ActivationType.GELU in ActivationType
        assert ActivationType.PRELU in ActivationType

    def test_activation_type_value_to_member(self):
        """Test converting string value to ActivationType member."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        assert ActivationType("relu") == ActivationType.RELU
        assert ActivationType("gelu") == ActivationType.GELU
        assert ActivationType("elu") == ActivationType.ELU
        assert ActivationType("leaky_relu") == ActivationType.LEAKY_RELU
        assert ActivationType("silu") == ActivationType.SILU
        assert ActivationType("tanh") == ActivationType.TANH
        assert ActivationType("prelu") == ActivationType.PRELU

    def test_activation_type_invalid_value_raises(self):
        """Test invalid value raises ValueError."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        with pytest.raises(ValueError):
            ActivationType("invalid_activation_type")

    def test_activation_type_name_attribute(self):
        """Test ActivationType name attribute."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        assert ActivationType.RELU.name == "RELU"
        assert ActivationType.GELU.name == "GELU"
        assert ActivationType.LEAKY_RELU.name == "LEAKY_RELU"

    def test_activation_type_equality(self):
        """Test ActivationType equality comparison."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        assert ActivationType.RELU == ActivationType.RELU
        assert ActivationType.GELU == ActivationType.GELU
        assert ActivationType.RELU != ActivationType.GELU

    def test_activation_type_hashable(self):
        """Test ActivationType members are hashable."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        act_set = {ActivationType.RELU, ActivationType.GELU, ActivationType.ELU}
        assert len(act_set) == 3
        assert ActivationType.RELU in act_set


# =============================================================================
# ENUM CROSS-TYPE TESTS
# =============================================================================


class TestEnumCrossTypeComparisons:
    """Test comparisons and interactions across different enum types."""

    def test_layer_type_not_equal_pooling_type(self):
        """Test LayerType and PoolingType members are not equal."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType, PoolingType

        # Both have similar string values but should not be equal
        assert LayerType.GCN != PoolingType.MEAN

    def test_pooling_type_not_equal_aggregation_type(self):
        """Test PoolingType and AggregationType members are not equal."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType, PoolingType

        # MEAN exists in both but should not be equal
        assert PoolingType.MEAN != AggregationType.MEAN

    def test_aggregation_type_not_equal_activation_type(self):
        """Test AggregationType and ActivationType are different enums."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType, AggregationType

        # Different enums should not be equal
        for agg in AggregationType:
            for act in ActivationType:
                assert agg != act

    def test_all_enum_types_have_unique_identity(self):
        """Test all enum types maintain their unique identity."""
        from milia_pipeline.models.hpo.nas.search_space import (
            ActivationType,
            AggregationType,
            LayerType,
            PoolingType,
        )

        assert LayerType is not PoolingType
        assert PoolingType is not AggregationType
        assert AggregationType is not ActivationType
        assert LayerType is not ActivationType

    def test_enums_can_be_used_together_in_dict(self):
        """Test different enum types can be used together as dict keys."""
        from milia_pipeline.models.hpo.nas.search_space import (
            ActivationType,
            AggregationType,
            LayerType,
            PoolingType,
        )

        config = {
            LayerType.GAT: "attention layer",
            PoolingType.MEAN: "mean pooling",
            AggregationType.SUM: "sum aggregation",
            ActivationType.RELU: "relu activation",
        }

        assert config[LayerType.GAT] == "attention layer"
        assert config[PoolingType.MEAN] == "mean pooling"
        assert config[AggregationType.SUM] == "sum aggregation"
        assert config[ActivationType.RELU] == "relu activation"

    def test_enums_can_be_stored_in_mixed_list(self):
        """Test different enum types can be stored in same list."""
        from milia_pipeline.models.hpo.nas.search_space import (
            ActivationType,
            AggregationType,
            LayerType,
            PoolingType,
        )

        mixed_list = [
            LayerType.GCN,
            PoolingType.MEAN,
            AggregationType.SUM,
            ActivationType.RELU,
        ]

        assert len(mixed_list) == 4
        assert isinstance(mixed_list[0], LayerType)
        assert isinstance(mixed_list[1], PoolingType)
        assert isinstance(mixed_list[2], AggregationType)
        assert isinstance(mixed_list[3], ActivationType)


# =============================================================================
# MODULE EXPORTS TESTS FOR ENUMS
# =============================================================================


class TestEnumModuleExports:
    """Test enum classes are properly exported from module."""

    def test_layer_type_in_all_exports(self):
        """Test LayerType is in module __all__."""
        from milia_pipeline.models.hpo.nas import search_space

        assert "LayerType" in search_space.__all__

    def test_pooling_type_in_all_exports(self):
        """Test PoolingType is in module __all__."""
        from milia_pipeline.models.hpo.nas import search_space

        assert "PoolingType" in search_space.__all__

    def test_aggregation_type_in_all_exports(self):
        """Test AggregationType is in module __all__."""
        from milia_pipeline.models.hpo.nas import search_space

        assert "AggregationType" in search_space.__all__

    def test_activation_type_in_all_exports(self):
        """Test ActivationType is in module __all__."""
        from milia_pipeline.models.hpo.nas import search_space

        assert "ActivationType" in search_space.__all__

    def test_can_import_all_enums_from_module(self):
        """Test all enums can be imported directly from module."""
        from milia_pipeline.models.hpo.nas.search_space import (
            ActivationType,
            AggregationType,
            LayerType,
            PoolingType,
        )

        assert LayerType is not None
        assert PoolingType is not None
        assert AggregationType is not None
        assert ActivationType is not None


# =============================================================================
# ENUM STRING REPRESENTATION TESTS
# =============================================================================


class TestEnumStringRepresentations:
    """Test string representations of enum members."""

    def test_layer_type_repr(self):
        """Test LayerType repr includes type name."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        repr_str = repr(LayerType.GCN)
        assert "LayerType" in repr_str
        assert "GCN" in repr_str

    def test_pooling_type_repr(self):
        """Test PoolingType repr includes type name."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        repr_str = repr(PoolingType.MEAN)
        assert "PoolingType" in repr_str
        assert "MEAN" in repr_str

    def test_aggregation_type_repr(self):
        """Test AggregationType repr includes type name."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType

        repr_str = repr(AggregationType.SUM)
        assert "AggregationType" in repr_str
        assert "SUM" in repr_str

    def test_activation_type_repr(self):
        """Test ActivationType repr includes type name."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType

        repr_str = repr(ActivationType.RELU)
        assert "ActivationType" in repr_str
        assert "RELU" in repr_str

    def test_layer_type_str_is_value(self):
        """Test str(LayerType) returns enum representation."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType

        # Default Enum str behavior
        str_result = str(LayerType.GCN)
        assert "GCN" in str_result

    def test_pooling_type_str_is_value(self):
        """Test str(PoolingType) returns enum representation."""
        from milia_pipeline.models.hpo.nas.search_space import PoolingType

        str_result = str(PoolingType.MEAN)
        assert "MEAN" in str_result


# =============================================================================
# LAYERCONFIG TEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_gcn_layer_data():
    """Provide sample data for GCN layer configuration."""
    from milia_pipeline.models.hpo.nas.search_space import LayerType

    return {
        "type": LayerType.GCN,
        "hidden_channels": 64,
    }


@pytest.fixture
def sample_gat_layer_data():
    """Provide sample data for GAT layer configuration."""
    from milia_pipeline.models.hpo.nas.search_space import LayerType

    return {
        "type": LayerType.GAT,
        "hidden_channels": 64,
        "heads": 4,
        "dropout": 0.2,
        "batch_norm": True,
        "residual": True,
    }


@pytest.fixture
def sample_layer_dict():
    """Provide sample layer configuration as dict."""
    return {
        "type": "gat",
        "hidden_channels": 64,
        "heads": 4,
        "dropout": 0.2,
        "activation": "relu",
        "batch_norm": True,
        "residual": False,
    }


# =============================================================================
# LAYERCONFIG INITIALIZATION TESTS
# =============================================================================


class TestLayerConfigInitialization:
    """Test LayerConfig initialization."""

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_minimal_init(self):
        """Test LayerConfig with minimal required parameters."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
        )

        assert config.type == LayerType.GCN
        assert config.hidden_channels == 64

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_full_init(self):
        """Test LayerConfig with all parameters."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GAT,
            hidden_channels=128,
            heads=8,
            dropout=0.3,
            activation="gelu",
            batch_norm=True,
            residual=True,
        )

        assert config.type == LayerType.GAT
        assert config.hidden_channels == 128
        assert config.heads == 8
        assert config.dropout == 0.3
        assert config.activation == "gelu"
        assert config.batch_norm is True
        assert config.residual is True

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_default_heads(self):
        """Test LayerConfig default heads is 1."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
        )

        assert config.heads == 1

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_default_dropout(self):
        """Test LayerConfig default dropout is 0.0."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
        )

        assert config.dropout == 0.0

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_default_activation(self):
        """Test LayerConfig default activation is 'relu'."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
        )

        assert config.activation == "relu"

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_default_batch_norm(self):
        """Test LayerConfig default batch_norm is True."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
        )

        assert config.batch_norm is True

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_default_residual(self):
        """Test LayerConfig default residual is False."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
        )

        assert config.residual is False


# =============================================================================
# LAYERCONFIG PYDANTIC MODEL PROPERTIES TESTS
# =============================================================================


class TestLayerConfigPydanticModelProperties:
    """Test LayerConfig Pydantic BaseModel properties.

    Note: LayerConfig is a Pydantic V2 BaseModel with frozen=True,
    not a dataclass. Modifying attributes raises pydantic.ValidationError.
    """

    def test_layer_config_is_pydantic_model(self):
        """Test LayerConfig is a Pydantic BaseModel subclass."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig

        assert is_pydantic_model(LayerConfig)
        assert issubclass(LayerConfig, BaseModel)

    def test_layer_config_is_frozen(self):
        """Test LayerConfig is frozen (immutable Pydantic model)."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
        )

        # Pydantic V2 frozen models raise ValidationError on modification
        with pytest.raises(ValidationError):
            config.hidden_channels = 128

    def test_layer_config_cannot_modify_type(self):
        """Test cannot modify type attribute."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
        )

        with pytest.raises(ValidationError):
            config.type = LayerType.GAT

    def test_layer_config_cannot_modify_heads(self):
        """Test cannot modify heads attribute."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GAT,
            hidden_channels=64,
            heads=4,
        )

        with pytest.raises(ValidationError):
            config.heads = 8

    def test_layer_config_cannot_modify_dropout(self):
        """Test cannot modify dropout attribute."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
            dropout=0.2,
        )

        with pytest.raises(ValidationError):
            config.dropout = 0.5

    def test_layer_config_cannot_add_new_attribute(self):
        """Test cannot add new attribute to frozen config."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
        )

        with pytest.raises(ValidationError):
            config.new_attribute = "value"


# =============================================================================
# LAYERCONFIG VALIDATION TESTS
# =============================================================================


class TestLayerConfigValidation:
    """Test LayerConfig __post_init__ validation."""

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_rejects_zero_hidden_channels(self):
        """Test LayerConfig rejects hidden_channels = 0."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        with pytest.raises(Exception) as exc_info:
            LayerConfig(
                type=LayerType.GCN,
                hidden_channels=0,
            )

        assert (
            "hidden_channels" in str(exc_info.value).lower()
            or "positive" in str(exc_info.value).lower()
        )

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_rejects_negative_hidden_channels(self):
        """Test LayerConfig rejects negative hidden_channels."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        with pytest.raises(Exception) as exc_info:
            LayerConfig(
                type=LayerType.GCN,
                hidden_channels=-64,
            )

        assert (
            "hidden_channels" in str(exc_info.value).lower()
            or "positive" in str(exc_info.value).lower()
        )

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_rejects_zero_heads(self):
        """Test LayerConfig rejects heads = 0."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        with pytest.raises(Exception) as exc_info:
            LayerConfig(
                type=LayerType.GAT,
                hidden_channels=64,
                heads=0,
            )

        assert "heads" in str(exc_info.value).lower() or "1" in str(exc_info.value)

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_rejects_negative_heads(self):
        """Test LayerConfig rejects negative heads."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        with pytest.raises(Exception) as exc_info:
            LayerConfig(
                type=LayerType.GAT,
                hidden_channels=64,
                heads=-4,
            )

        assert "heads" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_rejects_negative_dropout(self):
        """Test LayerConfig rejects negative dropout."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        with pytest.raises(Exception) as exc_info:
            LayerConfig(
                type=LayerType.GCN,
                hidden_channels=64,
                dropout=-0.1,
            )

        assert "dropout" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_rejects_dropout_greater_than_one(self):
        """Test LayerConfig rejects dropout > 1.0."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        with pytest.raises(Exception) as exc_info:
            LayerConfig(
                type=LayerType.GCN,
                hidden_channels=64,
                dropout=1.5,
            )

        assert "dropout" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_accepts_dropout_zero(self):
        """Test LayerConfig accepts dropout = 0.0 (boundary)."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
            dropout=0.0,
        )

        assert config.dropout == 0.0

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_accepts_dropout_one(self):
        """Test LayerConfig accepts dropout = 1.0 (boundary)."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
            dropout=1.0,
        )

        assert config.dropout == 1.0

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_accepts_heads_one(self):
        """Test LayerConfig accepts heads = 1 (boundary)."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GAT,
            hidden_channels=64,
            heads=1,
        )

        assert config.heads == 1

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_accepts_large_heads(self):
        """Test LayerConfig accepts large heads value."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.TRANSFORMER,
            hidden_channels=512,
            heads=16,
        )

        assert config.heads == 16

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_accepts_large_hidden_channels(self):
        """Test LayerConfig accepts large hidden_channels."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=2048,
        )

        assert config.hidden_channels == 2048


# =============================================================================
# LAYERCONFIG VALIDATION ERROR ATTRIBUTES TESTS
# =============================================================================


class TestLayerConfigValidationErrorAttributes:
    """Test LayerConfig validation error attributes."""

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_hidden_channels_error_has_config_key(self):
        """Test hidden_channels error includes config_key."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        with pytest.raises(Exception) as exc_info:
            LayerConfig(
                type=LayerType.GCN,
                hidden_channels=-1,
            )

        error = exc_info.value
        if hasattr(error, "config_key"):
            assert "hidden_channels" in error.config_key

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_hidden_channels_error_has_actual_value(self):
        """Test hidden_channels error includes actual_value."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        with pytest.raises(Exception) as exc_info:
            LayerConfig(
                type=LayerType.GCN,
                hidden_channels=-100,
            )

        error = exc_info.value
        if hasattr(error, "actual_value"):
            assert error.actual_value == -100

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_heads_error_has_config_key(self):
        """Test heads error includes config_key."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        with pytest.raises(Exception) as exc_info:
            LayerConfig(
                type=LayerType.GAT,
                hidden_channels=64,
                heads=0,
            )

        error = exc_info.value
        if hasattr(error, "config_key"):
            assert "heads" in error.config_key

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_dropout_error_has_config_key(self):
        """Test dropout error includes config_key."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        with pytest.raises(Exception) as exc_info:
            LayerConfig(
                type=LayerType.GCN,
                hidden_channels=64,
                dropout=2.0,
            )

        error = exc_info.value
        if hasattr(error, "config_key"):
            assert "dropout" in error.config_key


# =============================================================================
# LAYERCONFIG TO_DICT METHOD TESTS
# =============================================================================


class TestLayerConfigToDict:
    """Test LayerConfig.to_dict() method."""

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_to_dict_returns_dict(self):
        """Test to_dict returns a dictionary."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
        )

        result = config.to_dict()

        assert isinstance(result, dict)

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_to_dict_contains_type_as_string(self):
        """Test to_dict converts type to string value."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GAT,
            hidden_channels=64,
        )

        result = config.to_dict()

        assert result["type"] == "gat"
        assert isinstance(result["type"], str)

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_to_dict_contains_hidden_channels(self):
        """Test to_dict contains hidden_channels."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=128,
        )

        result = config.to_dict()

        assert result["hidden_channels"] == 128

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_to_dict_contains_heads(self):
        """Test to_dict contains heads."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GAT,
            hidden_channels=64,
            heads=8,
        )

        result = config.to_dict()

        assert result["heads"] == 8

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_to_dict_contains_dropout(self):
        """Test to_dict contains dropout."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
            dropout=0.3,
        )

        result = config.to_dict()

        assert result["dropout"] == 0.3

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_to_dict_contains_activation(self):
        """Test to_dict contains activation."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
            activation="gelu",
        )

        result = config.to_dict()

        assert result["activation"] == "gelu"

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_to_dict_contains_batch_norm(self):
        """Test to_dict contains batch_norm."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
            batch_norm=False,
        )

        result = config.to_dict()

        assert result["batch_norm"] is False

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_to_dict_contains_residual(self):
        """Test to_dict contains residual."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
            residual=True,
        )

        result = config.to_dict()

        assert result["residual"] is True

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_to_dict_contains_all_seven_keys(self):
        """Test to_dict contains all seven expected keys."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
        )

        result = config.to_dict()

        expected_keys = {
            "type",
            "hidden_channels",
            "heads",
            "dropout",
            "activation",
            "batch_norm",
            "residual",
        }
        assert set(result.keys()) == expected_keys

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_to_dict_full_example(self):
        """Test to_dict with full configuration."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.TRANSFORMER,
            hidden_channels=256,
            heads=8,
            dropout=0.1,
            activation="gelu",
            batch_norm=True,
            residual=True,
        )

        result = config.to_dict()

        assert result == {
            "type": "transformer",
            "hidden_channels": 256,
            "heads": 8,
            "dropout": 0.1,
            "activation": "gelu",
            "batch_norm": True,
            "residual": True,
        }


# =============================================================================
# LAYERCONFIG FROM_DICT CLASS METHOD TESTS
# =============================================================================


class TestLayerConfigFromDict:
    """Test LayerConfig.from_dict() class method."""

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_from_dict_returns_layer_config(self):
        """Test from_dict returns LayerConfig instance."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig

        config_dict = {
            "type": "gcn",
            "hidden_channels": 64,
        }

        result = LayerConfig.from_dict(config_dict)

        assert isinstance(result, LayerConfig)

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_from_dict_converts_string_type(self):
        """Test from_dict converts string type to LayerType enum."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config_dict = {
            "type": "gat",
            "hidden_channels": 64,
        }

        result = LayerConfig.from_dict(config_dict)

        assert result.type == LayerType.GAT
        assert isinstance(result.type, LayerType)

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_from_dict_preserves_layer_type_enum(self):
        """Test from_dict preserves LayerType enum if already enum."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config_dict = {
            "type": LayerType.SAGE,
            "hidden_channels": 128,
        }

        result = LayerConfig.from_dict(config_dict)

        assert result.type == LayerType.SAGE

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_from_dict_sets_hidden_channels(self):
        """Test from_dict sets hidden_channels correctly."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig

        config_dict = {
            "type": "gcn",
            "hidden_channels": 256,
        }

        result = LayerConfig.from_dict(config_dict)

        assert result.hidden_channels == 256

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_from_dict_sets_heads(self):
        """Test from_dict sets heads correctly."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig

        config_dict = {
            "type": "gat",
            "hidden_channels": 64,
            "heads": 4,
        }

        result = LayerConfig.from_dict(config_dict)

        assert result.heads == 4

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_from_dict_sets_dropout(self):
        """Test from_dict sets dropout correctly."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig

        config_dict = {
            "type": "gcn",
            "hidden_channels": 64,
            "dropout": 0.5,
        }

        result = LayerConfig.from_dict(config_dict)

        assert result.dropout == 0.5

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_from_dict_sets_activation(self):
        """Test from_dict sets activation correctly."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig

        config_dict = {
            "type": "gcn",
            "hidden_channels": 64,
            "activation": "elu",
        }

        result = LayerConfig.from_dict(config_dict)

        assert result.activation == "elu"

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_from_dict_sets_batch_norm(self):
        """Test from_dict sets batch_norm correctly."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig

        config_dict = {
            "type": "gcn",
            "hidden_channels": 64,
            "batch_norm": False,
        }

        result = LayerConfig.from_dict(config_dict)

        assert result.batch_norm is False

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_from_dict_sets_residual(self):
        """Test from_dict sets residual correctly."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig

        config_dict = {
            "type": "gcn",
            "hidden_channels": 64,
            "residual": True,
        }

        result = LayerConfig.from_dict(config_dict)

        assert result.residual is True

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_from_dict_full_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        original = LayerConfig(
            type=LayerType.GATV2,
            hidden_channels=128,
            heads=4,
            dropout=0.2,
            activation="gelu",
            batch_norm=True,
            residual=True,
        )

        dict_repr = original.to_dict()
        restored = LayerConfig.from_dict(dict_repr)

        assert restored.type == original.type
        assert restored.hidden_channels == original.hidden_channels
        assert restored.heads == original.heads
        assert restored.dropout == original.dropout
        assert restored.activation == original.activation
        assert restored.batch_norm == original.batch_norm
        assert restored.residual == original.residual

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_from_dict_does_not_modify_input(self):
        """Test from_dict does not modify input dictionary."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig

        config_dict = {
            "type": "gcn",
            "hidden_channels": 64,
        }

        _original_dict = config_dict.copy()
        LayerConfig.from_dict(config_dict)

        # Original dict keys should be unchanged
        assert "type" in config_dict
        assert "hidden_channels" in config_dict


# =============================================================================
# LAYERCONFIG EQUALITY AND HASHING TESTS
# =============================================================================


class TestLayerConfigEqualityAndHashing:
    """Test LayerConfig equality and hashing behavior."""

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_equal_same_values(self):
        """Test LayerConfigs with same values are equal."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config1 = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
            heads=1,
            dropout=0.0,
        )

        config2 = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
            heads=1,
            dropout=0.0,
        )

        assert config1 == config2

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_not_equal_different_type(self):
        """Test LayerConfigs with different types are not equal."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config1 = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
        )

        config2 = LayerConfig(
            type=LayerType.GAT,
            hidden_channels=64,
        )

        assert config1 != config2

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_not_equal_different_channels(self):
        """Test LayerConfigs with different hidden_channels are not equal."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config1 = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
        )

        config2 = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=128,
        )

        assert config1 != config2

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_hashable(self):
        """Test LayerConfig is hashable (can be used in sets/dicts)."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(
            type=LayerType.GCN,
            hidden_channels=64,
        )

        # Should be hashable
        hash_value = hash(config)
        assert isinstance(hash_value, int)

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_in_set(self):
        """Test LayerConfig can be added to set."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config1 = LayerConfig(type=LayerType.GCN, hidden_channels=64)
        config2 = LayerConfig(type=LayerType.GAT, hidden_channels=64)
        config3 = LayerConfig(type=LayerType.GCN, hidden_channels=64)  # Same as config1

        config_set = {config1, config2, config3}

        # config1 and config3 should be deduplicated
        assert len(config_set) == 2

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_as_dict_key(self):
        """Test LayerConfig can be used as dictionary key."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(type=LayerType.GCN, hidden_channels=64)

        config_dict = {config: "GCN layer"}

        assert config_dict[config] == "GCN layer"


# =============================================================================
# LAYERCONFIG WITH ALL LAYER TYPES TESTS
# =============================================================================


class TestLayerConfigAllLayerTypes:
    """Test LayerConfig with all LayerType values."""

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_with_gcn(self):
        """Test LayerConfig with GCN type."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(type=LayerType.GCN, hidden_channels=64)
        assert config.type == LayerType.GCN

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_with_gat(self):
        """Test LayerConfig with GAT type."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(type=LayerType.GAT, hidden_channels=64, heads=4)
        assert config.type == LayerType.GAT
        assert config.heads == 4

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_with_sage(self):
        """Test LayerConfig with SAGE type."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(type=LayerType.SAGE, hidden_channels=64)
        assert config.type == LayerType.SAGE

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_with_gin(self):
        """Test LayerConfig with GIN type."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(type=LayerType.GIN, hidden_channels=64)
        assert config.type == LayerType.GIN

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_with_gatv2(self):
        """Test LayerConfig with GATV2 type."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(type=LayerType.GATV2, hidden_channels=64, heads=8)
        assert config.type == LayerType.GATV2
        assert config.heads == 8

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_with_transformer(self):
        """Test LayerConfig with TRANSFORMER type."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(type=LayerType.TRANSFORMER, hidden_channels=256, heads=16)
        assert config.type == LayerType.TRANSFORMER
        assert config.heads == 16

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_with_pna(self):
        """Test LayerConfig with PNA type."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(type=LayerType.PNA, hidden_channels=64)
        assert config.type == LayerType.PNA


# =============================================================================
# LAYERCONFIG MODULE EXPORT TESTS
# =============================================================================


class TestLayerConfigModuleExport:
    """Test LayerConfig module export."""

    def test_layer_config_in_all_exports(self):
        """Test LayerConfig is in module __all__."""
        from milia_pipeline.models.hpo.nas import search_space

        assert "LayerConfig" in search_space.__all__

    def test_layer_config_importable_directly(self):
        """Test LayerConfig can be imported directly."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig

        assert LayerConfig is not None


# =============================================================================
# LAYERCONFIG REPR AND STR TESTS
# =============================================================================


class TestLayerConfigRepresentation:
    """Test LayerConfig repr and str methods."""

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_repr_includes_class_name(self):
        """Test LayerConfig repr includes class name."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(type=LayerType.GCN, hidden_channels=64)
        repr_str = repr(config)

        assert "LayerConfig" in repr_str

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_repr_includes_type(self):
        """Test LayerConfig repr includes type."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(type=LayerType.GAT, hidden_channels=64)
        repr_str = repr(config)

        assert "GAT" in repr_str or "gat" in repr_str.lower()

    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_repr_includes_hidden_channels(self):
        """Test LayerConfig repr includes hidden_channels."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, LayerType

        config = LayerConfig(type=LayerType.GCN, hidden_channels=128)
        repr_str = repr(config)

        assert "128" in repr_str


# =============================================================================
# GNNARCHITECTURESPACE TEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_arch_space_dict():
    """Provide sample architecture space as dictionary."""
    return {
        "min_layers": 2,
        "max_layers": 6,
        "layer_types": ["gcn", "gat"],
        "hidden_channels": [64, 128, 256],
        "heads": [1, 2, 4],
        "dropout_range": (0.0, 0.5),
        "allow_skip_connections": True,
        "allow_dense_connections": False,
        "allow_mixed_layers": True,
        "pooling_types": ["mean", "attention"],
        "aggregation_types": ["mean", "sum"],
        "activation_types": ["relu", "gelu"],
        "batch_norm_options": [True, False],
    }


@pytest.fixture
def minimal_arch_space_dict():
    """Provide minimal valid architecture space as dictionary."""
    return {
        "min_layers": 1,
        "max_layers": 2,
        "layer_types": ["gcn"],
        "hidden_channels": [64],
        "heads": [1],
        "dropout_range": (0.0, 0.3),
        "pooling_types": ["mean"],
        "aggregation_types": ["mean"],
    }


# =============================================================================
# GNNARCHITECTURESPACE INITIALIZATION TESTS - DEFAULT VALUES
# =============================================================================


class TestGNNArchitectureSpaceDefaultValues:
    """Test GNNArchitectureSpace default initialization values."""

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_default_min_layers(self):
        """Test default min_layers is 2."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()

        assert space.min_layers == 2

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_default_max_layers(self):
        """Test default max_layers is 8."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()

        assert space.max_layers == 8

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_default_layer_types(self):
        """Test default layer_types contains GCN, GAT, SAGE."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace()

        assert LayerType.GCN in space.layer_types
        assert LayerType.GAT in space.layer_types
        assert LayerType.SAGE in space.layer_types
        assert len(space.layer_types) == 3

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_default_hidden_channels(self):
        """Test default hidden_channels is [32, 64, 128, 256]."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()

        assert space.hidden_channels == [32, 64, 128, 256]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_default_heads(self):
        """Test default heads is [1, 2, 4, 8]."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()

        assert space.heads == [1, 2, 4, 8]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_default_dropout_range(self):
        """Test default dropout_range is (0.0, 0.6)."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()

        assert space.dropout_range == (0.0, 0.6)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_default_allow_skip_connections(self):
        """Test default allow_skip_connections is True."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()

        assert space.allow_skip_connections is True

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_default_allow_dense_connections(self):
        """Test default allow_dense_connections is False."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()

        assert space.allow_dense_connections is False

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_default_allow_mixed_layers(self):
        """Test default allow_mixed_layers is True."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()

        assert space.allow_mixed_layers is True

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_default_pooling_types(self):
        """Test default pooling_types contains MEAN, ATTENTION."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, PoolingType

        space = GNNArchitectureSpace()

        assert PoolingType.MEAN in space.pooling_types
        assert PoolingType.ATTENTION in space.pooling_types
        assert len(space.pooling_types) == 2

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_default_aggregation_types(self):
        """Test default aggregation_types contains MEAN, SUM."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType, GNNArchitectureSpace

        space = GNNArchitectureSpace()

        assert AggregationType.MEAN in space.aggregation_types
        assert AggregationType.SUM in space.aggregation_types
        assert len(space.aggregation_types) == 2

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_default_activation_types(self):
        """Test default activation_types contains RELU, GELU, ELU."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType, GNNArchitectureSpace

        space = GNNArchitectureSpace()

        assert ActivationType.RELU in space.activation_types
        assert ActivationType.GELU in space.activation_types
        assert ActivationType.ELU in space.activation_types
        assert len(space.activation_types) == 3

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_default_batch_norm_options(self):
        """Test default batch_norm_options is [True, False]."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()

        assert space.batch_norm_options == [True, False]


# =============================================================================
# GNNARCHITECTURESPACE INITIALIZATION TESTS - CUSTOM VALUES
# =============================================================================


class TestGNNArchitectureSpaceCustomValues:
    """Test GNNArchitectureSpace with custom initialization values."""

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_custom_min_max_layers(self):
        """Test custom min_layers and max_layers."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(min_layers=3, max_layers=10)

        assert space.min_layers == 3
        assert space.max_layers == 10

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_custom_layer_types(self):
        """Test custom layer_types."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(layer_types=[LayerType.GIN, LayerType.PNA])

        assert space.layer_types == [LayerType.GIN, LayerType.PNA]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_custom_hidden_channels(self):
        """Test custom hidden_channels."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(hidden_channels=[128, 256, 512, 1024])

        assert space.hidden_channels == [128, 256, 512, 1024]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_custom_heads(self):
        """Test custom heads."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(heads=[2, 4, 8, 16])

        assert space.heads == [2, 4, 8, 16]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_custom_dropout_range(self):
        """Test custom dropout_range."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(dropout_range=(0.1, 0.4))

        assert space.dropout_range == (0.1, 0.4)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_custom_allow_skip_connections_false(self):
        """Test allow_skip_connections set to False."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(allow_skip_connections=False)

        assert space.allow_skip_connections is False

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_custom_allow_dense_connections_true(self):
        """Test allow_dense_connections set to True."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(allow_dense_connections=True)

        assert space.allow_dense_connections is True

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_custom_allow_mixed_layers_false(self):
        """Test allow_mixed_layers set to False."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(allow_mixed_layers=False)

        assert space.allow_mixed_layers is False

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_custom_pooling_types(self):
        """Test custom pooling_types."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, PoolingType

        space = GNNArchitectureSpace(
            pooling_types=[PoolingType.MAX, PoolingType.SUM, PoolingType.TOPK]
        )

        assert space.pooling_types == [PoolingType.MAX, PoolingType.SUM, PoolingType.TOPK]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_custom_aggregation_types(self):
        """Test custom aggregation_types."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType, GNNArchitectureSpace

        space = GNNArchitectureSpace(
            aggregation_types=[AggregationType.MULTI, AggregationType.LSTM]
        )

        assert space.aggregation_types == [AggregationType.MULTI, AggregationType.LSTM]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_custom_activation_types(self):
        """Test custom activation_types."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType, GNNArchitectureSpace

        space = GNNArchitectureSpace(activation_types=[ActivationType.SILU, ActivationType.PRELU])

        assert space.activation_types == [ActivationType.SILU, ActivationType.PRELU]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_custom_batch_norm_options(self):
        """Test custom batch_norm_options."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(batch_norm_options=[True])

        assert space.batch_norm_options == [True]


# =============================================================================
# GNNARCHITECTURESPACE PYDANTIC MODEL PROPERTIES TESTS
# =============================================================================


class TestGNNArchitectureSpacePydanticModelProperties:
    """Test GNNArchitectureSpace Pydantic BaseModel properties.

    Note: GNNArchitectureSpace is a mutable Pydantic V2 BaseModel,
    not a dataclass or frozen model. Attributes can be modified after creation.
    """

    def test_gnn_architecture_space_is_pydantic_model(self):
        """Test GNNArchitectureSpace is a Pydantic BaseModel subclass."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        assert is_pydantic_model(GNNArchitectureSpace)
        assert issubclass(GNNArchitectureSpace, BaseModel)

    def test_gnn_architecture_space_not_frozen(self):
        """Test GNNArchitectureSpace is NOT frozen (mutable)."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()

        # Should be able to modify
        space.min_layers = 3
        assert space.min_layers == 3

    def test_can_modify_max_layers(self):
        """Test can modify max_layers after creation."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()
        space.max_layers = 12

        assert space.max_layers == 12

    def test_can_modify_hidden_channels(self):
        """Test can modify hidden_channels after creation."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()
        space.hidden_channels = [512, 1024]

        assert space.hidden_channels == [512, 1024]


# =============================================================================
# GNNARCHITECTURESPACE VALIDATION TESTS
# =============================================================================


class TestGNNArchitectureSpaceValidation:
    """Test GNNArchitectureSpace __post_init__ validation."""

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_rejects_min_layers_zero(self):
        """Test rejects min_layers = 0."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        with pytest.raises(Exception) as exc_info:
            GNNArchitectureSpace(min_layers=0)

        assert "min_layers" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_rejects_min_layers_negative(self):
        """Test rejects negative min_layers."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        with pytest.raises(Exception) as exc_info:
            GNNArchitectureSpace(min_layers=-1)

        assert "min_layers" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_rejects_max_layers_less_than_min(self):
        """Test rejects max_layers < min_layers."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        with pytest.raises(Exception) as exc_info:
            GNNArchitectureSpace(min_layers=5, max_layers=3)

        assert (
            "max_layers" in str(exc_info.value).lower()
            or "min_layers" in str(exc_info.value).lower()
        )

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_accepts_min_layers_equal_max_layers(self):
        """Test accepts min_layers == max_layers."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(min_layers=3, max_layers=3)

        assert space.min_layers == 3
        assert space.max_layers == 3

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_rejects_empty_layer_types(self):
        """Test rejects empty layer_types list."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        with pytest.raises(Exception) as exc_info:
            GNNArchitectureSpace(layer_types=[])

        assert "layer_types" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_rejects_empty_hidden_channels(self):
        """Test rejects empty hidden_channels list."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        with pytest.raises(Exception) as exc_info:
            GNNArchitectureSpace(hidden_channels=[])

        assert "hidden_channels" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_rejects_zero_hidden_channels(self):
        """Test rejects hidden_channels containing zero."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        with pytest.raises(Exception) as exc_info:
            GNNArchitectureSpace(hidden_channels=[64, 0, 128])

        assert "hidden_channels" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_rejects_negative_hidden_channels(self):
        """Test rejects hidden_channels containing negative value."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        with pytest.raises(Exception) as exc_info:
            GNNArchitectureSpace(hidden_channels=[64, -32, 128])

        assert "hidden_channels" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_rejects_empty_heads(self):
        """Test rejects empty heads list."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        with pytest.raises(Exception) as exc_info:
            GNNArchitectureSpace(heads=[])

        assert "heads" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_rejects_zero_heads(self):
        """Test rejects heads containing zero."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        with pytest.raises(Exception) as exc_info:
            GNNArchitectureSpace(heads=[1, 0, 4])

        assert "heads" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_rejects_negative_heads(self):
        """Test rejects heads containing negative value."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        with pytest.raises(Exception) as exc_info:
            GNNArchitectureSpace(heads=[1, -2, 4])

        assert "heads" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_rejects_invalid_dropout_range_length(self):
        """Test rejects dropout_range with wrong length."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        with pytest.raises(Exception) as exc_info:
            GNNArchitectureSpace(dropout_range=(0.1, 0.3, 0.5))

        assert "dropout_range" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_rejects_dropout_range_min_greater_than_max(self):
        """Test rejects dropout_range where min > max."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        with pytest.raises(Exception) as exc_info:
            GNNArchitectureSpace(dropout_range=(0.5, 0.2))

        assert "dropout_range" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_rejects_dropout_range_negative_min(self):
        """Test rejects dropout_range with negative min."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        with pytest.raises(Exception) as exc_info:
            GNNArchitectureSpace(dropout_range=(-0.1, 0.5))

        assert "dropout_range" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_rejects_dropout_range_max_greater_than_one(self):
        """Test rejects dropout_range with max > 1."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        with pytest.raises(Exception) as exc_info:
            GNNArchitectureSpace(dropout_range=(0.0, 1.5))

        assert "dropout_range" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_accepts_dropout_range_full_zero_to_one(self):
        """Test accepts dropout_range (0.0, 1.0)."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(dropout_range=(0.0, 1.0))

        assert space.dropout_range == (0.0, 1.0)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_accepts_dropout_range_zero_only(self):
        """Test accepts dropout_range (0.0, 0.0)."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(dropout_range=(0.0, 0.0))

        assert space.dropout_range == (0.0, 0.0)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_rejects_empty_pooling_types(self):
        """Test rejects empty pooling_types list."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        with pytest.raises(Exception) as exc_info:
            GNNArchitectureSpace(pooling_types=[])

        assert "pooling_types" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_rejects_empty_aggregation_types(self):
        """Test rejects empty aggregation_types list."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        with pytest.raises(Exception) as exc_info:
            GNNArchitectureSpace(aggregation_types=[])

        assert "aggregation_types" in str(exc_info.value).lower()


# =============================================================================
# GNNARCHITECTURESPACE TO_DICT METHOD TESTS
# =============================================================================


class TestGNNArchitectureSpaceToDict:
    """Test GNNArchitectureSpace.to_dict() method."""

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_dict_returns_dict(self):
        """Test to_dict returns a dictionary."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()
        result = space.to_dict()

        assert isinstance(result, dict)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_dict_contains_min_layers(self):
        """Test to_dict contains min_layers."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(min_layers=3)
        result = space.to_dict()

        assert result["min_layers"] == 3

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_dict_contains_max_layers(self):
        """Test to_dict contains max_layers."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(max_layers=10)
        result = space.to_dict()

        assert result["max_layers"] == 10

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_dict_layer_types_as_strings(self):
        """Test to_dict converts layer_types to string values."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(layer_types=[LayerType.GCN, LayerType.GAT])
        result = space.to_dict()

        assert result["layer_types"] == ["gcn", "gat"]
        assert all(isinstance(lt, str) for lt in result["layer_types"])

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_dict_contains_hidden_channels(self):
        """Test to_dict contains hidden_channels."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(hidden_channels=[128, 256])
        result = space.to_dict()

        assert result["hidden_channels"] == [128, 256]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_dict_contains_heads(self):
        """Test to_dict contains heads."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(heads=[2, 4, 8])
        result = space.to_dict()

        assert result["heads"] == [2, 4, 8]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_dict_contains_dropout_range(self):
        """Test to_dict contains dropout_range as tuple."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(dropout_range=(0.1, 0.5))
        result = space.to_dict()

        assert result["dropout_range"] == (0.1, 0.5)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_dict_contains_boolean_options(self):
        """Test to_dict contains boolean options."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(
            allow_skip_connections=True,
            allow_dense_connections=False,
            allow_mixed_layers=True,
        )
        result = space.to_dict()

        assert result["allow_skip_connections"] is True
        assert result["allow_dense_connections"] is False
        assert result["allow_mixed_layers"] is True

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_dict_pooling_types_as_strings(self):
        """Test to_dict converts pooling_types to string values."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, PoolingType

        space = GNNArchitectureSpace(pooling_types=[PoolingType.MEAN, PoolingType.MAX])
        result = space.to_dict()

        assert result["pooling_types"] == ["mean", "max"]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_dict_aggregation_types_as_strings(self):
        """Test to_dict converts aggregation_types to string values."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType, GNNArchitectureSpace

        space = GNNArchitectureSpace(aggregation_types=[AggregationType.SUM])
        result = space.to_dict()

        assert result["aggregation_types"] == ["sum"]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_dict_activation_types_as_strings(self):
        """Test to_dict converts activation_types to string values."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType, GNNArchitectureSpace

        space = GNNArchitectureSpace(activation_types=[ActivationType.RELU, ActivationType.GELU])
        result = space.to_dict()

        assert result["activation_types"] == ["relu", "gelu"]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_dict_contains_batch_norm_options(self):
        """Test to_dict contains batch_norm_options."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(batch_norm_options=[True])
        result = space.to_dict()

        assert result["batch_norm_options"] == [True]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_dict_has_all_expected_keys(self):
        """Test to_dict has all 13 expected keys."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()
        result = space.to_dict()

        expected_keys = {
            "min_layers",
            "max_layers",
            "layer_types",
            "hidden_channels",
            "heads",
            "dropout_range",
            "allow_skip_connections",
            "allow_dense_connections",
            "allow_mixed_layers",
            "pooling_types",
            "aggregation_types",
            "activation_types",
            "batch_norm_options",
        }
        assert set(result.keys()) == expected_keys


# =============================================================================
# GNNARCHITECTURESPACE FROM_DICT CLASS METHOD TESTS
# =============================================================================


class TestGNNArchitectureSpaceFromDict:
    """Test GNNArchitectureSpace.from_dict() class method."""

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_from_dict_returns_gnn_architecture_space(self):
        """Test from_dict returns GNNArchitectureSpace instance."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        config_dict = {
            "min_layers": 2,
            "max_layers": 6,
            "layer_types": ["gcn", "gat"],
            "hidden_channels": [64, 128],
            "heads": [1, 2],
            "dropout_range": [0.0, 0.5],
            "pooling_types": ["mean"],
            "aggregation_types": ["mean"],
        }

        result = GNNArchitectureSpace.from_dict(config_dict)

        assert isinstance(result, GNNArchitectureSpace)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_from_dict_converts_layer_types_strings(self):
        """Test from_dict converts layer_types strings to enums."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        config_dict = {
            "layer_types": ["gcn", "gat", "sage"],
        }

        result = GNNArchitectureSpace.from_dict(config_dict)

        assert result.layer_types == [LayerType.GCN, LayerType.GAT, LayerType.SAGE]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_from_dict_preserves_layer_type_enums(self):
        """Test from_dict preserves LayerType enums."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        config_dict = {
            "layer_types": [LayerType.GIN, LayerType.PNA],
        }

        result = GNNArchitectureSpace.from_dict(config_dict)

        assert result.layer_types == [LayerType.GIN, LayerType.PNA]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_from_dict_converts_pooling_types_strings(self):
        """Test from_dict converts pooling_types strings to enums."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, PoolingType

        config_dict = {
            "pooling_types": ["mean", "max", "attention"],
        }

        result = GNNArchitectureSpace.from_dict(config_dict)

        assert result.pooling_types == [PoolingType.MEAN, PoolingType.MAX, PoolingType.ATTENTION]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_from_dict_converts_aggregation_types_strings(self):
        """Test from_dict converts aggregation_types strings to enums."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType, GNNArchitectureSpace

        config_dict = {
            "aggregation_types": ["sum", "multi"],
        }

        result = GNNArchitectureSpace.from_dict(config_dict)

        assert result.aggregation_types == [AggregationType.SUM, AggregationType.MULTI]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_from_dict_converts_activation_types_strings(self):
        """Test from_dict converts activation_types strings to enums."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType, GNNArchitectureSpace

        config_dict = {
            "activation_types": ["relu", "silu", "tanh"],
        }

        result = GNNArchitectureSpace.from_dict(config_dict)

        assert result.activation_types == [
            ActivationType.RELU,
            ActivationType.SILU,
            ActivationType.TANH,
        ]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_from_dict_converts_list_dropout_range_to_tuple(self):
        """Test from_dict converts list dropout_range to tuple."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        config_dict = {
            "dropout_range": [0.1, 0.4],
        }

        result = GNNArchitectureSpace.from_dict(config_dict)

        assert result.dropout_range == (0.1, 0.4)
        assert isinstance(result.dropout_range, tuple)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_from_dict_preserves_tuple_dropout_range(self):
        """Test from_dict preserves tuple dropout_range."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        config_dict = {
            "dropout_range": (0.0, 0.3),
        }

        result = GNNArchitectureSpace.from_dict(config_dict)

        assert result.dropout_range == (0.0, 0.3)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_from_dict_sets_integer_fields(self):
        """Test from_dict sets integer fields correctly."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        config_dict = {
            "min_layers": 3,
            "max_layers": 7,
            "hidden_channels": [128, 256, 512],
            "heads": [2, 4, 8, 16],
        }

        result = GNNArchitectureSpace.from_dict(config_dict)

        assert result.min_layers == 3
        assert result.max_layers == 7
        assert result.hidden_channels == [128, 256, 512]
        assert result.heads == [2, 4, 8, 16]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_from_dict_sets_boolean_fields(self):
        """Test from_dict sets boolean fields correctly."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        config_dict = {
            "allow_skip_connections": False,
            "allow_dense_connections": True,
            "allow_mixed_layers": False,
            "batch_norm_options": [False],
        }

        result = GNNArchitectureSpace.from_dict(config_dict)

        assert result.allow_skip_connections is False
        assert result.allow_dense_connections is True
        assert result.allow_mixed_layers is False
        assert result.batch_norm_options == [False]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_from_dict_to_dict_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        from milia_pipeline.models.hpo.nas.search_space import (
            ActivationType,
            AggregationType,
            GNNArchitectureSpace,
            LayerType,
            PoolingType,
        )

        original = GNNArchitectureSpace(
            min_layers=2,
            max_layers=6,
            layer_types=[LayerType.GAT, LayerType.TRANSFORMER],
            hidden_channels=[64, 128, 256],
            heads=[2, 4, 8],
            dropout_range=(0.1, 0.4),
            allow_skip_connections=True,
            allow_dense_connections=True,
            allow_mixed_layers=False,
            pooling_types=[PoolingType.ATTENTION],
            aggregation_types=[AggregationType.MULTI],
            activation_types=[ActivationType.GELU, ActivationType.SILU],
            batch_norm_options=[True],
        )

        dict_repr = original.to_dict()
        restored = GNNArchitectureSpace.from_dict(dict_repr)

        assert restored.min_layers == original.min_layers
        assert restored.max_layers == original.max_layers
        assert restored.layer_types == original.layer_types
        assert restored.hidden_channels == original.hidden_channels
        assert restored.heads == original.heads
        assert restored.dropout_range == original.dropout_range
        assert restored.allow_skip_connections == original.allow_skip_connections
        assert restored.allow_dense_connections == original.allow_dense_connections
        assert restored.allow_mixed_layers == original.allow_mixed_layers
        assert restored.pooling_types == original.pooling_types
        assert restored.aggregation_types == original.aggregation_types
        assert restored.activation_types == original.activation_types
        assert restored.batch_norm_options == original.batch_norm_options


# =============================================================================
# GNNARCHITECTURESPACE GET_ATTENTION_LAYER_TYPES METHOD TESTS
# =============================================================================


class TestGNNArchitectureSpaceGetAttentionLayerTypes:
    """Test GNNArchitectureSpace.get_attention_layer_types() method."""

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_get_attention_layer_types_returns_list(self):
        """Test get_attention_layer_types returns a list."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()
        result = space.get_attention_layer_types()

        assert isinstance(result, list)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_get_attention_layer_types_with_gat(self):
        """Test get_attention_layer_types returns GAT when present."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(layer_types=[LayerType.GCN, LayerType.GAT])
        result = space.get_attention_layer_types()

        assert LayerType.GAT in result
        assert LayerType.GCN not in result

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_get_attention_layer_types_with_gatv2(self):
        """Test get_attention_layer_types returns GATV2 when present."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(layer_types=[LayerType.SAGE, LayerType.GATV2])
        result = space.get_attention_layer_types()

        assert LayerType.GATV2 in result
        assert LayerType.SAGE not in result

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_get_attention_layer_types_with_transformer(self):
        """Test get_attention_layer_types returns TRANSFORMER when present."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(layer_types=[LayerType.GIN, LayerType.TRANSFORMER])
        result = space.get_attention_layer_types()

        assert LayerType.TRANSFORMER in result
        assert LayerType.GIN not in result

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_get_attention_layer_types_multiple_attention_layers(self):
        """Test get_attention_layer_types with multiple attention layers."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(
            layer_types=[LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER]
        )
        result = space.get_attention_layer_types()

        assert LayerType.GAT in result
        assert LayerType.GATV2 in result
        assert LayerType.TRANSFORMER in result
        assert len(result) == 3

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_get_attention_layer_types_no_attention_layers(self):
        """Test get_attention_layer_types returns empty list when no attention layers."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(layer_types=[LayerType.GCN, LayerType.SAGE, LayerType.GIN])
        result = space.get_attention_layer_types()

        assert result == []

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_get_attention_layer_types_preserves_order(self):
        """Test get_attention_layer_types preserves layer order."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(
            layer_types=[LayerType.TRANSFORMER, LayerType.GCN, LayerType.GAT, LayerType.GATV2]
        )
        result = space.get_attention_layer_types()

        # Should be in same order as they appear in layer_types
        assert result == [LayerType.TRANSFORMER, LayerType.GAT, LayerType.GATV2]


# =============================================================================
# GNNARCHITECTURESPACE HAS_ATTENTION_LAYERS METHOD TESTS
# =============================================================================


class TestGNNArchitectureSpaceHasAttentionLayers:
    """Test GNNArchitectureSpace.has_attention_layers() method."""

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_has_attention_layers_returns_bool(self):
        """Test has_attention_layers returns a boolean."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()
        result = space.has_attention_layers()

        assert isinstance(result, bool)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_has_attention_layers_true_with_gat(self):
        """Test has_attention_layers returns True with GAT."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(layer_types=[LayerType.GCN, LayerType.GAT])
        result = space.has_attention_layers()

        assert result is True

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_has_attention_layers_true_with_gatv2(self):
        """Test has_attention_layers returns True with GATV2."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(layer_types=[LayerType.GATV2])
        result = space.has_attention_layers()

        assert result is True

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_has_attention_layers_true_with_transformer(self):
        """Test has_attention_layers returns True with TRANSFORMER."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(layer_types=[LayerType.TRANSFORMER])
        result = space.has_attention_layers()

        assert result is True

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_has_attention_layers_false_with_no_attention(self):
        """Test has_attention_layers returns False with no attention layers."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(
            layer_types=[LayerType.GCN, LayerType.SAGE, LayerType.GIN, LayerType.PNA]
        )
        result = space.has_attention_layers()

        assert result is False

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_has_attention_layers_default_space(self):
        """Test has_attention_layers on default space (includes GAT)."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()
        result = space.has_attention_layers()

        # Default includes GAT, so should be True
        assert result is True


# =============================================================================
# GNNARCHITECTURESPACE MODULE EXPORT TESTS
# =============================================================================


class TestGNNArchitectureSpaceModuleExport:
    """Test GNNArchitectureSpace module export."""

    def test_gnn_architecture_space_in_all_exports(self):
        """Test GNNArchitectureSpace is in module __all__."""
        from milia_pipeline.models.hpo.nas import search_space

        assert "GNNArchitectureSpace" in search_space.__all__

    def test_gnn_architecture_space_importable_directly(self):
        """Test GNNArchitectureSpace can be imported directly."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        assert GNNArchitectureSpace is not None


# =============================================================================
# PART 4: ADVANCED METHODS, FACTORY FUNCTIONS, AND INTEGRATION TESTS
# =============================================================================

# =============================================================================
# ADVANCED TESTS FIXTURES
# =============================================================================


@pytest.fixture
def attention_only_space():
    """Create space with only attention layers."""
    from milia_pipeline.models.hpo.nas.search_space import (
        ActivationType,
        AggregationType,
        GNNArchitectureSpace,
        LayerType,
        PoolingType,
    )

    return GNNArchitectureSpace(
        min_layers=2,
        max_layers=4,
        layer_types=[LayerType.GAT, LayerType.GATV2, LayerType.TRANSFORMER],
        hidden_channels=[64, 128],
        heads=[2, 4, 8],
        pooling_types=[PoolingType.ATTENTION],
        aggregation_types=[AggregationType.MEAN],
        activation_types=[ActivationType.GELU],
    )


@pytest.fixture
def non_attention_space():
    """Create space with no attention layers."""
    from milia_pipeline.models.hpo.nas.search_space import (
        ActivationType,
        AggregationType,
        GNNArchitectureSpace,
        LayerType,
        PoolingType,
    )

    return GNNArchitectureSpace(
        min_layers=2,
        max_layers=4,
        layer_types=[LayerType.GCN, LayerType.SAGE, LayerType.GIN],
        hidden_channels=[64, 128],
        heads=[1],
        pooling_types=[PoolingType.MEAN],
        aggregation_types=[AggregationType.SUM],
        activation_types=[ActivationType.RELU],
    )


# =============================================================================
# GNNARCHITECTURESPACE TO_OPTUNA_SEARCH_SPACE METHOD TESTS
# =============================================================================


class TestGNNArchitectureSpaceToOptunaSearchSpace:
    """Test GNNArchitectureSpace.to_optuna_search_space() method."""

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_returns_dict(self):
        """Test to_optuna_search_space returns a dictionary."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()
        result = space.to_optuna_search_space()

        assert isinstance(result, dict)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_has_architecture_key(self):
        """Test to_optuna_search_space has 'architecture' key."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()
        result = space.to_optuna_search_space()

        assert "architecture" in result

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_num_layers(self):
        """Test to_optuna_search_space contains num_layers config."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(min_layers=2, max_layers=6)
        result = space.to_optuna_search_space()

        assert "num_layers" in result["architecture"]
        assert result["architecture"]["num_layers"]["type"] == "int"
        assert result["architecture"]["num_layers"]["low"] == 2
        assert result["architecture"]["num_layers"]["high"] == 6

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_hidden_channels(self):
        """Test to_optuna_search_space contains hidden_channels config."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(hidden_channels=[64, 128, 256])
        result = space.to_optuna_search_space()

        assert "hidden_channels" in result["architecture"]
        assert result["architecture"]["hidden_channels"]["type"] == "categorical"
        assert result["architecture"]["hidden_channels"]["choices"] == [64, 128, 256]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_pooling(self):
        """Test to_optuna_search_space contains pooling config."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, PoolingType

        space = GNNArchitectureSpace(pooling_types=[PoolingType.MEAN, PoolingType.MAX])
        result = space.to_optuna_search_space()

        assert "pooling" in result["architecture"]
        assert result["architecture"]["pooling"]["type"] == "categorical"
        assert result["architecture"]["pooling"]["choices"] == ["mean", "max"]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_dropout(self):
        """Test to_optuna_search_space contains dropout config."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(dropout_range=(0.1, 0.5))
        result = space.to_optuna_search_space()

        assert "dropout" in result["architecture"]
        assert result["architecture"]["dropout"]["type"] == "float"
        assert result["architecture"]["dropout"]["low"] == 0.1
        assert result["architecture"]["dropout"]["high"] == 0.5

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_aggregation(self):
        """Test to_optuna_search_space contains aggregation config."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType, GNNArchitectureSpace

        space = GNNArchitectureSpace(aggregation_types=[AggregationType.MEAN, AggregationType.SUM])
        result = space.to_optuna_search_space()

        assert "aggregation" in result["architecture"]
        assert result["architecture"]["aggregation"]["type"] == "categorical"
        assert result["architecture"]["aggregation"]["choices"] == ["mean", "sum"]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_activation(self):
        """Test to_optuna_search_space contains activation config."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType, GNNArchitectureSpace

        space = GNNArchitectureSpace(activation_types=[ActivationType.RELU, ActivationType.GELU])
        result = space.to_optuna_search_space()

        assert "activation" in result["architecture"]
        assert result["architecture"]["activation"]["type"] == "categorical"
        assert result["architecture"]["activation"]["choices"] == ["relu", "gelu"]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_batch_norm(self):
        """Test to_optuna_search_space contains batch_norm config."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(batch_norm_options=[True, False])
        result = space.to_optuna_search_space()

        assert "batch_norm" in result["architecture"]
        assert result["architecture"]["batch_norm"]["type"] == "categorical"
        assert result["architecture"]["batch_norm"]["choices"] == [True, False]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_skip_connections_when_allowed(self):
        """Test to_optuna_search_space includes skip connections when allowed."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(allow_skip_connections=True)
        result = space.to_optuna_search_space()

        assert "use_skip_connections" in result["architecture"]
        assert result["architecture"]["use_skip_connections"]["type"] == "categorical"
        assert result["architecture"]["use_skip_connections"]["choices"] == [True, False]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_skip_connections_when_not_allowed(self):
        """Test to_optuna_search_space excludes skip connections when not allowed."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(allow_skip_connections=False)
        result = space.to_optuna_search_space()

        assert "use_skip_connections" not in result["architecture"]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_dense_connections_when_allowed(self):
        """Test to_optuna_search_space includes dense connections when allowed."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(allow_dense_connections=True)
        result = space.to_optuna_search_space()

        assert "use_dense_connections" in result["architecture"]
        assert result["architecture"]["use_dense_connections"]["type"] == "categorical"
        assert result["architecture"]["use_dense_connections"]["choices"] == [True, False]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_dense_connections_when_not_allowed(self):
        """Test to_optuna_search_space excludes dense connections when not allowed."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(allow_dense_connections=False)
        result = space.to_optuna_search_space()

        assert "use_dense_connections" not in result["architecture"]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_mixed_layers_per_layer_type(self):
        """Test to_optuna_search_space creates per-layer type when mixed layers allowed."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(
            max_layers=3,
            layer_types=[LayerType.GCN, LayerType.GAT],
            allow_mixed_layers=True,
        )
        result = space.to_optuna_search_space()

        # Should have layer_0_type, layer_1_type, layer_2_type
        assert "layer_0_type" in result["architecture"]
        assert "layer_1_type" in result["architecture"]
        assert "layer_2_type" in result["architecture"]
        assert result["architecture"]["layer_0_type"]["choices"] == ["gcn", "gat"]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_mixed_layers_per_layer_heads(self):
        """Test to_optuna_search_space creates per-layer heads when mixed with attention."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(
            max_layers=2,
            layer_types=[LayerType.GCN, LayerType.GAT],
            heads=[2, 4],
            allow_mixed_layers=True,
        )
        result = space.to_optuna_search_space()

        # Should have layer_0_heads, layer_1_heads (because GAT is present)
        assert "layer_0_heads" in result["architecture"]
        assert "layer_1_heads" in result["architecture"]
        assert result["architecture"]["layer_0_heads"]["choices"] == [2, 4]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_no_mixed_layers_single_type(self):
        """Test to_optuna_search_space creates single layer_type when mixed not allowed."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(
            max_layers=4,
            layer_types=[LayerType.GCN, LayerType.GAT],
            allow_mixed_layers=False,
        )
        result = space.to_optuna_search_space()

        # Should have single layer_type, not per-layer
        assert "layer_type" in result["architecture"]
        assert "layer_0_type" not in result["architecture"]
        assert result["architecture"]["layer_type"]["choices"] == ["gcn", "gat"]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_no_mixed_layers_single_heads(self):
        """Test to_optuna_search_space creates single heads when mixed not allowed."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(
            max_layers=4,
            layer_types=[LayerType.GAT],
            heads=[2, 4, 8],
            allow_mixed_layers=False,
        )
        result = space.to_optuna_search_space()

        # Should have single heads, not per-layer
        assert "heads" in result["architecture"]
        assert "layer_0_heads" not in result["architecture"]
        assert result["architecture"]["heads"]["choices"] == [2, 4, 8]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_to_optuna_search_space_no_heads_without_attention_layers(self):
        """Test to_optuna_search_space excludes heads when no attention layers."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(
            layer_types=[LayerType.GCN, LayerType.SAGE, LayerType.GIN],
            allow_mixed_layers=False,
        )
        result = space.to_optuna_search_space()

        assert "heads" not in result["architecture"]


# =============================================================================
# GNNARCHITECTURESPACE GET_SEARCH_DIMENSIONS METHOD TESTS
# =============================================================================


class TestGNNArchitectureSpaceGetSearchDimensions:
    """Test GNNArchitectureSpace.get_search_dimensions() method."""

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_get_search_dimensions_returns_int(self):
        """Test get_search_dimensions returns an integer."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()
        result = space.get_search_dimensions()

        assert isinstance(result, int)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_get_search_dimensions_positive(self):
        """Test get_search_dimensions returns positive value."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()
        result = space.get_search_dimensions()

        assert result > 0

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_get_search_dimensions_base_count(self):
        """Test get_search_dimensions has at least base dimensions (7)."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        # Minimal space without skip/dense connections
        space = GNNArchitectureSpace(
            layer_types=[LayerType.GCN],
            allow_skip_connections=False,
            allow_dense_connections=False,
            allow_mixed_layers=False,
        )
        result = space.get_search_dimensions()

        # Base is 7 dimensions
        assert result >= 7

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_get_search_dimensions_increases_with_skip_connections(self):
        """Test get_search_dimensions increases with skip connections."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space_without = GNNArchitectureSpace(
            layer_types=[LayerType.GCN],
            allow_skip_connections=False,
            allow_dense_connections=False,
            allow_mixed_layers=False,
        )
        space_with = GNNArchitectureSpace(
            layer_types=[LayerType.GCN],
            allow_skip_connections=True,
            allow_dense_connections=False,
            allow_mixed_layers=False,
        )

        dims_without = space_without.get_search_dimensions()
        dims_with = space_with.get_search_dimensions()

        assert dims_with > dims_without

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_get_search_dimensions_increases_with_dense_connections(self):
        """Test get_search_dimensions increases with dense connections."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space_without = GNNArchitectureSpace(
            layer_types=[LayerType.GCN],
            allow_skip_connections=False,
            allow_dense_connections=False,
            allow_mixed_layers=False,
        )
        space_with = GNNArchitectureSpace(
            layer_types=[LayerType.GCN],
            allow_skip_connections=False,
            allow_dense_connections=True,
            allow_mixed_layers=False,
        )

        dims_without = space_without.get_search_dimensions()
        dims_with = space_with.get_search_dimensions()

        assert dims_with > dims_without

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_get_search_dimensions_increases_with_mixed_layers(self):
        """Test get_search_dimensions increases significantly with mixed layers."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space_without = GNNArchitectureSpace(
            max_layers=4,
            layer_types=[LayerType.GCN, LayerType.GAT],
            allow_skip_connections=False,
            allow_dense_connections=False,
            allow_mixed_layers=False,
        )
        space_with = GNNArchitectureSpace(
            max_layers=4,
            layer_types=[LayerType.GCN, LayerType.GAT],
            allow_skip_connections=False,
            allow_dense_connections=False,
            allow_mixed_layers=True,
        )

        dims_without = space_without.get_search_dimensions()
        dims_with = space_with.get_search_dimensions()

        # Mixed layers adds per-layer dimensions
        assert dims_with > dims_without


# =============================================================================
# GNNARCHITECTURESPACE ESTIMATE_SEARCH_SPACE_SIZE METHOD TESTS
# =============================================================================


class TestGNNArchitectureSpaceEstimateSearchSpaceSize:
    """Test GNNArchitectureSpace.estimate_search_space_size() method."""

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_estimate_search_space_size_returns_int(self):
        """Test estimate_search_space_size returns an integer."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()
        result = space.estimate_search_space_size()

        assert isinstance(result, int)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_estimate_search_space_size_positive(self):
        """Test estimate_search_space_size returns positive value."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()
        result = space.estimate_search_space_size()

        assert result > 0

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_estimate_search_space_size_minimal_space(self):
        """Test estimate_search_space_size for minimal space."""
        from milia_pipeline.models.hpo.nas.search_space import (
            ActivationType,
            AggregationType,
            GNNArchitectureSpace,
            LayerType,
            PoolingType,
        )

        space = GNNArchitectureSpace(
            min_layers=1,
            max_layers=1,
            layer_types=[LayerType.GCN],
            hidden_channels=[64],
            heads=[1],
            pooling_types=[PoolingType.MEAN],
            aggregation_types=[AggregationType.MEAN],
            activation_types=[ActivationType.RELU],
            batch_norm_options=[True],
            allow_skip_connections=False,
            allow_dense_connections=False,
            allow_mixed_layers=False,
        )
        result = space.estimate_search_space_size()

        # Minimal space should have small size
        assert result > 0
        assert result < 1000

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_estimate_search_space_size_increases_with_layers(self):
        """Test estimate_search_space_size increases with more layer options."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space_small = GNNArchitectureSpace(min_layers=2, max_layers=3)
        space_large = GNNArchitectureSpace(min_layers=2, max_layers=8)

        size_small = space_small.estimate_search_space_size()
        size_large = space_large.estimate_search_space_size()

        assert size_large > size_small

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_estimate_search_space_size_increases_with_hidden_channels(self):
        """Test estimate_search_space_size increases with more hidden channel options."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space_small = GNNArchitectureSpace(hidden_channels=[64, 128])
        space_large = GNNArchitectureSpace(hidden_channels=[32, 64, 128, 256, 512])

        size_small = space_small.estimate_search_space_size()
        size_large = space_large.estimate_search_space_size()

        assert size_large > size_small

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_estimate_search_space_size_doubles_with_skip_connections(self):
        """Test estimate_search_space_size approximately doubles with skip connections."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space_without = GNNArchitectureSpace(
            allow_skip_connections=False,
            allow_dense_connections=False,
        )
        space_with = GNNArchitectureSpace(
            allow_skip_connections=True,
            allow_dense_connections=False,
        )

        size_without = space_without.estimate_search_space_size()
        size_with = space_with.estimate_search_space_size()

        # Should roughly double
        assert size_with == size_without * 2

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_estimate_search_space_size_doubles_with_dense_connections(self):
        """Test estimate_search_space_size approximately doubles with dense connections."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space_without = GNNArchitectureSpace(
            allow_skip_connections=False,
            allow_dense_connections=False,
        )
        space_with = GNNArchitectureSpace(
            allow_skip_connections=False,
            allow_dense_connections=True,
        )

        size_without = space_without.estimate_search_space_size()
        size_with = space_with.estimate_search_space_size()

        # Should roughly double
        assert size_with == size_without * 2

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_estimate_search_space_size_explodes_with_mixed_layers(self):
        """Test estimate_search_space_size grows exponentially with mixed layers."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space_without = GNNArchitectureSpace(
            max_layers=4,
            layer_types=[LayerType.GCN, LayerType.GAT],
            allow_mixed_layers=False,
        )
        space_with = GNNArchitectureSpace(
            max_layers=4,
            layer_types=[LayerType.GCN, LayerType.GAT],
            allow_mixed_layers=True,
        )

        size_without = space_without.estimate_search_space_size()
        size_with = space_with.estimate_search_space_size()

        # Mixed layers causes exponential growth
        assert size_with > size_without * 10


# =============================================================================
# GNNARCHITECTURESPACE CREATE_DEFAULT_LAYER_CONFIG METHOD TESTS
# =============================================================================


class TestGNNArchitectureSpaceCreateDefaultLayerConfig:
    """Test GNNArchitectureSpace.create_default_layer_config() method."""

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_create_default_layer_config_returns_layer_config(self):
        """Test create_default_layer_config returns LayerConfig instance."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerConfig

        space = GNNArchitectureSpace()
        result = space.create_default_layer_config()

        assert isinstance(result, LayerConfig)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_create_default_layer_config_uses_first_layer_type(self):
        """Test create_default_layer_config uses first layer type."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(layer_types=[LayerType.SAGE, LayerType.GCN])
        result = space.create_default_layer_config()

        assert result.type == LayerType.SAGE

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_create_default_layer_config_uses_first_hidden_channels(self):
        """Test create_default_layer_config uses first hidden channels."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(hidden_channels=[128, 256, 512])
        result = space.create_default_layer_config()

        assert result.hidden_channels == 128

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_create_default_layer_config_uses_first_dropout(self):
        """Test create_default_layer_config uses first dropout value (min)."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(dropout_range=(0.2, 0.6))
        result = space.create_default_layer_config()

        assert result.dropout == 0.2

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_create_default_layer_config_uses_first_activation(self):
        """Test create_default_layer_config uses first activation type."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType, GNNArchitectureSpace

        space = GNNArchitectureSpace(activation_types=[ActivationType.GELU, ActivationType.RELU])
        result = space.create_default_layer_config()

        assert result.activation == "gelu"

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_create_default_layer_config_uses_first_batch_norm(self):
        """Test create_default_layer_config uses first batch_norm option."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(batch_norm_options=[False, True])
        result = space.create_default_layer_config()

        assert result.batch_norm is False

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_create_default_layer_config_respects_skip_connections(self):
        """Test create_default_layer_config sets residual based on skip connections."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space_with = GNNArchitectureSpace(allow_skip_connections=True)
        space_without = GNNArchitectureSpace(allow_skip_connections=False)

        result_with = space_with.create_default_layer_config()
        result_without = space_without.create_default_layer_config()

        assert result_with.residual is True
        assert result_without.residual is False

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_create_default_layer_config_override_layer_type(self):
        """Test create_default_layer_config allows layer_type override."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(layer_types=[LayerType.GCN, LayerType.GAT])
        result = space.create_default_layer_config(layer_type=LayerType.GAT)

        assert result.type == LayerType.GAT

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_create_default_layer_config_override_hidden_channels(self):
        """Test create_default_layer_config allows hidden_channels override."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(hidden_channels=[64, 128])
        result = space.create_default_layer_config(hidden_channels=256)

        assert result.hidden_channels == 256

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_create_default_layer_config_attention_layer_gets_heads(self):
        """Test create_default_layer_config sets heads for attention layers."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(
            layer_types=[LayerType.GAT],
            heads=[2, 4, 8],
        )
        result = space.create_default_layer_config()

        assert result.heads == 2  # First in heads list

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_create_default_layer_config_non_attention_layer_gets_heads_1(self):
        """Test create_default_layer_config sets heads=1 for non-attention layers."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        space = GNNArchitectureSpace(
            layer_types=[LayerType.GCN, LayerType.GAT],
            heads=[4, 8],
        )
        result = space.create_default_layer_config(layer_type=LayerType.GCN)

        assert result.heads == 1


# =============================================================================
# CREATE_GNN_SEARCH_SPACE FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateGNNSearchSpaceFactory:
    """Test create_gnn_search_space() factory function."""

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_create_gnn_search_space_returns_gnn_architecture_space(self):
        """Test create_gnn_search_space returns GNNArchitectureSpace."""
        from milia_pipeline.models.hpo.nas.search_space import (
            GNNArchitectureSpace,
            create_gnn_search_space,
        )

        result = create_gnn_search_space()

        assert isinstance(result, GNNArchitectureSpace)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_create_gnn_search_space_no_args_default(self):
        """Test create_gnn_search_space with no args returns default space."""
        from milia_pipeline.models.hpo.nas.search_space import create_gnn_search_space

        result = create_gnn_search_space()

        assert result.min_layers == 2
        assert result.max_layers == 8

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_create_gnn_search_space_gcn_preset(self):
        """Test create_gnn_search_space with 'gcn' model_type."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType, create_gnn_search_space

        result = create_gnn_search_space(model_type="gcn")

        assert result.layer_types == [LayerType.GCN]
        assert result.allow_mixed_layers is False

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_create_gnn_search_space_gat_preset(self):
        """Test create_gnn_search_space with 'gat' model_type."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType, create_gnn_search_space

        result = create_gnn_search_space(model_type="gat")

        assert LayerType.GAT in result.layer_types
        assert LayerType.GATV2 in result.layer_types
        assert result.heads == [1, 2, 4, 8]
        assert result.allow_mixed_layers is False

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_create_gnn_search_space_sage_preset(self):
        """Test create_gnn_search_space with 'sage' model_type."""
        from milia_pipeline.models.hpo.nas.search_space import (
            AggregationType,
            LayerType,
            create_gnn_search_space,
        )

        result = create_gnn_search_space(model_type="sage")

        assert result.layer_types == [LayerType.SAGE]
        assert AggregationType.MEAN in result.aggregation_types
        assert AggregationType.MAX in result.aggregation_types
        assert AggregationType.LSTM in result.aggregation_types

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_create_gnn_search_space_gin_preset(self):
        """Test create_gnn_search_space with 'gin' model_type."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType, create_gnn_search_space

        result = create_gnn_search_space(model_type="gin")

        assert result.layer_types == [LayerType.GIN]
        assert result.allow_mixed_layers is False

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_create_gnn_search_space_transformer_preset(self):
        """Test create_gnn_search_space with 'transformer' model_type."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType, create_gnn_search_space

        result = create_gnn_search_space(model_type="transformer")

        assert result.layer_types == [LayerType.TRANSFORMER]
        assert result.heads == [2, 4, 8, 16]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_create_gnn_search_space_pna_preset(self):
        """Test create_gnn_search_space with 'pna' model_type."""
        from milia_pipeline.models.hpo.nas.search_space import (
            AggregationType,
            LayerType,
            create_gnn_search_space,
        )

        result = create_gnn_search_space(model_type="pna")

        assert result.layer_types == [LayerType.PNA]
        assert result.aggregation_types == [AggregationType.MULTI]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_create_gnn_search_space_mixed_preset(self):
        """Test create_gnn_search_space with 'mixed' model_type."""
        from milia_pipeline.models.hpo.nas.search_space import LayerType, create_gnn_search_space

        result = create_gnn_search_space(model_type="mixed")

        assert LayerType.GCN in result.layer_types
        assert LayerType.GAT in result.layer_types
        assert LayerType.SAGE in result.layer_types
        assert LayerType.GIN in result.layer_types
        assert result.allow_mixed_layers is True

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_create_gnn_search_space_invalid_model_type(self):
        """Test create_gnn_search_space raises error for invalid model_type."""
        from milia_pipeline.models.hpo.nas.search_space import create_gnn_search_space

        with pytest.raises(ValueError) as exc_info:
            create_gnn_search_space(model_type="invalid_model")

        assert "invalid_model" in str(exc_info.value).lower()

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_create_gnn_search_space_case_insensitive(self):
        """Test create_gnn_search_space is case-insensitive."""
        from milia_pipeline.models.hpo.nas.search_space import create_gnn_search_space

        result_lower = create_gnn_search_space(model_type="gcn")
        result_upper = create_gnn_search_space(model_type="GCN")
        result_mixed = create_gnn_search_space(model_type="GcN")

        assert result_lower.layer_types == result_upper.layer_types == result_mixed.layer_types

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_create_gnn_search_space_custom_kwargs(self):
        """Test create_gnn_search_space accepts custom kwargs."""
        from milia_pipeline.models.hpo.nas.search_space import create_gnn_search_space

        result = create_gnn_search_space(
            min_layers=3,
            max_layers=6,
            hidden_channels=[128, 256],
        )

        assert result.min_layers == 3
        assert result.max_layers == 6
        assert result.hidden_channels == [128, 256]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_create_gnn_search_space_preset_with_overrides(self):
        """Test create_gnn_search_space preset with kwargs overrides."""
        from milia_pipeline.models.hpo.nas.search_space import create_gnn_search_space

        result = create_gnn_search_space(
            model_type="gcn",
            min_layers=4,
            max_layers=10,
        )

        # Should have GCN preset but with overridden layers
        assert result.min_layers == 4
        assert result.max_layers == 10


# =============================================================================
# GET_DEFAULT_GNN_SEARCH_SPACE FACTORY FUNCTION TESTS
# =============================================================================


class TestGetDefaultGNNSearchSpaceFactory:
    """Test get_default_gnn_search_space() factory function."""

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_get_default_gnn_search_space_returns_gnn_architecture_space(self):
        """Test get_default_gnn_search_space returns GNNArchitectureSpace."""
        from milia_pipeline.models.hpo.nas.search_space import (
            GNNArchitectureSpace,
            get_default_gnn_search_space,
        )

        result = get_default_gnn_search_space()

        assert isinstance(result, GNNArchitectureSpace)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_get_default_gnn_search_space_default_values(self):
        """Test get_default_gnn_search_space has expected default values."""
        from milia_pipeline.models.hpo.nas.search_space import get_default_gnn_search_space

        result = get_default_gnn_search_space()

        assert result.min_layers == 2
        assert result.max_layers == 8
        assert result.hidden_channels == [32, 64, 128, 256]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_get_default_gnn_search_space_returns_new_instance(self):
        """Test get_default_gnn_search_space returns new instance each time."""
        from milia_pipeline.models.hpo.nas.search_space import get_default_gnn_search_space

        result1 = get_default_gnn_search_space()
        result2 = get_default_gnn_search_space()

        # Should be equal but not same object
        assert result1.min_layers == result2.min_layers
        assert result1 is not result2


# =============================================================================
# MODULE EXPORTS TESTS
# =============================================================================


class TestModuleExports:
    """Test module-level exports."""

    def test_all_exports_contains_all_classes(self):
        """Test __all__ contains all expected exports."""
        from milia_pipeline.models.hpo.nas import search_space

        expected_exports = [
            "LayerType",
            "PoolingType",
            "AggregationType",
            "ActivationType",
            "LayerConfig",
            "GNNArchitectureSpace",
            "create_gnn_search_space",
            "get_default_gnn_search_space",
        ]

        for export in expected_exports:
            assert export in search_space.__all__

    def test_module_version(self):
        """Test module has version attribute with valid semantic version format."""
        import re

        from milia_pipeline.models.hpo.nas import search_space

        assert hasattr(search_space, "__version__")
        # Check for valid semantic version format (e.g., '1.0.0', '1.1.0', '2.0.0-beta')
        version_pattern = r"^\d+\.\d+\.\d+(-[\w.]+)?$"
        assert re.match(version_pattern, search_space.__version__), (
            f"Version '{search_space.__version__}' does not match semantic version format"
        )

    def test_module_author(self):
        """Test module has author attribute."""
        from milia_pipeline.models.hpo.nas import search_space

        assert hasattr(search_space, "__author__")
        assert search_space.__author__ == "Milia Team"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegrationScenarios:
    """Test comprehensive integration scenarios."""

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_full_workflow_create_serialize_deserialize(self):
        """Test complete workflow: create, serialize, deserialize."""
        from milia_pipeline.models.hpo.nas.search_space import (
            ActivationType,
            AggregationType,
            GNNArchitectureSpace,
            LayerType,
            PoolingType,
        )

        # Create custom space
        original = GNNArchitectureSpace(
            min_layers=2,
            max_layers=6,
            layer_types=[LayerType.GAT, LayerType.TRANSFORMER],
            hidden_channels=[64, 128, 256],
            heads=[2, 4, 8],
            dropout_range=(0.1, 0.4),
            allow_skip_connections=True,
            allow_dense_connections=False,
            allow_mixed_layers=True,
            pooling_types=[PoolingType.ATTENTION, PoolingType.MEAN],
            aggregation_types=[AggregationType.SUM],
            activation_types=[ActivationType.GELU, ActivationType.SILU],
            batch_norm_options=[True],
        )

        # Serialize
        dict_repr = original.to_dict()

        # Deserialize
        restored = GNNArchitectureSpace.from_dict(dict_repr)

        # Verify
        assert restored.min_layers == original.min_layers
        assert restored.max_layers == original.max_layers
        assert restored.layer_types == original.layer_types
        assert restored.has_attention_layers() == original.has_attention_layers()

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_factory_to_optuna_workflow(self):
        """Test factory creation to Optuna conversion workflow."""
        from milia_pipeline.models.hpo.nas.search_space import create_gnn_search_space

        # Create using factory
        space = create_gnn_search_space(model_type="gat")

        # Convert to Optuna format
        optuna_space = space.to_optuna_search_space()

        # Verify Optuna space structure
        assert "architecture" in optuna_space
        assert "num_layers" in optuna_space["architecture"]
        assert "hidden_channels" in optuna_space["architecture"]
        assert optuna_space["architecture"]["num_layers"]["type"] == "int"

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_create_default_layer_from_factory_space(self):
        """Test creating default layer config from factory-created space."""
        from milia_pipeline.models.hpo.nas.search_space import LayerConfig, create_gnn_search_space

        # Create space using factory
        space = create_gnn_search_space(model_type="transformer")

        # Create default layer
        layer = space.create_default_layer_config()

        # Verify
        assert isinstance(layer, LayerConfig)
        assert layer.type.value == "transformer"
        assert layer.heads >= 1

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_search_space_metrics(self):
        """Test getting search space metrics."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace()

        # Get metrics
        dims = space.get_search_dimensions()
        size = space.estimate_search_space_size()
        has_attention = space.has_attention_layers()
        attention_types = space.get_attention_layer_types()

        # Verify all metrics are valid
        assert dims > 0
        assert size > 0
        assert isinstance(has_attention, bool)
        assert isinstance(attention_types, list)

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    @patch("milia_pipeline.exceptions.ConfigurationError", MockConfigurationError)
    def test_layer_config_from_space_to_dict_roundtrip(self):
        """Test LayerConfig created from space survives dict roundtrip."""
        from milia_pipeline.models.hpo.nas.search_space import (
            GNNArchitectureSpace,
            LayerConfig,
        )

        space = GNNArchitectureSpace()
        original_layer = space.create_default_layer_config()

        # Roundtrip through dict
        dict_repr = original_layer.to_dict()
        restored_layer = LayerConfig.from_dict(dict_repr)

        assert restored_layer.type == original_layer.type
        assert restored_layer.hidden_channels == original_layer.hidden_channels
        assert restored_layer.heads == original_layer.heads
        assert restored_layer.dropout == original_layer.dropout

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_multiple_presets_comparison(self):
        """Test comparing metrics across different presets."""
        from milia_pipeline.models.hpo.nas.search_space import create_gnn_search_space

        presets = ["gcn", "gat", "sage", "gin", "transformer", "pna", "mixed"]
        spaces = {preset: create_gnn_search_space(model_type=preset) for preset in presets}

        # All should be valid GNNArchitectureSpace instances
        for _preset, space in spaces.items():
            assert space.min_layers >= 1
            assert space.max_layers >= space.min_layers
            assert len(space.layer_types) >= 1
            assert space.estimate_search_space_size() > 0


# =============================================================================
# EDGE CASES TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_single_layer_space(self):
        """Test space with single layer (min==max)."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(min_layers=3, max_layers=3)

        assert space.min_layers == 3
        assert space.max_layers == 3

        optuna_space = space.to_optuna_search_space()
        assert optuna_space["architecture"]["num_layers"]["low"] == 3
        assert optuna_space["architecture"]["num_layers"]["high"] == 3

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_single_option_categorical(self):
        """Test space with single option in categorical fields."""
        from milia_pipeline.models.hpo.nas.search_space import (
            ActivationType,
            AggregationType,
            GNNArchitectureSpace,
            LayerType,
            PoolingType,
        )

        space = GNNArchitectureSpace(
            layer_types=[LayerType.GCN],
            hidden_channels=[64],
            heads=[1],
            pooling_types=[PoolingType.MEAN],
            aggregation_types=[AggregationType.SUM],
            activation_types=[ActivationType.RELU],
            batch_norm_options=[True],
        )

        optuna_space = space.to_optuna_search_space()

        assert optuna_space["architecture"]["hidden_channels"]["choices"] == [64]
        assert optuna_space["architecture"]["pooling"]["choices"] == ["mean"]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_large_max_layers(self):
        """Test space with large max_layers value."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(max_layers=100)

        assert space.max_layers == 100
        assert space.estimate_search_space_size() > 0

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_large_hidden_channels(self):
        """Test space with large hidden_channels values."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(hidden_channels=[1024, 2048, 4096])

        assert space.hidden_channels == [1024, 2048, 4096]

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_zero_dropout_range(self):
        """Test space with zero dropout range."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace

        space = GNNArchitectureSpace(dropout_range=(0.0, 0.0))

        assert space.dropout_range == (0.0, 0.0)

        optuna_space = space.to_optuna_search_space()
        assert optuna_space["architecture"]["dropout"]["low"] == 0.0
        assert optuna_space["architecture"]["dropout"]["high"] == 0.0

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_all_layer_types(self):
        """Test space with all layer types."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType

        all_types = list(LayerType)
        space = GNNArchitectureSpace(layer_types=all_types)

        assert len(space.layer_types) == 7
        assert space.has_attention_layers() is True
        assert len(space.get_attention_layer_types()) == 3

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_all_pooling_types(self):
        """Test space with all pooling types."""
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, PoolingType

        all_types = list(PoolingType)
        space = GNNArchitectureSpace(pooling_types=all_types)

        assert len(space.pooling_types) == 6

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_all_aggregation_types(self):
        """Test space with all aggregation types."""
        from milia_pipeline.models.hpo.nas.search_space import AggregationType, GNNArchitectureSpace

        all_types = list(AggregationType)
        space = GNNArchitectureSpace(aggregation_types=all_types)

        assert len(space.aggregation_types) == 5

    @patch("milia_pipeline.exceptions.SearchSpaceError", MockSearchSpaceError)
    def test_all_activation_types(self):
        """Test space with all activation types."""
        from milia_pipeline.models.hpo.nas.search_space import ActivationType, GNNArchitectureSpace

        all_types = list(ActivationType)
        space = GNNArchitectureSpace(activation_types=all_types)

        assert len(space.activation_types) == 7


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
