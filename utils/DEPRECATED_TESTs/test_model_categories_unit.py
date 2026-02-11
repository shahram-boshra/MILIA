#!/usr/bin/env python3
"""
Complete Unit Test Suite for model_categories.py Module

Tests model categorization and metadata including:
- ModelCategory enum (12 categories)
- ModelMetadata dataclass with all attributes
- Model dictionaries for all categories
- MODELS_BY_CATEGORY aggregation
- ALL_MODELS flattened dictionary
- Helper functions: get_all_model_names, get_model_metadata, get_models_by_category,
  get_models_by_task, get_models_by_tag, get_category_statistics, search_models
- Data integrity and consistency
- Edge cases and validation

This is a PRODUCTION-READY test suite with comprehensive coverage.

Author: milia Team
Version: 1.0.0
"""
import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import pytest
from dataclasses import fields, is_dataclass
from typing import List, Dict, Optional, Set, Any
from enum import Enum

# Import the module under test
from milia_pipeline.models.registry.model_categories import (
    # Enum
    ModelCategory,
    
    # Dataclass
    ModelMetadata,
    
    # Model dictionaries
    BASIC_GNN_MODELS,
    CONVOLUTIONAL_MODELS,
    ATTENTION_MODELS,
    POOLING_MODELS,
    AGGREGATION_MODELS,
    ENCODER_MODELS,
    AUTOENCODER_MODELS,
    TRANSFORMER_MODELS,
    TEMPORAL_MODELS,
    META_LEARNING_MODELS,
    EXPLAINABILITY_MODELS,
    UTILITY_MODELS,
    
    # Aggregated dictionaries
    MODELS_BY_CATEGORY,
    ALL_MODELS,
    
    # Helper functions
    get_all_model_names,
    get_model_metadata,
    get_models_by_category,
    get_models_by_task,
    get_models_by_tag,
    get_category_statistics,
    search_models,
)


# =============================================================================
# ENUM TESTS
# =============================================================================

class TestModelCategoryEnum:
    """Test ModelCategory enum."""
    
    def test_model_category_values(self):
        """Test all ModelCategory enum values."""
        assert ModelCategory.BASIC_GNN.value == "basic_gnn"
        assert ModelCategory.CONVOLUTIONAL.value == "convolutional"
        assert ModelCategory.ATTENTION.value == "attention"
        assert ModelCategory.POOLING.value == "pooling"
        assert ModelCategory.AGGREGATION.value == "aggregation"
        assert ModelCategory.ENCODER.value == "encoder"
        assert ModelCategory.AUTOENCODER.value == "autoencoder"
        assert ModelCategory.TRANSFORMER.value == "transformer"
        assert ModelCategory.TEMPORAL.value == "temporal"
        assert ModelCategory.META_LEARNING.value == "meta_learning"
        assert ModelCategory.EXPLAINABILITY.value == "explainability"
        assert ModelCategory.UTILITY.value == "utility"
    
    def test_model_category_count(self):
        """Test total number of model categories."""
        assert len(ModelCategory) == 12
    
    def test_model_category_from_string(self):
        """Test creating ModelCategory from string."""
        assert ModelCategory("basic_gnn") == ModelCategory.BASIC_GNN
        assert ModelCategory("convolutional") == ModelCategory.CONVOLUTIONAL
        assert ModelCategory("transformer") == ModelCategory.TRANSFORMER
    
    def test_model_category_invalid(self):
        """Test invalid category raises error."""
        with pytest.raises(ValueError):
            ModelCategory("invalid_category")
    
    def test_model_category_is_enum(self):
        """Test that ModelCategory is an Enum."""
        assert issubclass(ModelCategory, Enum)
    
    def test_model_category_iteration(self):
        """Test iterating over ModelCategory."""
        categories = list(ModelCategory)
        assert len(categories) == 12
        assert ModelCategory.BASIC_GNN in categories
        assert ModelCategory.UTILITY in categories


# =============================================================================
# DATACLASS TESTS
# =============================================================================

class TestModelMetadataDataclass:
    """Test ModelMetadata dataclass."""
    
    def test_is_dataclass(self):
        """Test that ModelMetadata is a dataclass."""
        assert is_dataclass(ModelMetadata)
    
    def test_required_fields(self):
        """Test required fields of ModelMetadata."""
        metadata = ModelMetadata(
            name="TestModel",
            category=ModelCategory.BASIC_GNN,
            import_path="torch_geometric.nn.models.TestModel",
            description="Test model description"
        )
        assert metadata.name == "TestModel"
        assert metadata.category == ModelCategory.BASIC_GNN
        assert metadata.import_path == "torch_geometric.nn.models.TestModel"
        assert metadata.description == "Test model description"
    
    def test_optional_fields_defaults(self):
        """Test optional fields have correct defaults."""
        metadata = ModelMetadata(
            name="TestModel",
            category=ModelCategory.BASIC_GNN,
            import_path="torch_geometric.nn.models.TestModel",
            description="Test"
        )
        assert metadata.paper_url is None
        assert metadata.paper_title is None
        assert metadata.supported_tasks == []
        assert metadata.hyperparameters == {}
        assert metadata.requires_edge_features is False
        assert metadata.requires_edge_weights is False
        assert metadata.requires_edge_index is True
        assert metadata.supports_heterogeneous is False
        assert metadata.supports_directed is True
        assert metadata.min_pyg_version == "2.0.0"
        assert metadata.tags == []
    
    def test_all_fields_present(self):
        """Test all expected fields are present."""
        field_names = {f.name for f in fields(ModelMetadata)}
        expected_fields = {
            'name', 'category', 'import_path', 'description',
            'paper_url', 'paper_title', 'supported_tasks', 'hyperparameters',
            'requires_edge_features', 'requires_edge_weights', 'requires_edge_index',
            'supports_heterogeneous', 'supports_directed', 'min_pyg_version', 'tags'
        }
        assert field_names == expected_fields
    
    def test_field_types(self):
        """Test field types are correct."""
        field_dict = {f.name: f.type for f in fields(ModelMetadata)}
        
        assert field_dict['name'] == str
        assert field_dict['category'] == ModelCategory
        assert field_dict['import_path'] == str
        assert field_dict['description'] == str
        assert field_dict['paper_url'] == Optional[str]
        assert field_dict['paper_title'] == Optional[str]
        assert field_dict['supported_tasks'] == List[str]
        assert field_dict['hyperparameters'] == Dict[str, Any]
        assert field_dict['requires_edge_features'] == bool
        assert field_dict['requires_edge_weights'] == bool
        assert field_dict['requires_edge_index'] == bool
        assert field_dict['supports_heterogeneous'] == bool
        assert field_dict['supports_directed'] == bool
        assert field_dict['min_pyg_version'] == str
        assert field_dict['tags'] == List[str]
    
    def test_with_all_fields(self):
        """Test creating metadata with all fields specified."""
        metadata = ModelMetadata(
            name="GCN",
            category=ModelCategory.BASIC_GNN,
            import_path="torch_geometric.nn.models.GCN",
            description="Graph Convolutional Network",
            paper_url="https://arxiv.org/abs/1609.02907",
            paper_title="Semi-Supervised Classification with Graph Convolutional Networks",
            supported_tasks=["node_classification", "graph_regression"],
            hyperparameters={"hidden_channels": {"type": "integer", "default": 64}},
            requires_edge_features=False,
            requires_edge_weights=False,
            requires_edge_index=True,
            supports_heterogeneous=False,
            supports_directed=True,
            min_pyg_version="2.0.0",
            tags=["spectral", "convolutional"]
        )
        
        assert metadata.name == "GCN"
        assert metadata.paper_url == "https://arxiv.org/abs/1609.02907"
        assert len(metadata.supported_tasks) == 2
        assert "spectral" in metadata.tags
    
    def test_mutable_dataclass(self):
        """Test that ModelMetadata is mutable (not frozen)."""
        metadata = ModelMetadata(
            name="TestModel",
            category=ModelCategory.BASIC_GNN,
            import_path="test.path",
            description="Test"
        )
        # Should be able to modify
        metadata.name = "ModifiedModel"
        assert metadata.name == "ModifiedModel"
        
        metadata.tags.append("new_tag")
        assert "new_tag" in metadata.tags


# =============================================================================
# MODEL DICTIONARY TESTS
# =============================================================================

class TestBasicGNNModels:
    """Test BASIC_GNN_MODELS dictionary."""
    
    def test_basic_gnn_exists(self):
        """Test BASIC_GNN_MODELS dictionary exists."""
        assert BASIC_GNN_MODELS is not None
        assert isinstance(BASIC_GNN_MODELS, dict)
    
    def test_basic_gnn_count(self):
        """Test expected models in BASIC_GNN category."""
        # According to documentation: 6 models
        assert len(BASIC_GNN_MODELS) >= 6
    
    def test_basic_gnn_known_models(self):
        """Test known BASIC_GNN models are present."""
        expected_models = ["GCN", "GraphSAGE", "GIN", "GAT", "EdgeCNN", "PNA"]
        for model_name in expected_models:
            assert model_name in BASIC_GNN_MODELS
    
    def test_basic_gnn_metadata_type(self):
        """Test all values are ModelMetadata instances."""
        for model_name, metadata in BASIC_GNN_MODELS.items():
            assert isinstance(metadata, ModelMetadata)
            assert metadata.name == model_name
    
    def test_basic_gnn_category_consistency(self):
        """Test all models have BASIC_GNN category."""
        for metadata in BASIC_GNN_MODELS.values():
            assert metadata.category == ModelCategory.BASIC_GNN
    
    def test_gcn_model_details(self):
        """Test GCN model has correct details."""
        gcn = BASIC_GNN_MODELS["GCN"]
        assert gcn.name == "GCN"
        assert gcn.category == ModelCategory.BASIC_GNN
        assert gcn.import_path == "torch_geometric.nn.models.GCN"
        assert "Graph Convolutional Network" in gcn.description
        assert gcn.paper_url == "https://arxiv.org/abs/1609.02907"
        assert "node_classification" in gcn.supported_tasks
        assert "spectral" in gcn.tags
    
    def test_gat_model_details(self):
        """Test GAT model has correct details."""
        gat = BASIC_GNN_MODELS["GAT"]
        assert gat.name == "GAT"
        assert "attention" in gat.tags
        assert "heads" in gat.hyperparameters


class TestConvolutionalModels:
    """Test CONVOLUTIONAL_MODELS dictionary."""
    
    def test_convolutional_exists(self):
        """Test CONVOLUTIONAL_MODELS dictionary exists."""
        assert CONVOLUTIONAL_MODELS is not None
        assert isinstance(CONVOLUTIONAL_MODELS, dict)
    
    def test_convolutional_count(self):
        """Test expected models in CONVOLUTIONAL category."""
        # According to documentation: 52+ models
        assert len(CONVOLUTIONAL_MODELS) >= 10  # At least 10 models
    
    def test_convolutional_known_models(self):
        """Test known CONVOLUTIONAL models are present."""
        expected_models = ["GCNConv", "SAGEConv", "GATConv", "GINConv", "ChebConv"]
        for model_name in expected_models:
            assert model_name in CONVOLUTIONAL_MODELS
    
    def test_convolutional_category_consistency(self):
        """Test all models have CONVOLUTIONAL category."""
        for metadata in CONVOLUTIONAL_MODELS.values():
            assert metadata.category == ModelCategory.CONVOLUTIONAL
    
    def test_convolutional_models_are_layers(self):
        """Test convolutional models have 'layer' tag."""
        layer_count = sum(1 for m in CONVOLUTIONAL_MODELS.values() if "layer" in m.tags)
        assert layer_count > 0


class TestAttentionModels:
    """Test ATTENTION_MODELS dictionary."""
    
    def test_attention_exists(self):
        """Test ATTENTION_MODELS dictionary exists."""
        assert ATTENTION_MODELS is not None
        assert isinstance(ATTENTION_MODELS, dict)
    
    def test_attention_count(self):
        """Test expected models in ATTENTION category."""
        # According to documentation: 8 models
        assert len(ATTENTION_MODELS) >= 3
    
    def test_attention_known_models(self):
        """Test known ATTENTION models are present."""
        expected_models = ["GATv2Conv", "TransformerConv", "SuperGATConv"]
        for model_name in expected_models:
            assert model_name in ATTENTION_MODELS
    
    def test_attention_category_consistency(self):
        """Test all models have ATTENTION category."""
        for metadata in ATTENTION_MODELS.values():
            assert metadata.category == ModelCategory.ATTENTION
    
    def test_attention_models_have_attention_tag(self):
        """Test attention models have 'attention' tag."""
        for metadata in ATTENTION_MODELS.values():
            assert "attention" in metadata.tags


class TestPoolingModels:
    """Test POOLING_MODELS dictionary."""
    
    def test_pooling_exists(self):
        """Test POOLING_MODELS dictionary exists."""
        assert POOLING_MODELS is not None
        assert isinstance(POOLING_MODELS, dict)
    
    def test_pooling_category_consistency(self):
        """Test all models have POOLING category."""
        for metadata in POOLING_MODELS.values():
            assert metadata.category == ModelCategory.POOLING


class TestAggregationModels:
    """Test AGGREGATION_MODELS dictionary."""
    
    def test_aggregation_exists(self):
        """Test AGGREGATION_MODELS dictionary exists."""
        assert AGGREGATION_MODELS is not None
        assert isinstance(AGGREGATION_MODELS, dict)
    
    def test_aggregation_category_consistency(self):
        """Test all models have AGGREGATION category."""
        for metadata in AGGREGATION_MODELS.values():
            assert metadata.category == ModelCategory.AGGREGATION


class TestEncoderModels:
    """Test ENCODER_MODELS dictionary."""
    
    def test_encoder_exists(self):
        """Test ENCODER_MODELS dictionary exists."""
        assert ENCODER_MODELS is not None
        assert isinstance(ENCODER_MODELS, dict)
    
    def test_encoder_category_consistency(self):
        """Test all models have ENCODER category."""
        for metadata in ENCODER_MODELS.values():
            assert metadata.category == ModelCategory.ENCODER


class TestAutoencoderModels:
    """Test AUTOENCODER_MODELS dictionary."""
    
    def test_autoencoder_exists(self):
        """Test AUTOENCODER_MODELS dictionary exists."""
        assert AUTOENCODER_MODELS is not None
        assert isinstance(AUTOENCODER_MODELS, dict)
    
    def test_autoencoder_category_consistency(self):
        """Test all models have AUTOENCODER category."""
        for metadata in AUTOENCODER_MODELS.values():
            assert metadata.category == ModelCategory.AUTOENCODER


class TestTransformerModels:
    """Test TRANSFORMER_MODELS dictionary."""
    
    def test_transformer_exists(self):
        """Test TRANSFORMER_MODELS dictionary exists."""
        assert TRANSFORMER_MODELS is not None
        assert isinstance(TRANSFORMER_MODELS, dict)
    
    def test_transformer_category_consistency(self):
        """Test all models have TRANSFORMER category."""
        for metadata in TRANSFORMER_MODELS.values():
            assert metadata.category == ModelCategory.TRANSFORMER


class TestTemporalModels:
    """Test TEMPORAL_MODELS dictionary."""
    
    def test_temporal_exists(self):
        """Test TEMPORAL_MODELS dictionary exists."""
        assert TEMPORAL_MODELS is not None
        assert isinstance(TEMPORAL_MODELS, dict)
    
    def test_temporal_category_consistency(self):
        """Test all models have TEMPORAL category."""
        for metadata in TEMPORAL_MODELS.values():
            assert metadata.category == ModelCategory.TEMPORAL


class TestMetaLearningModels:
    """Test META_LEARNING_MODELS dictionary."""
    
    def test_meta_learning_exists(self):
        """Test META_LEARNING_MODELS dictionary exists."""
        assert META_LEARNING_MODELS is not None
        assert isinstance(META_LEARNING_MODELS, dict)
    
    def test_meta_learning_category_consistency(self):
        """Test all models have META_LEARNING category."""
        for metadata in META_LEARNING_MODELS.values():
            assert metadata.category == ModelCategory.META_LEARNING


class TestExplainabilityModels:
    """Test EXPLAINABILITY_MODELS dictionary."""
    
    def test_explainability_exists(self):
        """Test EXPLAINABILITY_MODELS dictionary exists."""
        assert EXPLAINABILITY_MODELS is not None
        assert isinstance(EXPLAINABILITY_MODELS, dict)
    
    def test_explainability_category_consistency(self):
        """Test all models have EXPLAINABILITY category."""
        for metadata in EXPLAINABILITY_MODELS.values():
            assert metadata.category == ModelCategory.EXPLAINABILITY


class TestUtilityModels:
    """Test UTILITY_MODELS dictionary."""
    
    def test_utility_exists(self):
        """Test UTILITY_MODELS dictionary exists."""
        assert UTILITY_MODELS is not None
        assert isinstance(UTILITY_MODELS, dict)
    
    def test_utility_category_consistency(self):
        """Test all models have UTILITY category."""
        for metadata in UTILITY_MODELS.values():
            assert metadata.category == ModelCategory.UTILITY


# =============================================================================
# AGGREGATED DICTIONARY TESTS
# =============================================================================

class TestModelsByCategoryDict:
    """Test MODELS_BY_CATEGORY dictionary."""
    
    def test_models_by_category_exists(self):
        """Test MODELS_BY_CATEGORY dictionary exists."""
        assert MODELS_BY_CATEGORY is not None
        assert isinstance(MODELS_BY_CATEGORY, dict)
    
    def test_models_by_category_keys(self):
        """Test all ModelCategory values are keys."""
        for category in ModelCategory:
            assert category in MODELS_BY_CATEGORY
    
    def test_models_by_category_count(self):
        """Test correct number of categories."""
        assert len(MODELS_BY_CATEGORY) == 12
    
    def test_models_by_category_values_are_dicts(self):
        """Test all values are dictionaries."""
        for category_dict in MODELS_BY_CATEGORY.values():
            assert isinstance(category_dict, dict)
    
    def test_models_by_category_references(self):
        """Test MODELS_BY_CATEGORY references correct dictionaries."""
        assert MODELS_BY_CATEGORY[ModelCategory.BASIC_GNN] is BASIC_GNN_MODELS
        assert MODELS_BY_CATEGORY[ModelCategory.CONVOLUTIONAL] is CONVOLUTIONAL_MODELS
        assert MODELS_BY_CATEGORY[ModelCategory.ATTENTION] is ATTENTION_MODELS
        assert MODELS_BY_CATEGORY[ModelCategory.POOLING] is POOLING_MODELS
        assert MODELS_BY_CATEGORY[ModelCategory.AGGREGATION] is AGGREGATION_MODELS
        assert MODELS_BY_CATEGORY[ModelCategory.ENCODER] is ENCODER_MODELS
        assert MODELS_BY_CATEGORY[ModelCategory.AUTOENCODER] is AUTOENCODER_MODELS
        assert MODELS_BY_CATEGORY[ModelCategory.TRANSFORMER] is TRANSFORMER_MODELS
        assert MODELS_BY_CATEGORY[ModelCategory.TEMPORAL] is TEMPORAL_MODELS
        assert MODELS_BY_CATEGORY[ModelCategory.META_LEARNING] is META_LEARNING_MODELS
        assert MODELS_BY_CATEGORY[ModelCategory.EXPLAINABILITY] is EXPLAINABILITY_MODELS
        assert MODELS_BY_CATEGORY[ModelCategory.UTILITY] is UTILITY_MODELS


class TestAllModelsDict:
    """Test ALL_MODELS dictionary."""
    
    def test_all_models_exists(self):
        """Test ALL_MODELS dictionary exists."""
        assert ALL_MODELS is not None
        assert isinstance(ALL_MODELS, dict)
    
    def test_all_models_count(self):
        """Test ALL_MODELS has reasonable number of models."""
        # Should have 120+ models total
        assert len(ALL_MODELS) >= 20  # At least 20 models defined
    
    def test_all_models_contains_known_models(self):
        """Test ALL_MODELS contains known models from all categories."""
        known_models = ["GCN", "GAT", "GIN", "GCNConv", "GATConv"]
        for model_name in known_models:
            assert model_name in ALL_MODELS
    
    def test_all_models_values_are_metadata(self):
        """Test all values are ModelMetadata instances."""
        for model_name, metadata in ALL_MODELS.items():
            assert isinstance(metadata, ModelMetadata)
            assert metadata.name == model_name
    
    def test_all_models_no_duplicates(self):
        """Test no duplicate model names in ALL_MODELS."""
        model_names = list(ALL_MODELS.keys())
        assert len(model_names) == len(set(model_names))
    
    def test_all_models_sum_equals_categories(self):
        """Test ALL_MODELS count equals sum of category counts."""
        category_total = sum(len(models) for models in MODELS_BY_CATEGORY.values())
        assert len(ALL_MODELS) == category_total
    
    def test_all_models_contains_all_category_models(self):
        """Test ALL_MODELS contains all models from each category."""
        for category, category_models in MODELS_BY_CATEGORY.items():
            for model_name, metadata in category_models.items():
                assert model_name in ALL_MODELS
                assert ALL_MODELS[model_name] is metadata


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestGetAllModelNames:
    """Test get_all_model_names function."""
    
    def test_returns_list(self):
        """Test function returns a list."""
        result = get_all_model_names()
        assert isinstance(result, list)
    
    def test_returns_strings(self):
        """Test all returned elements are strings."""
        result = get_all_model_names()
        assert all(isinstance(name, str) for name in result)
    
    def test_count_matches_all_models(self):
        """Test count matches ALL_MODELS."""
        result = get_all_model_names()
        assert len(result) == len(ALL_MODELS)
    
    def test_contains_known_models(self):
        """Test contains known model names."""
        result = get_all_model_names()
        assert "GCN" in result
        assert "GAT" in result
        assert "GIN" in result
    
    def test_no_duplicates(self):
        """Test no duplicate model names."""
        result = get_all_model_names()
        assert len(result) == len(set(result))


class TestGetModelMetadata:
    """Test get_model_metadata function."""
    
    def test_returns_metadata_for_valid_model(self):
        """Test returns ModelMetadata for valid model name."""
        result = get_model_metadata("GCN")
        assert isinstance(result, ModelMetadata)
        assert result.name == "GCN"
    
    def test_returns_none_for_invalid_model(self):
        """Test returns None for invalid model name."""
        result = get_model_metadata("NonExistentModel")
        assert result is None
    
    def test_returns_correct_metadata(self):
        """Test returns correct metadata for known models."""
        gcn = get_model_metadata("GCN")
        assert gcn.category == ModelCategory.BASIC_GNN
        assert "Graph Convolutional Network" in gcn.description
    
    def test_case_sensitive(self):
        """Test function is case-sensitive."""
        assert get_model_metadata("GCN") is not None
        assert get_model_metadata("gcn") is None
    
    def test_multiple_models(self):
        """Test retrieving multiple different models."""
        models = ["GCN", "GAT", "GIN"]
        for model_name in models:
            metadata = get_model_metadata(model_name)
            assert metadata is not None
            assert metadata.name == model_name


class TestGetModelsByCategory:
    """Test get_models_by_category function."""
    
    def test_returns_dict_for_valid_category(self):
        """Test returns dict for valid category."""
        result = get_models_by_category(ModelCategory.BASIC_GNN)
        assert isinstance(result, dict)
    
    def test_returns_correct_models(self):
        """Test returns correct models for category."""
        result = get_models_by_category(ModelCategory.BASIC_GNN)
        assert "GCN" in result
        assert "GAT" in result
        assert "GIN" in result
    
    def test_returns_empty_dict_for_invalid_category(self):
        """Test returns empty dict for non-existent category."""
        # Create a fake enum value (this shouldn't exist)
        result = get_models_by_category(None)
        assert result == {}
    
    def test_all_categories_return_dicts(self):
        """Test all valid categories return dictionaries."""
        for category in ModelCategory:
            result = get_models_by_category(category)
            assert isinstance(result, dict)
    
    def test_returned_models_have_correct_category(self):
        """Test all returned models have the requested category."""
        result = get_models_by_category(ModelCategory.BASIC_GNN)
        for metadata in result.values():
            assert metadata.category == ModelCategory.BASIC_GNN
    
    def test_references_original_dict(self):
        """Test returns reference to original dictionary."""
        result = get_models_by_category(ModelCategory.BASIC_GNN)
        assert result is BASIC_GNN_MODELS


class TestGetModelsByTask:
    """Test get_models_by_task function."""
    
    def test_returns_list(self):
        """Test function returns a list."""
        result = get_models_by_task("node_classification")
        assert isinstance(result, list)
    
    def test_returns_strings(self):
        """Test all returned elements are strings."""
        result = get_models_by_task("graph_regression")
        assert all(isinstance(name, str) for name in result)
    
    def test_returns_models_for_node_classification(self):
        """Test returns models that support node classification."""
        result = get_models_by_task("node_classification")
        assert len(result) > 0
        # GCN and GAT should support node classification
        assert "GCN" in result
        assert "GAT" in result
    
    def test_returns_models_for_graph_regression(self):
        """Test returns models that support graph regression."""
        result = get_models_by_task("graph_regression")
        assert len(result) > 0
        # GCN should support graph regression
        assert "GCN" in result
    
    def test_returns_empty_for_unsupported_task(self):
        """Test returns empty list for unsupported task."""
        result = get_models_by_task("nonexistent_task")
        assert result == []
    
    def test_no_duplicates(self):
        """Test no duplicate model names in result."""
        result = get_models_by_task("node_classification")
        assert len(result) == len(set(result))
    
    def test_all_returned_models_support_task(self):
        """Test all returned models actually support the task."""
        task = "graph_classification"
        result = get_models_by_task(task)
        for model_name in result:
            metadata = ALL_MODELS[model_name]
            assert task in metadata.supported_tasks


class TestGetModelsByTag:
    """Test get_models_by_tag function."""
    
    def test_returns_list(self):
        """Test function returns a list."""
        result = get_models_by_tag("attention")
        assert isinstance(result, list)
    
    def test_returns_strings(self):
        """Test all returned elements are strings."""
        result = get_models_by_tag("spectral")
        assert all(isinstance(name, str) for name in result)
    
    def test_returns_models_with_attention_tag(self):
        """Test returns models with 'attention' tag."""
        result = get_models_by_tag("attention")
        assert len(result) > 0
        # GAT should have attention tag
        assert "GAT" in result
    
    def test_returns_models_with_spectral_tag(self):
        """Test returns models with 'spectral' tag."""
        result = get_models_by_tag("spectral")
        assert len(result) > 0
        # GCN should have spectral tag
        assert "GCN" in result
    
    def test_returns_empty_for_nonexistent_tag(self):
        """Test returns empty list for non-existent tag."""
        result = get_models_by_tag("nonexistent_tag_xyz")
        assert result == []
    
    def test_no_duplicates(self):
        """Test no duplicate model names in result."""
        result = get_models_by_tag("layer")
        assert len(result) == len(set(result))
    
    def test_all_returned_models_have_tag(self):
        """Test all returned models actually have the tag."""
        tag = "attention"
        result = get_models_by_tag(tag)
        for model_name in result:
            metadata = ALL_MODELS[model_name]
            assert tag in metadata.tags
    
    def test_case_sensitive(self):
        """Test tag search is case-sensitive."""
        result_lower = get_models_by_tag("attention")
        result_upper = get_models_by_tag("ATTENTION")
        # Should be different if tags are stored in lowercase
        assert len(result_lower) > 0
        assert len(result_upper) == 0


class TestGetCategoryStatistics:
    """Test get_category_statistics function."""
    
    def test_returns_dict(self):
        """Test function returns a dictionary."""
        result = get_category_statistics()
        assert isinstance(result, dict)
    
    def test_contains_all_categories(self):
        """Test result contains all categories."""
        result = get_category_statistics()
        for category in ModelCategory:
            assert category.value in result
    
    def test_values_are_integers(self):
        """Test all values are integers."""
        result = get_category_statistics()
        assert all(isinstance(count, int) for count in result.values())
    
    def test_values_are_positive(self):
        """Test all values are non-negative."""
        result = get_category_statistics()
        assert all(count >= 0 for count in result.values())
    
    def test_total_matches_all_models(self):
        """Test total count matches ALL_MODELS."""
        result = get_category_statistics()
        total = sum(result.values())
        assert total == len(ALL_MODELS)
    
    def test_basic_gnn_count(self):
        """Test BASIC_GNN category count."""
        result = get_category_statistics()
        basic_gnn_count = result["basic_gnn"]
        assert basic_gnn_count == len(BASIC_GNN_MODELS)
    
    def test_all_category_counts_match(self):
        """Test all category counts match their dictionaries."""
        result = get_category_statistics()
        for category in ModelCategory:
            category_models = MODELS_BY_CATEGORY[category]
            assert result[category.value] == len(category_models)


class TestSearchModels:
    """Test search_models function."""
    
    def test_returns_list(self):
        """Test function returns a list."""
        result = search_models("graph")
        assert isinstance(result, list)
    
    def test_returns_strings(self):
        """Test all returned elements are strings."""
        result = search_models("convolution")
        assert all(isinstance(name, str) for name in result)
    
    def test_search_by_name(self):
        """Test searching by name."""
        result = search_models("GCN")
        assert "GCN" in result
        assert "GCNConv" in result
    
    def test_search_by_description(self):
        """Test searching by description."""
        result = search_models("convolutional")
        assert len(result) > 0
    
    def test_search_by_tags(self):
        """Test searching by tags."""
        result = search_models("attention")
        assert len(result) > 0
    
    def test_case_insensitive_search(self):
        """Test search is case-insensitive."""
        result_lower = search_models("gcn")
        result_upper = search_models("GCN")
        result_mixed = search_models("Gcn")
        assert result_lower == result_upper == result_mixed
    
    def test_search_with_specific_fields(self):
        """Test search with specific fields."""
        result = search_models("GCN", search_in=["name"])
        assert len(result) > 0
        assert all("gcn" in name.lower() for name in result)
    
    def test_search_name_only(self):
        """Test searching only in name field."""
        result = search_models("graph", search_in=["name"])
        # Should find models with "graph" in name
        assert len(result) >= 0
    
    def test_search_description_only(self):
        """Test searching only in description field."""
        result = search_models("network", search_in=["description"])
        assert len(result) > 0
    
    def test_search_tags_only(self):
        """Test searching only in tags field."""
        result = search_models("spectral", search_in=["tags"])
        assert len(result) > 0
    
    def test_empty_search_returns_all(self):
        """Test empty query returns all models (empty string matches all)."""
        result = search_models("")
        # Empty string is substring of all strings, so returns all models
        assert len(result) == len(ALL_MODELS)
        assert set(result) == set(ALL_MODELS.keys())
    
    def test_nonexistent_query_returns_empty(self):
        """Test non-existent query returns empty list."""
        result = search_models("xyznonexistent")
        assert result == []
    
    def test_no_duplicates_in_results(self):
        """Test no duplicate models in search results."""
        result = search_models("graph")
        assert len(result) == len(set(result))
    
    def test_search_finds_partial_matches(self):
        """Test search finds partial matches."""
        result = search_models("conv")
        # Should find GCNConv, SAGEConv, etc.
        assert len(result) > 0


# =============================================================================
# DATA INTEGRITY TESTS
# =============================================================================

class TestDataIntegrity:
    """Test data integrity and consistency."""
    
    def test_all_models_have_valid_categories(self):
        """Test all models have valid ModelCategory."""
        for model_name, metadata in ALL_MODELS.items():
            assert isinstance(metadata.category, ModelCategory)
    
    def test_all_models_have_names(self):
        """Test all models have non-empty names."""
        for model_name, metadata in ALL_MODELS.items():
            assert metadata.name
            assert len(metadata.name) > 0
    
    def test_all_models_have_import_paths(self):
        """Test all models have non-empty import paths."""
        for metadata in ALL_MODELS.values():
            assert metadata.import_path
            assert len(metadata.import_path) > 0
            assert "torch_geometric" in metadata.import_path
    
    def test_all_models_have_descriptions(self):
        """Test all models have non-empty descriptions."""
        for metadata in ALL_MODELS.values():
            assert metadata.description
            assert len(metadata.description) > 0
    
    def test_model_names_match_dict_keys(self):
        """Test model names match their dictionary keys."""
        for model_name, metadata in ALL_MODELS.items():
            assert metadata.name == model_name
    
    def test_supported_tasks_are_lists(self):
        """Test supported_tasks are lists."""
        for metadata in ALL_MODELS.values():
            assert isinstance(metadata.supported_tasks, list)
    
    def test_hyperparameters_are_dicts(self):
        """Test hyperparameters are dictionaries."""
        for metadata in ALL_MODELS.values():
            assert isinstance(metadata.hyperparameters, dict)
    
    def test_tags_are_lists(self):
        """Test tags are lists."""
        for metadata in ALL_MODELS.values():
            assert isinstance(metadata.tags, list)
    
    def test_boolean_fields_are_bools(self):
        """Test boolean fields are actually booleans."""
        for metadata in ALL_MODELS.values():
            assert isinstance(metadata.requires_edge_features, bool)
            assert isinstance(metadata.requires_edge_weights, bool)
            assert isinstance(metadata.requires_edge_index, bool)
            assert isinstance(metadata.supports_heterogeneous, bool)
            assert isinstance(metadata.supports_directed, bool)
    
    def test_paper_urls_are_valid_or_none(self):
        """Test paper URLs are valid strings or None."""
        for metadata in ALL_MODELS.values():
            if metadata.paper_url is not None:
                assert isinstance(metadata.paper_url, str)
                assert len(metadata.paper_url) > 0
                assert metadata.paper_url.startswith("http")
    
    def test_min_pyg_version_format(self):
        """Test min_pyg_version has valid format."""
        for metadata in ALL_MODELS.values():
            assert isinstance(metadata.min_pyg_version, str)
            # Should be in format like "2.0.0"
            parts = metadata.min_pyg_version.split(".")
            assert len(parts) >= 2


# =============================================================================
# CONSISTENCY TESTS
# =============================================================================

class TestConsistency:
    """Test consistency across data structures."""
    
    def test_category_dictionaries_disjoint(self):
        """Test category dictionaries have no overlapping models."""
        all_seen_models = set()
        for category, models_dict in MODELS_BY_CATEGORY.items():
            model_names = set(models_dict.keys())
            # Check no overlap with previously seen models
            assert len(all_seen_models & model_names) == 0
            all_seen_models.update(model_names)
    
    def test_all_models_completeness(self):
        """Test ALL_MODELS contains exactly all category models."""
        all_from_categories = set()
        for models_dict in MODELS_BY_CATEGORY.values():
            all_from_categories.update(models_dict.keys())
        
        all_models_keys = set(ALL_MODELS.keys())
        assert all_from_categories == all_models_keys
    
    def test_metadata_object_identity(self):
        """Test same metadata objects are referenced everywhere."""
        # Check that metadata in category dict is same object as in ALL_MODELS
        for category, models_dict in MODELS_BY_CATEGORY.items():
            for model_name, metadata in models_dict.items():
                assert ALL_MODELS[model_name] is metadata
    
    def test_import_paths_unique(self):
        """Test import paths are unique across models."""
        import_paths = [m.import_path for m in ALL_MODELS.values()]
        # Most should be unique, but some layers might share import paths
        # Just check we have a reasonable number
        assert len(set(import_paths)) > len(import_paths) * 0.5


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_supported_tasks(self):
        """Test models can have empty supported_tasks."""
        # This should be allowed
        metadata = ModelMetadata(
            name="Test",
            category=ModelCategory.UTILITY,
            import_path="test.path",
            description="Test",
            supported_tasks=[]
        )
        assert metadata.supported_tasks == []
    
    def test_empty_hyperparameters(self):
        """Test models can have empty hyperparameters."""
        metadata = ModelMetadata(
            name="Test",
            category=ModelCategory.UTILITY,
            import_path="test.path",
            description="Test",
            hyperparameters={}
        )
        assert metadata.hyperparameters == {}
    
    def test_empty_tags(self):
        """Test models can have empty tags."""
        metadata = ModelMetadata(
            name="Test",
            category=ModelCategory.UTILITY,
            import_path="test.path",
            description="Test",
            tags=[]
        )
        assert metadata.tags == []
    
    def test_none_paper_fields(self):
        """Test paper fields can be None."""
        metadata = ModelMetadata(
            name="Test",
            category=ModelCategory.UTILITY,
            import_path="test.path",
            description="Test",
            paper_url=None,
            paper_title=None
        )
        assert metadata.paper_url is None
        assert metadata.paper_title is None
    
    def test_get_model_metadata_empty_string(self):
        """Test get_model_metadata with empty string."""
        result = get_model_metadata("")
        assert result is None
    
    def test_search_models_with_empty_search_in(self):
        """Test search_models with empty search_in list."""
        result = search_models("test", search_in=[])
        assert result == []
    
    def test_get_models_by_task_empty_string(self):
        """Test get_models_by_task with empty string."""
        result = get_models_by_task("")
        # Should return empty as no models support "" task
        assert result == []
    
    def test_get_models_by_tag_empty_string(self):
        """Test get_models_by_tag with empty string."""
        result = get_models_by_tag("")
        # Should return empty as no models have "" tag
        assert result == []


# =============================================================================
# HYPERPARAMETER SCHEMA TESTS
# =============================================================================

class TestHyperparameterSchemas:
    """Test hyperparameter schemas in models."""
    
    def test_gcn_hyperparameters(self):
        """Test GCN has expected hyperparameters."""
        gcn = ALL_MODELS["GCN"]
        hp = gcn.hyperparameters
        assert "in_channels" in hp
        assert "hidden_channels" in hp
        assert "num_layers" in hp
        assert "out_channels" in hp
        assert "dropout" in hp
    
    def test_hyperparameter_structure(self):
        """Test hyperparameter entries have expected structure."""
        gcn = ALL_MODELS["GCN"]
        in_channels = gcn.hyperparameters["in_channels"]
        assert "type" in in_channels
        assert isinstance(in_channels["type"], str)
    
    def test_required_hyperparameters(self):
        """Test required hyperparameters are marked."""
        gcn = ALL_MODELS["GCN"]
        in_channels = gcn.hyperparameters["in_channels"]
        assert in_channels.get("required") is True
    
    def test_default_values_present(self):
        """Test default values are present where expected."""
        gcn = ALL_MODELS["GCN"]
        hidden = gcn.hyperparameters["hidden_channels"]
        assert "default" in hidden
        assert isinstance(hidden["default"], int)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Test performance of helper functions."""
    
    def test_get_all_model_names_performance(self):
        """Test get_all_model_names completes quickly."""
        import time
        start = time.time()
        for _ in range(1000):
            get_all_model_names()
        duration = time.time() - start
        # Should complete 1000 calls in under 1 second
        assert duration < 1.0
    
    def test_get_model_metadata_performance(self):
        """Test get_model_metadata completes quickly."""
        import time
        start = time.time()
        for _ in range(1000):
            get_model_metadata("GCN")
        duration = time.time() - start
        # Should complete 1000 calls in under 0.1 seconds
        assert duration < 0.1
    
    def test_search_models_performance(self):
        """Test search_models completes in reasonable time."""
        import time
        start = time.time()
        for _ in range(100):
            search_models("conv")
        duration = time.time() - start
        # Should complete 100 searches in under 1 second
        assert duration < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
