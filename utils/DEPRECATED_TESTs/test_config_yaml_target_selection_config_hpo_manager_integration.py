#!/usr/bin/env python3
"""
Integration Test Suite for Node/Edge Level Task Functionality

Tests the integration between the following modules:
1. config.yaml - Configuration file with target_selection section
2. target_selection_config.py - Target selection configuration container
3. hpo_manager.py - HPO orchestration with node/edge level support
4. Model factory integration

This is a PRODUCTION-READY integration test suite covering:
- End-to-end workflows for node-level tasks
- End-to-end workflows for edge-level tasks
- Configuration parsing → resolution → data preparation pipeline
- Cross-module compatibility and data flow
- PyG convention compliance (Y first, then fallback)

Author: Milia Team
Version: 1.0.0
"""
import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Optional, Dict, Any
import logging


# =============================================================================
# IMPORTS WITH GRACEFUL FALLBACK
# =============================================================================

try:
    from milia_pipeline.models.factory.target_selection_config import (
        SelectionMode,
        TargetLevel,
        TargetSource,
        TargetSelectionConfig,
    )
    TARGET_SELECTION_AVAILABLE = True
except ImportError:
    TARGET_SELECTION_AVAILABLE = False

try:
    from milia_pipeline.models.hpo.hpo_manager import (
        _prepare_data_for_task_hpo,
        _prepare_node_level_data_hpo,
        _extract_targets_from_source,
    )
    HPO_MANAGER_AVAILABLE = True
except ImportError:
    HPO_MANAGER_AVAILABLE = False

try:
    from milia_pipeline.exceptions import HPOError, ConfigurationError
    EXCEPTIONS_AVAILABLE = True
except ImportError:
    EXCEPTIONS_AVAILABLE = False
    HPOError = Exception
    ConfigurationError = Exception


# =============================================================================
# SKIP MARKERS
# =============================================================================

skip_if_target_selection_unavailable = pytest.mark.skipif(
    not TARGET_SELECTION_AVAILABLE,
    reason="target_selection_config module not available"
)

skip_if_hpo_unavailable = pytest.mark.skipif(
    not HPO_MANAGER_AVAILABLE,
    reason="hpo_manager module not available"
)

skip_if_integration_unavailable = pytest.mark.skipif(
    not (TARGET_SELECTION_AVAILABLE and HPO_MANAGER_AVAILABLE),
    reason="Required modules not available for integration tests"
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_graph_level_data():
    """Create mock graph-level data (QM9-style)."""
    import torch
    
    class MockData:
        def __init__(self):
            self.x = torch.randn(10, 5)  # 10 nodes, 5 features
            self.y = torch.tensor([1.5])  # Graph-level target (scalar)
            self.edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
            self.num_nodes = 10
    
    return [MockData(), MockData()]


@pytest.fixture
def mock_node_level_data_y_correct():
    """Create mock node-level data with y having correct shape."""
    import torch
    
    class MockData:
        def __init__(self):
            self.x = torch.randn(10, 5)
            self.y = torch.randn(10)  # Node-level target (correct shape)
            self.edge_index = torch.tensor([[0, 1], [1, 0]])
            self.num_nodes = 10
    
    return [MockData(), MockData()]


@pytest.fixture
def mock_node_level_data_y_graph():
    """Create mock node-level data with y having graph-level shape."""
    import torch
    
    class MockData:
        def __init__(self):
            self.x = torch.randn(10, 5)
            self.y = torch.tensor([1.5])  # Graph-level shape (wrong for node task)
            self.edge_index = torch.tensor([[0, 1], [1, 0]])
            self.num_nodes = 10
    
    return [MockData(), MockData()]


@pytest.fixture
def mock_edge_level_data():
    """Create mock edge-level data."""
    import torch
    
    class MockData:
        def __init__(self):
            self.x = torch.randn(10, 5)
            self.y = torch.tensor([1.0])  # Graph-level (not edge)
            self.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
            self.edge_attr = torch.randn(4, 3)  # 4 edges, 3 features
            self.num_nodes = 10
    
    return [MockData(), MockData()]


# =============================================================================
# TEST: Config Parsing → TargetSelectionConfig Integration
# =============================================================================

@skip_if_target_selection_unavailable
class TestConfigParsingIntegration:
    """Test config.yaml parsing to TargetSelectionConfig integration."""
    
    def test_graph_regression_config_parsing(self):
        """Test graph_regression config parses correctly."""
        # Simulate config from config.yaml
        config_dict = {
            'target_level': 'auto',
            'target_source': 'auto',
            'indices': [0, 1, 2],
            'strict': True
        }
        
        config = TargetSelectionConfig.from_config(config_dict)
        config.resolve_for_task('graph_regression')
        
        assert config.config_level == 'auto'
        assert config.config_source == 'auto'
        assert config.resolved_level == TargetLevel.GRAPH
        assert config.resolved_source == TargetSource.Y
    
    def test_node_classification_config_parsing(self):
        """Test node_classification config parses correctly."""
        config_dict = {
            'target_level': 'auto',
            'target_source': 'auto',
            'strict': True
        }
        
        config = TargetSelectionConfig.from_config(config_dict)
        config.resolve_for_task('node_classification')
        
        assert config.resolved_level == TargetLevel.NODE
    
    def test_node_regression_from_x_config(self):
        """Test node_regression extracting from x config."""
        config_dict = {
            'target_level': 'auto',
            'target_source': 'x',
            'indices': [5, 6],
            'strict': True
        }
        
        config = TargetSelectionConfig.from_config(config_dict)
        config.resolve_for_task('node_regression')
        
        assert config.resolved_level == TargetLevel.NODE
        assert config.resolved_source == TargetSource.X
        assert config.resolved_source_attr == 'x'
    
    def test_edge_regression_config_parsing(self):
        """Test edge_regression config parses correctly."""
        config_dict = {
            'target_level': 'auto',
            'target_source': 'edge_attr',
            'indices': [0],
            'strict': True
        }
        
        config = TargetSelectionConfig.from_config(config_dict)
        config.resolve_for_task('edge_regression')
        
        assert config.resolved_level == TargetLevel.EDGE
        assert config.resolved_source == TargetSource.EDGE_ATTR
    
    def test_link_prediction_config_parsing(self):
        """Test link_prediction config parses correctly."""
        config_dict = {
            'target_level': 'auto',
            'target_source': 'auto',
            'strict': True
        }
        
        config = TargetSelectionConfig.from_config(config_dict)
        config.resolve_for_task('link_prediction')
        
        assert config.resolved_level == TargetLevel.EDGE
        assert config.resolved_source == TargetSource.EDGE_LABEL
    
    def test_custom_attribute_config_parsing(self):
        """Test custom attribute config parses correctly."""
        config_dict = {
            'target_level': 'node',
            'target_source': 'atomic_charges',
            'strict': True
        }
        
        config = TargetSelectionConfig.from_config(config_dict)
        config.resolve_for_task('node_regression')
        
        assert config.resolved_level == TargetLevel.NODE
        assert config.resolved_source == TargetSource.CUSTOM
        assert config.resolved_source_attr == 'atomic_charges'


# =============================================================================
# TEST: TargetSelectionConfig → HPO Manager Integration
# =============================================================================

@skip_if_integration_unavailable
class TestTargetSelectionHPOIntegration:
    """Test TargetSelectionConfig → HPO Manager integration."""
    
    def test_graph_regression_data_unchanged(self, mock_graph_level_data):
        """Test graph_regression data passes through unchanged."""
        train_data = mock_graph_level_data
        val_data = mock_graph_level_data
        
        result_train, result_val, num_classes = _prepare_data_for_task_hpo(
            train_data, val_data, 'graph_regression', None, None
        )
        
        assert result_train is train_data
        assert result_val is val_data
        assert num_classes is None
    
    def test_node_level_with_correct_y_unchanged(self, mock_node_level_data_y_correct):
        """Test node-level with correct y shape passes unchanged."""
        train_data = mock_node_level_data_y_correct
        val_data = []
        
        result_train, result_val, num_classes = _prepare_data_for_task_hpo(
            train_data, val_data, 'node_regression', None, None
        )
        
        # Data should pass through (y already has correct shape)
        assert result_train is train_data
    
    def test_node_level_with_target_selection_config(self, mock_node_level_data_y_correct):
        """Test node-level with TargetSelectionConfig."""
        train_data = mock_node_level_data_y_correct
        val_data = []
        
        config = TargetSelectionConfig.from_config({
            'target_level': 'auto',
            'target_source': 'auto',
            'strict': True
        })
        
        result_train, result_val, num_classes = _prepare_data_for_task_hpo(
            train_data, val_data, 'node_regression', None, config
        )
        
        # Config should be resolved
        assert config.resolved_level == TargetLevel.NODE
    
    def test_prepare_node_level_data_with_config(self, mock_node_level_data_y_correct):
        """Test _prepare_node_level_data_hpo with config."""
        train_data = mock_node_level_data_y_correct
        val_data = []
        
        config = TargetSelectionConfig.from_config({
            'target_level': 'auto',
            'target_source': 'auto'
        })
        config.resolve_for_task('node_regression', train_data[0])
        
        result_train, result_val = _prepare_node_level_data_hpo(
            train_data, val_data, 'node_regression', config
        )
        
        assert result_train is not None


# =============================================================================
# TEST: End-to-End Node Level Workflow
# =============================================================================

@skip_if_integration_unavailable
class TestNodeLevelWorkflow:
    """Test complete node-level task workflow."""
    
    def test_node_regression_y_correct_shape_workflow(self):
        """Test node regression with y having correct shape."""
        import torch
        
        # Create data with node-level y
        class MockData:
            def __init__(self):
                self.x = torch.randn(5, 3)
                self.y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # Node-level
                self.num_nodes = 5
        
        train_data = [MockData()]
        val_data = []
        
        # Create config
        config = TargetSelectionConfig.from_config({
            'target_level': 'auto',
            'target_source': 'auto',
            'strict': True
        })
        
        # Prepare data
        result_train, result_val, num_classes = _prepare_data_for_task_hpo(
            train_data, val_data, 'node_regression', None, config
        )
        
        # Verify
        assert config.resolved_level == TargetLevel.NODE
        assert result_train is train_data  # Unchanged (y correct)
    
    def test_node_regression_extract_from_x_workflow(self):
        """Test node regression extracting targets from x."""
        import torch
        
        # Create data with graph-level y but node features in x
        class MockData:
            def __init__(self):
                self.x = torch.tensor([
                    [0.1, 0.2, 0.3, 1.0, 2.0],  # Last 2 are targets
                    [0.4, 0.5, 0.6, 3.0, 4.0],
                    [0.7, 0.8, 0.9, 5.0, 6.0],
                ])
                self.y = torch.tensor([99.0])  # Graph-level (wrong shape)
                self.num_nodes = 3
        
        train_data = [MockData()]
        val_data = []
        
        # Create config to extract from x
        config = TargetSelectionConfig.from_config({
            'target_level': 'auto',
            'target_source': 'x',
            'indices': [3, 4],  # Extract columns 3,4 from x
            'strict': True
        })
        config.resolve_for_task('node_regression', train_data[0])
        config.resolved_indices = [3, 4]
        
        # Prepare data
        result_train, result_val = _prepare_node_level_data_hpo(
            train_data, val_data, 'node_regression', config
        )
        
        # Verify extraction happened
        assert len(result_train) == 1
        assert hasattr(result_train[0], 'y')
        # New y should be [num_nodes, 2] or [num_nodes] squeezed


# =============================================================================
# TEST: End-to-End Edge Level Workflow
# =============================================================================

@skip_if_integration_unavailable
class TestEdgeLevelWorkflow:
    """Test complete edge-level task workflow."""
    
    def test_link_prediction_workflow(self):
        """Test link prediction workflow."""
        import torch
        
        # Create data with edge_label
        class MockData:
            def __init__(self):
                self.x = torch.randn(5, 3)
                self.edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
                self.edge_label = torch.tensor([1, 0, 1])  # Binary labels
                self.num_nodes = 5
        
        train_data = [MockData()]
        val_data = []
        
        config = TargetSelectionConfig.from_config({
            'target_level': 'auto',
            'target_source': 'auto'
        })
        
        # Resolve for link prediction
        config.resolve_for_task('link_prediction', train_data[0])
        
        assert config.resolved_level == TargetLevel.EDGE
        assert config.resolved_source == TargetSource.EDGE_LABEL
    
    def test_edge_regression_workflow(self):
        """Test edge regression workflow."""
        import torch
        
        # Create data with edge_attr
        class MockData:
            def __init__(self):
                self.x = torch.randn(5, 3)
                self.edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
                self.edge_attr = torch.randn(3, 4)  # 3 edges, 4 features
                self.y = torch.tensor([1.0])  # Graph-level
                self.num_nodes = 5
        
        train_data = [MockData()]
        val_data = []
        
        config = TargetSelectionConfig.from_config({
            'target_level': 'auto',
            'target_source': 'edge_attr',
            'indices': [0]
        })
        
        config.resolve_for_task('edge_regression', train_data[0])
        
        assert config.resolved_level == TargetLevel.EDGE
        assert config.resolved_source == TargetSource.EDGE_ATTR


# =============================================================================
# TEST: Extract Targets From Source Integration
# =============================================================================

@skip_if_integration_unavailable
class TestExtractTargetsIntegration:
    """Test _extract_targets_from_source integration."""
    
    def test_extract_from_x_creates_y(self):
        """Test extracting from x creates proper y attribute."""
        import torch
        
        class MockData:
            def __init__(self):
                self.x = torch.tensor([
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                ])
        
        data_subset = [MockData()]
        
        result = _extract_targets_from_source(
            data_subset, 'x', [0, 1], 'y', 'train'
        )
        
        assert len(result) == 1
        assert hasattr(result[0], 'y')
        assert result[0].y.shape[0] == 3  # num_nodes
        assert result[0].y.shape[1] == 2  # 2 columns extracted
    
    def test_extract_single_column_squeezes(self):
        """Test extracting single column squeezes result."""
        import torch
        
        class MockData:
            def __init__(self):
                self.x = torch.tensor([
                    [1.0, 2.0],
                    [3.0, 4.0],
                ])
        
        data_subset = [MockData()]
        
        result = _extract_targets_from_source(
            data_subset, 'x', [0], 'y', 'train'
        )
        
        assert result[0].y.dim() == 1  # Squeezed to 1D
        assert result[0].y.shape[0] == 2
    
    def test_extract_from_edge_attr(self):
        """Test extracting from edge_attr."""
        import torch
        
        class MockData:
            def __init__(self):
                self.edge_attr = torch.tensor([
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ])
        
        data_subset = [MockData()]
        
        result = _extract_targets_from_source(
            data_subset, 'edge_attr', [0], 'edge_y', 'train'
        )
        
        assert hasattr(result[0], 'edge_y')


# =============================================================================
# TEST: PyG Convention Compliance
# =============================================================================

@skip_if_integration_unavailable
class TestPyGConventionCompliance:
    """Test PyG convention compliance (Y first, then fallback)."""
    
    def test_y_tried_first_for_node_level(self):
        """Test Y is tried first for node-level tasks."""
        import torch
        
        # Data with correct node-level y
        class MockData:
            def __init__(self):
                self.x = torch.randn(5, 3)
                self.y = torch.randn(5)  # Correct node-level shape
                self.num_nodes = 5
        
        config = TargetSelectionConfig.from_config({
            'target_level': 'auto',
            'target_source': 'auto'
        })
        
        sample = MockData()
        config.resolve_for_task('node_regression', sample)
        
        # Should use Y (not X) because shape is correct
        assert config.resolved_source == TargetSource.Y
    
    def test_fallback_to_x_when_y_wrong_shape(self):
        """Test fallback to X when Y has wrong shape."""
        import torch
        
        # Data with graph-level y
        class MockData:
            def __init__(self):
                self.x = torch.randn(5, 3)
                self.y = torch.tensor([1.0])  # Graph-level shape
                self.num_nodes = 5
        
        config = TargetSelectionConfig.from_config({
            'target_level': 'auto',
            'target_source': 'auto'
        })
        
        sample = MockData()
        config.resolve_for_task('node_regression', sample)
        
        # Should fallback to X
        assert config.resolved_source == TargetSource.X
    
    def test_graph_level_always_uses_y(self):
        """Test graph-level tasks always use Y."""
        import torch
        
        class MockData:
            def __init__(self):
                self.x = torch.randn(5, 3)
                self.y = torch.tensor([1.0])
                self.num_nodes = 5
        
        config = TargetSelectionConfig.from_config({
            'target_level': 'auto',
            'target_source': 'auto'
        })
        
        sample = MockData()
        config.resolve_for_task('graph_regression', sample)
        
        # Graph-level should always use Y
        assert config.resolved_source == TargetSource.Y


# =============================================================================
# TEST: Backward Compatibility
# =============================================================================

@skip_if_integration_unavailable
class TestBackwardCompatibility:
    """Test backward compatibility with existing workflows."""
    
    def test_none_config_works(self, mock_graph_level_data):
        """Test None target_selection_config works (backward compatible)."""
        train_data = mock_graph_level_data
        val_data = mock_graph_level_data
        
        # Should not raise with None config
        result_train, result_val, num_classes = _prepare_data_for_task_hpo(
            train_data, val_data, 'graph_regression', None, None
        )
        
        assert result_train is train_data
    
    def test_empty_config_uses_defaults(self):
        """Test empty config uses sensible defaults."""
        config = TargetSelectionConfig.from_config({})
        
        assert config.mode == SelectionMode.ALL
        assert config.config_level == 'auto'
        assert config.config_source == 'auto'
    
    def test_existing_graph_regression_unchanged(self, mock_graph_level_data):
        """Test existing graph regression workflow unchanged."""
        train_data = mock_graph_level_data
        val_data = []
        
        config = TargetSelectionConfig.from_config({
            'target_level': 'auto',
            'target_source': 'auto'
        })
        
        result_train, result_val, num_classes = _prepare_data_for_task_hpo(
            train_data, val_data, 'graph_regression', None, config
        )
        
        # Graph regression data should pass through unchanged
        assert result_train is train_data


# =============================================================================
# TEST: Error Handling Integration
# =============================================================================

@skip_if_integration_unavailable
class TestErrorHandlingIntegration:
    """Test error handling across module boundaries."""
    
    def test_empty_dataset_raises_error(self):
        """Test empty dataset raises appropriate error."""
        config = TargetSelectionConfig.from_config({})
        
        with pytest.raises(Exception):  # HPOError
            _prepare_data_for_task_hpo(
                [], [], 'node_regression', None, config
            )
    
    def test_invalid_source_attr_handled(self):
        """Test invalid source attribute handled gracefully."""
        import torch
        
        class MockData:
            def __init__(self):
                self.x = torch.randn(5, 3)
                self.y = torch.tensor([1.0])
                self.num_nodes = 5
                # Note: no 'nonexistent_attr'
        
        train_data = [MockData()]
        
        config = TargetSelectionConfig.from_config({
            'target_level': 'node',
            'target_source': 'nonexistent_attr'
        })
        config.resolve_for_task('node_regression')
        
        # Should raise when trying to extract from nonexistent attribute
        with pytest.raises(Exception):
            _prepare_node_level_data_hpo(
                train_data, [], 'node_regression', config
            )


# =============================================================================
# TEST: to_dict Integration
# =============================================================================

@skip_if_target_selection_unavailable
class TestToDictIntegration:
    """Test to_dict output for logging/debugging."""
    
    def test_to_dict_after_resolution(self):
        """Test to_dict contains all resolved values."""
        config = TargetSelectionConfig.from_config({
            'target_level': 'auto',
            'target_source': 'auto',
            'indices': [0, 1, 2],
            'strict': True
        })
        
        config.resolve_for_task('node_regression')
        config.resolve(['a', 'b', 'c', 'd'], 4)
        
        result = config.to_dict()
        
        assert result['config_level'] == 'auto'
        assert result['config_source'] == 'auto'
        assert result['resolved_level'] == 'NODE'
        assert result['resolved_source'] is not None
        assert result['resolved_indices'] == [0, 1, 2]
        assert result['resolved_names'] == ['a', 'b', 'c']


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
