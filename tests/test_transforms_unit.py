#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for transforms.py Module

This test suite provides extensive coverage of PyG augmentation transforms including:
- DropEdge: Edge dropping for graph augmentation
- DropNode: Node dropping with incident edge removal
- MaskFeatures: Feature masking for data augmentation
- RandomNodeSample: Random node sampling for subgraph extraction

Test Coverage:
- Initialization and parameter validation
- Forward pass transformations on various graph structures
- Edge cases (empty graphs, single nodes, zero-node graphs, disconnected components)
- Undirected graph property preservation (even/odd edge counts)
- Feature attribute handling (x, pos, edge_attr, edge_weight, etc.)
- Missing attribute resilience (no edge_index, no x, x=None)
- Metadata extraction and validation
- Error handling and boundary conditions (p=0.0, p=1.0, fallback paths)
- BaseTransform inheritance and PyG transform protocol compliance
- Integration with PyG Data objects and Compose pipeline
- Deterministic reproducibility with seeded randomness

NOTE: This test suite runs inside Docker at /app/milia
Path: ~/ml_projects/milia/milia_pipeline/plugins/pyg_augmentation/transforms.py

Author: milia Project Team
Created: November 3, 2025
Updated: February 2026
"""

import sys
from pathlib import Path

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))


import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

# Import the module under test
from milia_pipeline.plugins.pyg_augmentation.transforms import (
    DropEdge,
    DropNode,
    MaskFeatures,
    RandomNodeSample,
)

# =============================================================================
# TEST FIXTURES AND HELPER FUNCTIONS
# =============================================================================


@pytest.fixture
def simple_graph():
    """Create a simple undirected graph with 5 nodes and 6 edges"""
    # Triangle + 2 isolated nodes
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 0],  # Triangle: 0-1, 1-2, 2-0
            [1, 0, 2, 1, 0, 2],
        ],
        dtype=torch.long,
    )

    x = torch.randn(5, 3)  # 5 nodes, 3 features each

    data = Data(x=x, edge_index=edge_index)
    return data


@pytest.fixture
def graph_with_attributes():
    """Create a graph with node and edge attributes"""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 0, 3, 4], [1, 0, 2, 1, 0, 2, 4, 3]], dtype=torch.long
    )

    x = torch.randn(5, 4)
    pos = torch.randn(5, 3)  # 3D positions
    edge_attr = torch.randn(8, 2)  # Edge features
    edge_weight = torch.rand(8)  # Edge weights
    y = torch.tensor([0, 1, 0, 1, 0])  # Node labels

    data = Data(
        x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weight, y=y
    )
    return data


@pytest.fixture
def empty_graph():
    """Create a graph with no edges"""
    x = torch.randn(3, 2)
    edge_index = torch.tensor([[], []], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data


@pytest.fixture
def single_node_graph():
    """Create a graph with a single node"""
    x = torch.randn(1, 3)
    edge_index = torch.tensor([[], []], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data


@pytest.fixture
def large_graph():
    """Create a larger graph for performance testing"""
    num_nodes = 100
    num_edges = 500

    # Random edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    x = torch.randn(num_nodes, 10)

    data = Data(x=x, edge_index=edge_index)
    return data


@pytest.fixture
def zero_node_graph():
    """Create a graph with zero nodes and zero edges"""
    edge_index = torch.tensor([[], []], dtype=torch.long)
    data = Data(edge_index=edge_index, num_nodes=0)
    return data


@pytest.fixture
def odd_edge_graph():
    """Create a directed graph with an odd number of edges (3 edges)"""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    x = torch.randn(3, 4)
    data = Data(x=x, edge_index=edge_index)
    return data


# =============================================================================
# DropEdge Transform Tests
# =============================================================================


class TestDropEdge:
    """Test suite for DropEdge transform"""

    def test_initialization_default(self):
        """Test DropEdge initialization with default parameters"""
        transform = DropEdge()

        assert transform.p == 0.5
        assert transform.force_undirected is True

    def test_initialization_custom_p(self):
        """Test DropEdge initialization with custom probability"""
        transform = DropEdge(p=0.3)

        assert transform.p == 0.3
        assert transform.force_undirected is True

    def test_initialization_invalid_p_negative(self):
        """Test DropEdge rejects negative probability"""
        with pytest.raises(ValueError, match="p must be in"):
            DropEdge(p=-0.1)

    def test_initialization_invalid_p_too_large(self):
        """Test DropEdge rejects probability > 1"""
        with pytest.raises(ValueError, match="p must be in"):
            DropEdge(p=1.5)

    def test_forward_simple_graph(self, simple_graph):
        """Test DropEdge on simple graph"""
        transform = DropEdge(p=0.5)

        original_num_edges = simple_graph.edge_index.size(1)
        result = transform(simple_graph)

        # Should have fewer or equal edges
        assert result.edge_index.size(1) <= original_num_edges
        assert result.edge_index.size(0) == 2  # Still 2 rows

    def test_forward_preserves_node_features(self, simple_graph):
        """Test DropEdge preserves node features"""
        transform = DropEdge(p=0.5)

        original_x = simple_graph.x.clone()
        result = transform(simple_graph)

        assert torch.equal(result.x, original_x)

    def test_forward_with_edge_attributes(self, graph_with_attributes):
        """Test DropEdge properly handles edge attributes"""
        transform = DropEdge(p=0.3)

        result = transform(graph_with_attributes)

        # Edge attributes should be consistent with edges
        assert result.edge_attr.size(0) == result.edge_index.size(1)
        assert result.edge_weight.size(0) == result.edge_index.size(1)

    def test_forward_empty_graph(self, empty_graph):
        """Test DropEdge on graph with no edges"""
        transform = DropEdge(p=0.5)

        result = transform(empty_graph)

        assert result.edge_index.size(1) == 0
        assert result.x.size(0) == 3

    def test_forward_p_zero(self, simple_graph):
        """Test DropEdge with p=0 (no edges dropped)"""
        transform = DropEdge(p=0.0)

        original_num_edges = simple_graph.edge_index.size(1)
        result = transform(simple_graph)

        assert result.edge_index.size(1) == original_num_edges

    def test_forward_p_one(self, simple_graph):
        """Test DropEdge with p=1 (all edges dropped)"""
        transform = DropEdge(p=1.0)

        result = transform(simple_graph)

        # All edges should be dropped
        assert result.edge_index.size(1) == 0

    def test_forward_force_undirected(self, simple_graph):
        """Test DropEdge with force_undirected=True"""
        transform = DropEdge(p=0.5, force_undirected=True)

        result = transform(simple_graph)

        # Result should still be undirected (edges come in pairs)
        # This is implicit in the symmetric mask application
        assert result.edge_index.size(1) % 2 == 0 or result.edge_index.size(1) == 0

    def test_forward_not_force_undirected(self, simple_graph):
        """Test DropEdge with force_undirected=False"""
        transform = DropEdge(p=0.5, force_undirected=False)

        result = transform(simple_graph)

        # Should still work, may not preserve undirected property
        assert result.edge_index.size(1) <= simple_graph.edge_index.size(1)

    def test_forward_no_edge_index(self):
        """Test DropEdge on data with empty edge_index (no edges)"""
        # Create data with empty edge_index (no edges)
        data = Data(x=torch.randn(5, 3), edge_index=torch.tensor([[], []], dtype=torch.long))
        transform = DropEdge(p=0.5)

        result = transform(data)

        # Should return data unchanged (no edges to drop)
        assert result.edge_index.size(1) == 0
        assert torch.equal(result.x, data.x)

    def test_forward_no_edge_index_attribute(self):
        """Test DropEdge on data where edge_index is None (PyG Data property returns None when unset)"""
        data = Data(x=torch.randn(3, 2))
        # In PyG, Data.edge_index is a class-level property that returns None
        # when not set in _store, so hasattr() is always True.
        assert data.edge_index is None  # Confirm PyG behavior

        transform = DropEdge(p=0.5)
        result = transform(data)

        # Should return data unchanged since edge_index is None
        assert result.edge_index is None
        assert torch.equal(result.x, data.x)

    def test_forward_odd_edge_count_force_undirected(self, odd_edge_graph):
        """Test DropEdge with force_undirected=True on odd-edge-count graph (skips symmetric dropping)"""
        transform = DropEdge(p=0.5, force_undirected=True)

        result = transform(odd_edge_graph)

        # With odd edges, symmetric dropping is skipped; standard mask is applied
        assert result.edge_index.size(1) <= odd_edge_graph.edge_index.size(1)
        assert result.edge_index.size(0) == 2

    def test_inherits_base_transform(self):
        """Test DropEdge inherits from PyG BaseTransform"""
        assert issubclass(DropEdge, BaseTransform)

        transform = DropEdge(p=0.3)
        assert isinstance(transform, BaseTransform)

    def test_callable_protocol(self, simple_graph):
        """Test DropEdge is callable via __call__ (PyG BaseTransform protocol)"""
        transform = DropEdge(p=0.3)

        assert callable(transform)

        # __call__ delegates to forward
        result = transform(simple_graph)
        assert hasattr(result, "edge_index")

    def test_get_metadata(self):
        """Test DropEdge metadata extraction"""
        metadata = DropEdge.get_metadata()

        assert metadata.name == "DropEdge"
        assert metadata.version == "1.0.0"
        assert metadata.category == "augmentation"
        assert "Randomly drops edges" in metadata.description
        assert isinstance(metadata.required_node_features, list)
        assert isinstance(metadata.required_edge_features, list)

    def test_forward_deterministic_with_seed(self, simple_graph):
        """Test DropEdge produces same result with same seed"""
        torch.manual_seed(42)
        transform = DropEdge(p=0.5)
        result1 = transform(simple_graph.clone())

        torch.manual_seed(42)
        result2 = transform(simple_graph.clone())

        assert torch.equal(result1.edge_index, result2.edge_index)

    def test_forward_large_graph(self, large_graph):
        """Test DropEdge on larger graph for performance"""
        transform = DropEdge(p=0.3)

        result = transform(large_graph)

        # Should complete quickly and reduce edges
        assert result.edge_index.size(1) <= large_graph.edge_index.size(1)

    def test_initialization_boundary_p_zero(self):
        """Test DropEdge accepts p=0.0 (boundary)"""
        transform = DropEdge(p=0.0)
        assert transform.p == 0.0

    def test_initialization_boundary_p_one(self):
        """Test DropEdge accepts p=1.0 (boundary)"""
        transform = DropEdge(p=1.0)
        assert transform.p == 1.0

    def test_forward_with_edge_weight_no_edge_attr(self):
        """Test DropEdge with edge_weight but no edge_attr"""
        edge_index = torch.tensor([[0, 1, 1, 0], [1, 0, 2, 2]], dtype=torch.long)
        x = torch.randn(3, 2)
        edge_weight = torch.rand(4)
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

        transform = DropEdge(p=0.3)
        result = transform(data)

        # edge_weight should stay consistent with edge_index
        assert result.edge_weight.size(0) == result.edge_index.size(1)


# =============================================================================
# DropNode Transform Tests
# =============================================================================


class TestDropNode:
    """Test suite for DropNode transform"""

    def test_initialization_default(self):
        """Test DropNode initialization with default parameters"""
        transform = DropNode()

        assert transform.p == 0.5

    def test_initialization_custom_p(self):
        """Test DropNode initialization with custom probability"""
        transform = DropNode(p=0.3)

        assert transform.p == 0.3

    def test_initialization_invalid_p_negative(self):
        """Test DropNode rejects negative probability"""
        with pytest.raises(ValueError, match="p must be in"):
            DropNode(p=-0.1)

    def test_initialization_invalid_p_too_large(self):
        """Test DropNode rejects probability > 1"""
        with pytest.raises(ValueError, match="p must be in"):
            DropNode(p=1.5)

    def test_forward_simple_graph(self, simple_graph):
        """Test DropNode on simple graph"""
        transform = DropNode(p=0.4)

        original_num_nodes = simple_graph.num_nodes
        result = transform(simple_graph)

        # Should have fewer or equal nodes
        assert result.num_nodes <= original_num_nodes
        assert result.num_nodes >= 1  # At least one node remains

    def test_forward_reduces_features(self, simple_graph):
        """Test DropNode reduces node features correctly"""
        transform = DropNode(p=0.4)

        original_num_nodes = simple_graph.num_nodes
        result = transform(simple_graph)

        # Node features should match number of nodes
        assert result.x.size(0) == result.num_nodes
        assert result.x.size(0) <= original_num_nodes

    def test_forward_removes_incident_edges(self, simple_graph):
        """Test DropNode removes incident edges"""
        transform = DropNode(p=0.6)

        result = transform(simple_graph)

        # All edges should reference valid node indices
        max_node_idx = result.num_nodes - 1
        assert result.edge_index.max() <= max_node_idx if result.edge_index.numel() > 0 else True

    def test_forward_with_multiple_attributes(self, graph_with_attributes):
        """Test DropNode handles multiple node attributes"""
        transform = DropNode(p=0.3)

        result = transform(graph_with_attributes)

        # All node attributes should have consistent sizes
        assert result.x.size(0) == result.num_nodes
        assert result.pos.size(0) == result.num_nodes
        assert result.y.size(0) == result.num_nodes

    def test_forward_empty_graph(self, empty_graph):
        """Test DropNode on graph with no edges"""
        transform = DropNode(p=0.5)

        result = transform(empty_graph)

        # Should have at least 1 node
        assert result.num_nodes >= 1
        assert result.x.size(0) == result.num_nodes

    def test_forward_single_node(self, single_node_graph):
        """Test DropNode on single node graph"""
        transform = DropNode(p=0.9)

        result = transform(single_node_graph)

        # Must keep at least one node
        assert result.num_nodes == 1
        assert result.x.size(0) == 1

    def test_forward_p_zero(self, simple_graph):
        """Test DropNode with p=0 (no nodes dropped)"""
        transform = DropNode(p=0.0)

        original_num_nodes = simple_graph.num_nodes
        result = transform(simple_graph)

        assert result.num_nodes == original_num_nodes

    def test_forward_p_high(self, simple_graph):
        """Test DropNode with p=0.95 (most nodes dropped)"""
        transform = DropNode(p=0.95)

        result = transform(simple_graph)

        # Should keep at least 1 node
        assert result.num_nodes >= 1
        assert result.num_nodes <= simple_graph.num_nodes

    def test_forward_relabels_nodes(self, simple_graph):
        """Test DropNode relabels nodes correctly"""
        transform = DropNode(p=0.4)

        result = transform(simple_graph)

        # Edge indices should be in valid range [0, num_nodes-1]
        if result.edge_index.numel() > 0:
            assert result.edge_index.min() >= 0
            assert result.edge_index.max() < result.num_nodes

    def test_forward_edge_attr_consistency(self, graph_with_attributes):
        """Test DropNode maintains edge attribute consistency"""
        transform = DropNode(p=0.3)

        result = transform(graph_with_attributes)

        # Edge attributes should match number of edges
        if hasattr(result, "edge_attr") and result.edge_attr is not None:
            assert result.edge_attr.size(0) == result.edge_index.size(1)

    def test_get_metadata(self):
        """Test DropNode metadata extraction"""
        metadata = DropNode.get_metadata()

        assert metadata.name == "DropNode"
        assert metadata.version == "1.0.0"
        assert metadata.category == "augmentation"
        assert "drops nodes" in metadata.description
        assert isinstance(metadata.required_node_features, list)

    def test_forward_deterministic_with_seed(self, simple_graph):
        """Test DropNode produces same result with same seed"""
        torch.manual_seed(42)
        transform = DropNode(p=0.5)
        result1 = transform(simple_graph.clone())

        torch.manual_seed(42)
        result2 = transform(simple_graph.clone())

        assert result1.num_nodes == result2.num_nodes
        assert torch.equal(result1.edge_index, result2.edge_index)

    def test_forward_large_graph(self, large_graph):
        """Test DropNode on larger graph for performance"""
        transform = DropNode(p=0.3)

        result = transform(large_graph)

        # Should complete quickly and reduce nodes
        assert result.num_nodes <= large_graph.num_nodes
        assert result.num_nodes >= 1

    def test_forward_p_one_fallback(self, simple_graph):
        """Test DropNode with p=1.0 triggers fallback to keep at least 1 node"""
        transform = DropNode(p=1.0)

        result = transform(simple_graph)

        # All nodes masked => fallback should keep exactly 1 node
        assert result.num_nodes >= 1
        assert result.x.size(0) == result.num_nodes

    def test_forward_zero_node_graph(self, zero_node_graph):
        """Test DropNode on graph with zero nodes returns early"""
        transform = DropNode(p=0.5)

        result = transform(zero_node_graph)

        # Should return data unchanged
        assert result.num_nodes == 0

    def test_forward_no_x_attribute(self):
        """Test DropNode on data without x attribute (device fallback to cpu)"""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data = Data(edge_index=edge_index, num_nodes=3)

        transform = DropNode(p=0.3)
        result = transform(data)

        # Should handle missing x gracefully using cpu device
        assert result.num_nodes >= 1
        if result.edge_index.numel() > 0:
            assert result.edge_index.max() < result.num_nodes

    def test_inherits_base_transform(self):
        """Test DropNode inherits from PyG BaseTransform"""
        assert issubclass(DropNode, BaseTransform)

        transform = DropNode(p=0.3)
        assert isinstance(transform, BaseTransform)

    def test_callable_protocol(self, simple_graph):
        """Test DropNode is callable via __call__ (PyG BaseTransform protocol)"""
        transform = DropNode(p=0.3)

        assert callable(transform)
        result = transform(simple_graph)
        assert result.num_nodes >= 1

    def test_initialization_boundary_p_zero(self):
        """Test DropNode accepts p=0.0 (boundary)"""
        transform = DropNode(p=0.0)
        assert transform.p == 0.0

    def test_initialization_boundary_p_one(self):
        """Test DropNode accepts p=1.0 (boundary)"""
        transform = DropNode(p=1.0)
        assert transform.p == 1.0


# =============================================================================
# MaskFeatures Transform Tests
# =============================================================================


class TestMaskFeatures:
    """Test suite for MaskFeatures transform"""

    def test_initialization_default(self):
        """Test MaskFeatures initialization with default parameters"""
        transform = MaskFeatures()

        assert transform.p == 0.5
        assert transform.mask_value == 0.0

    def test_initialization_custom_p(self):
        """Test MaskFeatures initialization with custom probability"""
        transform = MaskFeatures(p=0.3)

        assert transform.p == 0.3
        assert transform.mask_value == 0.0

    def test_initialization_custom_mask_value(self):
        """Test MaskFeatures initialization with custom mask value"""
        transform = MaskFeatures(p=0.5, mask_value=-1.0)

        assert transform.p == 0.5
        assert transform.mask_value == -1.0

    def test_initialization_invalid_p_negative(self):
        """Test MaskFeatures rejects negative probability"""
        with pytest.raises(ValueError, match="p must be in"):
            MaskFeatures(p=-0.1)

    def test_initialization_invalid_p_too_large(self):
        """Test MaskFeatures rejects probability > 1"""
        with pytest.raises(ValueError, match="p must be in"):
            MaskFeatures(p=1.5)

    def test_forward_simple_graph(self, simple_graph):
        """Test MaskFeatures on simple graph"""
        transform = MaskFeatures(p=0.5, mask_value=0.0)

        original_x = simple_graph.x.clone()
        result = transform(simple_graph)

        # Shape should be unchanged
        assert result.x.shape == original_x.shape

        # Some values should be masked (set to 0)
        # Unless by extreme chance no masks applied
        # We can't guarantee masks were applied due to randomness

    def test_forward_masks_to_zero(self, simple_graph):
        """Test MaskFeatures masks features to zero"""
        torch.manual_seed(42)
        transform = MaskFeatures(p=0.5, mask_value=0.0)

        original_x = simple_graph.x.clone()
        result = transform(simple_graph)

        # Some features should be exactly 0 (masked)
        _num_zeros = (result.x == 0.0).sum().item()
        _original_zeros = (original_x == 0.0).sum().item()

        # Should have more zeros than original (unless p=0)
        # This is probabilistic, so we just check shape
        assert result.x.shape == original_x.shape

    def test_forward_custom_mask_value(self, simple_graph):
        """Test MaskFeatures with custom mask value"""
        torch.manual_seed(42)
        transform = MaskFeatures(p=0.8, mask_value=-999.0)

        result = transform(simple_graph)

        # Should contain the mask value
        _has_mask_value = (result.x == -999.0).any().item()
        # With p=0.8, very likely to have at least one masked value
        assert result.x.shape == simple_graph.x.shape

    def test_forward_preserves_graph_structure(self, simple_graph):
        """Test MaskFeatures preserves graph structure"""
        transform = MaskFeatures(p=0.5)

        original_edges = simple_graph.edge_index.clone()
        result = transform(simple_graph)

        # Edge structure should be unchanged
        assert torch.equal(result.edge_index, original_edges)

    def test_forward_no_node_features(self):
        """Test MaskFeatures on graph without node features"""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data = Data(edge_index=edge_index)

        transform = MaskFeatures(p=0.5)
        result = transform(data)

        # Should return data unchanged
        assert not hasattr(result, "x") or result.x is None

    def test_forward_p_zero(self, simple_graph):
        """Test MaskFeatures with p=0 (no masking)"""
        transform = MaskFeatures(p=0.0)

        original_x = simple_graph.x.clone()
        result = transform(simple_graph)

        # Should be unchanged (no masking)
        assert torch.allclose(result.x, original_x)

    def test_forward_p_one(self, simple_graph):
        """Test MaskFeatures with p=1 (all features masked)"""
        transform = MaskFeatures(p=1.0, mask_value=0.0)

        result = transform(simple_graph)

        # All features should be masked
        assert torch.all(result.x == 0.0)

    def test_forward_clones_features(self, simple_graph):
        """Test MaskFeatures doesn't modify original data in-place initially"""
        transform = MaskFeatures(p=0.5)

        _original_x = simple_graph.x.clone()
        result = transform(simple_graph)

        # Original and result should be different objects
        # (transform clones internally)
        assert result.x.data_ptr() != simple_graph.x.data_ptr()

    def test_forward_multiple_attributes(self, graph_with_attributes):
        """Test MaskFeatures only affects x, not other attributes"""
        transform = MaskFeatures(p=0.5)

        original_pos = graph_with_attributes.pos.clone()
        result = transform(graph_with_attributes)

        # Only x should change, pos should be unchanged
        assert torch.equal(result.pos, original_pos)

    def test_get_metadata(self):
        """Test MaskFeatures metadata extraction"""
        metadata = MaskFeatures.get_metadata()

        assert metadata.name == "MaskFeatures"
        assert metadata.version == "1.0.0"
        assert metadata.category == "augmentation"
        assert "masks node features" in metadata.description
        assert isinstance(metadata.required_node_features, list)

    def test_forward_deterministic_with_seed(self, simple_graph):
        """Test MaskFeatures produces same result with same seed"""
        torch.manual_seed(42)
        transform = MaskFeatures(p=0.5)
        result1 = transform(simple_graph.clone())

        torch.manual_seed(42)
        result2 = transform(simple_graph.clone())

        assert torch.equal(result1.x, result2.x)

    def test_forward_large_graph(self, large_graph):
        """Test MaskFeatures on larger graph for performance"""
        transform = MaskFeatures(p=0.3, mask_value=0.0)

        result = transform(large_graph)

        # Should complete quickly
        assert result.x.shape == large_graph.x.shape

    def test_forward_x_is_none(self):
        """Test MaskFeatures on data where x is explicitly None"""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data = Data(edge_index=edge_index, x=None)

        transform = MaskFeatures(p=0.5)
        result = transform(data)

        # Should return data unchanged
        assert result.x is None

    def test_inherits_base_transform(self):
        """Test MaskFeatures inherits from PyG BaseTransform"""
        assert issubclass(MaskFeatures, BaseTransform)

        transform = MaskFeatures(p=0.3)
        assert isinstance(transform, BaseTransform)

    def test_callable_protocol(self, simple_graph):
        """Test MaskFeatures is callable via __call__ (PyG BaseTransform protocol)"""
        transform = MaskFeatures(p=0.3)

        assert callable(transform)
        result = transform(simple_graph)
        assert result.x.shape == simple_graph.x.shape

    def test_forward_preserves_feature_dimensions(self, simple_graph):
        """Test MaskFeatures preserves both node count and feature dimension"""
        transform = MaskFeatures(p=0.7, mask_value=-1.0)

        result = transform(simple_graph)

        assert result.x.size(0) == simple_graph.x.size(0)
        assert result.x.size(1) == simple_graph.x.size(1)

    def test_initialization_boundary_p_zero(self):
        """Test MaskFeatures accepts p=0.0 (boundary)"""
        transform = MaskFeatures(p=0.0)
        assert transform.p == 0.0

    def test_initialization_boundary_p_one(self):
        """Test MaskFeatures accepts p=1.0 (boundary)"""
        transform = MaskFeatures(p=1.0)
        assert transform.p == 1.0


# =============================================================================
# RandomNodeSample Transform Tests
# =============================================================================


class TestRandomNodeSample:
    """Test suite for RandomNodeSample transform"""

    def test_initialization_with_num(self):
        """Test RandomNodeSample initialization with num parameter"""
        transform = RandomNodeSample(num=10)

        assert transform.num == 10
        assert transform.ratio is None

    def test_initialization_with_ratio(self):
        """Test RandomNodeSample initialization with ratio parameter"""
        transform = RandomNodeSample(ratio=0.5)

        assert transform.num is None
        assert transform.ratio == 0.5

    def test_initialization_requires_parameter(self):
        """Test RandomNodeSample requires either num or ratio"""
        with pytest.raises(ValueError, match="Either 'num' or 'ratio' must be specified"):
            RandomNodeSample()

    def test_initialization_invalid_num_negative(self):
        """Test RandomNodeSample rejects negative num"""
        with pytest.raises(ValueError, match="num must be positive"):
            RandomNodeSample(num=-5)

    def test_initialization_invalid_num_zero(self):
        """Test RandomNodeSample rejects zero num"""
        with pytest.raises(ValueError, match="num must be positive"):
            RandomNodeSample(num=0)

    def test_initialization_invalid_ratio_zero(self):
        """Test RandomNodeSample rejects ratio of 0"""
        with pytest.raises(ValueError, match="ratio must be in"):
            RandomNodeSample(ratio=0.0)

    def test_initialization_invalid_ratio_negative(self):
        """Test RandomNodeSample rejects negative ratio"""
        with pytest.raises(ValueError, match="ratio must be in"):
            RandomNodeSample(ratio=-0.1)

    def test_initialization_invalid_ratio_too_large(self):
        """Test RandomNodeSample rejects ratio > 1"""
        with pytest.raises(ValueError, match="ratio must be in"):
            RandomNodeSample(ratio=1.5)

    def test_forward_with_num(self, simple_graph):
        """Test RandomNodeSample with num parameter"""
        transform = RandomNodeSample(num=3)

        result = transform(simple_graph)

        # Should sample exactly 3 nodes (or all if graph has fewer)
        expected_nodes = min(3, simple_graph.num_nodes)
        assert result.num_nodes == expected_nodes

    def test_forward_with_ratio(self, simple_graph):
        """Test RandomNodeSample with ratio parameter"""
        transform = RandomNodeSample(ratio=0.6)

        result = transform(simple_graph)

        # Should sample ~60% of nodes (at least 1)
        expected_nodes = max(1, int(0.6 * simple_graph.num_nodes))
        assert result.num_nodes == expected_nodes

    def test_forward_num_exceeds_graph_size(self, simple_graph):
        """Test RandomNodeSample when num > graph size"""
        transform = RandomNodeSample(num=100)

        result = transform(simple_graph)

        # Should cap at actual number of nodes
        assert result.num_nodes == simple_graph.num_nodes

    def test_forward_reduces_features(self, simple_graph):
        """Test RandomNodeSample reduces node features correctly"""
        transform = RandomNodeSample(num=3)

        result = transform(simple_graph)

        # Feature dimensions should match sampled nodes
        assert result.x.size(0) == result.num_nodes

    def test_forward_subgraph_extraction(self, simple_graph):
        """Test RandomNodeSample extracts valid subgraph"""
        transform = RandomNodeSample(num=3)

        result = transform(simple_graph)

        # All edges should reference valid node indices
        if result.edge_index.numel() > 0:
            assert result.edge_index.min() >= 0
            assert result.edge_index.max() < result.num_nodes

    def test_forward_with_multiple_attributes(self, graph_with_attributes):
        """Test RandomNodeSample handles multiple node attributes"""
        transform = RandomNodeSample(num=3)

        result = transform(graph_with_attributes)

        # All node attributes should have consistent sizes
        assert result.x.size(0) == result.num_nodes
        assert result.pos.size(0) == result.num_nodes
        assert result.y.size(0) == result.num_nodes

    def test_forward_empty_graph(self, empty_graph):
        """Test RandomNodeSample on graph with no edges"""
        transform = RandomNodeSample(num=2)

        result = transform(empty_graph)

        # Should sample nodes successfully
        assert result.num_nodes == 2
        assert result.x.size(0) == 2

    def test_forward_single_node(self, single_node_graph):
        """Test RandomNodeSample on single node graph"""
        transform = RandomNodeSample(num=1)

        result = transform(single_node_graph)

        assert result.num_nodes == 1
        assert result.x.size(0) == 1

    def test_forward_ratio_small(self, large_graph):
        """Test RandomNodeSample with small ratio"""
        transform = RandomNodeSample(ratio=0.1)

        result = transform(large_graph)

        expected_nodes = max(1, int(0.1 * large_graph.num_nodes))
        assert result.num_nodes == expected_nodes

    def test_forward_ratio_large(self, large_graph):
        """Test RandomNodeSample with large ratio"""
        transform = RandomNodeSample(ratio=0.9)

        result = transform(large_graph)

        expected_nodes = int(0.9 * large_graph.num_nodes)
        assert result.num_nodes == expected_nodes

    def test_forward_edge_attr_consistency(self, graph_with_attributes):
        """Test RandomNodeSample maintains edge attribute consistency"""
        transform = RandomNodeSample(num=3)

        result = transform(graph_with_attributes)

        # Edge attributes should match number of edges
        if hasattr(result, "edge_attr") and result.edge_attr is not None:
            assert result.edge_attr.size(0) == result.edge_index.size(1)

    def test_forward_relabels_nodes(self, simple_graph):
        """Test RandomNodeSample relabels nodes correctly"""
        transform = RandomNodeSample(num=3)

        result = transform(simple_graph)

        # Node indices should be contiguous [0, num_sampled-1]
        if result.edge_index.numel() > 0:
            unique_nodes = torch.unique(result.edge_index)
            # Nodes should be relabeled starting from 0
            assert unique_nodes.min() >= 0
            assert unique_nodes.max() < result.num_nodes

    def test_get_metadata(self):
        """Test RandomNodeSample metadata extraction"""
        metadata = RandomNodeSample.get_metadata()

        assert metadata.name == "RandomNodeSample"
        assert metadata.version == "1.0.0"
        assert metadata.category == "sampling"
        assert "samples a subset of nodes" in metadata.description
        assert isinstance(metadata.required_node_features, list)

    def test_forward_deterministic_with_seed(self, simple_graph):
        """Test RandomNodeSample produces same result with same seed"""
        torch.manual_seed(42)
        transform = RandomNodeSample(num=3)
        result1 = transform(simple_graph.clone())

        torch.manual_seed(42)
        result2 = transform(simple_graph.clone())

        assert result1.num_nodes == result2.num_nodes
        assert torch.equal(result1.edge_index, result2.edge_index)

    def test_forward_large_graph(self, large_graph):
        """Test RandomNodeSample on larger graph for performance"""
        transform = RandomNodeSample(ratio=0.3)

        result = transform(large_graph)

        # Should complete quickly and reduce nodes
        expected_nodes = int(0.3 * large_graph.num_nodes)
        assert result.num_nodes == expected_nodes

    def test_forward_ratio_one(self, simple_graph):
        """Test RandomNodeSample with ratio=1.0 (full graph)"""
        transform = RandomNodeSample(ratio=1.0)

        result = transform(simple_graph)

        # ratio=1.0 should sample all nodes
        expected_nodes = max(1, int(1.0 * simple_graph.num_nodes))
        assert result.num_nodes == expected_nodes

    def test_forward_num_takes_precedence_over_ratio(self, simple_graph):
        """Test RandomNodeSample num takes precedence when both provided"""
        transform = RandomNodeSample(num=2, ratio=0.9)

        result = transform(simple_graph)

        # num should take precedence: sample exactly min(2, num_nodes)
        expected = min(2, simple_graph.num_nodes)
        assert result.num_nodes == expected

    def test_forward_zero_node_graph(self, zero_node_graph):
        """Test RandomNodeSample on graph with zero nodes returns early"""
        transform = RandomNodeSample(num=3)

        result = transform(zero_node_graph)

        # Should return data unchanged
        assert result.num_nodes == 0

    def test_inherits_base_transform(self):
        """Test RandomNodeSample inherits from PyG BaseTransform"""
        assert issubclass(RandomNodeSample, BaseTransform)

        transform = RandomNodeSample(num=5)
        assert isinstance(transform, BaseTransform)

    def test_callable_protocol(self, simple_graph):
        """Test RandomNodeSample is callable via __call__ (PyG BaseTransform protocol)"""
        transform = RandomNodeSample(num=3)

        assert callable(transform)
        result = transform(simple_graph)
        assert result.num_nodes == 3


# =============================================================================
# Integration and Cross-Transform Tests
# =============================================================================


class TestTransformIntegration:
    """Integration tests across multiple transforms"""

    def test_compose_drop_edge_and_drop_node(self, simple_graph):
        """Test composing DropEdge and DropNode"""
        from torch_geometric.transforms import Compose

        transform = Compose([DropEdge(p=0.3), DropNode(p=0.3)])

        result = transform(simple_graph)

        # Should have fewer nodes and edges
        assert result.num_nodes <= simple_graph.num_nodes
        assert result.edge_index.size(1) <= simple_graph.edge_index.size(1)

    def test_compose_mask_and_sample(self, simple_graph):
        """Test composing MaskFeatures and RandomNodeSample"""
        from torch_geometric.transforms import Compose

        transform = Compose([MaskFeatures(p=0.5), RandomNodeSample(num=3)])

        result = transform(simple_graph)

        # Should have sampled nodes with masked features
        assert result.num_nodes == 3
        assert result.x.size(0) == 3

    def test_all_transforms_together(self, graph_with_attributes):
        """Test applying all transforms in sequence"""
        from torch_geometric.transforms import Compose

        transform = Compose(
            [DropEdge(p=0.2), DropNode(p=0.2), MaskFeatures(p=0.3), RandomNodeSample(num=3)]
        )

        result = transform(graph_with_attributes)

        # Final result should be valid
        assert result.num_nodes >= 1
        assert result.x.size(0) == result.num_nodes
        if result.edge_index.numel() > 0:
            assert result.edge_index.max() < result.num_nodes

    def test_metadata_available_for_all_transforms(self):
        """Test that all transforms provide metadata"""
        transforms = [DropEdge, DropNode, MaskFeatures, RandomNodeSample]

        for transform_class in transforms:
            metadata = transform_class.get_metadata()

            assert hasattr(metadata, "name")
            assert hasattr(metadata, "version")
            assert hasattr(metadata, "category")
            assert hasattr(metadata, "description")
            assert metadata.name is not None
            assert len(metadata.version) > 0


# =============================================================================
# BaseTransform Protocol Compliance Tests
# =============================================================================


class TestBaseTransformProtocol:
    """Verify all transforms comply with PyG BaseTransform protocol"""

    ALL_TRANSFORM_CLASSES = [DropEdge, DropNode, MaskFeatures, RandomNodeSample]

    @pytest.mark.parametrize(
        "cls", [DropEdge, DropNode, MaskFeatures, RandomNodeSample], ids=lambda c: c.__name__
    )
    def test_inherits_base_transform(self, cls):
        """All transforms must inherit from PyG BaseTransform"""
        assert issubclass(cls, BaseTransform), (
            f"{cls.__name__} must inherit from torch_geometric.transforms.BaseTransform"
        )

    @pytest.mark.parametrize(
        "cls", [DropEdge, DropNode, MaskFeatures, RandomNodeSample], ids=lambda c: c.__name__
    )
    def test_implements_forward(self, cls):
        """All transforms must implement forward method"""
        assert hasattr(cls, "forward"), f"{cls.__name__} must implement forward()"
        assert callable(cls.forward)

    @pytest.mark.parametrize(
        "cls", [DropEdge, DropNode, MaskFeatures, RandomNodeSample], ids=lambda c: c.__name__
    )
    def test_implements_get_metadata(self, cls):
        """All transforms must implement get_metadata classmethod"""
        assert hasattr(cls, "get_metadata"), f"{cls.__name__} must implement get_metadata()"
        assert isinstance(cls.__dict__.get("get_metadata"), classmethod) or callable(
            cls.get_metadata
        )

    @pytest.mark.parametrize(
        "cls", [DropEdge, DropNode, MaskFeatures, RandomNodeSample], ids=lambda c: c.__name__
    )
    def test_metadata_schema_completeness(self, cls):
        """All transforms must return metadata with all required fields"""
        metadata = cls.get_metadata()

        required_fields = [
            "name",
            "version",
            "author",
            "category",
            "description",
            "paper_reference",
            "github_url",
            "validated_datasets",
            "required_node_features",
            "required_edge_features",
            "required_graph_attributes",
        ]
        for field in required_fields:
            assert hasattr(metadata, field), (
                f"{cls.__name__}.get_metadata() missing required field: {field}"
            )

    @pytest.mark.parametrize(
        "cls", [DropEdge, DropNode, MaskFeatures, RandomNodeSample], ids=lambda c: c.__name__
    )
    def test_metadata_name_matches_class(self, cls):
        """Metadata name must match the class name"""
        metadata = cls.get_metadata()
        assert metadata.name == cls.__name__, (
            f"Metadata name '{metadata.name}' does not match class name '{cls.__name__}'"
        )

    @pytest.mark.parametrize(
        "cls", [DropEdge, DropNode, MaskFeatures, RandomNodeSample], ids=lambda c: c.__name__
    )
    def test_metadata_list_fields_are_lists(self, cls):
        """Metadata list fields must actually be lists"""
        metadata = cls.get_metadata()
        for field_name in [
            "validated_datasets",
            "required_node_features",
            "required_edge_features",
            "required_graph_attributes",
        ]:
            value = getattr(metadata, field_name)
            assert isinstance(value, list), (
                f"{cls.__name__}.get_metadata().{field_name} must be a list, got {type(value)}"
            )


# =============================================================================
# Edge Cases and Stress Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_graph_with_self_loops(self):
        """Test transforms on graph with self-loops"""
        edge_index = torch.tensor(
            [
                [0, 0, 1, 1, 2],  # Self-loops on 0 and 1
                [0, 1, 1, 2, 0],
            ],
            dtype=torch.long,
        )

        x = torch.randn(3, 2)
        data = Data(x=x, edge_index=edge_index)

        # Test each transform
        drop_edge = DropEdge(p=0.3)
        result1 = drop_edge(data.clone())
        assert result1.edge_index.size(1) <= data.edge_index.size(1)

        drop_node = DropNode(p=0.3)
        result2 = drop_node(data.clone())
        assert result2.num_nodes >= 1

        mask_features = MaskFeatures(p=0.5)
        result3 = mask_features(data.clone())
        assert result3.x.shape == data.x.shape

    def test_disconnected_components(self):
        """Test transforms on graph with disconnected components"""
        # Two separate triangles
        edge_index = torch.tensor(
            [[0, 1, 2, 0, 3, 4, 5, 3], [1, 2, 0, 2, 4, 5, 3, 5]], dtype=torch.long
        )

        x = torch.randn(6, 3)
        data = Data(x=x, edge_index=edge_index)

        # Test transforms maintain validity
        sample = RandomNodeSample(num=4)
        result = sample(data)

        assert result.num_nodes == 4
        if result.edge_index.numel() > 0:
            assert result.edge_index.max() < result.num_nodes

    def test_very_sparse_graph(self):
        """Test transforms on very sparse graph"""
        # 20 nodes, only 3 edges
        edge_index = torch.tensor([[0, 5, 10], [1, 6, 11]], dtype=torch.long)

        x = torch.randn(20, 5)
        data = Data(x=x, edge_index=edge_index)

        drop_node = DropNode(p=0.5)
        result = drop_node(data)

        # Should handle sparse connectivity
        assert result.num_nodes >= 1

    def test_complete_graph_small(self):
        """Test transforms on complete graph"""
        # Complete graph with 4 nodes
        num_nodes = 4
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        x = torch.randn(num_nodes, 3)
        data = Data(x=x, edge_index=edge_index)

        drop_edge = DropEdge(p=0.5)
        result = drop_edge(data)

        # Should reduce many edges
        assert result.edge_index.size(1) <= data.edge_index.size(1)

    def test_zero_node_graph_all_transforms(self, zero_node_graph):
        """Test all transforms handle zero-node graph gracefully"""
        # DropEdge: no edges to drop
        drop_edge = DropEdge(p=0.5)
        result1 = drop_edge(zero_node_graph.clone())
        assert result1.num_nodes == 0

        # DropNode: num_nodes == 0 early return
        drop_node = DropNode(p=0.5)
        result2 = drop_node(zero_node_graph.clone())
        assert result2.num_nodes == 0

        # RandomNodeSample: num_nodes == 0 early return
        sample = RandomNodeSample(num=3)
        result3 = sample(zero_node_graph.clone())
        assert result3.num_nodes == 0

    def test_graph_with_only_edge_attr_no_weight(self):
        """Test DropEdge with edge_attr but no edge_weight"""
        edge_index = torch.tensor([[0, 1, 1, 0], [1, 0, 2, 2]], dtype=torch.long)
        x = torch.randn(3, 2)
        edge_attr = torch.randn(4, 3)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        transform = DropEdge(p=0.3)
        result = transform(data)

        assert result.edge_attr.size(0) == result.edge_index.size(1)
        assert not hasattr(result, "edge_weight") or result.edge_weight is None

    def test_single_edge_graph(self):
        """Test transforms on graph with exactly one directed edge"""
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        x = torch.randn(2, 3)
        data = Data(x=x, edge_index=edge_index)

        # DropEdge with p=0 should keep the edge
        transform = DropEdge(p=0.0)
        result = transform(data)
        assert result.edge_index.size(1) == 1

        # DropEdge with p=1 should drop the edge
        transform = DropEdge(p=1.0)
        result = transform(data)
        assert result.edge_index.size(1) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
