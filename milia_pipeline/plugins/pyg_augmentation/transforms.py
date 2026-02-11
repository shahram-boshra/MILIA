"""
PyG Augmentation Transform Implementations

These transforms provide data augmentation capabilities missing from PyG 2.6.1.
Standard PyG-style transforms that don't depend on Milia custom transform system.
"""

from torch_geometric.transforms import BaseTransform


class DropEdge(BaseTransform):
    """
    Randomly drops edges from the graph.
    
    Missing from PyG 2.6.1. Essential for graph contrastive learning.
    
    Args:
        p (float): Probability of dropping each edge (0-1). Default: 0.5
        force_undirected (bool): Maintain undirected property. Default: True
    
    Algorithm:
        1. Generate random mask for edges
        2. If force_undirected, drop edges symmetrically
        3. Apply mask to edge_index and attributes
    
    Complexity: O(E)
    """
    
    def __init__(self, p: float = 0.5, force_undirected: bool = True):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p
        self.force_undirected = force_undirected
    
    def forward(self, data):
        import torch
        
        if data.edge_index is None or data.edge_index.size(1) == 0:
            return data
        
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        
        # Create dropout mask (True = keep)
        mask = torch.rand(num_edges, device=edge_index.device) >= self.p
        
        if self.force_undirected and num_edges % 2 == 0:
            # Symmetric dropping for undirected graphs
            mask = mask.view(-1, 2).all(dim=1).repeat_interleave(2)
        
        # Apply mask
        data.edge_index = edge_index[:, mask]
        
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr[mask]
        
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            data.edge_weight = data.edge_weight[mask]
        
        return data
    
    @classmethod
    def get_metadata(cls):
        from collections import namedtuple
        Metadata = namedtuple('Metadata', [
            'name', 'version', 'author', 'category', 'description',
            'paper_reference', 'github_url', 'validated_datasets',
            'required_node_features', 'required_edge_features', 'required_graph_attributes'
        ])
        return Metadata(
            name='DropEdge',
            version='1.0.0',
            author='Shahram Boshra, Ilia Boshra',
            category='augmentation',
            description='Randomly drops edges for graph augmentation',
            paper_reference=None,
            github_url=None,
            validated_datasets=[],
            required_node_features=[],
            required_edge_features=[],
            required_graph_attributes=[]
        )


class DropNode(BaseTransform):
    """
    Randomly drops nodes and their incident edges.
    
    Missing from PyG 2.6.1. Used for structural augmentation.
    
    Args:
        p (float): Probability of dropping each node (0-1). Default: 0.5
    
    Algorithm:
        1. Generate random mask for nodes
        2. Ensure at least 1 node remains
        3. Extract subgraph with remaining nodes
        4. Relabel nodes and update features
    
    Complexity: O(V + E)
    """
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p
    
    def forward(self, data):
        import torch
        from torch_geometric.utils import subgraph
        
        num_nodes = data.num_nodes
        if num_nodes == 0:
            return data
        
        # Node mask (True = keep)
        device = data.x.device if data.x is not None else 'cpu'
        mask = torch.rand(num_nodes, device=device) >= self.p
        
        # Ensure at least one node
        if not mask.any():
            mask[torch.randint(0, num_nodes, (1,), device=device)] = True
        
        subset = mask.nonzero(as_tuple=False).view(-1)
        
        # Extract subgraph with relabeling
        edge_index, edge_attr = subgraph(
            subset,
            data.edge_index,
            data.edge_attr if hasattr(data, 'edge_attr') else None,
            relabel_nodes=True,
            num_nodes=num_nodes
        )
        
        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        
        # Update node features
        for key in ['x', 'pos', 'norm', 'y']:
            if hasattr(data, key) and getattr(data, key) is not None:
                setattr(data, key, getattr(data, key)[mask])
        
        return data
    
    @classmethod
    def get_metadata(cls):
        from collections import namedtuple
        Metadata = namedtuple('Metadata', [
            'name', 'version', 'author', 'category', 'description',
            'paper_reference', 'github_url', 'validated_datasets',
            'required_node_features', 'required_edge_features', 'required_graph_attributes'
        ])
        return Metadata(
            name='DropNode',
            version='1.0.0',
            author='Shahram Boshra, Ilia Boshra',
            category='augmentation',
            description='Randomly drops nodes and incident edges',
            paper_reference=None,
            github_url=None,
            validated_datasets=[],
            required_node_features=[],
            required_edge_features=[],
            required_graph_attributes=[]
        )


class MaskFeatures(BaseTransform):
    """
    Randomly masks node features.
    
    Missing from PyG 2.6.1. Used for feature-level augmentation.
    
    Args:
        p (float): Probability of masking each feature (0-1). Default: 0.5
        mask_value (float): Value for masked features. Default: 0.0
    
    Algorithm:
        1. Generate random mask for feature elements
        2. Replace masked elements with mask_value
    
    Complexity: O(V × F)
    """
    
    def __init__(self, p: float = 0.5, mask_value: float = 0.0):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p
        self.mask_value = mask_value
    
    def forward(self, data):
        import torch
        
        if data.x is None:
            return data
        
        # Element-wise masking
        mask = torch.rand_like(data.x) < self.p
        
        data.x = data.x.clone()
        data.x[mask] = self.mask_value
        
        return data
    
    @classmethod
    def get_metadata(cls):
        from collections import namedtuple
        Metadata = namedtuple('Metadata', [
            'name', 'version', 'author', 'category', 'description',
            'paper_reference', 'github_url', 'validated_datasets',
            'required_node_features', 'required_edge_features', 'required_graph_attributes'
        ])
        return Metadata(
            name='MaskFeatures',
            version='1.0.0',
            author='Shahram Boshra, Ilia Boshra',
            category='augmentation',
            description='Randomly masks node features',
            paper_reference=None,
            github_url=None,
            validated_datasets=[],
            required_node_features=[],
            required_edge_features=[],
            required_graph_attributes=[]
        )


class RandomNodeSample(BaseTransform):
    """
    Randomly samples nodes from the graph.
    
    Missing from PyG 2.6.1. Used for subgraph sampling.
    
    Args:
        num (int): Absolute number of nodes to sample. Takes precedence.
        ratio (float): Fraction of nodes to sample (0-1). Used if num is None.
    
    Algorithm:
        1. Determine sample size (num or ratio)
        2. Random permutation of nodes
        3. Select first k nodes
        4. Extract subgraph
    
    Complexity: O(V + E)
    """
    
    def __init__(self, num: int = None, ratio: float = None):
        super().__init__()
        if num is None and ratio is None:
            raise ValueError("Either 'num' or 'ratio' must be specified")
        if num is not None and num <= 0:
            raise ValueError(f"num must be positive, got {num}")
        if ratio is not None and not 0 < ratio <= 1:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        
        self.num = num
        self.ratio = ratio
    
    def forward(self, data):
        import torch
        from torch_geometric.utils import subgraph
        
        num_nodes = data.num_nodes
        if num_nodes == 0:
            return data
        
        # Determine sample size
        if self.num is not None:
            sample_size = min(self.num, num_nodes)
        else:
            sample_size = max(1, int(self.ratio * num_nodes))
        
        # Random sampling
        perm = torch.randperm(num_nodes)[:sample_size]
        
        # Extract subgraph
        edge_index, edge_attr = subgraph(
            perm,
            data.edge_index,
            data.edge_attr if hasattr(data, 'edge_attr') else None,
            relabel_nodes=True,
            num_nodes=num_nodes
        )
        
        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        
        # Update node features
        for key in ['x', 'pos', 'norm', 'y']:
            if hasattr(data, key) and getattr(data, key) is not None:
                setattr(data, key, getattr(data, key)[perm])
        
        return data
    
    @classmethod
    def get_metadata(cls):
        from collections import namedtuple
        Metadata = namedtuple('Metadata', [
            'name', 'version', 'author', 'category', 'description',
            'paper_reference', 'github_url', 'validated_datasets',
            'required_node_features', 'required_edge_features', 'required_graph_attributes'
        ])
        return Metadata(
            name='RandomNodeSample',
            version='1.0.0',
            author='Shahram Boshra, Ilia Boshra',
            category='sampling',
            description='Randomly samples a subset of nodes',
            paper_reference=None,
            github_url=None,
            validated_datasets=[],
            required_node_features=[],
            required_edge_features=[],
            required_graph_attributes=[]
        )
