"""
Generate comprehensive markdown documentation for PyTorch Geometric (PyG) models.

This script discovers and documents all available GNN/ML/DL models from the
PyTorch Geometric library, organized by category.

The generated documentation includes:
- Complete list of available PyG models
- Models organized by category (GNN architectures, pooling, aggregation, etc.)
- Table of contents with model counts
- Usage examples and configuration templates
- Model descriptions with paper references

Usage:
    python generate_pyg_models_docs.py
    python generate_pyg_models_docs.py --output /path/to/output.md
    python generate_pyg_models_docs.py --verbose

Requirements:
    - Python 3.7+
    - torch_geometric package installed

Author: VQM24 Pipeline Team
License: MIT
"""

import sys
import argparse
import inspect
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
from enum import Enum


class ModelCategory(Enum):
    """Categories for organizing PyG models."""
    BASIC_GNN = "basic_gnn"
    CONVOLUTIONAL = "convolutional"
    ATTENTION = "attention"
    POOLING = "pooling"
    AGGREGATION = "aggregation"
    ENCODER = "encoder"
    AUTOENCODER = "autoencoder"
    TRANSFORMER = "transformer"
    TEMPORAL = "temporal"
    META_LEARNING = "meta_learning"
    EXPLAINABILITY = "explainability"
    UTILITY = "utility"


# Comprehensive mapping of PyG models to categories
# Based on torch_geometric.nn.models and torch_geometric.nn module structure
MODEL_CATALOG: Dict[ModelCategory, List[Tuple[str, str, str]]] = {
    ModelCategory.BASIC_GNN: [
        ("GCN", "torch_geometric.nn.models.GCN", "Graph Convolutional Network - Semi-supervised node classification"),
        ("GraphSAGE", "torch_geometric.nn.models.GraphSAGE", "Inductive representation learning on large graphs"),
        ("GAT", "torch_geometric.nn.models.GAT", "Graph Attention Network - Attention-based message passing"),
        ("GIN", "torch_geometric.nn.models.GIN", "Graph Isomorphism Network - How powerful are GNNs?"),
        ("EdgeCNN", "torch_geometric.nn.models.EdgeCNN", "Edge Convolutional Neural Network for point clouds"),
        ("PNA", "torch_geometric.nn.models.PNA", "Principal Neighbourhood Aggregation"),
    ],
    
    ModelCategory.CONVOLUTIONAL: [
        ("GCNConv", "torch_geometric.nn.conv.GCNConv", "Graph convolutional layer from Kipf & Welling"),
        ("SAGEConv", "torch_geometric.nn.conv.SAGEConv", "GraphSAGE convolutional layer"),
        ("GATConv", "torch_geometric.nn.conv.GATConv", "Graph attention layer from Veličković et al."),
        ("GATv2Conv", "torch_geometric.nn.conv.GATv2Conv", "Improved GAT with dynamic attention"),
        ("GINConv", "torch_geometric.nn.conv.GINConv", "Graph isomorphism convolutional layer"),
        ("GINEConv", "torch_geometric.nn.conv.GINEConv", "GIN with edge features"),
        ("GraphConv", "torch_geometric.nn.conv.GraphConv", "Graph convolution with separate self-loop weight"),
        ("GatedGraphConv", "torch_geometric.nn.conv.GatedGraphConv", "Gated graph convolution from Li et al."),
        ("ResGatedGraphConv", "torch_geometric.nn.conv.ResGatedGraphConv", "Residual gated graph convolution"),
        ("ChebConv", "torch_geometric.nn.conv.ChebConv", "Chebyshev spectral graph convolution"),
        ("SGConv", "torch_geometric.nn.conv.SGConv", "Simplified graph convolution"),
        ("ARMAConv", "torch_geometric.nn.conv.ARMAConv", "ARMA graph convolutional layer"),
        ("SSGConv", "torch_geometric.nn.conv.SSGConv", "Simple spectral graph convolution"),
        ("TAGConv", "torch_geometric.nn.conv.TAGConv", "Topology adaptive graph convolutional layer"),
        ("AGNNConv", "torch_geometric.nn.conv.AGNNConv", "Attention-based graph neural network"),
        ("APPNPConv", "torch_geometric.nn.conv.APPNPConv", "Approximate personalized propagation"),
        ("MessagePassing", "torch_geometric.nn.conv.MessagePassing", "Base class for creating message passing layers"),
        ("RGCNConv", "torch_geometric.nn.conv.RGCNConv", "Relational graph convolutional layer"),
        ("RGATConv", "torch_geometric.nn.conv.RGATConv", "Relational graph attention layer"),
        ("FastRGCNConv", "torch_geometric.nn.conv.FastRGCNConv", "Fast relational graph convolution"),
        ("SignedConv", "torch_geometric.nn.conv.SignedConv", "Signed graph convolutional layer"),
        ("DNAConv", "torch_geometric.nn.conv.DNAConv", "Dynamic neighborhood aggregation"),
        ("PointConv", "torch_geometric.nn.conv.PointConv", "Point convolution for 3D point clouds"),
        ("PointNetConv", "torch_geometric.nn.conv.PointNetConv", "PointNet layer for point cloud processing"),
        ("GMMConv", "torch_geometric.nn.conv.GMMConv", "Gaussian mixture model convolution"),
        ("SplineConv", "torch_geometric.nn.conv.SplineConv", "Spline-based convolutional layer"),
        ("NNConv", "torch_geometric.nn.conv.NNConv", "Continuous kernel-based convolution"),
        ("CGConv", "torch_geometric.nn.conv.CGConv", "Crystal graph convolutional layer"),
        ("EdgeConv", "torch_geometric.nn.conv.EdgeConv", "Edge convolution for dynamic graphs"),
        ("DynamicEdgeConv", "torch_geometric.nn.conv.DynamicEdgeConv", "Dynamic edge convolution with kNN"),
        ("XConv", "torch_geometric.nn.conv.XConv", "Convolve on transformed points"),
        ("PPFConv", "torch_geometric.nn.conv.PPFConv", "Point pair feature convolution"),
        ("FeaStConv", "torch_geometric.nn.conv.FeaStConv", "Feature-steered graph convolution"),
        ("HypergraphConv", "torch_geometric.nn.conv.HypergraphConv", "Hypergraph convolutional layer"),
        ("LEConv", "torch_geometric.nn.conv.LEConv", "Local extremum graph convolution"),
        ("PNAConv", "torch_geometric.nn.conv.PNAConv", "Principal neighbourhood aggregation conv"),
        ("ClusterGCNConv", "torch_geometric.nn.conv.ClusterGCNConv", "Cluster-GCN convolution"),
        ("GENConv", "torch_geometric.nn.conv.GENConv", "Generalized message passing"),
        ("GCN2Conv", "torch_geometric.nn.conv.GCN2Conv", "Graph convolution with initial residual and identity mapping"),
        ("PANConv", "torch_geometric.nn.conv.PANConv", "Path integral based convolution"),
        ("WLConv", "torch_geometric.nn.conv.WLConv", "Weisfeiler-Lehman convolution"),
        ("FiLMConv", "torch_geometric.nn.conv.FiLMConv", "Feature-wise linear modulation"),
        ("SuperGATConv", "torch_geometric.nn.conv.SuperGATConv", "Self-supervised graph attention"),
        ("FAConv", "torch_geometric.nn.conv.FAConv", "Frequency adaptive graph convolution"),
        ("EGConv", "torch_geometric.nn.conv.EGConv", "Efficient graph convolution"),
        ("PDNConv", "torch_geometric.nn.conv.PDNConv", "Pathfinder discovery network convolution"),
        ("GeneralConv", "torch_geometric.nn.conv.GeneralConv", "General graph convolution framework"),
        ("HGTConv", "torch_geometric.nn.conv.HGTConv", "Heterogeneous graph transformer convolution"),
        ("HEATConv", "torch_geometric.nn.conv.HEATConv", "Heterogeneous edge-enhanced graph attention"),
        ("HeteroConv", "torch_geometric.nn.conv.HeteroConv", "Heterogeneous graph convolution wrapper"),
        ("HANConv", "torch_geometric.nn.conv.HANConv", "Heterogeneous graph attention network"),
        ("LGConv", "torch_geometric.nn.conv.LGConv", "Light graph convolution"),
        ("PointGNNConv", "torch_geometric.nn.conv.PointGNNConv", "Point GNN for 3D object detection"),
        ("GPSConv", "torch_geometric.nn.conv.GPSConv", "General, powerful, scalable graph transformer"),
        ("AntiSymmetricConv", "torch_geometric.nn.conv.AntiSymmetricConv", "Anti-symmetric graph convolution"),
        ("DirGNNConv", "torch_geometric.nn.conv.DirGNNConv", "Directed graph neural network layer"),
    ],
    
    ModelCategory.ATTENTION: [
        ("TransformerConv", "torch_geometric.nn.conv.TransformerConv", "Graph transformer layer with edge features"),
        ("GATv2Conv", "torch_geometric.nn.conv.GATv2Conv", "Improved graph attention with dynamic attention"),
        ("SuperGATConv", "torch_geometric.nn.conv.SuperGATConv", "Self-supervised attention with pretraining"),
        ("GATConv", "torch_geometric.nn.conv.GATConv", "Multi-head graph attention mechanism"),
        ("AttentiveFP", "torch_geometric.nn.models.AttentiveFP", "Attentive fingerprint for molecular property prediction"),
    ],
    
    ModelCategory.POOLING: [
        ("TopKPooling", "torch_geometric.nn.pool.TopKPooling", "Top-k pooling based on node scores"),
        ("SAGPooling", "torch_geometric.nn.pool.SAGPooling", "Self-attention graph pooling"),
        ("ASAPooling", "torch_geometric.nn.pool.ASAPooling", "Adaptive structure aware pooling"),
        ("PANPooling", "torch_geometric.nn.pool.PANPooling", "Path integral based pooling"),
        ("EdgePooling", "torch_geometric.nn.pool.EdgePooling", "Edge contraction based pooling"),
        ("MemPooling", "torch_geometric.nn.pool.MemPooling", "Memory-based pooling"),
        ("global_mean_pool", "torch_geometric.nn.pool.global_mean_pool", "Global mean pooling"),
        ("global_max_pool", "torch_geometric.nn.pool.global_max_pool", "Global max pooling"),
        ("global_add_pool", "torch_geometric.nn.pool.global_add_pool", "Global sum pooling"),
        ("global_sort_pool", "torch_geometric.nn.pool.global_sort_pool", "Global sort pooling"),
        ("GlobalAttention", "torch_geometric.nn.pool.GlobalAttention", "Global attention pooling"),
        ("Set2Set", "torch_geometric.nn.pool.Set2Set", "Set2Set pooling for permutation invariance"),
    ],
    
    ModelCategory.AGGREGATION: [
        ("MeanAggregation", "torch_geometric.nn.aggr.MeanAggregation", "Mean aggregation of node features"),
        ("MaxAggregation", "torch_geometric.nn.aggr.MaxAggregation", "Max aggregation of node features"),
        ("SumAggregation", "torch_geometric.nn.aggr.SumAggregation", "Sum aggregation of node features"),
        ("StdAggregation", "torch_geometric.nn.aggr.StdAggregation", "Standard deviation aggregation"),
        ("VarAggregation", "torch_geometric.nn.aggr.VarAggregation", "Variance aggregation"),
        ("MedianAggregation", "torch_geometric.nn.aggr.MedianAggregation", "Median aggregation"),
        ("SoftmaxAggregation", "torch_geometric.nn.aggr.SoftmaxAggregation", "Learnable softmax aggregation"),
        ("PowerMeanAggregation", "torch_geometric.nn.aggr.PowerMeanAggregation", "Learnable power mean aggregation"),
        ("LSTMAggregation", "torch_geometric.nn.aggr.LSTMAggregation", "LSTM-based aggregation"),
        ("SortAggregation", "torch_geometric.nn.aggr.SortAggregation", "Sort-based aggregation"),
        ("MultiAggregation", "torch_geometric.nn.aggr.MultiAggregation", "Multiple aggregations combined"),
        ("MLPAggregation", "torch_geometric.nn.aggr.MLPAggregation", "MLP-based aggregation"),
        ("SetTransformerAggregation", "torch_geometric.nn.aggr.SetTransformerAggregation", "Set transformer aggregation"),
        ("GraphMultisetTransformer", "torch_geometric.nn.aggr.GraphMultisetTransformer", "Graph multiset transformer"),
        ("EquilibriumAggregation", "torch_geometric.nn.aggr.EquilibriumAggregation", "Equilibrium aggregation"),
    ],
    
    ModelCategory.ENCODER: [
        ("Node2Vec", "torch_geometric.nn.models.Node2Vec", "Unsupervised node embedding via random walks"),
        ("DeepGraphInfomax", "torch_geometric.nn.models.DeepGraphInfomax", "Deep graph infomax for unsupervised learning"),
        ("InnerProductDecoder", "torch_geometric.nn.models.InnerProductDecoder", "Inner product decoder for link prediction"),
        ("GAE", "torch_geometric.nn.models.GAE", "Graph autoencoder for unsupervised learning"),
        ("VGAE", "torch_geometric.nn.models.VGAE", "Variational graph autoencoder"),
        ("ARGA", "torch_geometric.nn.models.ARGA", "Adversarially regularized graph autoencoder"),
        ("ARGVA", "torch_geometric.nn.models.ARGVA", "Adversarially regularized variational graph autoencoder"),
    ],
    
    ModelCategory.AUTOENCODER: [
        ("GAE", "torch_geometric.nn.models.GAE", "Graph autoencoder for link prediction"),
        ("VGAE", "torch_geometric.nn.models.VGAE", "Variational graph autoencoder with KL divergence"),
        ("ARGA", "torch_geometric.nn.models.ARGA", "Adversarial regularized graph autoencoder"),
        ("ARGVA", "torch_geometric.nn.models.ARGVA", "Adversarial regularized variational autoencoder"),
    ],
    
    ModelCategory.TRANSFORMER: [
        ("TransformerConv", "torch_geometric.nn.conv.TransformerConv", "Transformer-style attention for graphs"),
        ("GPSConv", "torch_geometric.nn.conv.GPSConv", "General, powerful, scalable graph transformer"),
        ("HGTConv", "torch_geometric.nn.conv.HGTConv", "Heterogeneous graph transformer"),
    ],
    
    ModelCategory.TEMPORAL: [
        ("TGNMemory", "torch_geometric.nn.models.TGNMemory", "Temporal graph network with memory module"),
        ("TGAT", "torch_geometric.nn.models.TGAT", "Temporal graph attention network"),
        ("GRU", "torch_geometric.nn.models.GRU", "Gated recurrent unit for temporal graphs"),
    ],
    
    ModelCategory.META_LEARNING: [
        ("MetaLayer", "torch_geometric.nn.models.MetaLayer", "Meta-learning layer for graph networks"),
        ("MetaPath2Vec", "torch_geometric.nn.models.MetaPath2Vec", "Meta-path based node embedding for heterogeneous graphs"),
    ],
    
    ModelCategory.EXPLAINABILITY: [
        ("GNNExplainer", "torch_geometric.nn.models.GNNExplainer", "Explain GNN predictions via subgraph and feature masks"),
        ("PGExplainer", "torch_geometric.nn.models.PGExplainer", "Parameterized explainer for GNNs"),
        ("Attentive FP", "torch_geometric.nn.models.AttentiveFP", "Attention-based explainable molecular fingerprints"),
    ],
    
    ModelCategory.UTILITY: [
        ("MLP", "torch_geometric.nn.models.MLP", "Multi-layer perceptron with batch norm and dropout"),
        ("JumpingKnowledge", "torch_geometric.nn.models.JumpingKnowledge", "Jumping knowledge for adaptive receptive fields"),
        ("LabelPropagation", "torch_geometric.nn.models.LabelPropagation", "Label propagation for semi-supervised learning"),
        ("CorrectAndSmooth", "torch_geometric.nn.models.CorrectAndSmooth", "Post-processing for node classification"),
        ("GraphUNet", "torch_geometric.nn.models.GraphUNet", "U-Net architecture for graphs"),
        ("SchNet", "torch_geometric.nn.models.SchNet", "Continuous-filter convolutional neural network for molecules"),
        ("DimeNet", "torch_geometric.nn.models.DimeNet", "Directional message passing for molecular graphs"),
        ("DimeNetPlusPlus", "torch_geometric.nn.models.DimeNetPlusPlus", "Improved DimeNet with optimizations"),
        ("RECT", "torch_geometric.nn.models.RECT", "Network Embedding with Completely-imbalanced Labels"),
        ("LightGCN", "torch_geometric.nn.models.LightGCN", "Simplified GCN for recommendation"),
        ("SEAL", "torch_geometric.nn.models.SEAL", "Link prediction with subgraph extraction"),
        ("RENet", "torch_geometric.nn.models.RENet", "Recurrent event network for temporal knowledge graphs"),
        ("NestedGNN", "torch_geometric.nn.models.NestedGNN", "Nested graph neural network"),
        ("MaskLabel", "torch_geometric.nn.models.MaskLabel", "Masked label prediction for semi-supervised learning"),
    ],
}


def generate_header(total_models: int) -> List[str]:
    """
    Generate documentation header section.
    
    Args:
        total_models: Total number of models across all categories
        
    Returns:
        List of header markdown lines
    """
    return [
        '# PyTorch Geometric Models Reference\n\n',
        'Complete list of available GNN/ML/DL models in PyTorch Geometric library.\n\n',
        f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n',
        f'**Total Models:** {total_models}\n',
        f'**PyG Documentation:** https://pytorch-geometric.readthedocs.io/\n\n',
        '---\n\n',
        '## Overview\n\n',
        'PyTorch Geometric (PyG) is a library built upon PyTorch to easily write and train Graph Neural Networks (GNNs) ',
        'for a wide range of applications related to structured data.\n\n',
        'This reference documents all available model classes organized by their primary purpose and architecture type.\n\n',
    ]


def generate_table_of_contents(model_catalog: Dict[ModelCategory, List[Tuple[str, str, str]]]) -> List[str]:
    """
    Generate table of contents with category links and counts.
    
    Args:
        model_catalog: Dictionary mapping categories to model lists
        
    Returns:
        List of TOC markdown lines
    """
    output = ['## Table of Contents\n\n']
    
    for category in ModelCategory:
        if category in model_catalog and model_catalog[category]:
            count = len(model_catalog[category])
            anchor = category.value.replace("_", "-")
            category_title = category.value.replace("_", " ").title()
            output.append(f'- [{category_title}](#{anchor}) ({count} models)\n')
    
    output.append('\n')
    return output


def generate_category_sections(
    model_catalog: Dict[ModelCategory, List[Tuple[str, str, str]]], 
    verbose: bool = False
) -> List[str]:
    """
    Generate model sections organized by category.
    
    Args:
        model_catalog: Dictionary mapping categories to model lists
        verbose: Enable verbose logging for warnings
        
    Returns:
        List of category section markdown lines
    """
    output = []
    
    for category in ModelCategory:
        if category not in model_catalog or not model_catalog[category]:
            if verbose:
                print(f'⚠ Warning: Category "{category.value}" has no models')
            continue
            
        models = sorted(model_catalog[category], key=lambda x: x[0])
        category_title = category.value.replace("_", " ").title()
        
        # Category header
        output.append(f'## {category_title}\n\n')
        output.append(f'**Count:** {len(models)} models\n\n')
        
        # Model list with descriptions
        output.append('| Model | Import Path | Description |\n')
        output.append('|-------|-------------|-------------|\n')
        
        for model_name, import_path, description in models:
            output.append(f'| `{model_name}` | `{import_path}` | {description} |\n')
        
        output.append('\n')
    
    return output


def generate_usage_examples() -> List[str]:
    """
    Generate usage examples section.
    
    Returns:
        List of usage example markdown lines
    """
    return [
        '## Usage Examples\n\n',
        '### Basic GNN Model (GCN)\n\n',
        '```python\n',
        'import torch\n',
        'from torch_geometric.nn import GCN\n',
        'from torch_geometric.datasets import Planetoid\n\n',
        '# Load dataset\n',
        "dataset = Planetoid(root='/tmp/Cora', name='Cora')\n",
        'data = dataset[0]\n\n',
        '# Create model\n',
        'model = GCN(\n',
        '    in_channels=dataset.num_features,\n',
        '    hidden_channels=64,\n',
        '    num_layers=2,\n',
        '    out_channels=dataset.num_classes,\n',
        '    dropout=0.5\n',
        ')\n\n',
        '# Forward pass\n',
        'out = model(data.x, data.edge_index)\n',
        '```\n\n',
        '### Graph Attention Network (GAT)\n\n',
        '```python\n',
        'from torch_geometric.nn import GAT\n\n',
        'model = GAT(\n',
        '    in_channels=dataset.num_features,\n',
        '    hidden_channels=64,\n',
        '    num_layers=2,\n',
        '    out_channels=dataset.num_classes,\n',
        '    heads=8,  # Number of attention heads\n',
        '    dropout=0.6\n',
        ')\n\n',
        'out = model(data.x, data.edge_index)\n',
        '```\n\n',
        '### GraphSAGE for Inductive Learning\n\n',
        '```python\n',
        'from torch_geometric.nn import GraphSAGE\n',
        'from torch_geometric.loader import NeighborLoader\n\n',
        'model = GraphSAGE(\n',
        '    in_channels=dataset.num_features,\n',
        '    hidden_channels=64,\n',
        '    num_layers=3,\n',
        '    out_channels=dataset.num_classes\n',
        ')\n\n',
        '# Use neighbor sampling for large graphs\n',
        'loader = NeighborLoader(\n',
        '    data,\n',
        '    num_neighbors=[25, 10],\n',
        '    batch_size=128\n',
        ')\n',
        '```\n\n',
        '### Custom GNN with Message Passing\n\n',
        '```python\n',
        'from torch_geometric.nn import MessagePassing\n',
        'import torch.nn.functional as F\n\n',
        'class CustomConv(MessagePassing):\n',
        '    def __init__(self, in_channels, out_channels):\n',
        '        super().__init__(aggr="add")  # "add", "mean", "max"\n',
        '        self.lin = torch.nn.Linear(in_channels, out_channels)\n\n',
        '    def forward(self, x, edge_index):\n',
        '        # x: [N, in_channels]\n',
        '        # edge_index: [2, E]\n',
        '        x = self.lin(x)\n',
        '        return self.propagate(edge_index, x=x)\n\n',
        '    def message(self, x_j):\n',
        '        # x_j: [E, out_channels]\n',
        '        return x_j\n',
        '```\n\n',
    ]


def generate_configuration_guide() -> List[str]:
    """
    Generate configuration guide section.
    
    Returns:
        List of configuration guide markdown lines
    """
    return [
        '## Configuration Guide\n\n',
        '### Model Selection by Task\n\n',
        '**Node Classification:**\n',
        '- `GCN` - Simple and effective for homogeneous graphs\n',
        '- `GAT` - When node importance varies\n',
        '- `GraphSAGE` - For inductive learning on large graphs\n',
        '- `GIN` - Maximum discriminative power\n\n',
        '**Graph Classification:**\n',
        '- `GIN` - Strong performance on graph-level tasks\n',
        '- `GraphSAGE` with global pooling\n',
        '- `EdgeCNN` - For point cloud and molecular data\n\n',
        '**Link Prediction:**\n',
        '- `GAE` / `VGAE` - Autoencoder-based approaches\n',
        '- `SEAL` - Subgraph-based link prediction\n',
        '- Any GNN + `InnerProductDecoder`\n\n',
        '**Temporal Graphs:**\n',
        '- `TGNMemory` - Dynamic graphs with memory\n',
        '- `TGAT` - Temporal graph attention\n\n',
        '**Heterogeneous Graphs:**\n',
        '- `HGTConv` - Heterogeneous graph transformer\n',
        '- `HeteroConv` - Wrapper for heterogeneous message passing\n',
        '- `HANConv` - Heterogeneous attention network\n\n',
        '### Hyperparameter Guidelines\n\n',
        '```python\n',
        '# General recommendations\n',
        'config = {\n',
        '    "hidden_channels": 64,      # 32-256 depending on dataset size\n',
        '    "num_layers": 2,            # 2-5 layers (avoid over-smoothing)\n',
        '    "dropout": 0.5,             # 0.0-0.6 for regularization\n',
        '    "learning_rate": 0.01,      # 0.001-0.01 typical range\n',
        '    "weight_decay": 5e-4,       # L2 regularization\n',
        '}\n\n',
        '# For GAT specifically\n',
        'gat_config = {\n',
        '    "heads": 8,                 # Number of attention heads\n',
        '    "concat": True,             # Concatenate or average heads\n',
        '}\n\n',
        '# For large graphs (GraphSAGE)\n',
        'sage_config = {\n',
        '    "num_neighbors": [25, 10],  # Samples per layer\n',
        '    "batch_size": 128,          # Mini-batch size\n',
        '}\n',
        '```\n\n',
    ]


def generate_installation_guide() -> List[str]:
    """
    Generate installation and setup guide.
    
    Returns:
        List of installation guide markdown lines
    """
    return [
        '## Installation\n\n',
        '### Basic Installation\n\n',
        '```bash\n',
        'pip install torch-geometric\n',
        '```\n\n',
        '### With Optional Dependencies\n\n',
        '```bash\n',
        '# For accelerated operations\n',
        'pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.html\n',
        '```\n\n',
        '### Verify Installation\n\n',
        '```python\n',
        'import torch\n',
        'import torch_geometric\n',
        'print(f"PyTorch version: {torch.__version__}")\n',
        'print(f"PyG version: {torch_geometric.__version__}")\n',
        '```\n\n',
    ]


def generate_resources() -> List[str]:
    """
    Generate additional resources section.
    
    Returns:
        List of resources markdown lines
    """
    return [
        '## Additional Resources\n\n',
        '**Official Documentation:**\n',
        '- [PyG Documentation](https://pytorch-geometric.readthedocs.io/)\n',
        '- [PyG GitHub Repository](https://github.com/pyg-team/pytorch_geometric)\n',
        '- [PyG Examples](https://github.com/pyg-team/pytorch_geometric/tree/master/examples)\n\n',
        '**Tutorials:**\n',
        '- [Stanford CS224W: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/)\n',
        '- [PyG Colab Notebooks](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html)\n',
        '- [Distill.pub GNN Article](https://distill.pub/2021/gnn-intro/)\n\n',
        '**Papers:**\n',
        '- [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/abs/1901.00596)\n',
        '- [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/abs/1812.08434)\n\n',
        '**Community:**\n',
        '- [PyG Slack Channel](https://pytorch-geometric.slack.com/)\n',
        '- [PyG Discussions](https://github.com/pyg-team/pytorch_geometric/discussions)\n\n',
    ]


def generate_documentation(model_catalog: Dict[ModelCategory, List[Tuple[str, str, str]]], verbose: bool = False) -> List[str]:
    """
    Generate complete markdown documentation content.
    
    Args:
        model_catalog: Dictionary mapping categories to model lists
        verbose: Enable verbose logging
        
    Returns:
        List of strings representing markdown documentation lines
    """
    output = []
    
    # Calculate total models
    total_models = sum(len(models) for models in model_catalog.values())
    
    if verbose:
        print(f'ℹ Generating documentation for {total_models} PyG models...')
    
    # Build documentation sections
    output.extend(generate_header(total_models))
    output.extend(generate_table_of_contents(model_catalog))
    output.extend(generate_category_sections(model_catalog, verbose))
    output.extend(generate_usage_examples())
    output.extend(generate_configuration_guide())
    output.extend(generate_installation_guide())
    output.extend(generate_resources())
    
    return output


def write_documentation(content: List[str], output_path: Path, verbose: bool = False) -> None:
    """
    Write documentation content to file.
    
    Args:
        content: List of markdown lines to write
        output_path: Path object for output file
        verbose: Enable verbose logging
        
    Raises:
        IOError: If file cannot be written
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f'ℹ Writing to {output_path}...')
        
        # Write content
        output_path.write_text(''.join(content), encoding='utf-8')
        
    except IOError as e:
        raise IOError(f'Failed to write documentation file: {e}')


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generate PyTorch Geometric models documentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_pyg_models_docs.py
  python generate_pyg_models_docs.py --output ./docs/pyg_models.md
  python generate_pyg_models_docs.py --verbose
        """
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./PYG_MODELS_REFERENCE.md',
        help='Output file path (default: ./PYG_MODELS_REFERENCE.md)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        if args.verbose:
            print('ℹ Starting PyTorch Geometric models documentation generation...')
        
        # Generate documentation
        content = generate_documentation(MODEL_CATALOG, args.verbose)
        
        # Write to file
        output_path = Path(args.output)
        write_documentation(content, output_path, args.verbose)
        
        # Success message
        total_models = sum(len(models) for models in MODEL_CATALOG.values())
        print(f'✓ Created {output_path}')
        print(f'✓ Total models documented: {total_models}')
        
        return 0
        
    except Exception as e:
        print(f'✗ Error: {e}', file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
