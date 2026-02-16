"""
Architecture Templates

Pre-built common GNN architectures for quick start and experimentation.
All templates return ArchitectureBuilder instances that can be further customized.

Author: Milia Team
Version: 1.0.0
"""

import logging
from typing import Any

from .architecture_builder import ArchitectureBuilder

logger = logging.getLogger(__name__)


# =============================================================================
# ARCHITECTURE TEMPLATES
# =============================================================================


class ArchitectureTemplates:
    """
    Collection of pre-built architecture templates.

    Templates are parameterized and return ArchitectureBuilder instances
    ready for customization or immediate building.

    All templates support all 6 task types:
    - node_regression, node_classification
    - graph_regression, graph_classification
    - link_prediction, edge_regression

    Usage:
        >>> # Create template
        >>> builder = ArchitectureTemplates.simple_gcn(16, 1, num_layers=3)
        >>>
        >>> # Optionally customize further
        >>> builder.add_layer('Dropout', p=0.5)
        >>>
        >>> # Build model
        >>> model = builder.build()

        >>> # Or use directly
        >>> model = ArchitectureTemplates.attention_network(16, 1).build()
    """

    # =========================================================================
    # BASIC TEMPLATES
    # =========================================================================

    @staticmethod
    def simple_gcn(
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
        hidden_channels: int = 64,
        dropout: float = 0.0,
        task_type: str = "graph_regression",
    ) -> ArchitectureBuilder:
        """
        Simple GCN architecture.

        Structure: (Conv → ReLU → Dropout) × N → Pool → Linear

        Best for:
        - Quick prototyping
        - Baseline models
        - Small to medium graphs
        - When interpretability is important

        Args:
            in_channels: Input feature dimension
            out_channels: Output dimension
            num_layers: Number of GCN layers (default: 3)
            hidden_channels: Hidden dimension (default: 64)
            dropout: Dropout rate (default: 0.0)
            task_type: Task type (default: "graph_regression")

        Returns:
            ArchitectureBuilder instance configured with simple GCN

        Examples:
            >>> # Graph-level regression
            >>> builder = ArchitectureTemplates.simple_gcn(16, 1, num_layers=3)
            >>> model = builder.build()

            >>> # Node-level classification with dropout
            >>> builder = ArchitectureTemplates.simple_gcn(
            ...     32, 10, num_layers=2, dropout=0.5,
            ...     task_type="node_classification"
            ... )
            >>> model = builder.build()
        """
        builder = ArchitectureBuilder(task_type, in_channels, out_channels, name="SimpleGCN")

        # Add GCN layers
        for _i in range(num_layers):
            builder.add_layer("GCNConv", out_channels=hidden_channels)
            builder.add_layer("ReLU")
            if dropout > 0:
                builder.add_layer("Dropout", p=dropout)

        # Add output layers
        if "graph" in task_type:
            builder.add_layer("global_mean_pool")

        builder.add_layer("Linear", out_features=out_channels)

        logger.debug(f"Created simple_gcn template: {num_layers} layers, hidden={hidden_channels}")
        return builder

    @staticmethod
    def attention_network(
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
        hidden_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.0,
        task_type: str = "graph_regression",
    ) -> ArchitectureBuilder:
        """
        GAT-based attention network.

        Structure: (GAT → ELU → Dropout) × N → Pool → Linear

        Best for:
        - Heterogeneous graphs
        - When node importance varies
        - Molecular property prediction
        - Social networks

        Args:
            in_channels: Input feature dimension
            out_channels: Output dimension
            num_layers: Number of GAT layers (default: 3)
            hidden_channels: Hidden dimension per head (default: 64)
            heads: Number of attention heads (default: 4)
            dropout: Dropout rate (default: 0.0)
            task_type: Task type (default: "graph_regression")

        Returns:
            ArchitectureBuilder instance configured with GAT layers

        Examples:
            >>> # Multi-head attention for molecular graphs
            >>> builder = ArchitectureTemplates.attention_network(
            ...     in_channels=9, out_channels=1,
            ...     num_layers=3, heads=8
            ... )
            >>> model = builder.build()

            >>> # Node classification with attention
            >>> builder = ArchitectureTemplates.attention_network(
            ...     32, 7, num_layers=2, heads=4,
            ...     task_type="node_classification"
            ... )
        """
        builder = ArchitectureBuilder(task_type, in_channels, out_channels, name="AttentionNetwork")

        # Add GAT layers
        for i in range(num_layers):
            # Last layer: average heads, earlier layers: concatenate
            concat = i < num_layers - 1
            builder.add_layer(
                "GATConv", out_channels=hidden_channels, heads=heads, concat=concat, dropout=dropout
            )
            builder.add_layer("ELU")
            if dropout > 0:
                builder.add_layer("Dropout", p=dropout)

        # Add output layers
        if "graph" in task_type:
            builder.add_layer("global_mean_pool")

        builder.add_layer("Linear", out_features=out_channels)

        logger.debug(f"Created attention_network template: {num_layers} layers, {heads} heads")
        return builder

    @staticmethod
    def deep_residual(
        in_channels: int,
        out_channels: int,
        depth: int = 10,
        hidden_channels: int = 64,
        dropout: float = 0.0,
        task_type: str = "graph_regression",
    ) -> ArchitectureBuilder:
        """
        Deep residual architecture with skip connections.

        Structure: (Conv → ReLU → Conv → ReLU + Skip) × N → Pool → Linear

        Best for:
        - Very deep networks (10+ layers)
        - Complex graph structures
        - When training stability is crucial
        - Large-scale graphs

        Args:
            in_channels: Input feature dimension
            out_channels: Output dimension
            depth: Number of residual blocks (default: 10)
            hidden_channels: Hidden dimension (default: 64)
            dropout: Dropout rate (default: 0.0)
            task_type: Task type (default: "graph_regression")

        Returns:
            ArchitectureBuilder instance with residual connections

        Examples:
            >>> # Deep network with 10 residual blocks
            >>> builder = ArchitectureTemplates.deep_residual(
            ...     16, 1, depth=10, hidden_channels=128
            ... )
            >>> model = builder.build()
            >>> print(f"Skip connections: {len(builder.residual_connections)}")

            >>> # Shallow residual for node tasks
            >>> builder = ArchitectureTemplates.deep_residual(
            ...     32, 10, depth=3, task_type="node_classification"
            ... )
        """
        builder = ArchitectureBuilder(task_type, in_channels, out_channels, name="DeepResidual")

        # Initial projection
        builder.add_layer("GCNConv", out_channels=hidden_channels)
        builder.add_layer("ReLU")

        # Residual blocks
        for _i in range(depth):
            start_pos = len(builder.layers)

            builder.add_layer("GCNConv", out_channels=hidden_channels)
            builder.add_layer("ReLU")
            if dropout > 0:
                builder.add_layer("Dropout", p=dropout)
            builder.add_layer("GCNConv", out_channels=hidden_channels)
            builder.add_layer("ReLU")

            end_pos = len(builder.layers) - 1

            # Add skip connection
            builder.add_residual_connection(start_pos, end_pos, connection_type="add")

        # Add output layers
        if "graph" in task_type:
            builder.add_layer("global_mean_pool")

        builder.add_layer("Linear", out_features=out_channels)

        logger.debug(f"Created deep_residual template: {depth} residual blocks")
        return builder

    @staticmethod
    def hybrid_conv_attention(
        in_channels: int,
        out_channels: int,
        conv_layers: int = 2,
        attention_layers: int = 2,
        hidden_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.0,
        task_type: str = "graph_regression",
    ) -> ArchitectureBuilder:
        """
        Hybrid architecture: GCN layers followed by GAT layers.

        Structure: (GCN → ReLU) × N → (GAT → ELU) × M → Pool → Linear

        Best for:
        - Combining local and global features
        - Multi-scale learning
        - When both structure and attention matter
        - Heterogeneous graphs with varying importance

        Args:
            in_channels: Input feature dimension
            out_channels: Output dimension
            conv_layers: Number of GCN layers (default: 2)
            attention_layers: Number of GAT layers (default: 2)
            hidden_channels: Hidden dimension (default: 64)
            heads: Number of attention heads (default: 4)
            dropout: Dropout rate (default: 0.0)
            task_type: Task type (default: "graph_regression")

        Returns:
            ArchitectureBuilder instance with hybrid architecture

        Examples:
            >>> # Balanced hybrid architecture
            >>> builder = ArchitectureTemplates.hybrid_conv_attention(
            ...     16, 1, conv_layers=2, attention_layers=2
            ... )
            >>> model = builder.build()

            >>> # Attention-heavy hybrid
            >>> builder = ArchitectureTemplates.hybrid_conv_attention(
            ...     32, 1, conv_layers=1, attention_layers=4, heads=8
            ... )
        """
        builder = ArchitectureBuilder(
            task_type, in_channels, out_channels, name="HybridConvAttention"
        )

        # GCN layers
        for i in range(conv_layers):
            builder.add_layer("GCNConv", out_channels=hidden_channels)
            builder.add_layer("ReLU")
            if dropout > 0:
                builder.add_layer("Dropout", p=dropout)

        # GAT layers
        for i in range(attention_layers):
            concat = i < attention_layers - 1
            builder.add_layer("GATConv", out_channels=hidden_channels, heads=heads, concat=concat)
            builder.add_layer("ELU")
            if dropout > 0:
                builder.add_layer("Dropout", p=dropout)

        # Add output layers
        if "graph" in task_type:
            builder.add_layer("global_mean_pool")

        builder.add_layer("Linear", out_features=out_channels)

        logger.debug(
            f"Created hybrid_conv_attention template: "
            f"{conv_layers} conv + {attention_layers} attention"
        )
        return builder

    @staticmethod
    def hierarchical_pooling(
        in_channels: int,
        out_channels: int,
        num_levels: int = 3,
        hidden_channels: int = 64,
        pooling_ratio: float = 0.5,
        task_type: str = "graph_regression",
    ) -> ArchitectureBuilder:
        """
        Hierarchical architecture with pooling at each level.

        Structure: (Conv → ReLU → TopKPool) × N → Global Pool → Linear

        Best for:
        - Multi-scale graph analysis
        - Hierarchical graph structures
        - When coarse-to-fine learning is beneficial
        - Large graphs requiring compression

        Args:
            in_channels: Input feature dimension
            out_channels: Output dimension
            num_levels: Number of hierarchical levels (default: 3)
            hidden_channels: Hidden dimension (default: 64)
            pooling_ratio: Ratio of nodes to keep at each pooling (default: 0.5)
            task_type: Task type (default: "graph_regression")

        Returns:
            ArchitectureBuilder instance with hierarchical pooling

        Examples:
            >>> # 3-level hierarchical architecture
            >>> builder = ArchitectureTemplates.hierarchical_pooling(
            ...     16, 1, num_levels=3, pooling_ratio=0.5
            ... )
            >>> model = builder.build()

            >>> # Aggressive pooling for large graphs
            >>> builder = ArchitectureTemplates.hierarchical_pooling(
            ...     32, 1, num_levels=4, pooling_ratio=0.3
            ... )
        """
        builder = ArchitectureBuilder(
            task_type, in_channels, out_channels, name="HierarchicalPooling"
        )

        # Hierarchical levels
        for _i in range(num_levels):
            builder.add_layer("GCNConv", out_channels=hidden_channels)
            builder.add_layer("ReLU")
            builder.add_layer("TopKPooling", in_channels=hidden_channels, ratio=pooling_ratio)

        # Final global pooling
        builder.add_layer("global_mean_pool")
        builder.add_layer("Linear", out_features=out_channels)

        logger.debug(
            f"Created hierarchical_pooling template: {num_levels} levels, ratio={pooling_ratio}"
        )
        return builder

    @staticmethod
    def graph_sage_network(
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
        hidden_channels: int = 64,
        aggr: str = "mean",
        dropout: float = 0.0,
        task_type: str = "graph_regression",
    ) -> ArchitectureBuilder:
        """
        GraphSAGE-based architecture.

        Structure: (SAGE → ReLU → Dropout) × N → Pool → Linear

        Best for:
        - Large-scale graphs
        - Inductive learning (generalizing to unseen nodes)
        - Mini-batch training
        - Dynamic graphs

        Args:
            in_channels: Input feature dimension
            out_channels: Output dimension
            num_layers: Number of SAGE layers (default: 3)
            hidden_channels: Hidden dimension (default: 64)
            aggr: Aggregation method - "mean", "max", "lstm" (default: "mean")
            dropout: Dropout rate (default: 0.0)
            task_type: Task type (default: "graph_regression")

        Returns:
            ArchitectureBuilder instance with GraphSAGE layers

        Examples:
            >>> # Mean aggregation for balanced learning
            >>> builder = ArchitectureTemplates.graph_sage_network(
            ...     16, 1, num_layers=3, aggr="mean"
            ... )
            >>> model = builder.build()

            >>> # Max aggregation for node classification
            >>> builder = ArchitectureTemplates.graph_sage_network(
            ...     32, 10, aggr="max", task_type="node_classification"
            ... )
        """
        builder = ArchitectureBuilder(task_type, in_channels, out_channels, name="GraphSAGENetwork")

        # SAGE layers
        for _i in range(num_layers):
            builder.add_layer("SAGEConv", out_channels=hidden_channels, aggr=aggr)
            builder.add_layer("ReLU")
            if dropout > 0:
                builder.add_layer("Dropout", p=dropout)

        # Add output layers
        if "graph" in task_type:
            builder.add_layer("global_mean_pool")

        builder.add_layer("Linear", out_features=out_channels)

        logger.debug(f"Created graph_sage_network template: {num_layers} layers, aggr={aggr}")
        return builder

    @staticmethod
    def gin_network(
        in_channels: int,
        out_channels: int,
        num_layers: int = 5,
        hidden_channels: int = 64,
        dropout: float = 0.0,
        train_eps: bool = False,
        task_type: str = "graph_regression",
    ) -> ArchitectureBuilder:
        """
        Graph Isomorphism Network (GIN) architecture.

        Structure: (GIN → ReLU → Dropout) × N → Sum Pool → Linear

        Best for:
        - Graph isomorphism testing
        - Molecular property prediction
        - When maximum expressive power is needed
        - Graph classification tasks

        Note: Uses GCNConv as placeholder for GINConv (which requires MLP).
        For production, consider implementing proper GIN layers.

        Args:
            in_channels: Input feature dimension
            out_channels: Output dimension
            num_layers: Number of GIN layers (default: 5)
            hidden_channels: Hidden dimension (default: 64)
            dropout: Dropout rate (default: 0.0)
            train_eps: Whether to train epsilon parameter (default: False)
            task_type: Task type (default: "graph_regression")

        Returns:
            ArchitectureBuilder instance with GIN-inspired architecture

        Examples:
            >>> # Standard GIN for graph classification
            >>> builder = ArchitectureTemplates.gin_network(
            ...     9, 2, num_layers=5, hidden_channels=64
            ... )
            >>> model = builder.build()

            >>> # Deep GIN with dropout
            >>> builder = ArchitectureTemplates.gin_network(
            ...     16, 1, num_layers=7, dropout=0.5
            ... )
        """
        builder = ArchitectureBuilder(task_type, in_channels, out_channels, name="GINNetwork")

        # GIN layers - using GCNConv as placeholder
        # In production, implement proper GINConv with MLP
        for _i in range(num_layers):
            builder.add_layer("GCNConv", out_channels=hidden_channels)
            builder.add_layer("ReLU")
            if dropout > 0:
                builder.add_layer("Dropout", p=dropout)

        # Add output layers
        if "graph" in task_type:
            builder.add_layer("global_add_pool")  # GIN typically uses sum pooling

        builder.add_layer("Linear", out_features=out_channels)

        logger.debug(f"Created gin_network template: {num_layers} layers")
        return builder

    @staticmethod
    def molecular_network(
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        task_type: str = "graph_regression",
    ) -> ArchitectureBuilder:
        """
        Architecture optimized for molecular property prediction.

        Combines convolutional and attention mechanisms with MLP readout.
        Structure: Conv → (Conv/GAT alternating) × N → Pool → MLP

        Best for:
        - Molecular property prediction
        - Drug discovery
        - Chemical reaction prediction
        - Protein-ligand binding

        Args:
            in_channels: Input feature dimension (e.g., atom features)
            out_channels: Output dimension (e.g., property value)
            hidden_channels: Hidden dimension (default: 128)
            num_layers: Number of message passing layers (default: 4)
            dropout: Dropout rate (default: 0.1)
            task_type: Task type (default: "graph_regression")

        Returns:
            ArchitectureBuilder instance optimized for molecules

        Examples:
            >>> # Molecular property prediction
            >>> builder = ArchitectureTemplates.molecular_network(
            ...     9, 1, hidden_channels=128, num_layers=4
            ... )
            >>> model = builder.build()

            >>> # Multi-task molecular prediction
            >>> builder = ArchitectureTemplates.molecular_network(
            ...     11, 5, hidden_channels=256, num_layers=5
            ... )
        """
        builder = ArchitectureBuilder(task_type, in_channels, out_channels, name="MolecularNetwork")

        # Initial convolution
        builder.add_layer("GCNConv", out_channels=hidden_channels)
        builder.add_layer("ReLU")
        builder.add_layer("Dropout", p=dropout)

        # Mixed conv and attention layers
        for i in range(num_layers - 1):
            if i % 2 == 0:
                builder.add_layer("GCNConv", out_channels=hidden_channels)
            else:
                builder.add_layer("GATConv", out_channels=hidden_channels, heads=4, concat=False)
            builder.add_layer("ReLU")
            builder.add_layer("Dropout", p=dropout)

        # Global pooling
        builder.add_layer("global_mean_pool")

        # Readout MLP
        builder.add_layer("Linear", out_features=hidden_channels)
        builder.add_layer("ReLU")
        builder.add_layer("Dropout", p=dropout)
        builder.add_layer("Linear", out_features=out_channels)

        logger.debug(
            f"Created molecular_network template: {num_layers} layers, hidden={hidden_channels}"
        )
        return builder

    # =========================================================================
    # TASK-SPECIFIC TEMPLATES
    # =========================================================================

    @staticmethod
    def node_classification_network(
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> ArchitectureBuilder:
        """
        Architecture optimized for node classification.

        Structure: (Conv → ReLU → Dropout) × N → Linear

        Best for:
        - Semi-supervised node classification
        - Citation networks
        - Social networks
        - Knowledge graphs

        Args:
            in_channels: Input feature dimension
            num_classes: Number of classes
            hidden_channels: Hidden dimension (default: 64)
            num_layers: Number of layers (default: 2)
            dropout: Dropout rate (default: 0.5)

        Returns:
            ArchitectureBuilder instance for node classification

        Examples:
            >>> # Citation network classification (Cora, CiteSeer)
            >>> builder = ArchitectureTemplates.node_classification_network(
            ...     1433, 7, hidden_channels=16, num_layers=2, dropout=0.5
            ... )
            >>> model = builder.build()

            >>> # Deep network with more layers
            >>> builder = ArchitectureTemplates.node_classification_network(
            ...     128, 10, hidden_channels=64, num_layers=3
            ... )
        """
        builder = ArchitectureBuilder(
            "node_classification", in_channels, num_classes, name="NodeClassification"
        )

        for _i in range(num_layers):
            builder.add_layer("GCNConv", out_channels=hidden_channels)
            builder.add_layer("ReLU")
            builder.add_layer("Dropout", p=dropout)

        builder.add_layer("Linear", out_features=num_classes)

        logger.debug(
            f"Created node_classification_network: {num_layers} layers, classes={num_classes}"
        )
        return builder

    @staticmethod
    def graph_classification_network(
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.5,
    ) -> ArchitectureBuilder:
        """
        Architecture optimized for graph classification.

        Structure: (Conv → ReLU → Dropout) × N → Pool → Linear

        Best for:
        - Protein function prediction
        - Molecule classification
        - Social network analysis
        - Chemical compound categorization

        Args:
            in_channels: Input feature dimension
            num_classes: Number of classes
            hidden_channels: Hidden dimension (default: 64)
            num_layers: Number of layers (default: 3)
            dropout: Dropout rate (default: 0.5)

        Returns:
            ArchitectureBuilder instance for graph classification

        Examples:
            >>> # Protein function prediction
            >>> builder = ArchitectureTemplates.graph_classification_network(
            ...     20, 5, hidden_channels=128, num_layers=3
            ... )
            >>> model = builder.build()

            >>> # Binary molecule classification
            >>> builder = ArchitectureTemplates.graph_classification_network(
            ...     9, 2, hidden_channels=64, num_layers=4, dropout=0.3
            ... )
        """
        builder = ArchitectureBuilder(
            "graph_classification", in_channels, num_classes, name="GraphClassification"
        )

        for _i in range(num_layers):
            builder.add_layer("GCNConv", out_channels=hidden_channels)
            builder.add_layer("ReLU")
            builder.add_layer("Dropout", p=dropout)

        builder.add_layer("global_mean_pool")
        builder.add_layer("Linear", out_features=num_classes)

        logger.debug(
            f"Created graph_classification_network: {num_layers} layers, classes={num_classes}"
        )
        return builder

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def list_templates() -> list[str]:
        """
        List all available templates.

        Returns:
            List of template names

        Examples:
            >>> templates = ArchitectureTemplates.list_templates()
            >>> print(f"Available: {len(templates)} templates")
            >>> for name in templates:
            ...     print(f"  - {name}")
        """
        templates = [
            "simple_gcn",
            "attention_network",
            "deep_residual",
            "hybrid_conv_attention",
            "hierarchical_pooling",
            "graph_sage_network",
            "gin_network",
            "molecular_network",
            "node_classification_network",
            "graph_classification_network",
        ]
        return templates

    @staticmethod
    def get_template_info(template_name: str) -> dict[str, Any]:
        """
        Get information about a template.

        Args:
            template_name: Template name

        Returns:
            Template information dictionary with:
            - name: Template name
            - description: Brief description
            - parameters: List of parameter names
            - suitable_for: List of suitable task types
            - best_use_cases: List of ideal use cases

        Examples:
            >>> info = ArchitectureTemplates.get_template_info("simple_gcn")
            >>> print(info["description"])
            >>> print(f"Parameters: {info['parameters']}")
            >>> print(f"Suitable for: {info['suitable_for']}")
        """
        info_map = {
            "simple_gcn": {
                "name": "simple_gcn",
                "description": "Simple GCN architecture with configurable depth",
                "parameters": [
                    "in_channels",
                    "out_channels",
                    "num_layers",
                    "hidden_channels",
                    "dropout",
                    "task_type",
                ],
                "suitable_for": [
                    "graph_regression",
                    "graph_classification",
                    "node_regression",
                    "node_classification",
                ],
                "best_use_cases": [
                    "Quick prototyping",
                    "Baseline models",
                    "Small to medium graphs",
                    "When interpretability matters",
                ],
            },
            "attention_network": {
                "name": "attention_network",
                "description": "GAT-based multi-head attention network",
                "parameters": [
                    "in_channels",
                    "out_channels",
                    "num_layers",
                    "hidden_channels",
                    "heads",
                    "dropout",
                    "task_type",
                ],
                "suitable_for": [
                    "graph_regression",
                    "graph_classification",
                    "node_classification",
                    "molecular_prediction",
                ],
                "best_use_cases": [
                    "Heterogeneous graphs",
                    "Molecular property prediction",
                    "Social networks",
                    "When node importance varies",
                ],
            },
            "deep_residual": {
                "name": "deep_residual",
                "description": "Deep architecture with skip connections",
                "parameters": [
                    "in_channels",
                    "out_channels",
                    "depth",
                    "hidden_channels",
                    "dropout",
                    "task_type",
                ],
                "suitable_for": [
                    "graph_regression",
                    "graph_classification",
                    "node_regression",
                    "node_classification",
                ],
                "best_use_cases": [
                    "Very deep networks (10+ layers)",
                    "Complex graph structures",
                    "Training stability crucial",
                    "Large-scale graphs",
                ],
            },
            "hybrid_conv_attention": {
                "name": "hybrid_conv_attention",
                "description": "Hybrid GCN and GAT layers",
                "parameters": [
                    "in_channels",
                    "out_channels",
                    "conv_layers",
                    "attention_layers",
                    "hidden_channels",
                    "heads",
                    "dropout",
                    "task_type",
                ],
                "suitable_for": ["graph_regression", "graph_classification", "node_classification"],
                "best_use_cases": [
                    "Multi-scale learning",
                    "Heterogeneous graphs",
                    "Combining local and global features",
                ],
            },
            "hierarchical_pooling": {
                "name": "hierarchical_pooling",
                "description": "Multi-level architecture with pooling",
                "parameters": [
                    "in_channels",
                    "out_channels",
                    "num_levels",
                    "hidden_channels",
                    "pooling_ratio",
                    "task_type",
                ],
                "suitable_for": ["graph_regression", "graph_classification"],
                "best_use_cases": [
                    "Multi-scale analysis",
                    "Large graphs requiring compression",
                    "Hierarchical structures",
                    "Coarse-to-fine learning",
                ],
            },
            "graph_sage_network": {
                "name": "graph_sage_network",
                "description": "GraphSAGE-based architecture",
                "parameters": [
                    "in_channels",
                    "out_channels",
                    "num_layers",
                    "hidden_channels",
                    "aggr",
                    "dropout",
                    "task_type",
                ],
                "suitable_for": [
                    "graph_regression",
                    "graph_classification",
                    "node_regression",
                    "node_classification",
                ],
                "best_use_cases": [
                    "Large-scale graphs",
                    "Inductive learning",
                    "Mini-batch training",
                    "Dynamic graphs",
                ],
            },
            "gin_network": {
                "name": "gin_network",
                "description": "Graph Isomorphism Network architecture",
                "parameters": [
                    "in_channels",
                    "out_channels",
                    "num_layers",
                    "hidden_channels",
                    "dropout",
                    "train_eps",
                    "task_type",
                ],
                "suitable_for": [
                    "graph_regression",
                    "graph_classification",
                    "molecular_prediction",
                ],
                "best_use_cases": [
                    "Graph isomorphism testing",
                    "Molecular property prediction",
                    "Maximum expressive power needed",
                ],
            },
            "molecular_network": {
                "name": "molecular_network",
                "description": "Optimized for molecular property prediction",
                "parameters": [
                    "in_channels",
                    "out_channels",
                    "hidden_channels",
                    "num_layers",
                    "dropout",
                    "task_type",
                ],
                "suitable_for": [
                    "graph_regression",
                    "graph_classification",
                    "molecular_prediction",
                ],
                "best_use_cases": [
                    "Drug discovery",
                    "Chemical reaction prediction",
                    "Protein-ligand binding",
                    "Molecular properties",
                ],
            },
            "node_classification_network": {
                "name": "node_classification_network",
                "description": "Optimized for node classification",
                "parameters": [
                    "in_channels",
                    "num_classes",
                    "hidden_channels",
                    "num_layers",
                    "dropout",
                ],
                "suitable_for": ["node_classification"],
                "best_use_cases": [
                    "Citation networks",
                    "Social networks",
                    "Knowledge graphs",
                    "Semi-supervised learning",
                ],
            },
            "graph_classification_network": {
                "name": "graph_classification_network",
                "description": "Optimized for graph classification",
                "parameters": [
                    "in_channels",
                    "num_classes",
                    "hidden_channels",
                    "num_layers",
                    "dropout",
                ],
                "suitable_for": ["graph_classification"],
                "best_use_cases": [
                    "Protein function prediction",
                    "Molecule classification",
                    "Social network analysis",
                    "Chemical compound categorization",
                ],
            },
        }
        return info_map.get(template_name, {})


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info(
    f"templates module loaded - {len(ArchitectureTemplates.list_templates())} templates available"
)
