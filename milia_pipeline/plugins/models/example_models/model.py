"""
Example model plugin: a small message-passing regressor.

``ExampleRegressor`` follows the model plugin conventions, so MILIA can build it,
chain it in ensembles and pool it for graph-level tasks:

- constructor parameters ``in_channels``, ``hidden_channels`` and ``out_channels``;
- ``self.in_channels`` and ``self.out_channels`` set from them;
- ``forward(x, edge_index, batch=None)`` returning one row per node
  (``[num_nodes, out_channels]``). For graph-level tasks the model factory pools
  the rows per graph, as for PyG's own models.

It uses only PyTorch and PyTorch Geometric's ``MessagePassing`` base class (no
compiled extensions), and is deterministic under ``torch.manual_seed``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class MeanNeighbourConv(MessagePassing):
    """``x_i' = W_self x_i + mean_{j in N(i)} W_neigh x_j``."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="mean")
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.lin_self(x) + self.propagate(edge_index, x=self.lin_neigh(x))


class ExampleRegressor(nn.Module):
    """Stacked mean-neighbour convolutions with ReLU, then a linear read-out.

    Args:
        in_channels: Node feature size.
        hidden_channels: Size of the hidden node representations.
        out_channels: Output size per node.
        num_layers: Number of message-passing layers (at least 1).
        dropout: Dropout probability between layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        out_channels: int = 1,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        sizes = [in_channels] + [hidden_channels] * num_layers
        self.convs = nn.ModuleList(
            MeanNeighbourConv(sizes[i], sizes[i + 1]) for i in range(num_layers)
        )
        self.readout = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Node outputs ``[num_nodes, out_channels]``; ``batch`` is not needed per node."""
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.readout(x)
