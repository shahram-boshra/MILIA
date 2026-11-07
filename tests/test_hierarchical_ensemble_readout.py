"""Hierarchical ensembles read out once: only the last level pools for graph-level tasks."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from milia_pipeline.models.builders.model_composer import (
    HierarchicalComposition,
    ModelSpec,
    ParallelEnsemble,
)
from milia_pipeline.models.factory.model_factory import ModelFactory

NUM_GRAPHS, NUM_NODES, NUM_FEATURES = 3, 6, 4


def _graphs(task_type: str) -> list[Data]:
    generator = torch.Generator().manual_seed(11)
    ring = torch.arange(NUM_NODES)
    edge_index = torch.cat(
        [torch.stack([ring, (ring + 1) % NUM_NODES]), torch.stack([(ring + 1) % NUM_NODES, ring])],
        dim=1,
    )
    graphs = []
    for _ in range(NUM_GRAPHS):
        y = (
            torch.randint(0, 3, (1,), generator=generator)
            if task_type == "graph_classification"
            else torch.randn(1, 1, generator=generator)
        )
        x = torch.randn(NUM_NODES, NUM_FEATURES, generator=generator)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return graphs


class _NodeLinear(nn.Module):
    """Node-wise linear map that also checks it receives one row per node."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.lin = nn.Linear(in_channels, out_channels)
        self.rows_seen: list[int] = []

    def forward(self, x, edge_index, batch=None):
        self.rows_seen.append(x.size(0))
        assert int(edge_index.max()) < x.size(0), "node edge_index applied to pooled rows"
        return self.lin(x)


def test_parallel_ensemble_reads_out_by_default_and_not_when_told():
    batch = Batch.from_data_list(_graphs("graph_regression"))

    def ensemble(readout: bool) -> ParallelEnsemble:
        return ParallelEnsemble(
            models=[_NodeLinear(NUM_FEATURES, 2)],
            weights=[1.0],
            fusion="mean",
            task_type="graph_regression",
            readout=readout,
        )

    assert ensemble(True)(batch.x, batch.edge_index, batch=batch.batch).shape == (NUM_GRAPHS, 2)
    assert ensemble(False)(batch.x, batch.edge_index, batch=batch.batch).shape == (
        batch.num_nodes,
        2,
    )


def test_only_the_last_level_reads_out():
    """Levels 0 and 1 see and produce node rows; level 2 pools per graph."""
    batch = Batch.from_data_list(_graphs("graph_regression"))
    members = [_NodeLinear(NUM_FEATURES, 5), _NodeLinear(5, 5), _NodeLinear(5, 1)]
    hierarchy = HierarchicalComposition(
        levels_dict={level: [ModelSpec(model=m, weight=1.0)] for level, m in enumerate(members)},
        fusion="mean",
        task_type="graph_regression",
    )

    assert [e.readout for e in hierarchy.level_ensembles] == [False, False, True]
    out = hierarchy(batch.x, batch.edge_index, batch=batch.batch, num_graphs=batch.num_graphs)
    assert out.shape == (NUM_GRAPHS, 1)
    assert [m.rows_seen for m in members] == [[batch.num_nodes]] * 3


@pytest.mark.parametrize(
    ("task_type", "out_channels"), [("graph_regression", 1), ("graph_classification", 3)]
)
def test_hierarchical_pyg_ensemble_on_graph_level_tasks(task_type, out_channels):
    """GCN at level 0 feeds GraphSAGE at level 1 by message passing over node rows."""
    torch.manual_seed(0)
    graphs = _graphs(task_type)
    batch = Batch.from_data_list(graphs)
    ensemble_config = {
        "name": "HierarchicalPyG",
        "strategy": "hierarchical",
        "fusion": {"method": "mean"},
        "models": [
            {"name": "GCN", "level": 0, "hyperparameters": {"hidden_channels": 16}},
            {"name": "GraphSAGE", "level": 1, "hyperparameters": {"hidden_channels": 16}},
        ],
    }
    hyperparameters = {"ensemble_config": ensemble_config}
    if task_type == "graph_classification":
        hyperparameters["out_channels"] = out_channels  # as main.py injects num_classes

    model = ModelFactory().create_model(
        "ensemble", hyperparameters, task_type, sample_data=graphs[0]
    )

    out = model(batch.x, batch.edge_index, batch=batch.batch)
    assert out.shape == (NUM_GRAPHS, out_channels)
    if task_type == "graph_classification":
        loss_fn = lambda: nn.functional.cross_entropy(  # noqa: E731
            model(batch.x, batch.edge_index, batch=batch.batch), batch.y
        )
    else:
        loss_fn = lambda: nn.functional.mse_loss(  # noqa: E731
            model(batch.x, batch.edge_index, batch=batch.batch), batch.y
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = loss_fn()
    optimizer.zero_grad()
    before.backward()
    optimizer.step()
    with torch.no_grad():
        assert loss_fn().item() < before.item()
