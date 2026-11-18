"""Graph-level output is declared by MILIA's wrapper and ensembles, not inferred from shapes.

``GraphLevelModelWrapper`` used to decide whether its inner model's output still needed pooling
by comparing the output's row count with the node count. For an inner MILIA ensemble the row
count is the number of graphs, which depends on the values of ``batch``: correct in eager mode,
but a data-dependent branch that ``torch.export`` cannot trace. The ensembles now declare
``produces_graph_output`` and the wrapper trusts the declaration.
"""

import warnings

import pytest
import torch
from torch.export import Dim
from torch_geometric.data import Batch, Data

from milia_pipeline.models._graph_output import produces_graph_output
from milia_pipeline.models.factory.model_factory import GraphLevelModelWrapper, ModelFactory


def _graph(num_nodes: int) -> Data:
    src = torch.arange(max(num_nodes - 1, 0))
    edge_index = (
        torch.stack([src, src + 1]) if num_nodes > 1 else torch.empty(2, 0, dtype=torch.long)
    )
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return Data(
        x=torch.randn(num_nodes, 8),
        edge_index=edge_index,
        edge_attr=torch.randn(edge_index.size(1), 3),
        y=torch.randn(1, 2),
    )


def _ensemble(strategy: str = "parallel", task_type: str = "graph_regression"):
    torch.manual_seed(0)
    levels = (0, 0, 1) if strategy == "hierarchical" else (0, 0, 0)
    config = {
        "enabled": True,
        "strategy": strategy,
        "models": [
            {"name": name, "hyperparameters": {}, "weight": 1.0, "level": level}
            for name, level in zip(("GCN", "GAT", "GraphSAGE"), levels, strict=True)
        ],
        "fusion": {"method": "weighted"},
    }
    return (
        ModelFactory()
        .create_model(
            name="ensemble",
            hyperparameters={"ensemble_config": config, "in_channels": 8, "out_channels": 2},
            task_type=task_type,
            sample_data=_graph(5),
        )
        .eval()
    )


class TestDeclarations:
    def test_wrapper_declares_graph_output_for_graph_tasks_only(self):
        inner = torch.nn.Linear(8, 2)
        assert GraphLevelModelWrapper(inner, task_type="graph_regression").produces_graph_output
        assert not GraphLevelModelWrapper(inner, task_type="node_regression").produces_graph_output

    @pytest.mark.parametrize("strategy", ["parallel", "hierarchical"])
    def test_graph_task_ensembles_declare_graph_output(self, strategy):
        model = _ensemble(strategy)
        assert isinstance(model, GraphLevelModelWrapper)
        assert produces_graph_output(model.model)

    def test_plain_modules_declare_nothing(self):
        assert not produces_graph_output(torch.nn.Linear(2, 2))


class TestBehaviour:
    def test_ensemble_output_is_not_pooled_again(self):
        """Single-node molecules: graph count == node count, the case the shape rule cannot tell."""
        model = _ensemble()
        batch = Batch.from_data_list([_graph(1), _graph(1), _graph(1)])
        args = (batch.x, batch.edge_index)
        kwargs = {"edge_attr": batch.edge_attr, "batch": batch.batch}
        with torch.no_grad():
            wrapped = model(*args, **kwargs)
            inner = model.model(*args, **kwargs)
        assert wrapped.shape == (3, 2)
        torch.testing.assert_close(wrapped, inner)

    def test_wrapped_ensemble_exports_with_dynamic_sizes(self):
        """torch.export (strict) captures the wrapped ensemble; the program matches eager."""

        class Inputs(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x, edge_index, edge_attr, batch):
                return self.model(x, edge_index, edge_attr=edge_attr, batch=batch)

        module = Inputs(_ensemble()).eval()
        example = Batch.from_data_list([_graph(5), _graph(4), _graph(4)])
        other = Batch.from_data_list([_graph(6), _graph(3)])
        nodes, edges = Dim("num_nodes"), Dim("num_edges")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            program = torch.export.export(
                module,
                (example.x, example.edge_index, example.edge_attr, example.batch),
                dynamic_shapes={
                    "x": {0: nodes},
                    "edge_index": {1: edges},
                    "edge_attr": {0: edges},
                    "batch": {0: nodes},
                },
                strict=True,
            )
        args = (other.x, other.edge_index, other.edge_attr, other.batch)
        with torch.no_grad():
            torch.testing.assert_close(program.module()(*args), module(*args))
