"""Graph models are never traced with ``torch.jit.trace``.

A trace keeps the example batch's sizes and the branches taken for it, so a traced MILIA graph
model returned wrong shapes for other batches (a traced GIN gave one row for a two-graph batch).
``ComputationOptimizer`` and ``ModelOptimizer.optimize_for_mobile`` now refuse to trace one.
"""

import warnings
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from milia_pipeline.exceptions import ExportError, OptimizationError
from milia_pipeline.models._graph_trace import is_graph_model
from milia_pipeline.models.acceleration.computation_optimization import ComputationOptimizer
from milia_pipeline.models.deployment.model_optimization import ModelOptimizer
from milia_pipeline.models.factory.model_factory import GraphLevelModelWrapper, ModelFactory

CPU = torch.device("cpu")


def _graph(num_nodes: int) -> Data:
    src = torch.arange(num_nodes - 1)
    edge_index = torch.stack([src, src + 1])
    return Data(
        x=torch.randn(num_nodes, 8),
        edge_index=torch.cat([edge_index, edge_index.flip(0)], dim=1),
        y=torch.randn(1, 2),
    )


@pytest.fixture
def graph_model() -> nn.Module:
    torch.manual_seed(0)
    return (
        ModelFactory()
        .create_model(
            name="GIN",
            hyperparameters={"hidden_channels": 16, "num_layers": 2},
            task_type="graph_regression",
            sample_data=_graph(5),
        )
        .eval()
    )


@pytest.fixture
def jit_optimizer() -> ComputationOptimizer:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ComputationOptimizer(jit_compile=True, verbose=False, device=CPU)


class TestIsGraphModel:
    def test_milia_graph_model(self, graph_model):
        assert is_graph_model(graph_model)

    def test_wrapped_torch_nn_model(self):
        """A plain ``torch.nn`` model pooled per graph by MILIA's wrapper is a graph model."""
        model = GraphLevelModelWrapper(nn.Linear(8, 2), "graph_regression")
        assert not type(model).__module__.startswith("torch.")
        assert is_graph_model(model)

    def test_plain_module(self):
        assert not is_graph_model(nn.Sequential(nn.Linear(4, 4), nn.ReLU()))


class TestComputationOptimizer:
    def test_optimize_model_refuses_to_trace_a_graph_model(self, graph_model, jit_optimizer):
        batch = Batch.from_data_list([_graph(5), _graph(7)])
        with (
            patch.object(torch.jit, "trace") as trace,
            pytest.raises(OptimizationError, match="torch.jit.trace cannot convert a graph model"),
        ):
            jit_optimizer.optimize_model(
                graph_model, example_inputs=(batch.x, batch.edge_index, batch.batch)
            )
        trace.assert_not_called()

    def test_jit_script_model_refuses_to_trace_a_graph_model(self, graph_model, jit_optimizer):
        batch = Batch.from_data_list([_graph(4)])
        with (
            warnings.catch_warnings(),
            pytest.raises(OptimizationError, match="torch.jit.trace cannot convert a graph model"),
        ):
            warnings.simplefilter("ignore", DeprecationWarning)
            jit_optimizer.jit_script_model(graph_model, (batch.x, batch.edge_index))

    def test_plain_module_is_still_traced(self, jit_optimizer):
        model = nn.Linear(3, 2).eval()
        x = torch.randn(4, 3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # torch.jit's own warnings
            result = jit_optimizer.optimize_model(model, example_inputs=(x,))
        assert isinstance(result, torch.jit.ScriptModule)
        torch.testing.assert_close(result(x), model(x))

    def test_unscripted_graph_model_keeps_its_outputs(self, graph_model, jit_optimizer):
        """Without example inputs scripting a MILIA model fails and the eager model is kept."""
        batch = Batch.from_data_list([_graph(6), _graph(3)])
        with torch.no_grad():
            expected = graph_model(batch.x, batch.edge_index, batch=batch.batch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = jit_optimizer.optimize_model(graph_model)
        with torch.no_grad():
            out = result(batch.x, batch.edge_index, batch=batch.batch)
        assert out.shape == (2, 2)
        torch.testing.assert_close(out, expected)


class TestOptimizeForMobile:
    def test_refuses_a_graph_model(self, graph_model, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            optimizer = ModelOptimizer(optimize_for_mobile=True, verbose=False)
            batch = Batch.from_data_list([_graph(5)])
            with (
                patch.object(torch.jit, "trace") as trace,
                pytest.raises(ExportError, match="torch.jit.trace cannot convert a graph model"),
            ):
                optimizer.optimize_for_mobile(
                    graph_model, (batch.x, batch.edge_index, batch.batch), tmp_path / "m.ptl"
                )
        trace.assert_not_called()
        assert not (tmp_path / "m.ptl").exists()
