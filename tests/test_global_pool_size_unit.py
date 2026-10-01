"""PA-1d (F19): graph pooling with ``size=`` and sync-free graph counts.

PyG's scatter infers the output size with ``int(index.max()) + 1`` when ``size`` is omitted — a
device→host sync. These tests pin that MILIA's wrappers and ensembles pool with the known graph
count (``Batch.num_graphs``, a Python int), never forward ``num_graphs`` to plain PyG models, and
produce exactly the same outputs as before.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from milia_pipeline.models.builders.model_composer import (
    ParallelEnsemble,
    SequentialStack,
    _global_pool,
    _needs_graph_pooling,
    _resolve_num_graphs,
)
from milia_pipeline.models.factory.model_factory import GraphLevelModelWrapper
from milia_pipeline.models.training.trainer import Trainer

N_GRAPHS = 4


class NodeGCN(nn.Module):
    """Plain node-level model with a PyG signature and no **kwargs (rejects unknown keywords).

    A per-node Linear keeps the forward value-free, so the sync detector tests MILIA's pooling
    code only — not message-passing internals.
    """

    def __init__(self, out: int = 3) -> None:
        super().__init__()
        self.lin = nn.Linear(4, out)

    def forward(self, x, edge_index, batch=None):
        return self.lin(x)


def _batch(n_graphs: int = N_GRAPHS, nodes: int = 5):
    gen = torch.Generator().manual_seed(0)
    ei = torch.tensor([[i, (i + 1) % nodes] for i in range(nodes)]).t()
    graphs = [Data(x=torch.randn(nodes, 4, generator=gen), edge_index=ei) for _ in range(n_graphs)]
    return next(iter(DataLoader(graphs, batch_size=n_graphs)))


@contextlib.contextmanager
def no_host_sync():
    """Fail if a tensor value is read on the host (``.item()`` / ``int(tensor)``)."""

    def boom(*_a, **_k):
        raise AssertionError("device→host sync (Tensor.item / int(tensor)) during forward")

    with patch.object(torch.Tensor, "item", boom), patch.object(torch.Tensor, "__int__", boom):
        yield


# --- helpers -----------------------------------------------------------------------------------
def test_batch_num_graphs_is_python_int():
    assert isinstance(_batch().num_graphs, int)


@pytest.mark.parametrize("method", ["mean", "max", "add"])
def test_global_pool_with_size_equals_inferred(method):
    b = _batch()
    x = torch.randn(b.num_nodes, 3)
    assert torch.equal(
        _global_pool(x, b.batch, method, size=b.num_graphs), _global_pool(x, b.batch, method)
    )


def test_global_pool_honours_size():
    x = torch.ones(3, 2)
    out = _global_pool(x, torch.tensor([0, 0, 1]), "add", size=3)
    assert out.shape == (3, 2) and out[2].abs().sum() == 0  # trailing empty graph


def test_global_pool_with_size_reads_no_tensor_values():
    b = _batch()
    x = torch.randn(b.num_nodes, 3)
    with no_host_sync():
        _global_pool(x, b.batch, "mean", size=b.num_graphs)


def test_resolve_num_graphs():
    batch = torch.tensor([0, 0, 1])
    assert _resolve_num_graphs(2, batch) == 2
    assert _resolve_num_graphs(None, None) == 1
    assert _resolve_num_graphs(None, torch.empty(0, dtype=torch.long)) == 1
    assert _resolve_num_graphs(None, batch) is None
    assert _resolve_num_graphs(MagicMock(), batch) is None  # only a real int is trusted


def test_needs_graph_pooling():
    batch = torch.tensor([0, 0, 1])
    assert _needs_graph_pooling(3, 2, batch) is True  # known: rows > graphs
    assert _needs_graph_pooling(2, 2, batch) is False
    assert _needs_graph_pooling(3, None, batch) is True  # unknown: one row per node
    assert _needs_graph_pooling(2, None, batch) is False


# --- GraphLevelModelWrapper --------------------------------------------------------------------
def test_wrapper_consumes_num_graphs_never_reaches_plain_model():
    torch.manual_seed(0)
    b = _batch()
    wrapper = GraphLevelModelWrapper(NodeGCN(), "graph_regression")
    out = wrapper(
        b.x, b.edge_index, batch=b.batch, num_graphs=b.num_graphs
    )  # NodeGCN has no **kwargs
    assert out.shape == (N_GRAPHS, 3)


def test_wrapper_known_count_is_sync_free_and_matches_unknown():
    torch.manual_seed(0)
    b = _batch()
    wrapper = GraphLevelModelWrapper(NodeGCN(), "graph_regression")
    with no_host_sync():
        known = wrapper(b.x, b.edge_index, batch=b.batch, num_graphs=b.num_graphs)
    unknown = wrapper(b.x, b.edge_index, batch=b.batch)  # legacy call: count decided from shapes
    assert torch.equal(known, unknown)


def test_wrapper_forwards_count_only_to_capable_inner_model():
    class Capable(nn.Module):
        accepts_num_graphs = True

        def __init__(self) -> None:
            super().__init__()
            self.seen = None

        def forward(self, x, edge_index, batch=None, num_graphs=None):
            self.seen = num_graphs
            return torch.zeros(num_graphs, 2)

    b = _batch()
    inner = Capable()
    GraphLevelModelWrapper(inner, "graph_regression")(
        b.x, b.edge_index, batch=b.batch, num_graphs=4
    )
    assert inner.seen == 4


def test_wrapper_single_node_graphs_unchanged():
    """num_nodes == num_graphs: the unknown-count rule pools, which is the identity here."""
    torch.manual_seed(0)
    b = _batch(n_graphs=3, nodes=1)
    wrapper = GraphLevelModelWrapper(NodeGCN(), "graph_regression")
    known = wrapper(b.x, b.edge_index, batch=b.batch, num_graphs=3)
    assert torch.equal(known, wrapper(b.x, b.edge_index, batch=b.batch))


# --- ensembles ---------------------------------------------------------------------------------
def _ensemble(kind: str):
    torch.manual_seed(0)
    if kind == "parallel":
        return ParallelEnsemble(
            [NodeGCN(), NodeGCN()], weights=[0.5, 0.5], fusion="mean", task_type="graph_regression"
        )
    return SequentialStack([NodeGCN(out=4), NodeGCN()], task_type="graph_regression")


@pytest.mark.parametrize("kind", ["parallel", "sequential"])
def test_ensemble_known_count_sync_free_and_matches_unknown(kind):
    b = _batch()
    model = _ensemble(kind)
    with no_host_sync():
        known = model(b.x, edge_index=b.edge_index, batch=b.batch, num_graphs=b.num_graphs)
    unknown = model(b.x, edge_index=b.edge_index, batch=b.batch)
    assert known.shape == (N_GRAPHS, 3)
    assert torch.equal(known, unknown)


@pytest.mark.parametrize("kind", ["parallel", "sequential"])
def test_ensemble_reads_count_from_batch_object(kind):
    b = _batch()
    with no_host_sync():
        out = _ensemble(kind)(b)
    assert out.shape == (N_GRAPHS, 3)


def test_wrapped_ensemble_threads_count_end_to_end():
    b = _batch()
    model = GraphLevelModelWrapper(_ensemble("parallel"), "graph_regression")
    with no_host_sync():
        out = model(b.x, b.edge_index, batch=b.batch, num_graphs=b.num_graphs)
    assert out.shape == (N_GRAPHS, 3)


# --- Trainer -----------------------------------------------------------------------------------
def _num_graphs_kwargs(model, batch):
    trainer = Trainer.__new__(Trainer)  # only the helper is exercised
    trainer.model = model
    return trainer._num_graphs_kwargs(batch)


def test_trainer_passes_count_only_to_declaring_models():
    b = _batch()
    assert _num_graphs_kwargs(GraphLevelModelWrapper(NodeGCN(), "graph_regression"), b) == {
        "num_graphs": N_GRAPHS
    }
    assert _num_graphs_kwargs(NodeGCN(), b) == {}
    assert _num_graphs_kwargs(MagicMock(), b) == {}  # truthy mock attributes are not an opt-in


def test_trainer_requires_an_int_count():
    fake = MagicMock()
    fake.num_graphs = torch.tensor(4)
    assert _num_graphs_kwargs(GraphLevelModelWrapper(NodeGCN(), "graph_regression"), fake) == {}
