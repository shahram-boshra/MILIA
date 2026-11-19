"""The graph-level wrapper consumes ``batch``, whatever the model it wraps accepts.

PyG ``PMLP.forward(x, edge_index=None)`` takes no ``batch``. The Trainer and the Predictor used to
filter keyword arguments by that inner signature, so ``batch`` never reached
``GraphLevelModelWrapper``, which then pooled every node of the mini-batch into one row (PyG's
``global_mean_pool(x, batch=None)`` averages all rows). These tests pin that a model declaring
``consumes_batch`` receives ``batch`` from both callers, that the wrapper forwards it only to a
model whose forward accepts it, and that a missing ``batch`` for several graphs raises.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import PMLP

from milia_pipeline.models._batch_input import consumes_batch, forward_accepts
from milia_pipeline.models.factory.model_factory import GraphLevelModelWrapper
from milia_pipeline.models.post_training.inference.predictor import Predictor
from milia_pipeline.models.training.trainer import Trainer

N_GRAPHS = 3


class NoBatch(nn.Module):
    """Node-level model whose forward has no ``batch`` and no ``**kwargs`` (like PMLP)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 2)
        self.calls: list[set[str]] = []

    def forward(self, x, edge_index):
        self.calls.append(set())
        return self.lin(x)


class WithBatch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 2)
        self.seen = None

    def forward(self, x, edge_index, batch=None):
        self.seen = batch
        return self.lin(x)


def _batch() -> Batch:
    gen = torch.Generator().manual_seed(0)
    graphs = []
    for nodes in (3, 5, 4):  # different sizes: one row per node never equals one row per graph
        ei = torch.tensor([[i, (i + 1) % nodes] for i in range(nodes)]).t()
        graphs.append(Data(x=torch.randn(nodes, 4, generator=gen), edge_index=ei))
    return Batch.from_data_list(graphs)


# --- helpers -----------------------------------------------------------------------------------
def test_forward_accepts():
    class Kw(nn.Module):
        def forward(self, x, **kwargs):
            return x

    class KwOnly(nn.Module):
        def forward(self, x, *, batch=None):
            return x

    class PosOnly(nn.Module):
        def forward(self, x, batch=None, /):
            return x

    assert forward_accepts(WithBatch(), "batch")
    assert forward_accepts(Kw(), "batch")
    assert forward_accepts(KwOnly(), "batch")
    assert not forward_accepts(PosOnly(), "batch")  # cannot be passed as a keyword
    assert not forward_accepts(NoBatch(), "batch")
    assert not forward_accepts(PMLP(4, 8, 2, num_layers=2), "batch")


def test_consumes_batch_is_an_explicit_class_opt_in():
    assert consumes_batch(GraphLevelModelWrapper(NoBatch(), "graph_regression"))
    assert not consumes_batch(NoBatch())
    assert not consumes_batch(MagicMock())  # truthy mock attributes are not an opt-in


# --- GraphLevelModelWrapper --------------------------------------------------------------------
def test_wrapper_pools_per_graph_for_a_model_without_batch():
    torch.manual_seed(0)
    b = _batch()
    inner = NoBatch()
    out = GraphLevelModelWrapper(inner, "graph_regression")(b.x, b.edge_index, batch=b.batch)
    assert out.shape == (N_GRAPHS, 2)
    assert torch.allclose(out, global_mean_pool(inner.lin(b.x), b.batch))


def test_wrapper_forwards_batch_to_a_model_that_accepts_it():
    b = _batch()
    inner = WithBatch()
    GraphLevelModelWrapper(inner, "graph_regression")(b.x, b.edge_index, batch=b.batch)
    assert inner.seen is b.batch


def test_wrapper_raises_when_several_graphs_have_no_batch():
    b = _batch()
    wrapper = GraphLevelModelWrapper(NoBatch(), "graph_regression")
    with pytest.raises(ValueError, match="no `batch` vector"):
        wrapper(b.x, b.edge_index, num_graphs=N_GRAPHS)


def test_wrapper_single_graph_without_batch_still_pools():
    b = _batch()
    out = GraphLevelModelWrapper(NoBatch(), "graph_regression")(b.x, b.edge_index, num_graphs=1)
    assert out.shape == (1, 2)


# --- callers -----------------------------------------------------------------------------------
def test_predictor_passes_batch_to_the_wrapper(tmp_path):
    torch.manual_seed(0)
    b = _batch()
    inner = PMLP(4, 8, 2, num_layers=2)
    wrapper = GraphLevelModelWrapper(inner, "graph_regression").eval()
    predictor = Predictor(wrapper, working_root_dir=tmp_path, task_type="graph_regression")
    with torch.no_grad():
        out = predictor._forward(b)
        expected = global_mean_pool(inner(b.x, b.edge_index), b.batch)
    assert out.shape == (N_GRAPHS, 2)
    assert torch.allclose(out, expected)


def test_trainer_dynamic_forward_passes_batch_to_the_wrapper():
    torch.manual_seed(0)
    b = _batch()
    inner = NoBatch()
    trainer = Trainer.__new__(Trainer)  # only the forward helpers are exercised
    trainer.model = GraphLevelModelWrapper(inner, "graph_regression")
    trainer._forward_params = None
    trainer._uses_edge_features = None
    out = trainer._forward_with_dynamic_signature(b)
    assert out.shape == (N_GRAPHS, 2)
    assert torch.allclose(out, global_mean_pool(inner.lin(b.x), b.batch))


def test_trainer_batch_kwargs_only_for_declaring_models():
    b = _batch()
    trainer = Trainer.__new__(Trainer)
    trainer.model = GraphLevelModelWrapper(NoBatch(), "graph_regression")
    assert trainer._batch_kwargs(b)["batch"] is b.batch
    trainer.model = NoBatch()
    assert trainer._batch_kwargs(b) == {}
    trainer.model = MagicMock()
    assert trainer._batch_kwargs(b) == {}
