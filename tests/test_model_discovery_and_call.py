"""PyG model discovery lists only PyG's public models; MLP is called without edge_index.

Discovery scanned ``dir()`` of ``torch_geometric.nn.models`` and its submodules, so what those
modules import (``torch.nn.GRU``/``LSTM``/``Linear``/``Embedding``/``GRUCell``) and internal
helpers (``TimeEncoder``, ``IdentityMessage``, ``LastAggregator``, ``MeanAggregator``,
``SignedConv``) were registered as models for every task. And PyG ``MLP.forward(x, batch=None,
...)`` received MILIA's ``(x, edge_index)`` call: ``edge_index`` landed in ``batch``.
"""

from __future__ import annotations

import pytest
import torch
import torch_geometric.nn.models as pyg_models
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from milia_pipeline.models.capabilities import attached_capabilities, default_capabilities
from milia_pipeline.models.factory.model_factory import ModelFactory
from milia_pipeline.models.post_training import convert_to_pyg
from milia_pipeline.models.post_training.inference.predictor import Predictor
from milia_pipeline.models.registry.pyg_introspector import discover_pyg_models
from milia_pipeline.models.training.trainer import Trainer

NOT_MODELS = (
    "Embedding",
    "GRU",
    "GRUCell",
    "IdentityMessage",
    "LSTM",
    "LastAggregator",
    "Linear",
    "MeanAggregator",
    "SignedConv",
    "TimeEncoder",
)
H = {"hidden_channels": 16, "num_layers": 2, "out_channels": 2}


@pytest.fixture(scope="module")
def batch():
    return Batch.from_data_list(
        [convert_to_pyg(s, format="smiles") for s in ("CCO", "c1ccccc1O", "CC(=O)O")]
    )


# --- discovery -------------------------------------------------------------------------------
def test_discovery_is_the_public_api_of_pyg_models():
    discovered = discover_pyg_models()
    public = set(pyg_models.__all__)
    assert set(discovered) <= public
    expected = {
        name
        for name in public
        if isinstance(getattr(pyg_models, name), type)
        and issubclass(getattr(pyg_models, name), torch.nn.Module)
    }
    assert set(discovered) == expected
    assert all(path == f"torch_geometric.nn.models.{name}" for name, path in discovered.items())


@pytest.mark.parametrize("name", NOT_MODELS)
def test_imported_and_internal_classes_are_not_models(name):
    assert name not in discover_pyg_models()


# --- the call convention follows the capabilities ----------------------------------------------
def test_default_capabilities_of_mlp_and_variadic_models():
    assert not default_capabilities(pyg_models.MLP).uses_graph_structure
    # forward(*args, **kwargs) passes everything on: assumed to read the edges
    assert default_capabilities(pyg_models.GAE).uses_graph_structure
    assert default_capabilities(pyg_models.GCN).uses_graph_structure


def _mlp(batch, task):
    torch.manual_seed(0)
    return (
        ModelFactory()
        .create_model(
            name="MLP", hyperparameters=H, task_type=task, sample_data=batch.get_example(0)
        )
        .eval()
    )


def test_mlp_graph_task_through_the_wrapper(batch, tmp_path):
    model = _mlp(batch, "graph_regression")
    assert not attached_capabilities(model.model).uses_graph_structure
    with torch.no_grad():
        out = Predictor(model, working_root_dir=tmp_path, task_type="graph_regression")._forward(
            batch
        )
        expected = global_mean_pool(model.model(batch.x), batch.batch)
    assert out.shape == (batch.num_graphs, 2)
    assert torch.allclose(out, expected, atol=1e-6)


def test_mlp_node_task_called_directly(batch, tmp_path):
    model = _mlp(batch, "node_regression")
    with torch.no_grad():
        out = Predictor(model, working_root_dir=tmp_path, task_type="node_regression")._forward(
            batch
        )
        expected = model(batch.x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_mlp_trainer_forward(batch):
    model = _mlp(batch, "graph_regression")
    trainer = Trainer.__new__(Trainer)  # only the forward helper is exercised
    trainer.model = model
    trainer._forward_params = None
    trainer._uses_edge_features = None
    with torch.no_grad():
        out = trainer._forward_with_dynamic_signature(batch)
    assert out.shape == (batch.num_graphs, 2)


def test_gcn_still_receives_edge_index(batch):
    torch.manual_seed(0)
    model = (
        ModelFactory()
        .create_model(
            name="GCN",
            hyperparameters=H,
            task_type="graph_regression",
            sample_data=batch.get_example(0),
        )
        .eval()
    )
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch=batch.batch)
        expected = global_mean_pool(model.model(batch.x, batch.edge_index), batch.batch)
    assert torch.allclose(out, expected, atol=1e-6)
