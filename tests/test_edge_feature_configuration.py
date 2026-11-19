"""Edge features reach a model only when it was built to use them.

PyG ``PNA`` and ``GAT`` set ``supports_edge_attr``, so ``BasicGNN`` hands ``edge_attr`` to every
conv; the convs use it only when built with ``edge_dim`` (``PNAConv.edge_encoder`` and
``GATConv.lin_edge`` exist only then). MILIA passed a molecule's bond features to a PNA built
without ``edge_dim`` from the Predictor, ensembles and ONNX export, and PNAConv failed with
``'PNAConv' object has no attribute 'edge_encoder'``. The factory now attaches the capabilities
resolved for the built hyperparameters, and every caller follows them.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn.models import GAT, PNA

from milia_pipeline.models.builders.model_composer import ParallelEnsemble
from milia_pipeline.models.capabilities import (
    ModelCapabilities,
    ModelCapabilityError,
    OutputSpec,
    attach_capabilities,
    attached_capabilities,
    get_model_capabilities,
)
from milia_pipeline.models.factory.model_factory import GraphLevelModelWrapper, ModelFactory
from milia_pipeline.models.post_training import convert_to_pyg
from milia_pipeline.models.post_training.inference.predictor import Predictor

H = {"hidden_channels": 16, "num_layers": 2, "out_channels": 2}


@pytest.fixture(scope="module")
def molecules():
    return [convert_to_pyg(s, format="smiles") for s in ("CCO", "c1ccccc1O", "CC(=O)O")]


# --- capabilities for the built hyperparameters ----------------------------------------------
@pytest.mark.parametrize("model_class", [GAT, PNA])
def test_edge_features_follow_edge_dim(model_class):
    assert not get_model_capabilities(model_class, {}).accepts_edge_attr
    assert get_model_capabilities(model_class, {"edge_dim": 3}).accepts_edge_attr


def test_attached_capabilities_belong_to_the_instance():
    caps = ModelCapabilities(outputs=(OutputSpec("features", "node", "features"),))
    inner = nn.Linear(2, 2)
    attach_capabilities(inner, caps)
    assert attached_capabilities(inner) is caps
    assert "_milia_capabilities" not in str(list(inner.state_dict()))
    # the wrapper delegates unknown attributes to the model it holds, but does not report them
    assert attached_capabilities(GraphLevelModelWrapper(inner, "graph_regression")) is None
    assert attached_capabilities(nn.Linear(2, 2)) is None
    with pytest.raises(ModelCapabilityError):
        attach_capabilities(inner, {"accepts_edge_attr": False})


def test_compiled_module_is_looked_through():
    caps = ModelCapabilities(outputs=(OutputSpec("features", "node", "features"),))
    inner = nn.Linear(2, 2)
    attach_capabilities(inner, caps)

    class Compiled(nn.Module):  # stands in for torch.compile's OptimizedModule
        def __init__(self, module):
            super().__init__()
            self._orig_mod = module

    assert attached_capabilities(Compiled(inner)) is caps


# --- the factory attaches, the callers follow --------------------------------------------------
def _pna(sample, **extra):
    torch.manual_seed(0)
    return ModelFactory().create_model(
        name="PNA",
        hyperparameters={**H, **extra},
        task_type="graph_regression",
        sample_data=sample,
    )


def test_pna_without_edge_dim_predicts(molecules, tmp_path):
    model = _pna(molecules[0]).eval()
    assert not attached_capabilities(model.model).accepts_edge_attr
    batch = Batch.from_data_list(molecules)
    with torch.no_grad():
        out = Predictor(model, working_root_dir=tmp_path, task_type="graph_regression")._forward(
            batch
        )
    assert out.shape == (len(molecules), 2)


def test_pna_with_edge_dim_uses_bond_features(molecules):
    edge_dim = molecules[0].edge_attr.size(1)
    model = _pna(molecules[0], edge_dim=edge_dim).eval()
    assert attached_capabilities(model.model).accepts_edge_attr
    batch = Batch.from_data_list(molecules)
    kwargs = {"batch": batch.batch, "num_graphs": batch.num_graphs}
    with torch.no_grad():
        a = model(batch.x, batch.edge_index, edge_attr=batch.edge_attr, **kwargs)
        b = model(batch.x, batch.edge_index, edge_attr=torch.zeros_like(batch.edge_attr), **kwargs)
    assert not torch.allclose(a, b)  # the bond features change the prediction


def test_gat_without_edge_dim_is_unchanged_by_dropping_edge_features(molecules):
    # GATConv ignores edge_attr without lin_edge, so dropping it changes nothing (trained GAT
    # models keep their predictions).
    torch.manual_seed(0)
    model = (
        ModelFactory()
        .create_model(
            name="GAT", hyperparameters=H, task_type="graph_regression", sample_data=molecules[0]
        )
        .eval()
    )
    batch = Batch.from_data_list(molecules)
    with torch.no_grad():
        dropped = model(batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
        direct = model.model(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
    from torch_geometric.nn import global_mean_pool

    assert torch.allclose(dropped, global_mean_pool(direct, batch.batch), atol=1e-6)


def test_ensemble_member_pna_without_edge_dim(molecules):
    member = ModelFactory().create_model(
        name="PNA",
        hyperparameters=H,
        task_type="graph_regression",
        sample_data=molecules[0],
        _skip_wrapper=True,
    )
    ensemble = ParallelEnsemble(
        models=[member], weights=[1.0], fusion="mean", task_type="graph_regression"
    ).eval()
    batch = Batch.from_data_list(molecules)
    with torch.no_grad():
        out = ensemble(batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
    assert out.shape[0] == len(molecules)
