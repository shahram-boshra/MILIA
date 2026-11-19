"""Model capability tags: resolution, the tasks they imply, and verification against real outputs.

The contract tests build every PyG model MILIA describes (and the node-level defaults) and check the
resolved capabilities against what the model actually returns on a two-molecule batch whose
molecules differ in size (scikit-learn ``check_estimator`` pattern). A PyG release that changes a
model's outputs fails here instead of producing wrong rows in training.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch_geometric.nn.models as pyg_models
import torch_geometric.typing as pyg_typing

from milia_pipeline.models.capabilities import (
    EDGE_TASKS,
    GRAPH_TASKS,
    NODE_TASKS,
    ModelCapabilities,
    ModelCapabilityError,
    OutputSpec,
    check_model_capabilities,
    default_capabilities,
    get_model_capabilities,
    register_capabilities,
    supported_tasks,
)
from milia_pipeline.models.post_training import convert_to_pyg

GRAPH_PROPERTY = OutputSpec("property", "graph", "property")


@pytest.fixture(scope="module")
def molecules():
    gen = torch.Generator().manual_seed(0)
    graphs = [convert_to_pyg(s, format="smiles") for s in ("CCO", "c1ccccc1O")]
    for g in graphs:
        g.z = torch.randint(1, 10, (g.num_nodes,), generator=gen)
        g.pos = torch.randn(g.num_nodes, 3, generator=gen) * 1.5
    assert graphs[0].num_nodes != graphs[1].num_nodes
    return graphs


# --- the capability type and the tasks it implies -----------------------------------------------
def test_tasks_follow_the_primary_output():
    features = ModelCapabilities(outputs=(OutputSpec("features", "node", "features"),))
    assert supported_tasks(features) == NODE_TASKS + GRAPH_TASKS + EDGE_TASKS
    assert supported_tasks(ModelCapabilities(outputs=(GRAPH_PROPERTY,))) == GRAPH_TASKS
    forces = ModelCapabilities(outputs=(OutputSpec("forces", "node", "vector", width=3),))
    assert supported_tasks(forces) == ("node_regression",)
    embedding = ModelCapabilities(outputs=(OutputSpec("embedding", "node", "embedding"),))
    assert supported_tasks(embedding) == ("link_prediction", "node_embedding")


@pytest.mark.parametrize(
    "outputs",
    [(), (OutputSpec("x", "graph", "features"),), (OutputSpec("x", "graph", "vector"),)],
)
def test_invalid_primary_output_is_rejected(outputs):
    with pytest.raises(ModelCapabilityError):
        ModelCapabilities(outputs=outputs)


# --- resolution order --------------------------------------------------------------------------
class _NodeModel(nn.Module):
    def forward(self, x, edge_index, batch=None):
        return x


class _GeometricModel(nn.Module):
    supports_edge_attr = False

    def forward(self, z, pos, batch=None):
        return pos


def test_default_reads_the_forward_signature():
    caps = default_capabilities(_NodeModel)
    assert (caps.primary.level, caps.primary.kind) == ("node", "features")
    assert caps.uses_graph_structure and not caps.requires_positions
    geometric = default_capabilities(_GeometricModel)
    assert geometric.requires_positions and not geometric.uses_graph_structure


def test_own_declaration_wins_and_receives_hyperparameters():
    seen = {}

    class Declared(_NodeModel):
        @classmethod
        def milia_capabilities(cls, hyperparameters):
            seen.update(hyperparameters)
            return ModelCapabilities(outputs=(GRAPH_PROPERTY,))

    assert get_model_capabilities(Declared, {"readout": "sum"}).primary == GRAPH_PROPERTY
    assert seen == {"readout": "sum"}


def test_provider_applies_to_subclasses_and_rejects_duplicates(monkeypatch):
    from milia_pipeline.models import capabilities

    monkeypatch.setattr(capabilities, "_PROVIDERS", dict(capabilities._PROVIDERS))

    class Base(_NodeModel):
        pass

    class Child(Base):
        pass

    @register_capabilities(Base)
    def provider(hyperparameters):
        return ModelCapabilities(outputs=(GRAPH_PROPERTY,))

    assert get_model_capabilities(Child).primary == GRAPH_PROPERTY
    with pytest.raises(ModelCapabilityError, match="already registered"):
        register_capabilities(Base)(lambda hp: ModelCapabilities(outputs=(GRAPH_PROPERTY,)))


def test_declaration_must_return_capabilities():
    class Wrong(_NodeModel):
        @classmethod
        def milia_capabilities(cls, hyperparameters):
            return {"level": "graph"}

    with pytest.raises(ModelCapabilityError, match="must be ModelCapabilities"):
        get_model_capabilities(Wrong)


def test_every_pyg_provider_names_an_existing_class():
    import importlib

    from milia_pipeline.models.capabilities import _PROVIDERS

    for key in [k for k in _PROVIDERS if k.startswith("torch_geometric.")]:
        module, _, name = key.rpartition(".")
        assert hasattr(importlib.import_module(module), name), key


# --- verification against real outputs ---------------------------------------------------------
def test_check_detects_a_wrong_declaration(molecules):
    class Misdeclared(pyg_models.GCN):
        @classmethod
        def milia_capabilities(cls, hyperparameters):
            return ModelCapabilities(outputs=(GRAPH_PROPERTY,))

    with pytest.raises(ModelCapabilityError, match="declared graph-level"):
        check_model_capabilities(
            Misdeclared, {"in_channels": 6, "hidden_channels": 8, "num_layers": 2}, molecules
        )


def test_check_needs_graphs_of_different_sizes(molecules):
    with pytest.raises(ModelCapabilityError, match="different sizes"):
        check_model_capabilities(pyg_models.GCN, {}, [molecules[0], molecules[0]])


H = {"hidden_channels": 16, "out_channels": 2}
EDGE_DIM = 3
NEEDS_RADIUS = pytest.mark.skipif(
    not pyg_typing.WITH_RADIUS, reason="radius_graph needs pyg-lib with the radius op"
)
CASES = [
    ("GCN", {"in_channels": 6, "num_layers": 2, **H}, ("node", "features")),
    ("GAT", {"in_channels": 6, "num_layers": 2, **H}, ("node", "features")),
    ("GIN", {"in_channels": 6, "num_layers": 2, **H}, ("node", "features")),
    ("GraphSAGE", {"in_channels": 6, "num_layers": 2, **H}, ("node", "features")),
    ("EdgeCNN", {"in_channels": 6, "num_layers": 2, **H}, ("node", "features")),
    ("PMLP", {"in_channels": 6, "num_layers": 2, **H}, ("node", "features")),
    ("MLP", {"in_channels": 6, "num_layers": 2, **H}, ("node", "features")),
    ("Polynormer", {"in_channels": 6, **H}, ("node", "features")),
    (
        "AttentiveFP",
        {"in_channels": 6, "num_layers": 2, "num_timesteps": 2, "edge_dim": EDGE_DIM, **H},
        ("graph", "property"),
    ),
    ("NeuralFingerprint", {"in_channels": 6, "num_layers": 2, **H}, ("graph", "property")),
    pytest.param(
        "SchNet",
        {"hidden_channels": 16, "num_filters": 16, "num_interactions": 2},
        ("graph", "property"),
        marks=NEEDS_RADIUS,
    ),
    pytest.param(
        "DimeNet",
        {"num_blocks": 2, "num_bilinear": 4, "num_spherical": 4, "num_radial": 4, **H},
        ("graph", "property"),
        marks=NEEDS_RADIUS,
    ),
    pytest.param(
        "DimeNetPlusPlus",
        {
            "num_blocks": 2,
            "int_emb_size": 8,
            "basis_emb_size": 4,
            "out_emb_channels": 16,
            "num_spherical": 4,
            "num_radial": 4,
            **H,
        },
        ("graph", "property"),
        marks=NEEDS_RADIUS,
    ),
    pytest.param(
        "ViSNet",
        {"hidden_channels": 16, "num_layers": 2},
        ("graph", "property"),
        marks=NEEDS_RADIUS,
    ),
    pytest.param(
        "ViSNet",
        {"hidden_channels": 16, "num_layers": 2, "derivative": True},
        ("graph", "property"),
        marks=NEEDS_RADIUS,
        id="ViSNet-derivative",
    ),
    pytest.param(
        "GNNFF",
        {"hidden_node_channels": 16, "hidden_edge_channels": 16, "num_layers": 2},
        ("node", "vector"),
        marks=NEEDS_RADIUS,
    ),
]


@pytest.mark.parametrize(("name", "hyperparameters", "primary"), CASES)
def test_capabilities_match_real_outputs(name, hyperparameters, primary, molecules):
    torch.manual_seed(0)
    caps = check_model_capabilities(getattr(pyg_models, name), hyperparameters, molecules)
    assert (caps.primary.level, caps.primary.kind) == primary
