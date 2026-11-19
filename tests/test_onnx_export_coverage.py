"""ONNX export coverage of MILIA's molecular architectures for graph-level tasks.

Each architecture is built by ``ModelFactory`` for ``graph_regression`` and exported with
``export_onnx`` on a batch of two molecules; ``export_onnx`` runs the file with ONNX Runtime on the
example and compares it with PyTorch before keeping it. Architectures whose export is blocked by an
upstream data-dependent operation are pinned as strict xfails: when a PyG or PyTorch release lifts
the limit, the test turns XPASS and fails, so the documented limits are revisited.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
import torch_geometric.typing as pyg_typing
from torch_geometric.data import Batch

from milia_pipeline.exceptions import ExportError
from milia_pipeline.models._graph_output import produces_graph_output
from milia_pipeline.models.factory.model_factory import ModelFactory
from milia_pipeline.models.post_training import convert_to_pyg

pytestmark = pytest.mark.skipif(
    not all(importlib.util.find_spec(name) for name in ("onnx", "onnxscript", "onnxruntime")),
    reason="the `onnx` extra is not installed",
)

H = {"hidden_channels": 16, "num_layers": 2, "out_channels": 2}
NEEDS_RADIUS = pytest.mark.skipif(
    not pyg_typing.WITH_RADIUS, reason="radius_graph needs pyg-lib with the radius op"
)


def _upstream_limit(reason: str, *marks):
    return [pytest.mark.xfail(strict=True, raises=ExportError, reason=reason), *marks]


TOPK = "GraphUNet: TopKPooling keeps ceil(ratio * N) nodes, a data-dependent size (strict and non-strict capture fail)"
SGFORMER = "SGFormer: capture succeeds, ONNX decomposition fails on a size it expects constant"
RADIUS = (
    "3D models: radius_graph builds the interaction graph with a data-dependent size (index2ptr)"
)
H3D = {"hidden_channels": 16}
CASES = [
    ("GCN", H, ()),
    ("GAT", H, ()),
    ("GIN", H, ()),
    ("GraphSAGE", H, ()),
    ("EdgeCNN", H, ()),
    ("PNA", H, ()),
    ("MLP", H, ()),
    ("PMLP", {**H, "norm": False}, ()),
    ("Polynormer", {"hidden_channels": 16, "out_channels": 2}, ()),
    ("AttentiveFP", {**H, "num_timesteps": 2, "edge_dim": "from_data"}, ()),
    ("NeuralFingerprint", H, ()),
    ("GraphUNet", {"hidden_channels": 16, "out_channels": 2, "depth": 2}, _upstream_limit(TOPK)),
    ("SGFormer", {"hidden_channels": 16, "out_channels": 2}, _upstream_limit(SGFORMER)),
    (
        "SchNet",
        {**H3D, "num_filters": 16, "num_interactions": 2},
        _upstream_limit(RADIUS, NEEDS_RADIUS),
    ),
    (
        "DimeNet",
        {
            **H3D,
            "out_channels": 2,
            "num_blocks": 2,
            "num_bilinear": 4,
            "num_spherical": 4,
            "num_radial": 4,
        },
        _upstream_limit(RADIUS, NEEDS_RADIUS),
    ),
    ("ViSNet", {**H3D, "num_layers": 2}, _upstream_limit(RADIUS, NEEDS_RADIUS)),
    (
        "DimeNetPlusPlus",
        {
            **H3D,
            "out_channels": 2,
            "num_blocks": 2,
            "int_emb_size": 8,
            "basis_emb_size": 4,
            "out_emb_channels": 16,
            "num_spherical": 4,
            "num_radial": 4,
        },
        _upstream_limit(RADIUS, NEEDS_RADIUS),
    ),
]


@pytest.fixture(scope="module")
def molecules():
    gen = torch.Generator().manual_seed(0)
    graphs = [convert_to_pyg(s, format="smiles") for s in ("CCO", "c1ccccc1O", "CC(=O)O")]
    for g in graphs:
        g.z = torch.randint(1, 10, (g.num_nodes,), generator=gen)
        g.pos = torch.randn(g.num_nodes, 3, generator=gen) * 1.5
    return graphs


@pytest.mark.parametrize(
    ("name", "hyperparameters"),
    [pytest.param(name, hp, marks=marks, id=name) for name, hp, marks in CASES],
)
def test_graph_regression_model_exports(name, hyperparameters, molecules, tmp_path):
    from milia_pipeline.models.deployment.onnx_export import export_onnx

    hp = dict(hyperparameters)
    if hp.get("edge_dim") == "from_data":
        hp["edge_dim"] = molecules[0].edge_attr.size(1)
    torch.manual_seed(0)
    model = (
        ModelFactory()
        .create_model(
            name=name, hyperparameters=hp, task_type="graph_regression", sample_data=molecules[0]
        )
        .eval()
    )
    path = export_onnx(model, tmp_path / f"{name}.onnx", Batch.from_data_list(molecules[:2]))
    assert path.exists()


@pytest.mark.parametrize("name", ["AttentiveFP", "NeuralFingerprint"])
def test_built_in_readout_models_declare_graph_output(name, molecules):
    hp = (
        dict(H, num_timesteps=2, edge_dim=molecules[0].edge_attr.size(1))
        if name == "AttentiveFP"
        else H
    )
    wrapper = ModelFactory().create_model(
        name=name, hyperparameters=hp, task_type="graph_regression", sample_data=molecules[0]
    )
    # declared by the capabilities the factory attached, not inferred from the output's rows
    assert produces_graph_output(wrapper.model)
    assert not produces_graph_output(
        ModelFactory()
        .create_model(
            name="GCN", hyperparameters=H, task_type="graph_regression", sample_data=molecules[0]
        )
        .model
    )
