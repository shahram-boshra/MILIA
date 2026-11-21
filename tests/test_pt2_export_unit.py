"""``.pt2`` export of MILIA models (milia_pipeline.models.deployment.pt2_export).

The exported file is loaded back with PyTorch's own loaders (``torch.export.load``,
``torch._inductor.aoti_load_package``) and run on batches of other sizes than the example.
"""

from __future__ import annotations

import json
import shutil
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from milia_pipeline.exceptions import ExportError
from milia_pipeline.models.deployment.onnx_export import graph_example_inputs
from milia_pipeline.models.deployment.pt2_export import METADATA_FILE, export_pt2
from milia_pipeline.models.factory.model_factory import ModelFactory


def _graph(num_nodes: int, edge_dim: int = 0) -> Data:
    src = torch.arange(num_nodes - 1)
    edge_index = torch.stack([src, src + 1])
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    data = Data(x=torch.randn(num_nodes, 8), edge_index=edge_index, y=torch.randn(1, 2))
    if edge_dim:
        data.edge_attr = torch.randn(edge_index.size(1), edge_dim)
    return data


def _model(name: str, data_edge_dim: int = 0, **extra) -> nn.Module:
    torch.manual_seed(0)
    return (
        ModelFactory()
        .create_model(
            name=name,
            hyperparameters={"hidden_channels": 16, "num_layers": 2, "out_channels": 2, **extra},
            task_type="graph_regression",
            sample_data=_graph(5, data_edge_dim),
        )
        .eval()
    )


def _batches(edge_dim: int = 0) -> tuple[Batch, Batch]:
    torch.manual_seed(1)
    example = Batch.from_data_list([_graph(4, edge_dim), _graph(6, edge_dim)])
    other = Batch.from_data_list([_graph(n, edge_dim) for n in (3, 7, 5)])
    return example, other


def _eager(model: nn.Module, batch: Batch) -> torch.Tensor:
    with torch.no_grad():
        return model(
            batch.x,
            batch.edge_index,
            edge_attr=getattr(batch, "edge_attr", None),
            batch=batch.batch,
        )


@pytest.mark.parametrize(
    ("name", "edge_dim", "extra"),
    [
        ("GCN", 0, {}),
        ("GIN", 0, {}),
        ("GAT", 3, {"edge_dim": 3}),
        ("AttentiveFP", 3, {"edge_dim": 3, "num_timesteps": 2}),
    ],
    ids=["GCN", "GIN", "GAT-edge_attr", "AttentiveFP"],
)
def test_exported_program_runs_other_batch_sizes(name, edge_dim, extra, tmp_path):
    model = _model(name, edge_dim, **extra)
    example, other = _batches(edge_dim)
    path = export_pt2(model, tmp_path / f"{name}.pt2", example)

    loaded = torch.export.load(path).module()
    _, tensors = graph_example_inputs(other)
    torch.testing.assert_close(loaded(*tensors), _eager(model, other), rtol=1e-4, atol=1e-5)
    assert loaded(*tensors).shape == (3, 2)  # one row per graph


def test_metadata_records_the_versions_and_inputs(tmp_path):
    from milia_pipeline import __version__

    example, _ = _batches()
    path = export_pt2(_model("GCN"), tmp_path / "m.pt2", example)
    extra = {METADATA_FILE: ""}
    torch.export.load(path, extra_files=extra)
    assert json.loads(extra[METADATA_FILE]) == {
        "format": "torch.export.ExportedProgram",
        "milia_version": __version__,
        "torch_version": torch.__version__,
        "input_names": ["x", "edge_index", "batch"],
    }


def test_plain_module_exports_from_a_tensor_example(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU()).eval()
    path = export_pt2(model, tmp_path / "mlp.pt2", torch.randn(2, 4))
    x = torch.randn(2, 4)
    torch.testing.assert_close(torch.export.load(path).module()(x), model(x))


@pytest.mark.skipif(shutil.which("g++") is None, reason="AOTInductor needs a C++ compiler")
def test_aot_inductor_package_runs_other_batch_sizes(tmp_path):
    model = _model("GCN")
    example, other = _batches()
    path = export_pt2(model, tmp_path / "gcn_aoti.pt2", example, aot_compile=True)

    compiled = torch._inductor.aoti_load_package(str(path))
    _, tensors = graph_example_inputs(other)
    out = compiled(*tensors)
    out = out[0] if isinstance(out, list | tuple) else out
    torch.testing.assert_close(out, _eager(model, other), rtol=1e-4, atol=1e-5)


class TestErrors:
    def test_suffix_must_be_pt2(self, tmp_path):
        with pytest.raises(ExportError, match=r"ending in \.pt2, got m\.pt"):
            export_pt2(nn.Linear(2, 2), tmp_path / "m.pt", torch.randn(1, 2))

    def test_inductor_configs_need_aot_compile(self, tmp_path):
        with pytest.raises(ExportError, match="only with aot_compile=True"):
            export_pt2(nn.Linear(2, 2), tmp_path / "m.pt2", torch.randn(1, 2), inductor_configs={})

    def test_graph_model_needs_a_graph_example(self, tmp_path):
        with pytest.raises(ExportError, match="PyG Data or Batch example"):
            export_pt2(_model("GCN"), tmp_path / "m.pt2", torch.randn(3, 8))

    def test_capture_failure_names_both_attempts(self, tmp_path):
        with (
            patch("torch.export.export", side_effect=RuntimeError("boom")),
            pytest.raises(
                ExportError, match=r"\.pt2 export failed: .*strict=True: .*strict=False: "
            ),
        ):
            export_pt2(nn.Linear(2, 2), tmp_path / "m.pt2", torch.randn(1, 2))
        assert not (tmp_path / "m.pt2").exists()

    def test_verification_mismatch_removes_the_file(self, tmp_path):
        with (
            patch("torch.testing.assert_close", side_effect=AssertionError("differs")),
            pytest.raises(ExportError, match="does not reproduce PyTorch's output"),
        ):
            export_pt2(nn.Linear(2, 2), tmp_path / "m.pt2", torch.randn(1, 2))
        assert not (tmp_path / "m.pt2").exists()
