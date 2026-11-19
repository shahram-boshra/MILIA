"""ONNX export of MILIA models (milia_pipeline.models.deployment.onnx_export).

The export tests need the `onnx` extra (onnx, onnxscript, onnxruntime) and run the exported file
with ONNX Runtime on batches of other sizes than the example. The error-path tests run without it.
"""

import importlib.util
import warnings
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from milia_pipeline.exceptions import ExportError, OptimizationError
from milia_pipeline.models.deployment import optimize_model_for_deployment
from milia_pipeline.models.deployment.onnx_export import export_onnx, graph_example_inputs
from milia_pipeline.models.factory.model_factory import ModelFactory


def _graph(num_nodes: int, edge_dim: int = 0) -> Data:
    src = torch.arange(num_nodes - 1)
    edge_index = torch.stack([src, src + 1])
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    data = Data(x=torch.randn(num_nodes, 8), edge_index=edge_index, y=torch.randn(1, 2))
    if edge_dim:
        data.edge_attr = torch.randn(edge_index.size(1), edge_dim)
    return data


def _model(name: str, **extra) -> nn.Module:
    torch.manual_seed(0)
    return (
        ModelFactory()
        .create_model(
            name=name,
            hyperparameters={"hidden_channels": 16, "num_layers": 2, **extra},
            task_type="graph_regression",
            sample_data=_graph(5, extra.get("edge_dim", 0)),
        )
        .eval()
    )


class TestWithoutTheExtra:
    def test_missing_packages_name_the_extra(self, tmp_path):
        with (
            patch("importlib.util.find_spec", return_value=None),
            pytest.raises(ExportError, match=r"pip install 'milia-py\[onnx\]'"),
        ):
            export_onnx(nn.Linear(2, 2), tmp_path / "m.onnx", torch.randn(1, 2))

    def test_dynamic_shapes_and_axes_together_are_rejected(self, tmp_path):
        with pytest.raises(ExportError, match="not both"):
            export_onnx(
                nn.Linear(2, 2),
                tmp_path / "m.onnx",
                torch.randn(1, 2),
                dynamic_shapes={"input": {0: None}},
                dynamic_axes={"input": {0: "n"}},
            )

    def test_graph_model_needs_a_graph_example(self, tmp_path):
        with (
            patch("milia_pipeline.models.deployment.onnx_export._require"),
            pytest.raises(ExportError, match="PyG Data or Batch example"),
        ):
            export_onnx(_model("GIN"), tmp_path / "m.onnx", torch.randn(5, 8))

    def test_graph_example_inputs(self):
        names, tensors = graph_example_inputs(_graph(4, edge_dim=3))
        assert names == ("x", "edge_index", "edge_attr", "batch")
        assert torch.equal(tensors[-1], torch.zeros(4, dtype=torch.long))  # one graph
        names, _ = graph_example_inputs(Batch.from_data_list([_graph(4), _graph(3)]))
        assert names == ("x", "edge_index", "batch")


needs_onnx = pytest.mark.skipif(
    not all(importlib.util.find_spec(name) for name in ("onnx", "onnxscript", "onnxruntime")),
    reason="the `onnx` extra is not installed",
)


def _run(path, names, batch: Batch) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {name: getattr(batch, name).numpy() for name in names}
    return session.run(None, feeds)[0]


@needs_onnx
class TestGraphModelExport:
    @pytest.fixture(autouse=True)
    def _quiet(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # torch.export / onnxscript progress warnings
            yield

    @pytest.mark.parametrize("name", ["GCN", "GIN", "GraphSAGE"])
    def test_exported_model_accepts_other_batch_sizes(self, name, tmp_path):
        model = _model(name)
        example = Batch.from_data_list([_graph(5), _graph(7), _graph(4)])
        path = export_onnx(model, tmp_path / f"{name}.onnx", example)

        import onnx

        assert [i.name for i in onnx.load(str(path)).graph.input] == ["x", "edge_index", "batch"]
        for graphs in ([_graph(6), _graph(3)], [_graph(9)] * 5):
            batch = Batch.from_data_list(graphs)
            with torch.no_grad():
                expected = model(batch.x, batch.edge_index, batch=batch.batch).numpy()
            actual = _run(path, ["x", "edge_index", "batch"], batch)
            assert actual.shape == (len(graphs), 2)
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_single_graph_example(self, tmp_path):
        model = _model("GIN")
        path = export_onnx(model, tmp_path / "gin.onnx", _graph(6))
        batch = Batch.from_data_list([_graph(4), _graph(8)])
        assert _run(path, ["x", "edge_index", "batch"], batch).shape == (2, 2)

    def test_gat_with_edge_features(self, tmp_path):
        """GAT adds self-loops with edge-feature fill: only a strict capture exports it."""
        model = _model("GAT", edge_dim=3)
        names = ["x", "edge_index", "edge_attr", "batch"]
        path = export_onnx(model, tmp_path / "gat.onnx", Batch.from_data_list([_graph(5, 3)] * 2))
        batch = Batch.from_data_list([_graph(6, 3), _graph(3, 3), _graph(8, 3)])
        with torch.no_grad():
            expected = model(
                batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch
            ).numpy()
        np.testing.assert_allclose(_run(path, names, batch), expected, rtol=1e-4, atol=1e-5)

    def test_parallel_ensemble(self, tmp_path):
        """A MILIA ensemble declares graph-level output, so its wrapper does not branch on shapes."""
        torch.manual_seed(0)
        config = {
            "enabled": True,
            "strategy": "parallel",
            "models": [
                {"name": name, "hyperparameters": {}, "weight": weight}
                for name, weight in (("GCN", 0.4), ("GAT", 0.3), ("GraphSAGE", 0.3))
            ],
            "fusion": {"method": "weighted"},
        }
        model = (
            ModelFactory()
            .create_model(
                name="ensemble",
                hyperparameters={"ensemble_config": config, "in_channels": 8, "out_channels": 2},
                task_type="graph_regression",
                sample_data=_graph(5, 3),
            )
            .eval()
        )
        names = ["x", "edge_index", "edge_attr", "batch"]
        path = export_onnx(model, tmp_path / "ens.onnx", Batch.from_data_list([_graph(5, 3)] * 3))
        batch = Batch.from_data_list([_graph(6, 3), _graph(3, 3)])
        with torch.no_grad():
            expected = model(
                batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch
            ).numpy()
        np.testing.assert_allclose(_run(path, names, batch), expected, rtol=1e-4, atol=1e-5)

    def test_uncapturable_model_names_both_attempts(self, tmp_path):
        """Python control flow on tensor values cannot be captured in either mode."""

        class ValueBranch(nn.Module):
            def forward(self, x):
                return x * 2 if x.sum() > 0 else x

        with pytest.raises(ExportError, match=r"strict=True: .*strict=False: "):
            export_onnx(ValueBranch(), tmp_path / "v.onnx", torch.ones(3))

    def test_verification_mismatch_removes_the_file(self, tmp_path):
        path = tmp_path / "gin.onnx"
        with (
            patch("torch.testing.assert_close", side_effect=AssertionError("differs")),
            pytest.raises(ExportError, match="does not reproduce PyTorch's output"),
        ):
            export_onnx(_model("GIN"), path, Batch.from_data_list([_graph(5), _graph(4)]))
        assert not path.exists()

    def test_optimize_model_for_deployment_exports(self, tmp_path):
        path = tmp_path / "deploy.onnx"
        example = Batch.from_data_list([_graph(5), _graph(4)])
        optimize_model_for_deployment(
            _model("GIN"), export_onnx=True, onnx_path=path, dummy_input=example, verbose=False
        )
        assert path.exists()


@needs_onnx
class TestModuleExport:
    def test_plain_module(self, tmp_path):
        import onnxruntime as ort

        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
        x = torch.randn(5, 3)
        path = export_onnx(model, tmp_path / "mlp.onnx", x)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        actual = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
        np.testing.assert_allclose(actual, model(x).detach().numpy(), rtol=1e-4, atol=1e-5)

    def test_export_error_is_an_optimization_error_in_the_deployment_helper(self, tmp_path):
        with pytest.raises(OptimizationError):
            optimize_model_for_deployment(
                _model("GIN"),
                export_onnx=True,
                onnx_path=tmp_path / "x.onnx",
                dummy_input=torch.randn(5, 8),
                verbose=False,
            )
