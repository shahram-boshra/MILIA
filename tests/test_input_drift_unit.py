"""Input drift: training-input reference and the serving-time monitor (F65 P8)."""

from __future__ import annotations

import json
import logging
import threading
import time

import pytest
import torch
import torch.nn as nn
from checkpoint_records import IDENTIFIER_DATA_INFO
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from milia_pipeline.exceptions import DeploymentError
from milia_pipeline.models.checkpoint_io import load_checkpoint_file
from milia_pipeline.models.deployment import drift
from milia_pipeline.models.deployment.drift import (
    DriftMonitor,
    build_input_reference,
    drift_settings,
    jensen_shannon_distance,
)
from milia_pipeline.models.training.callbacks import ModelCheckpoint
from milia_pipeline.models.training.trainer import Trainer


def _graphs(n, seed=0, shift=0.0, atoms=(3, 6)):
    """Molecules with a one-hot column (x_0), a continuous column (x_1) and 3..5 atoms."""
    generator = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        size = int(torch.randint(atoms[0], atoms[1], (1,), generator=generator))
        onehot = (torch.rand(size, generator=generator) < 0.3).float()
        continuous = torch.randn(size, generator=generator) + shift
        x = torch.stack([onehot, continuous], dim=1)
        src = torch.arange(size - 1)
        edges = torch.cat([torch.stack([src, src + 1]), torch.stack([src + 1, src])], 1)
        out.append(Data(x=x, edge_index=edges, y=torch.zeros(1, 1)))
    return out


def _reference(n=600, seed=0):
    return build_input_reference(DataLoader(_graphs(n, seed), batch_size=32))


class TestDistance:
    def test_bounds_and_a_known_value(self):
        assert jensen_shannon_distance([1, 2, 3], [2, 4, 6]) == pytest.approx(0.0, abs=1e-9)
        assert jensen_shannon_distance([1, 0], [0, 1]) == pytest.approx(1.0)
        # p = (1/2, 1/2), q = (1, 0), m = (3/4, 1/4): JSD = (KL(p||m) + KL(q||m)) / 2
        p, q, m = [0.5, 0.5], [1.0, 0.0], [0.75, 0.25]
        expected = (
            0.5 * (0.5 * torch.log2(torch.tensor(0.5 / 0.75)) + 0.5 * torch.log2(torch.tensor(2.0)))
            + 0.5 * (1.0 * torch.log2(torch.tensor(1 / m[0])))
        ).item() ** 0.5
        assert jensen_shannon_distance(p, q) == pytest.approx(expected)


class TestReference:
    def test_kinds_and_frequencies(self):
        reference = _reference()
        features = reference["features"]
        assert set(features) == {"x_0", "x_1", "num_atoms"}
        assert features["x_0"]["kind"] == "categorical" and features["x_0"]["values"] == [0.0, 1.0]
        assert features["x_0"]["probabilities"][-1] == 0.0  # the bucket for unseen values
        assert features["x_1"]["kind"] == "numerical" and len(features["x_1"]["edges"]) == 9
        # deciles: each of the 10 bins holds about a tenth of the training atoms
        assert all(abs(p - 0.1) < 0.02 for p in features["x_1"]["probabilities"])
        assert features["num_atoms"]["kind"] == "categorical"
        assert features["num_atoms"]["values"] == [3.0, 4.0, 5.0]
        assert reference["molecules"] == 600

    def test_deterministic_and_bounded(self, monkeypatch):
        assert _reference() == _reference()
        monkeypatch.setattr(drift, "MAX_REFERENCE_ROWS", 50)
        reference = _reference()
        assert reference["atoms"] > 50  # all atoms counted, a uniform sample of 50 kept
        assert abs(sum(reference["features"]["x_1"]["probabilities"]) - 1) < 1e-9

    def test_no_node_features_give_no_reference(self):
        assert build_input_reference([Data(edge_index=torch.zeros(2, 0).long())]) is None


class TestMonitor:
    def test_collects_until_the_window_is_full(self):
        monitor = DriftMonitor(_reference(), window=100)
        monitor.observe(_graphs(99, seed=1))
        assert monitor.report()["status"] == "collecting"
        monitor.observe(_graphs(1, seed=2))
        report = monitor.report()
        # evaluated once full (100 molecules is too few for the 0.1 threshold: noise alone
        # reaches it, hence the default window of 1000)
        assert report["status"] in {"no_drift", "drift"} and report["molecules"] == 100
        assert set(report["distances"]) == {"x_0", "x_1", "num_atoms"}

    def test_training_like_requests_do_not_drift_and_shifted_ones_do(self, caplog):
        monitor = DriftMonitor(_reference(), window=500)
        monitor.observe(_graphs(500, seed=3))
        report = monitor.report()
        assert report["status"] == "no_drift", report
        assert max(report["distances"].values()) < 0.1
        with caplog.at_level(logging.WARNING, logger=drift.__name__):
            monitor.observe(_graphs(500, seed=4, shift=1.5, atoms=(6, 9)))
        report = monitor.report()
        assert report["status"] == "drift"
        assert report["drifted"] == ["num_atoms", "x_1"]  # not the unchanged one-hot column
        record = json.loads(caplog.records[-1].getMessage())
        assert record["event"] == "input_drift" and record["status"] == "drift"
        assert set(record["drifted"]) == {"num_atoms", "x_1"}

    def test_the_window_slides(self):
        monitor = DriftMonitor(_reference(), window=200)
        monitor.observe(_graphs(200, seed=5, shift=2.0))
        assert monitor.report()["status"] == "drift"
        monitor.observe(_graphs(200, seed=6))
        assert monitor.report()["status"] == "no_drift"
        assert monitor.report()["molecules"] == 200

    def test_unseen_categorical_values_count_as_drift(self):
        monitor = DriftMonitor(_reference(), window=100)
        graphs = _graphs(100, seed=7)
        for g in graphs:
            g.x[:, 0] = 2.0  # a value the one-hot column never had
        monitor.observe(graphs)
        assert "x_0" in monitor.report()["drifted"]

    def test_without_reference(self):
        monitor = DriftMonitor(None)
        monitor.observe(_graphs(3))
        assert monitor.report()["status"] == "not_recorded"
        with pytest.raises(DeploymentError, match="input reference version"):
            DriftMonitor({"version": 2, "features": {}})


class TestSettings:
    def test_defaults_and_validation(self, monkeypatch):
        assert drift_settings() == (0.1, 1000)
        monkeypatch.setenv("MILIA_DRIFT_THRESHOLD", "0.2")
        monkeypatch.setenv("MILIA_DRIFT_WINDOW", "50")
        assert drift_settings() == (0.2, 50)
        monkeypatch.setenv("MILIA_DRIFT_WINDOW", "0")
        with pytest.raises(DeploymentError, match="at least 1"):
            drift_settings()
        monkeypatch.setenv("MILIA_DRIFT_THRESHOLD", "high")
        with pytest.raises(DeploymentError, match="must be a number"):
            drift_settings()


class _Pooled(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 1)

    def forward(self, x, edge_index, batch=None):
        return self.lin(global_mean_pool(x, batch))


def _trainer(tmp_path, record=True):
    model = _Pooled()
    return Trainer(
        model=model,
        train_loader=DataLoader(_graphs(40, 8), batch_size=8),
        val_loader=DataLoader(_graphs(10, 9), batch_size=8),
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        device=torch.device("cpu"),
        callbacks=[ModelCheckpoint(dirpath=tmp_path / "checkpoints", verbose=False)],
        model_info={"task_type": "graph_regression"},
        max_epochs=1,
        record_input_reference=record,
    )


class TestTraining:
    def test_the_reference_of_the_training_split_is_stored(self, tmp_path):
        trainer = _trainer(tmp_path)
        trainer.fit(test=False)
        best = load_checkpoint_file(tmp_path / "checkpoints" / "best.pt", map_location="cpu")
        assert best["input_reference"] == trainer.input_reference
        assert best["input_reference"]["molecules"] == 40
        resumed = _trainer(tmp_path / "r")
        resumed.load_checkpoint(tmp_path / "checkpoints" / "best.pt")
        assert resumed.input_reference == trainer.input_reference

    def test_off_by_default_in_the_api(self, tmp_path):
        trainer = _trainer(tmp_path, record=False)
        trainer.fit(test=False)
        best = load_checkpoint_file(tmp_path / "checkpoints" / "best.pt", map_location="cpu")
        assert "input_reference" not in best


class TestServing:
    def _package(self, tmp_path, reference):
        from milia_pipeline.models.deployment.package import package_checkpoint
        from milia_pipeline.models.factory.model_factory import ModelFactory
        from milia_pipeline.models.post_training import convert_to_pyg
        from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
            CheckpointManager,
        )

        torch.manual_seed(0)
        in_channels = convert_to_pyg("CCO", format="smiles").x.shape[1]
        model, info = ModelFactory().create_model_with_info(
            name="GCN",
            hyperparameters={
                "in_channels": in_channels,
                "hidden_channels": 8,
                "out_channels": 1,
                "num_layers": 2,
            },
            task_type="graph_regression",
        )
        extra = {"input_reference": reference} if reference else {}
        checkpoint = CheckpointManager(working_root_dir=tmp_path).save(
            tmp_path / "checkpoints" / "best.pt",
            model,
            hyper_parameters={
                "model_name": "GCN",
                "task_type": "graph_regression",
                "hyperparameters": info["hyperparameters_values"],
                "model_info": info,
            },
            data_info=IDENTIFIER_DATA_INFO,
            **extra,
        )
        return package_checkpoint(checkpoint, tmp_path / "package")

    def _smiles_reference(self):
        from milia_pipeline.models.post_training import convert_to_pyg

        graphs = [convert_to_pyg(s, format="smiles") for s in ["CCO", "CCN", "CCC", "CO"] * 50]
        return build_input_reference(DataLoader(graphs, batch_size=16))

    def test_package_and_card_carry_the_reference(self, tmp_path):
        from milia_pipeline.models.deployment.package import MODEL_CARD, PACKAGE_MANIFEST

        reference = self._smiles_reference()
        package = self._package(tmp_path, reference)
        assert json.loads((package / PACKAGE_MANIFEST).read_text())["input_reference"] == reference
        assert "over 200 training molecules" in (package / MODEL_CARD).read_text()
        old = self._package(tmp_path / "old", None)
        assert "Input drift reference: not recorded" in (old / MODEL_CARD).read_text()

    def test_the_server_reports_drift_over_http(self, tmp_path, monkeypatch):
        uvicorn = pytest.importorskip("uvicorn", reason="the `serve` extra is not installed")
        import requests

        from milia_pipeline.models.deployment.serving import create_app

        monkeypatch.setenv("MILIA_DRIFT_WINDOW", "20")
        package = self._package(tmp_path, self._smiles_reference())
        server = uvicorn.Server(
            uvicorn.Config(create_app(package), host="127.0.0.1", port=0, log_level="warning")
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 60
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline
            time.sleep(0.05)
        url = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
        try:
            assert requests.get(f"{url}/monitoring/drift", timeout=30).json()["status"] == (
                "collecting"
            )
            like = {"instances": ["CCO", "CCN", "CCC", "CO"] * 5}
            assert requests.post(f"{url}/predict", json=like, timeout=60).status_code == 200
            report = requests.get(f"{url}/monitoring/drift", timeout=30).json()
            assert report["status"] == "no_drift" and report["molecules"] == 20
            shifted = {"instances": ["c1ccc2ccccc2c1", "ClC(Cl)(Cl)Cl", "c1ccncc1C(=O)O"] * 7}
            assert requests.post(f"{url}/predict", json=shifted, timeout=60).status_code == 200
            report = requests.get(f"{url}/monitoring/drift", timeout=30).json()
            assert report["status"] == "drift" and "num_atoms" in report["drifted"]
        finally:
            server.should_exit = True
            thread.join(timeout=30)
