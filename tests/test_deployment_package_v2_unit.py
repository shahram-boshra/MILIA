"""Deployment package format 2: signature, input example, metrics, lineage and model card.

An untrained GCN's checkpoint is written by ``CheckpointManager.save`` with the
entries MILIA's trainer writes (``metrics_history``, ``monitored_*``, ``test_metrics``,
``run_info``), so the manifest is checked field by field against known values.
"""

from __future__ import annotations

import json

import pytest
import torch

from milia_pipeline.exceptions import DeploymentError
from milia_pipeline.models.deployment.package import (
    MAX_SMILES_LENGTH,
    MODEL_CARD,
    PACKAGE_FORMAT,
    PACKAGE_MANIFEST,
    load_package,
    package_checkpoint,
    verify_package,
)
from milia_pipeline.models.factory.model_factory import ModelFactory
from milia_pipeline.models.post_training import convert_to_pyg
from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import CheckpointManager

RUN_INFO = {
    "run_info_version": 1,
    "dataset": {"type": "DFT", "source_units": {"energy": "hartree", "coordinates": "angstrom"}},
    "task_type": "graph_regression",
    "targets": {"names": ["Etot", "gap"], "count": 2},
    "data_split": {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1, "random_seed": 42},
    "config_sha256": "a" * 64,
    "git_commit": "b" * 40,
    "versions": {"python": "3.12.0", "milia-py": "1.21.0", "torch": "2.12.1"},
    "input_example": ["CCO", "c1ccccc1"],
    "model_source": {
        "model": "GCN",
        "class": "torch_geometric.nn.models.basic_gnn.GCN",
        "kind": "pyg",
        "requirement": "torch-geometric==2.8.1",
        "requirements": ["milia-py==1.21.0", "torch-geometric==2.8.1"],
    },
}
HISTORY = {"train_loss": [3.0, 2.0], "val_loss": [2.5, 1.5], "val_mae": [1.2, 0.9]}
INPUT_REFERENCE = {  # deployment/drift.py: one node-feature column and atoms per molecule
    "version": 1,
    "method": "jensen_shannon_distance",
    "molecules": 3,
    "atoms": 9,
    "features": {
        "x_0": {"kind": "categorical", "values": [0.0, 1.0], "probabilities": [0.5, 0.5, 0.0]},
        "num_atoms": {"kind": "categorical", "values": [3.0], "probabilities": [1.0, 0.0]},
    },
}


def _checkpoint(root, **extra):
    torch.manual_seed(0)
    in_channels = convert_to_pyg("CCO", format="smiles").x.shape[1]
    model, info = ModelFactory().create_model_with_info(
        name="GCN",
        hyperparameters={
            "in_channels": in_channels,
            "hidden_channels": 8,
            "out_channels": 2,
            "num_layers": 2,
        },
        task_type="graph_regression",
    )
    path = CheckpointManager(working_root_dir=root).save(
        root / "checkpoints" / "best.pt",
        model,
        epoch=1,
        metrics_history=HISTORY,
        hyper_parameters={
            "model_name": "GCN",
            "task_type": "graph_regression",
            "hyperparameters": info["hyperparameters_values"],
            "model_info": info,
        },
        **extra,
    )
    return path, in_channels


@pytest.fixture
def recorded(tmp_path):
    """A checkpoint with everything a MILIA training run records, and its package."""
    checkpoint, in_channels = _checkpoint(
        tmp_path,
        monitored_metric="val_loss",
        monitored_score=1.5,
        test_metrics={"test_loss": 1.75, "test_mae": 1.0},
        run_info=RUN_INFO,
        # trained on heavy-atom graphs without a pre-transform
        data_info={
            "input_transforms": None,
            "explicit_hydrogens": False,
            "qm_input_features": [],
            "input_filters": {"max_atoms": None, "min_atoms": None, "heavy_atom_filter": None},
        },
        input_reference=INPUT_REFERENCE,
    )
    return checkpoint, package_checkpoint(checkpoint, tmp_path / "package"), in_channels


def _manifest(package):
    return json.loads((package / PACKAGE_MANIFEST).read_text())


def test_manifest_records_signature_example_metrics_and_lineage(recorded):
    checkpoint, package, in_channels = recorded
    manifest = _manifest(package)

    assert manifest["format"] == PACKAGE_FORMAT == "milia-deployment/4"
    kinds = manifest["signature"]["inputs"]["kinds"]
    assert kinds["accepted"] == ["identifier", "dataset"]
    assert set(kinds["refused"]) == {"structure"}
    assert manifest["signature"]["inputs"]["max_length"] == MAX_SMILES_LENGTH
    assert manifest["signature"]["inputs"]["node_features"] == in_channels
    assert manifest["signature"]["outputs"] == {
        "count": 2,
        "names": ["Etot", "gap"],
        "task_type": "graph_regression",
    }
    assert manifest["input_example"] == ["CCO", "c1ccccc1"]
    assert manifest["metrics"] == {
        "epoch": 1,
        "monitored": {"name": "val_loss", "value": 1.5},
        "validation": {"val_loss": 1.5, "val_mae": 0.9},
        "test": {"test_loss": 1.75, "test_mae": 1.0},
    }
    lineage = manifest["lineage"]
    assert lineage["dataset"] == RUN_INFO["dataset"]
    assert lineage["data_split"] == RUN_INFO["data_split"]
    assert (lineage["config_sha256"], lineage["git_commit"]) == ("a" * 64, "b" * 40)
    assert lineage["versions"] == RUN_INFO["versions"]
    assert lineage["torch_version"] == torch.__version__
    import hashlib

    assert (
        lineage["source_checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )


def test_model_card_states_the_recorded_facts(recorded):
    _, package, in_channels = recorded
    card = (package / MODEL_CARD).read_text()

    assert card.startswith("# GCN (graph_regression)")
    for text in (
        "## Inputs and outputs",
        f"into `{in_channels}` node features",
        "names: `Etot`, `gap`",
        "Example input: `CCO`, `c1ccccc1`",
        "Dataset: `DFT`",
        "| `val_mae` | 0.9 |",
        "| `test_loss` | 1.75 |",
        f"Git commit: `{'b' * 40}`",
        "## Limitations",
    ):
        assert text in card, text
    assert "MILIA's default featurization" in card  # the test checkpoint records none
    assert "Input transforms: none (the training data had no pre-transform)" in card
    assert "Explicit hydrogens: no: the training graphs hold heavy atoms only" in card
    assert "histograms of 1 node-feature columns and of atoms per molecule over 3" in card
    assert "not recorded" not in card


def test_a_checkpoint_without_run_info_packages_with_nulls(tmp_path):
    checkpoint, in_channels = _checkpoint(tmp_path)
    package = package_checkpoint(checkpoint, tmp_path / "package")
    manifest = _manifest(package)

    assert manifest["signature"]["inputs"]["node_features"] == in_channels
    assert manifest["signature"]["outputs"]["names"] is None
    assert manifest["input_example"] == []
    assert manifest["metrics"]["test"] is None
    assert manifest["metrics"]["validation"] == {"val_loss": 1.5, "val_mae": 0.9}
    assert all(
        manifest["lineage"][key] is None
        for key in ("dataset", "data_split", "config_sha256", "git_commit", "versions")
    )
    card = (package / MODEL_CARD).read_text()
    assert "names: not recorded" in card and "Git commit: not recorded" in card
    load_package(package)  # still loads


def test_a_format_1_package_still_verifies_and_loads(recorded):
    _, package, _ = recorded
    manifest = _manifest(package)
    v1 = {
        key: manifest[key]
        for key in (
            "milia_version",
            "checkpoint",
            "sha256",
            "model_name",
            "task_type",
            "checkpoint_format_version",
        )
    }
    v1["format"] = "milia-deployment/1"
    (package / PACKAGE_MANIFEST).write_text(json.dumps(v1))
    (package / MODEL_CARD).unlink()

    _, read = verify_package(package)
    assert read["format"] == "milia-deployment/1"
    model, _ = load_package(package)
    assert not model.training


class TestServingSignature:
    @pytest.fixture(autouse=True)
    def _serve_extra(self):
        pytest.importorskip("fastapi", reason="the `serve` extra is not installed")

    def test_the_response_names_the_outputs(self, recorded):
        from milia_pipeline.models.deployment.serving import PackagedPredictor

        packaged = PackagedPredictor.from_package(recorded[1])
        assert packaged.output_names == ["Etot", "gap"]

    def test_a_featurization_that_differs_from_the_signature_refuses_to_start(self, recorded):
        from milia_pipeline.models.deployment.serving import PackagedPredictor

        package = recorded[1]
        manifest = _manifest(package)
        manifest["signature"]["inputs"]["node_features"] += 1
        (package / PACKAGE_MANIFEST).write_text(json.dumps(manifest))
        with pytest.raises(DeploymentError, match="node features; the package's model was trained"):
            PackagedPredictor.from_package(package)

    def test_an_output_count_that_differs_from_the_signature_refuses_to_start(self, recorded):
        from milia_pipeline.models.deployment.serving import PackagedPredictor

        package = recorded[1]
        manifest = _manifest(package)
        manifest["signature"]["outputs"]["count"] = 3
        (package / PACKAGE_MANIFEST).write_text(json.dumps(manifest))
        with pytest.raises(DeploymentError, match="returns 2 values per molecule"):
            PackagedPredictor.from_package(package)

    def test_predict_route_returns_output_names(self, recorded):
        import threading
        import time

        requests = pytest.importorskip("requests")
        uvicorn = pytest.importorskip("uvicorn")
        from milia_pipeline.models.deployment.serving import create_app

        app = create_app(recorded[1])
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 60
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline, "server did not start"
            time.sleep(0.05)
        port = server.servers[0].sockets[0].getsockname()[1]
        try:
            response = requests.post(
                f"http://127.0.0.1:{port}/predict", json={"instances": ["CCO"]}, timeout=30
            )
        finally:
            server.should_exit = True
            thread.join(timeout=10)
        assert response.status_code == 200
        body = response.json()
        assert body["output_names"] == ["Etot", "gap"] and len(body["predictions"][0]) == 2
