"""
The serving app answers every target's routes over real HTTP (F57-2).

A real MILIA checkpoint is packaged and served by Uvicorn in a background thread. Requests go
over TCP with ``requests``; nothing in the request path is mocked. Predictions must equal
``Predictor.predict_batch`` on the same SMILES featurized as ``main.py --predict`` does.
"""

import threading
import time

import pytest
import requests
import torch

from milia_pipeline.exceptions import DeploymentError
from milia_pipeline.models.deployment.package import PACKAGE_CHECKPOINT, package_checkpoint
from milia_pipeline.models.deployment.serving import (
    PackagedPredictor,
    create_app,
    default_base_image,
    main,
)
from milia_pipeline.models.factory.model_factory import ModelFactory
from milia_pipeline.models.post_training import convert_to_pyg
from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import CheckpointManager

uvicorn = pytest.importorskip("uvicorn", reason="the `serve` extra is not installed")

SMILES = ["CCO", "c1ccccc1", "CC(=O)O"]


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    """A package of a real GCN whose input width matches MILIA's default SMILES featurization."""
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("serve")
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
    checkpoint = CheckpointManager(working_root_dir=root).save(
        root / "checkpoints" / "best.pt",
        model,
        hyper_parameters={
            "model_name": "GCN",
            "task_type": "graph_regression",
            "hyperparameters": info["hyperparameters_values"],
            "model_info": info,
        },
    )
    return package_checkpoint(checkpoint, root / "package")


def _serve(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("server did not start")
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, f"http://127.0.0.1:{port}"


@pytest.fixture
def served(package, monkeypatch):
    """Start the app (with Vertex AI routes and a small instance limit) and yield its base URL."""
    monkeypatch.setenv("AIP_HEALTH_ROUTE", "/v1/endpoints/e/deployedModels/m")
    monkeypatch.setenv("AIP_PREDICT_ROUTE", "/v1/endpoints/e/deployedModels/m:predict")
    monkeypatch.setenv("MILIA_SERVE_MAX_INSTANCES", "4")
    server, thread, base = _serve(create_app(package))
    yield base
    server.should_exit = True
    thread.join(timeout=10)


def test_packaged_predictor_matches_the_cli_prediction_path(package):
    packaged = PackagedPredictor.from_package(package)
    config = packaged.predictor.structural_features_config
    graphs = [convert_to_pyg(s, format="smiles", structural_features_config=config) for s in SMILES]
    expected = packaged.predictor.predict_batch(graphs, return_numpy=True)

    rows = packaged.predict_smiles(SMILES)

    assert len(rows) == len(SMILES) and all(len(row) == 2 for row in rows)
    assert torch.allclose(torch.tensor(rows), torch.tensor(expected).reshape(len(SMILES), -1))


def test_invalid_smiles_names_its_index(package):
    with pytest.raises(ValueError, match=r"instances\[1\]"):
        PackagedPredictor.from_package(package).predict_smiles(["CCO", "not_a_smiles("])


def test_a_tampered_package_is_not_served(package, tmp_path):
    copy = tmp_path / "package"
    copy.mkdir()
    for path in package.iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    with (copy / PACKAGE_CHECKPOINT).open("ab") as stream:
        stream.write(b"\0")
    with pytest.raises(DeploymentError, match="SHA-256"):
        PackagedPredictor.from_package(copy)


def test_health_routes_for_every_target(served):
    for route in ("/ping", "/health", "/v1/endpoints/e/deployedModels/m"):
        response = requests.get(served + route, timeout=30)
        assert response.status_code == 200, route
        assert response.json() == {"status": "healthy"}


def test_predict_routes_for_every_target_return_the_same_rows(served, package):
    expected = PackagedPredictor.from_package(package).predict_smiles(SMILES)
    for route in ("/invocations", "/predict", "/v1/endpoints/e/deployedModels/m:predict"):
        response = requests.post(served + route, json={"instances": SMILES}, timeout=60)
        assert response.status_code == 200, route
        body = response.json()
        assert (body["model_name"], body["task_type"]) == ("GCN", "graph_regression")
        assert torch.allclose(torch.tensor(body["predictions"]), torch.tensor(expected))


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ({"instances": ["CCO", "not_a_smiles("]}, "instances[1]"),
        ({"instances": ["/etc/passwd"]}, "Invalid SMILES"),  # never read as a file path
        ({"instances": ["C" * 2001]}, "exceed 2000 characters"),
        ({"instances": ["C"] * 5}, "at most 4 per request"),
    ],
)
def test_bad_instances_are_rejected_with_the_reason(served, body, detail):
    response = requests.post(served + "/predict", json=body, timeout=60)
    assert response.status_code == 422
    assert detail in str(response.json()["detail"])


@pytest.mark.parametrize("body", [{"instances": []}, {"instances": ["CCO"], "extra": 1}, {}])
def test_malformed_bodies_fail_schema_validation(served, body):
    assert requests.post(served + "/predict", json=body, timeout=30).status_code == 422


def test_invalid_instance_limit_is_a_deployment_error(package, monkeypatch):
    monkeypatch.setenv("MILIA_SERVE_MAX_INSTANCES", "0")
    with pytest.raises(DeploymentError, match="at least 1"):
        create_app(package)


def test_main_accepts_sagemakers_serve_argument_and_vertex_port(package, monkeypatch):
    calls = {}
    monkeypatch.setenv("AIP_HTTP_PORT", "9123")
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.update(app=app, **kwargs))

    assert main(["serve", "--package", str(package)]) == 0
    assert (calls["host"], calls["port"], calls["workers"]) == ("0.0.0.0", 9123, 1)


def test_default_base_image_is_the_running_milia_version():
    from milia_pipeline import __version__

    assert default_base_image() == f"ghcr.io/shahram-boshra/milia:{__version__}"
