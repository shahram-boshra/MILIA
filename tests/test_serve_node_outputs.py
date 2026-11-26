"""Serving a node-level model: per instance, one row of outputs per atom.

* A node-level package is servable: its signature records the outputs per atom, and the start-up
  check counts them per atom.
* ``predict_instances`` nests the model's atom rows per instance, in node order, and the response
  names each instance's atomic numbers; intervals are nested alike.
* The predict route answers ``output_level: node`` with ``atomic_numbers``; a graph-level model
  answers ``output_level: graph`` as before.
"""

from __future__ import annotations

import contextlib
import threading
import time

import numpy as np
import pytest
import torch
from checkpoint_records import IDENTIFIER_DATA_INFO

from milia_pipeline.exceptions import DeploymentError
from milia_pipeline.models.deployment.package import package_checkpoint
from milia_pipeline.models.deployment.serving import PackagedPredictor

CALIBRATION = {
    "method": "split_conformal_absolute",
    "alpha": 0.1,
    "n": 10,
    "quantiles": [0.5],
    "scope": "molecule",
}


def _checkpoint(tmp_path, task_type="node_regression", calibration=None):
    from milia_pipeline.models.factory.model_factory import ModelFactory
    from milia_pipeline.models.post_training import convert_to_pyg
    from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
        CheckpointManager,
    )

    torch.manual_seed(0)
    model, info = ModelFactory().create_model_with_info(
        name="GCN",
        hyperparameters={
            "in_channels": convert_to_pyg("CCO", format="smiles").x.shape[1],
            "hidden_channels": 8,
            "out_channels": 1,
            "num_layers": 2,
        },
        task_type=task_type,
    )
    path = tmp_path / "checkpoints" / "best.pt"
    CheckpointManager(working_root_dir=tmp_path).save(
        path,
        model,
        data_info=IDENTIFIER_DATA_INFO,
        hyper_parameters={
            "model_name": "GCN",
            "task_type": task_type,
            "hyperparameters": info["hyperparameters_values"],
            "model_info": info,
        },
    )
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["run_info"] = {"input_example": ["CCO", "CC"], "targets": {"names": ["q"]}}
    if calibration is not None:
        checkpoint["calibration"] = calibration
    torch.save(checkpoint, path)
    return path


def _served(tmp_path, **kwargs):
    return PackagedPredictor.from_package(
        package_checkpoint(_checkpoint(tmp_path, **kwargs), tmp_path / "package")
    )


def test_a_node_level_package_is_served_with_one_row_per_atom(tmp_path):
    served = _served(tmp_path)
    assert served.output_level == "node" and served.unservable_reason() is None
    graphs = served.featurize(["CCO", "CC", "CCO"], observe=False)
    rows = served.predict_instances(["CCO", "CC", "CCO"])
    assert [len(instance) for instance in rows] == [g.num_nodes for g in graphs]
    expected = served.predictor.predict_batch(graphs, return_numpy=True).reshape(-1, 1)
    np.testing.assert_array_equal(np.concatenate([np.asarray(r) for r in rows]), expected)
    np.testing.assert_array_equal(rows[0], rows[2])  # the same molecule, the same atom rows
    assert served.atomic_numbers(graphs) == [g.z.tolist() for g in graphs]


def test_node_level_intervals_are_nested_like_the_predictions(tmp_path):
    served = _served(tmp_path, calibration=CALIBRATION)
    rows, intervals, atomic_numbers = served.predict_response(["CCO", "CC"])
    assert (
        [len(r) for r in intervals["lower"]]
        == [len(r) for r in rows]
        == list(map(len, atomic_numbers))
    )
    for mean, lower, upper in zip(rows, intervals["lower"], intervals["upper"], strict=True):
        np.testing.assert_allclose(np.asarray(mean) - 0.5, lower, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(mean) + 0.5, upper, rtol=1e-6)


def test_rows_that_are_not_the_atoms_are_refused(tmp_path):
    served = _served(tmp_path)
    graphs = served.featurize(["CCO"], observe=False)
    with pytest.raises(ValueError, match="output rows for"):
        served.per_instance(np.zeros((graphs[0].num_nodes + 1, 1)), graphs)


def test_the_start_up_check_counts_outputs_per_atom(tmp_path):
    served = _served(tmp_path)
    served.check_signature()  # one output per atom, as the signature records
    served.manifest["signature"]["outputs"]["count"] = 2
    with pytest.raises(DeploymentError, match="1 values per atom"):
        served.check_signature()


def test_a_graph_level_model_still_answers_one_row_per_instance(tmp_path):
    served = _served(tmp_path, task_type="graph_regression")
    rows, _, atomic_numbers = served.predict_response(["CCO", "CC"])
    assert served.output_level == "graph" and atomic_numbers is None
    assert [len(r) for r in rows] == [1, 1]


@contextlib.contextmanager
def _serving(package):
    requests = pytest.importorskip("requests")
    uvicorn = pytest.importorskip("uvicorn")
    from milia_pipeline.models.deployment.serving import create_app

    server = uvicorn.Server(
        uvicorn.Config(create_app(package), host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60
    while not server.started:
        assert thread.is_alive() and time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield lambda body: requests.post(f"http://127.0.0.1:{port}/predict", json=body, timeout=60)
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_the_predict_route_answers_node_level_outputs_per_atom(tmp_path):
    pytest.importorskip("fastapi", reason="the `serve` extra is not installed")
    package = package_checkpoint(
        _checkpoint(tmp_path, calibration=CALIBRATION), tmp_path / "package"
    )
    served = PackagedPredictor.from_package(package)
    rows, intervals, atomic_numbers = served.predict_response(["CCO", "CC"])
    with _serving(package) as post:
        response = post({"instances": ["CCO", "CC"]})
    assert response.status_code == 200
    body = response.json()
    assert body["output_level"] == "node" and body["output_names"] == ["q"]
    assert body["atomic_numbers"] == atomic_numbers
    np.testing.assert_allclose(np.concatenate(body["predictions"]), np.concatenate(rows))
    assert [len(r) for r in body["intervals"]["upper"]] == [len(r) for r in rows]
