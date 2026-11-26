"""Prediction outputs named by the model's targets, and the output level the server answers.

* ``milia --predict`` writes its columns under the names the training run recorded
  (``run_info.targets.names``), interval columns as ``lower_<name>``; without usable names the
  columns stay ``prediction_<i>``.
* The server answers per molecule (one row, or one row per atom), so it serves graph- and
  node-level models and refuses to start for edge-level ones and a task type naming no level.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from checkpoint_records import IDENTIFIER_DATA_INFO

from milia_pipeline.exceptions import DeploymentError
from milia_pipeline.models.deployment.serving import output_level_refusal
from milia_pipeline.models.post_training.inference.predictor import Predictor

NAMES = ["Etot", "gap"]
PREDICTIONS = np.array([[1.0, 2.0], [3.0, 4.0]])


def _predictor(tmp_path, names=None):
    return Predictor(model=nn.Identity(), working_root_dir=tmp_path, output_names=names)


# ----------------------------------------------------------------------------- --predict files


def test_csv_columns_and_interval_columns_are_named_by_target(tmp_path):
    intervals = {"lower": PREDICTIONS - 1, "upper": PREDICTIONS + 1, "alpha": 0.1}
    path = _predictor(tmp_path, NAMES).save_predictions(
        PREDICTIONS,
        "p.csv",
        include_inputs=True,
        input_identifiers=["CCO", "CCN"],
        intervals=intervals,
    )
    frame = pd.read_csv(path)
    assert list(frame.columns) == [
        "input",
        "Etot",
        "gap",
        "lower_Etot",
        "lower_gap",
        "upper_Etot",
        "upper_gap",
    ]
    np.testing.assert_allclose(frame[["Etot", "gap"]].to_numpy(), PREDICTIONS)
    np.testing.assert_allclose(frame["upper_gap"].to_numpy(), [3.0, 5.0])


def test_a_single_named_output_names_its_column(tmp_path):
    path = _predictor(tmp_path, ["Etot"]).save_predictions(np.array([1.0, 2.0]), "p.csv")
    assert list(pd.read_csv(path).columns) == ["Etot"]


def test_json_records_the_output_names(tmp_path):
    path = _predictor(tmp_path, NAMES).save_predictions(PREDICTIONS, "p.json", format="json")
    data = json.loads(path.read_text())
    assert data["output_names"] == NAMES
    assert data["predictions"] == PREDICTIONS.tolist()
    unnamed = _predictor(tmp_path).save_predictions(PREDICTIONS, "u.json", format="json")
    assert "output_names" not in json.loads(unnamed.read_text())


def test_without_names_the_columns_keep_their_positions(tmp_path):
    intervals = {"lower": PREDICTIONS - 1, "upper": PREDICTIONS + 1}
    path = _predictor(tmp_path).save_predictions(PREDICTIONS, "p.csv", intervals=intervals)
    assert list(pd.read_csv(path).columns) == [
        "prediction_0",
        "prediction_1",
        "lower_0",
        "lower_1",
        "upper_0",
        "upper_1",
    ]


@pytest.mark.parametrize(
    "names",
    [
        ["Etot"],  # one name for two columns (a classifier's class index is one column)
        ["Etot", "Etot"],  # duplicated
        ["input", "gap"],  # the identifier column's name
    ],
)
def test_names_that_cannot_name_the_columns_leave_them_unnamed(tmp_path, names, caplog):
    with caplog.at_level(logging.WARNING):
        path = _predictor(tmp_path, names).save_predictions(PREDICTIONS, "p.csv")
    assert list(pd.read_csv(path).columns) == ["prediction_0", "prediction_1"]
    assert "columns unnamed" in caplog.text


def test_binary_formats_are_unchanged(tmp_path):
    path = _predictor(tmp_path, NAMES).save_predictions(PREDICTIONS, "p.npy", format="npy")
    np.testing.assert_allclose(np.load(path), PREDICTIONS)


def _checkpoint(tmp_path, task_type="graph_regression", names=("a", "b")):
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
            "out_channels": 2,
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
    checkpoint["run_info"] = {
        "input_example": ["CCO"],
        "targets": {"names": list(names) if names else None, "count": 2},
    }
    torch.save(checkpoint, path)
    return path


def test_from_checkpoint_reads_the_recorded_target_names(tmp_path):
    named = Predictor.from_checkpoint(_checkpoint(tmp_path), working_root_dir=tmp_path)
    assert named.output_names == ["a", "b"]
    other = tmp_path / "unnamed"
    unnamed = Predictor.from_checkpoint(_checkpoint(other, names=None), working_root_dir=other)
    assert unnamed.output_names is None


# ----------------------------------------------------------------------------- output level


@pytest.mark.parametrize(
    "task_type",
    [
        "graph_regression",
        "graph_classification",
        "Graph_Regression",
        "node_regression",
        "node_embedding",
    ],
)
def test_graph_and_node_level_models_are_served(task_type):
    assert output_level_refusal(task_type) is None


@pytest.mark.parametrize(
    ("task_type", "reason"),
    [
        ("edge_regression", "edge-level model (edge_regression) gives one row per bond"),
        ("link_prediction", "one row per bond"),
        ("regression", "names no output level"),
        (None, "names no output level"),
    ],
)
def test_other_output_levels_are_refused_with_the_reason(task_type, reason):
    refusal = output_level_refusal(task_type)
    assert reason in refusal
    if task_type and task_type.startswith(("edge_", "link_")):
        assert "which it does not map to its molecule" in refusal


def test_an_edge_level_package_is_not_served(tmp_path):
    from milia_pipeline.models.deployment.package import package_checkpoint
    from milia_pipeline.models.deployment.serving import PackagedPredictor, require_servable

    package = package_checkpoint(_checkpoint(tmp_path, "edge_regression"), tmp_path / "package")
    with pytest.raises(DeploymentError, match="edge-level model"):
        PackagedPredictor.from_package(package)
    with pytest.raises(DeploymentError, match="'local' target"):
        require_servable(package)


def test_a_graph_level_package_is_served_with_its_output_names(tmp_path):
    from milia_pipeline.models.deployment.package import package_checkpoint
    from milia_pipeline.models.deployment.serving import PackagedPredictor

    served = PackagedPredictor.from_package(
        package_checkpoint(_checkpoint(tmp_path), tmp_path / "package")
    )
    assert served.unservable_reason() is None
    assert served.output_names == ["a", "b"]
    assert len(served.predict_instances(["CCO", "CCN"])) == 2
