"""An unbounded conformal interval in JSON outputs: ``null`` bounds, named in ``unbounded``.

A quantile is infinite when the calibration set is too small for 1 - alpha, so that output has
no finite bound for any molecule. JSON has no number for it (RFC 8259 §6): ``--predict`` JSON
and the server response both write such a bound as ``null`` and name the output in
``intervals.unbounded``; the ``--predict`` JSON file is strict JSON.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch.nn as nn

from milia_pipeline.models.post_training.conformal import json_bounds, unbounded_outputs
from milia_pipeline.models.post_training.inference.predictor import Predictor


def _strict(text: str):
    def refuse(constant):
        raise ValueError(f"not JSON: {constant}")

    return json.loads(text, parse_constant=refuse)


def test_unbounded_outputs_names_the_columns_without_a_finite_bound():
    lower = np.array([[0.0, -math.inf, 1.0], [0.5, -math.inf, 1.0]])
    upper = np.array([[1.0, math.inf, math.inf], [1.5, math.inf, 2.0]])
    assert unbounded_outputs(lower, upper) == [1, 2]
    assert unbounded_outputs(np.array([0.0, 1.0])) == []
    assert unbounded_outputs(np.array([math.inf, 1.0])) == [0]


def test_json_bounds_writes_a_non_finite_bound_as_null():
    values = np.array([[1.5, math.inf], [-math.inf, 2.0]], dtype=np.float32)
    assert json_bounds(values) == [[1.5, None], [None, 2.0]]
    assert json.dumps(json_bounds(values), allow_nan=False) == "[[1.5, null], [null, 2.0]]"


def _intervals(rows):
    lower, upper = rows.copy(), rows.copy()
    lower[:, 1], upper[:, 1] = -math.inf, math.inf
    return {"lower": lower, "upper": upper, "alpha": 0.1, "method": "split_conformal_absolute"}


def test_predict_json_is_strict_and_names_the_unbounded_output(tmp_path):
    rows = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    predictor = Predictor(
        model=nn.Identity(), working_root_dir=tmp_path, output_names=["Etot", "gap"]
    )
    path = predictor.save_predictions(rows, "p.json", format="json", intervals=_intervals(rows))
    data = _strict(path.read_text())
    assert data["intervals"]["unbounded"] == ["gap"]
    assert data["intervals"]["lower"] == [[1.0, None], [3.0, None]]
    assert data["intervals"]["upper"] == [[1.0, None], [3.0, None]]


def test_unnamed_outputs_are_named_by_index(tmp_path):
    rows = np.array([[1.0, 2.0]], dtype=np.float32)
    predictor = Predictor(model=nn.Identity(), working_root_dir=tmp_path)
    path = predictor.save_predictions(rows, "p.json", format="json", intervals=_intervals(rows))
    assert _strict(path.read_text())["intervals"]["unbounded"] == [1]


def test_a_node_level_json_nests_null_bounds_per_molecule(tmp_path):
    rows = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    predictor = Predictor(
        model=nn.Identity(),
        working_root_dir=tmp_path,
        task_type="node_regression",
        output_names=["q", "s"],
    )
    path = predictor.save_predictions(
        rows,
        "p.json",
        format="json",
        intervals=_intervals(rows),
        molecule_atoms=[[6, 8], [1]],
    )
    data = _strict(path.read_text())
    assert data["intervals"]["unbounded"] == ["s"]
    assert [[bound[1] for bound in molecule] for molecule in data["intervals"]["upper"]] == [
        [None, None],
        [None],
    ]


def test_finite_intervals_have_no_unbounded_entry(tmp_path):
    rows = np.array([[1.0, 2.0]], dtype=np.float32)
    intervals = {"lower": rows - 1, "upper": rows + 1, "alpha": 0.1}
    path = Predictor(model=nn.Identity(), working_root_dir=tmp_path).save_predictions(
        rows, "p.json", format="json", intervals=intervals
    )
    assert "unbounded" not in _strict(path.read_text())["intervals"]


def test_a_non_finite_prediction_is_refused_rather_than_written_as_nan(tmp_path):
    predictor = Predictor(model=nn.Identity(), working_root_dir=tmp_path)
    with pytest.raises(ValueError, match="JSON compliant"):
        predictor.save_predictions(np.array([[math.nan]]), "p.json", format="json")


def test_the_server_names_the_unbounded_output(tmp_path):
    from test_serve_node_outputs import _served

    calibration = {
        "method": "split_conformal_absolute",
        "alpha": 0.1,
        "n": 3,
        "quantiles": [math.inf],
        "scope": "molecule",
    }
    served = _served(tmp_path, calibration=calibration)
    rows, intervals, _ = served.predict_response(["CCO", "CC"])
    assert intervals["unbounded"] == ["q"]
    assert all(bound == [None] for molecule in intervals["upper"] for bound in molecule)
    assert all(isinstance(row[0], float) for molecule in rows for row in molecule)
