"""Node-level outputs: names, per-molecule calibration, and prediction files one row per atom.

* ``output_level`` reads what one output row describes from the task type's prefix, and refuses
  to guess an unknown one.
* ``run_info.target_names_for`` names a node-level model's outputs from its per-atom targets
  (``node_y_property_names``), not from the graph targets.
* A node-level model is calibrated per molecule: a molecule's score is its largest atom score, so
  ``n`` counts molecules (the exchangeable units) and the calibration records ``scope``.
* ``--predict`` files of a node-level model: CSV one row per atom, JSON one entry per molecule.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from milia_pipeline.exceptions import TrainingError
from milia_pipeline.models.capabilities import output_level
from milia_pipeline.models.checkpoint_io import load_checkpoint_file
from milia_pipeline.models.post_training.conformal import conformal_quantiles
from milia_pipeline.models.post_training.inference.predictor import Predictor
from milia_pipeline.models.training.callbacks import ModelCheckpoint
from milia_pipeline.models.training.run_info import target_names_for
from milia_pipeline.models.training.trainer import Trainer

# ----------------------------------------------------------------------------- output level


@pytest.mark.parametrize(
    ("task_type", "level"),
    [
        ("graph_regression", "graph"),
        ("Graph_Classification", "graph"),
        ("node_regression", "node"),
        ("node_embedding", "node"),
        ("edge_regression", "edge"),
        ("link_prediction", "edge"),
        ("regression", None),
        (None, None),
    ],
)
def test_output_level_reads_the_task_prefix(task_type, level):
    assert output_level(task_type) == level


# ----------------------------------------------------------------------------- target names


def _sample(**attrs):
    data = Data(x=torch.zeros(3, 1))
    data.y_property_names = ["Etot", "gap"]
    data.node_y_property_names = ["Qmulliken"]
    for key, value in attrs.items():
        setattr(data, key, value)
    return data


def test_a_node_level_model_is_named_by_its_per_atom_targets():
    assert target_names_for(None, _sample(), task_type="node_regression") == ["Qmulliken"]
    assert target_names_for(None, _sample(), task_type="graph_regression") == ["Etot", "gap"]
    assert target_names_for(None, _sample()) == ["Etot", "gap"]


def test_the_selection_source_and_names_decide_first():
    selection = {"resolved_source_attr": "node_y", "resolved_indices": [0]}
    assert target_names_for(selection, _sample(), task_type="graph_regression") == ["Qmulliken"]
    named = {"resolved_names": ["q"], "resolved_source_attr": None}
    assert target_names_for(named, _sample(), task_type="edge_regression") == ["q"]


def test_an_edge_level_or_unknown_task_without_a_source_has_no_names():
    assert target_names_for(None, _sample(), task_type="edge_regression") is None
    assert target_names_for(None, _sample(), task_type="regression") is None


# ----------------------------------------------------------------------------- calibration


def test_a_grouped_calibration_scores_each_molecule_by_its_largest_atom_score():
    # 3 molecules of 2, 3 and 1 atoms; one target
    targets = torch.zeros(6, 1)
    predictions = torch.tensor([[1.0], [3.0], [2.0], [5.0], [4.0], [7.0]])
    groups = torch.tensor([0, 0, 1, 1, 1, 2])
    calibration = conformal_quantiles(targets, predictions, alpha=0.5, groups=groups)
    assert calibration["n"] == 3 and calibration["scope"] == "molecule"
    # molecule scores 3, 5, 7; rank ceil(4 * 0.5) = 2 -> 5
    assert calibration["quantiles"] == [5.0]
    pooled = conformal_quantiles(targets, predictions, alpha=0.5)
    assert pooled["n"] == 6 and "scope" not in pooled


def test_group_indices_need_not_be_contiguous_and_must_match_the_rows():
    targets, predictions = torch.zeros(4, 2), torch.arange(8.0).reshape(4, 2)
    calibration = conformal_quantiles(
        targets, predictions, alpha=0.1, groups=torch.tensor([7, 7, 2, 2])
    )
    assert calibration["n"] == 2 and calibration["quantiles"] == [math.inf, math.inf]
    with pytest.raises(ValueError, match="3 group indices for 4 rows"):
        conformal_quantiles(targets, predictions, groups=torch.tensor([0, 0, 1]))


class _PerAtom(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)

    def forward(self, x, edge_index, batch=None):
        return self.lin(x)


def _atoms_graphs(n: int, seed: int) -> list[Data]:
    generator = torch.Generator().manual_seed(seed)
    graphs = []
    for i in range(n):
        x = torch.randn(2 + i % 3, 4, generator=generator)
        graphs.append(Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]), y=x[:, :2] * 2))
    return graphs


def _trainer(tmp_path, task_type):
    model = _PerAtom()
    return Trainer(
        model=model,
        train_loader=DataLoader(_atoms_graphs(16, 1), batch_size=4),
        val_loader=DataLoader(_atoms_graphs(19, 2), batch_size=4),
        test_loader=DataLoader(_atoms_graphs(8, 3), batch_size=4),
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        device=torch.device("cpu"),
        callbacks=[ModelCheckpoint(dirpath=tmp_path / "checkpoints", verbose=False)],
        model_info={"task_type": task_type},
        max_epochs=2,
        calibration_alpha=0.1,
    )


def test_a_node_level_model_is_calibrated_per_molecule(tmp_path):
    torch.manual_seed(0)
    trainer = _trainer(tmp_path, "node_regression")
    calibration = trainer.fit()["calibration"]
    assert calibration["scope"] == "molecule"
    assert calibration["n"] == 19  # molecules, not their atoms
    best = load_checkpoint_file(tmp_path / "checkpoints" / "best.pt", map_location="cpu")
    assert best["calibration"] == calibration


def test_an_edge_level_model_is_not_calibrated(tmp_path):
    trainer = _trainer(tmp_path, "edge_regression")
    assert trainer.fit()["calibration"] is None
    with pytest.raises(TrainingError, match="not mapped to molecules"):
        trainer.calibrate()


# ----------------------------------------------------------------------------- --predict files

ATOMS = [[6, 8], [7, 1, 1]]
ROWS = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32)


def _predictor(tmp_path, task_type="node_regression", names=("Qmulliken",), calibration=None):
    return Predictor(
        model=nn.Identity(),
        working_root_dir=tmp_path,
        task_type=task_type,
        output_names=list(names) if names else None,
        calibration=calibration,
    )


def test_a_node_level_csv_has_one_row_per_atom(tmp_path):
    intervals = {"lower": ROWS - 1, "upper": ROWS + 1, "alpha": 0.1}
    path = _predictor(tmp_path).save_predictions(
        ROWS,
        "p.csv",
        include_inputs=True,
        input_identifiers=["CO", "N"],
        intervals=intervals,
        molecule_atoms=ATOMS,
    )
    frame = pd.read_csv(path)
    assert list(frame.columns) == [
        "input",
        "molecule",
        "atom",
        "atomic_number",
        "Qmulliken",
        "lower_Qmulliken",
        "upper_Qmulliken",
    ]
    assert frame["input"].tolist() == ["CO", "CO", "N", "N", "N"]
    assert frame["molecule"].tolist() == [0, 0, 1, 1, 1]
    assert frame["atom"].tolist() == [0, 1, 0, 1, 2]
    assert frame["atomic_number"].tolist() == [6, 8, 7, 1, 1]
    assert np.array_equal(frame["Qmulliken"].to_numpy().astype(np.float32), ROWS[:, 0])


def test_a_node_level_json_has_one_entry_per_molecule(tmp_path):
    intervals = {"lower": ROWS - 1, "upper": ROWS + 1, "alpha": 0.1, "method": "m"}
    path = _predictor(tmp_path).save_predictions(
        ROWS, "p.json", format="json", intervals=intervals, molecule_atoms=ATOMS
    )
    data = json.loads(path.read_text())
    assert data["output_level"] == "node" and data["atomic_numbers"] == ATOMS
    assert [len(molecule) for molecule in data["predictions"]] == [2, 3]
    assert np.allclose(sum(data["predictions"], []), ROWS)
    assert [len(m) for m in data["intervals"]["upper"]] == [2, 3]
    assert data["intervals"]["alpha"] == 0.1 and data["intervals"]["method"] == "m"


def test_node_level_files_need_the_atoms_of_each_molecule(tmp_path):
    predictor = _predictor(tmp_path)
    with pytest.raises(ValueError, match="pass molecule_atoms"):
        predictor.save_predictions(ROWS, "p.csv")
    with pytest.raises(ValueError, match="5 prediction rows for 4 atoms in 2 molecules"):
        predictor.save_predictions(ROWS, "p.csv", molecule_atoms=[[6, 8], [7, 1]])
    np.testing.assert_array_equal(
        np.load(predictor.save_predictions(ROWS, "p.npy", format="npy")), ROWS
    )


@pytest.mark.parametrize("name", ["molecule", "atom", "atomic_number", "input"])
def test_an_output_named_like_a_written_column_stays_unnamed(tmp_path, name):
    assert _predictor(tmp_path, names=(name,)).output_labels(1) is None


def test_a_pooled_node_level_calibration_gives_no_intervals(tmp_path):
    pooled = {"method": "split_conformal_absolute", "alpha": 0.1, "n": 9, "quantiles": [1.0]}
    with pytest.raises(ValueError, match="calibrates per molecule"):
        _predictor(tmp_path, calibration=pooled).predict_with_intervals([])


# ----------------------------------------------------------------------------- packaging


def test_a_node_level_model_packages_with_molecules_of_different_sizes(tmp_path):
    """Packaging compares the source's and the package's predictions as the model returns them:
    one row per atom here (2 molecules of 3 and 2 atoms give 5 rows, not 2 molecule rows)."""
    from checkpoint_records import IDENTIFIER_DATA_INFO

    from milia_pipeline.models.deployment.package import PACKAGE_MANIFEST, package_checkpoint
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
        task_type="node_regression",
    )
    path = tmp_path / "checkpoints" / "best.pt"
    CheckpointManager(working_root_dir=tmp_path).save(
        path,
        model,
        data_info=IDENTIFIER_DATA_INFO,
        hyper_parameters={
            "model_name": "GCN",
            "task_type": "node_regression",
            "hyperparameters": info["hyperparameters_values"],
            "model_info": info,
        },
    )
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["run_info"] = {"input_example": ["CCO", "CC"], "targets": {"names": ["q"]}}
    torch.save(checkpoint, path)
    package = package_checkpoint(path, tmp_path / "package")
    assert json.loads((package / PACKAGE_MANIFEST).read_text())["task_type"] == "node_regression"
