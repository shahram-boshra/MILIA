"""Model families: versions and the deployment gate are keyed by what a model predicts.

A family is the dataset type, the task type, the targets (names and count) and the source units,
not the architecture. ``publish_package`` writes ``<output_dir>/<family>/<N>/`` with the family in
``family.json``; ``evaluate_gate(compare_to_current=True)`` compares only with the newest version
of the same family, scored on the same data split (and, for ``val_loss``, the same target
scaling), and records why it did not compare otherwise.
"""

from __future__ import annotations

import json

import pytest
import torch

from milia_pipeline.exceptions import DeploymentError
from milia_pipeline.models.deployment.after_training import evaluate_gate
from milia_pipeline.models.deployment.package import (
    FAMILY_FILE,
    PACKAGE_MANIFEST,
    checkpoint_family,
    manifest_family,
    model_family,
    publish_package,
)
from milia_pipeline.models.deployment.settings import GateSettings
from milia_pipeline.models.factory.model_factory import ModelFactory
from milia_pipeline.models.post_training import convert_to_pyg
from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import CheckpointManager

SPLIT = {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1, "random_seed": 42}
DFT = {"type": "DFT", "source_units": {"Etot": "hartree", "gap": "hartree"}}
QM9 = {"type": "QM9", "source_units": {"Etot": "hartree", "gap": "hartree"}}


def _checkpoint(
    root,
    name="best.pt",
    *,
    val_loss=1.0,
    model_name="GCN",
    dataset=DFT,
    names=("Etot", "gap"),
    split=SPLIT,
    scaler=None,
):
    torch.manual_seed(0)
    model, info = ModelFactory().create_model_with_info(
        name=model_name,
        hyperparameters={
            "in_channels": convert_to_pyg("CCO", format="smiles").x.shape[1],
            "hidden_channels": 8,
            "out_channels": 2,
            "num_layers": 2,
        },
        task_type="graph_regression",
    )
    path = CheckpointManager(working_root_dir=root).save(
        root / "checkpoints" / name,
        model,
        epoch=1,
        metrics_history={"val_loss": [9.0, val_loss]},
        monitored_metric="val_loss",
        monitored_score=val_loss,
        hyper_parameters={
            "model_name": model_name,
            "task_type": "graph_regression",
            "hyperparameters": info["hyperparameters_values"],
            "model_info": info,
        },
    )
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["run_info"] = {
        "dataset": dataset,
        "targets": {"names": list(names), "count": 2},
        "data_split": split,
    }
    checkpoint["target_scaler"] = scaler
    torch.save(checkpoint, path)
    return path


def _family(path):
    return checkpoint_family(torch.load(path, weights_only=False))


# ----------------------------------------------------------------------------- the family


def test_a_family_is_what_the_model_predicts_not_its_architecture():
    base = {
        "dataset_type": "DFT",
        "task_type": "graph_regression",
        "target_names": ["Etot", "gap"],
        "target_count": 2,
        "source_units": {"Etot": "hartree"},
    }
    family = model_family(**base)
    assert family == model_family(**base)
    assert (
        family["id"].startswith("DFT-graph_regression-") and len(family["id"].split("-")[-1]) == 12
    )
    for change in (
        {"dataset_type": "QM9"},
        {"task_type": "node_regression"},
        {"target_names": ["gap", "Etot"]},
        {"target_count": 3},
        {"source_units": {"Etot": "eV"}},
    ):
        assert model_family(**{**base, **change})["id"] != family["id"], change


def test_a_package_records_the_family_of_its_checkpoint(tmp_path):
    path = _checkpoint(tmp_path)
    version = publish_package(path, tmp_path / "out")
    manifest = json.loads((version / PACKAGE_MANIFEST).read_text())
    assert manifest_family(manifest) == _family(path)
    assert json.loads((version.parent / FAMILY_FILE).read_text()) == json.loads(
        json.dumps(_family(path))
    )


# ----------------------------------------------------------------------------- versions


def test_versions_are_numbered_per_family(tmp_path):
    out = tmp_path / "out"
    dft_gcn = publish_package(_checkpoint(tmp_path, "a.pt"), out)
    qm9_gcn = publish_package(_checkpoint(tmp_path, "b.pt", dataset=QM9), out)
    dft_gat = publish_package(_checkpoint(tmp_path, "c.pt", model_name="GAT"), out)
    assert dft_gcn.name == "1" and qm9_gcn.name == "1"  # a GCN on QM9 is another family
    assert dft_gat.parent == dft_gcn.parent and dft_gat.name == "2"  # a GAT on DFT is the same
    assert json.loads((dft_gat / PACKAGE_MANIFEST).read_text())["model_name"] == "GAT"


def test_a_folder_recording_another_family_is_not_published_into(tmp_path):
    out = tmp_path / "out"
    folder = publish_package(_checkpoint(tmp_path, "a.pt"), out).parent
    record = json.loads((folder / FAMILY_FILE).read_text())
    record["key"]["dataset_type"] = "edited"
    (folder / FAMILY_FILE).write_text(json.dumps(record))
    with pytest.raises(DeploymentError, match="records another model family"):
        publish_package(_checkpoint(tmp_path, "b.pt"), out)
    assert sorted(p.name for p in folder.iterdir()) == ["1", FAMILY_FILE]


# ----------------------------------------------------------------------------- the gate

GATE = GateSettings(compare_to_current=True)


def test_the_gate_compares_within_the_family(tmp_path):
    out = tmp_path / "out"
    publish_package(_checkpoint(tmp_path, "old.pt", val_loss=1.0), out)
    worse = evaluate_gate(_checkpoint(tmp_path, val_loss=1.5, model_name="GAT"), GATE, out)
    assert not worse.passed and worse.current_value == 1.0 and worse.not_compared is None
    assert "does not beat 1" in worse.reasons[0]
    assert worse.family == _family(tmp_path / "checkpoints" / "best.pt")["id"]


def test_another_family_is_not_compared(tmp_path):
    out = tmp_path / "out"
    publish_package(_checkpoint(tmp_path, "qm9.pt", val_loss=0.1, dataset=QM9), out)
    decision = evaluate_gate(_checkpoint(tmp_path, val_loss=1.5), GATE, out)
    assert decision.passed and decision.current_version is None
    assert "no published version of this model family" in decision.not_compared


def test_another_data_split_is_not_compared(tmp_path):
    out = tmp_path / "out"
    publish_package(_checkpoint(tmp_path, "old.pt", val_loss=0.1), out)
    other = {**SPLIT, "random_seed": 7}
    decision = evaluate_gate(_checkpoint(tmp_path, val_loss=1.5, split=other), GATE, out)
    assert decision.passed and decision.current_value is None
    assert "another data split" in decision.not_compared


def test_another_target_scaling_is_not_compared_for_val_loss(tmp_path):
    out = tmp_path / "out"
    scaler = {"method": "standard", "n": 8, "mean": [0.0, 0.0], "std": [1.0, 1.0]}
    publish_package(_checkpoint(tmp_path, "old.pt", val_loss=0.1, scaler=scaler), out)
    other = {**scaler, "mean": [1.0, 0.0], "std": [2.0, 1.0]}
    decision = evaluate_gate(_checkpoint(tmp_path, val_loss=1.5, scaler=other), GATE, out)
    assert decision.passed and "another target scale" in decision.not_compared


def test_the_threshold_still_applies_when_not_compared(tmp_path):
    gate = GateSettings(compare_to_current=True, threshold=1.0)
    decision = evaluate_gate(_checkpoint(tmp_path, val_loss=1.5), gate, tmp_path / "out")
    assert not decision.passed and decision.not_compared is not None
    assert "the threshold is at most 1" in decision.reasons[0]
