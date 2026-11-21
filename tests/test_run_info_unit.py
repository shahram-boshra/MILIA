"""What a training run records in its checkpoints (milia_pipeline.models.training.run_info)."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from milia_pipeline.models.checkpoint_io import load_checkpoint_file
from milia_pipeline.models.training import run_info as run_info_module
from milia_pipeline.models.training.callbacks import ModelCheckpoint
from milia_pipeline.models.training.run_info import (
    collect_run_info,
    config_sha256,
    example_smiles,
    target_names_for,
)
from milia_pipeline.models.training.trainer import Trainer


def test_config_hash_ignores_key_order():
    assert config_sha256({"a": 1, "b": {"c": 2}}) == config_sha256({"b": {"c": 2}, "a": 1})
    assert config_sha256({"a": 1}) != config_sha256({"a": 2})


def test_example_smiles_uses_smiles_then_inchi_and_stops_at_five():
    graphs = [
        Data(smiles="N/A", inchi="InChI=1S/CH4/h1H4"),
        Data(smiles="CCO"),
        Data(inchi="not an inchi"),
    ] + [Data(smiles="C" * n) for n in range(1, 10)]
    assert example_smiles(graphs) == ["C", "CCO", "C", "CC", "CCC"]


def test_target_names_come_from_the_data_or_the_selection():
    sample = Data(y_property_names=["Etot", "U0", "gap"])
    assert target_names_for(None, sample) == ["Etot", "U0", "gap"]
    assert target_names_for({"resolved_names": ["gap"]}, sample) == ["gap"]
    assert target_names_for({"resolved_names": None, "resolved_indices": [0, 2]}, sample) == [
        "Etot",
        "gap",
    ]
    assert target_names_for({"resolved_indices": [5]}, sample) is None
    assert target_names_for(None, Data()) is None


def test_collect_records_the_run_and_drops_names_that_do_not_fit():
    config = {"dataset_type": "DFT", "models": {"training": {"epochs": 3}}}
    info = collect_run_info(
        config,
        task_type="graph_regression",
        out_channels=2,
        target_names=["Etot", "U0", "gap"],
        data_split={"train_ratio": 0.8},
        train_graphs=[Data(smiles="CCO")],
    )
    assert info["targets"] == {"names": None, "count": 2}  # 3 names for 2 outputs
    assert info["dataset"] == {
        "type": "DFT",
        "source_units": {"energy": "hartree", "coordinates": "angstrom"},
    }
    assert info["config_sha256"] == config_sha256(config)
    assert info["data_split"] == {"train_ratio": 0.8}
    assert info["input_example"] == ["CCO"]
    assert info["versions"]["torch"] is not None and "python" in info["versions"]
    assert info["git_commit"] is None or len(info["git_commit"]) == 40


def test_git_commit_is_none_outside_a_checkout():
    failed = subprocess.CompletedProcess(args=[], returncode=128, stdout="", stderr="not a repo")
    with patch.object(run_info_module.subprocess, "run", return_value=failed):
        assert run_info_module._git_commit() is None
    with patch.object(run_info_module.subprocess, "run", side_effect=FileNotFoundError("git")):
        assert run_info_module._git_commit() is None


def test_unknown_dataset_has_no_units():
    info = collect_run_info(
        {"dataset_type": "NotADataset"},
        task_type="graph_regression",
        out_channels=1,
        target_names=None,
        data_split=None,
    )
    assert info["dataset"]["source_units"] == {"energy": None, "coordinates": None}
    assert info["input_example"] == [] and info["data_split"] is None


class _Pooled(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, x, edge_index, batch=None):
        return self.lin(global_mean_pool(x, batch))


def test_every_checkpoint_the_trainer_writes_carries_run_info(tmp_path):
    torch.manual_seed(0)
    graphs = [
        Data(x=torch.randn(3, 4), edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.randn(1, 1))
        for _ in range(4)
    ]
    model = _Pooled()
    recorded = {"run_info_version": 1, "config_sha256": "c" * 64}
    trainer = Trainer(
        model=model,
        train_loader=DataLoader(graphs, batch_size=2),
        val_loader=DataLoader(graphs, batch_size=2),
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        device=torch.device("cpu"),
        callbacks=[ModelCheckpoint(dirpath=tmp_path / "checkpoints", verbose=False)],
        max_epochs=1,
        run_info=recorded,
    )
    trainer.fit()
    for name in ("best.pt", "last.pt"):
        stored = load_checkpoint_file(tmp_path / "checkpoints" / name, map_location="cpu")
        assert stored["run_info"] == recorded, name
    trainer.save_checkpoint(tmp_path / "manual.pt")
    assert load_checkpoint_file(tmp_path / "manual.pt", map_location="cpu")["run_info"] == recorded


def test_milia_training_records_run_info(tmp_path):
    """main.py's standard training on a small dataset writes run_info into best.pt."""
    import argparse
    import logging

    import main

    class _Dataset(list):
        structural_features_config = {"atom": ["degree"], "bond": []}

    torch.manual_seed(0)
    dataset = _Dataset()
    for i in range(20):
        n = 3 + i % 3
        src = torch.arange(n - 1)
        edges = torch.cat([torch.stack([src, src + 1]), torch.stack([src + 1, src])], 1)
        dataset.append(
            Data(
                x=torch.randn(n, 4),
                edge_index=edges,
                y=torch.randn(1, 2),
                smiles="C" * n,
                y_property_names=["Etot", "gap"],
            )
        )
    split = {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2, "random_seed": 1}
    config = {
        "dataset_type": "DFT",
        "global_paths": {"working_root_dir": str(tmp_path)},
        "models": {
            "selection": {"mode": "single", "model_name": "GCN", "task_type": "graph_regression"},
            "training": {"epochs": 2, "batch_size": 4, "data_split": split},
            "GCN": {"hidden_channels": 8, "num_layers": 2},
        },
    }
    args = argparse.Namespace(
        checkpoint=None,
        mode=None,
        task_type=None,
        model_name=None,
        epochs=None,
        batch_size=None,
        learning_rate=None,
        evaluate_only=False,
    )
    assert main._run_standard_training(args, logging.getLogger("test"), dataset, config) == 0

    best = load_checkpoint_file(tmp_path / "checkpoints" / "best.pt", map_location="cpu")
    recorded = best["run_info"]
    assert recorded["targets"] == {"names": ["Etot", "gap"], "count": 2}
    assert recorded["data_split"] == split and recorded["config_sha256"] == config_sha256(config)
    assert len(recorded["input_example"]) == 5 and "test_metrics" in best
    assert recorded["dataset"]["type"] == "DFT" and recorded["task_type"] == "graph_regression"
