"""Standard training reads ``models.evaluation`` (where configs/models.yaml and ModelConfig put it)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Data

from milia_pipeline.models.checkpoint_io import load_checkpoint_file

MODELS_YAML = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"


def _train(tmp_path, evaluation):
    import main

    class _Dataset(list):
        structural_features_config = None

    torch.manual_seed(0)
    dataset = _Dataset()
    for i in range(20):
        n = 3 + i % 3
        src = torch.arange(n - 1)
        edges = torch.cat([torch.stack([src, src + 1]), torch.stack([src + 1, src])], 1)
        dataset.append(Data(x=torch.randn(n, 4), edge_index=edges, y=torch.randn(1, 2)))
    config = {
        "global_paths": {"working_root_dir": str(tmp_path)},
        "models": {
            "selection": {"mode": "single", "model_name": "GCN", "task_type": "graph_regression"},
            "training": {"epochs": 2, "batch_size": 4},
            "evaluation": evaluation,
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
    return load_checkpoint_file(tmp_path / "checkpoints" / "best.pt", map_location="cpu")


def test_the_shipped_config_has_evaluation_at_models_level():
    models = yaml.safe_load(MODELS_YAML.read_text())["models"]
    assert "evaluation" in models and "evaluation" not in models["training"]


def test_test_after_training_false_is_honoured(tmp_path):
    best = _train(tmp_path, {"test_after_training": False, "visualization": {"enabled": False}})
    assert "test_metrics" not in best


def test_the_configured_metrics_are_computed(tmp_path):
    best = _train(tmp_path, {"metrics": ["mae"], "visualization": {"enabled": False}})
    # the configured metric, plus the per-target MAE and RMSE of the two targets
    assert {name for name in best["metrics_history"] if name.startswith("val_")} == {
        "val_loss",
        "val_mae",
        "val_mae_t0",
        "val_rmse_t0",
        "val_mae_t1",
        "val_rmse_t1",
    }
