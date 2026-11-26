"""Deployment at the end of training (milia_pipeline.models.deployment.after_training)."""

from __future__ import annotations

import argparse
import json
import logging

import pytest
import torch
from torch_geometric.data import Data

from milia_pipeline.models.deployment.after_training import (
    GATE_FILE,
    deploy_after_training,
    evaluate_gate,
)
from milia_pipeline.models.deployment.package import PACKAGE_MANIFEST, model_family, publish_package
from milia_pipeline.models.deployment.settings import GateSettings, load_deployment_settings
from milia_pipeline.models.factory.model_factory import ModelFactory
from milia_pipeline.models.post_training import convert_to_pyg
from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import CheckpointManager

# The family folder of these tests' checkpoints (no run_info: dataset type and target names unknown)
FAMILY = model_family(
    dataset_type=None, task_type="graph_regression", target_names=None, target_count=2
)["id"]


def _checkpoint(root, val_loss: float, name: str = "best.pt"):
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
    return CheckpointManager(working_root_dir=root).save(
        root / "checkpoints" / name,
        model,
        epoch=1,
        metrics_history={"val_loss": [9.0, val_loss], "val_r2": [0.1, 0.5]},
        monitored_metric="val_loss",
        monitored_score=val_loss,
        test_metrics={"test_loss": val_loss * 2},
        hyper_parameters={
            "model_name": "GCN",
            "task_type": "graph_regression",
            "hyperparameters": info["hyperparameters_values"],
            "model_info": info,
        },
    )


class TestGate:
    def test_no_threshold_and_no_comparison_passes(self, tmp_path):
        decision = evaluate_gate(_checkpoint(tmp_path, 1.5), GateSettings(), tmp_path / "out")
        assert decision.passed and decision.value == 1.5 and decision.reasons == []

    @pytest.mark.parametrize(
        ("metric", "mode", "threshold", "passed"),
        [
            ("val_loss", "min", 2.0, True),
            ("val_loss", "min", 1.0, False),
            ("val_r2", "max", 0.4, True),
            ("val_r2", "max", 0.6, False),
            ("test_loss", "min", 3.0, True),
        ],
    )
    def test_threshold(self, tmp_path, metric, mode, threshold, passed):
        gate = GateSettings(metric=metric, mode=mode, threshold=threshold)
        decision = evaluate_gate(_checkpoint(tmp_path, 1.5), gate, tmp_path / "out")
        assert decision.passed is passed
        if not passed:
            assert "the threshold is" in decision.reasons[0]

    def test_an_unrecorded_metric_fails(self, tmp_path):
        decision = evaluate_gate(
            _checkpoint(tmp_path, 1.5), GateSettings(metric="val_mae"), tmp_path / "out"
        )
        assert not decision.passed and "records no 'val_mae'" in decision.reasons[0]

    def test_compare_to_current(self, tmp_path):
        out = tmp_path / "out"
        gate = GateSettings(compare_to_current=True)
        assert evaluate_gate(_checkpoint(tmp_path, 1.5), gate, out).passed  # nothing published
        publish_package(_checkpoint(tmp_path, 1.0, "old.pt"), out)

        worse = evaluate_gate(_checkpoint(tmp_path, 1.5), gate, out)
        better = evaluate_gate(_checkpoint(tmp_path, 0.5), gate, out)
        assert not worse.passed and worse.current_value == 1.0
        assert worse.current_version == str(out / FAMILY / "1")
        assert "does not beat 1" in worse.reasons[0]
        assert better.passed


class TestDeployAfterTraining:
    def test_a_passing_gate_publishes_and_records_the_decision(self, tmp_path):
        settings = load_deployment_settings({"enabled": True, "target": "local"})
        decision, result = deploy_after_training(
            _checkpoint(tmp_path, 1.5), settings, tmp_path, tmp_path / "training_output"
        )
        assert decision.passed and result.version_dir == tmp_path / "deployments" / FAMILY / "1"
        gate = json.loads((tmp_path / "training_output" / GATE_FILE).read_text())
        assert gate["passed"] is True and gate["metric"] == "val_loss" and gate["value"] == 1.5

    def test_a_failing_gate_publishes_nothing_and_records_why(self, tmp_path, caplog):
        settings = load_deployment_settings({"enabled": True, "gate": {"threshold": 0.1}})
        with caplog.at_level(logging.WARNING):
            decision, result = deploy_after_training(
                _checkpoint(tmp_path, 1.5), settings, tmp_path, tmp_path / "training_output"
            )
        assert result is None and not decision.passed
        assert not (tmp_path / "deployments").exists()
        gate = json.loads((tmp_path / "training_output" / GATE_FILE).read_text())
        assert gate["passed"] is False and gate["threshold"] == 0.1 and gate["reasons"]
        assert "Deployment skipped" in caplog.text


def _train(tmp_path, deployment):
    """main.py's standard training on a small dataset with a models.deployment section."""
    import main

    class _Dataset(list):
        structural_features_config = None

    in_channels = convert_to_pyg("CCO", format="smiles").x.shape[1]
    torch.manual_seed(0)
    dataset = _Dataset()
    for i in range(20):
        smiles = "C" * (2 + i % 3)
        graph = convert_to_pyg(smiles, format="smiles")
        assert graph.x.shape[1] == in_channels
        dataset.append(
            Data(
                x=graph.x,
                edge_index=graph.edge_index,
                y=torch.randn(1, 2),
                smiles=smiles,
                y_property_names=["Etot", "gap"],
            )
        )
    config = {
        "dataset_type": "DFT",
        "global_paths": {"working_root_dir": str(tmp_path)},
        "models": {
            "selection": {"mode": "single", "model_name": "GCN", "task_type": "graph_regression"},
            "training": {
                "epochs": 2,
                "batch_size": 4,
                "data_split": {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2},
            },
            "GCN": {"hidden_channels": 8, "num_layers": 2},
            "deployment": deployment,
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
    return main.handle_training_mode(args, logging.getLogger("test"), dataset, config)


class TestTrainingWorkflow:
    def test_enabled_training_deploys_and_predicts(self, tmp_path):
        molecules = tmp_path / "molecules.csv"
        molecules.write_text("smiles\nCCO\nCCC\n")
        code = _train(
            tmp_path,
            {"enabled": True, "target": "container", "predict": {"input_path": str(molecules)}},
        )
        (version,) = (tmp_path / "deployments").glob("*/1")
        # trained on a DFT dataset: its family names the dataset type and the task
        assert version.parent.name.startswith("DFT-graph_regression-")
        assert code == 0 and (version / "Dockerfile").is_file()
        manifest = json.loads((version / PACKAGE_MANIFEST).read_text())
        assert manifest["signature"]["outputs"]["names"] == ["Etot", "gap"]
        assert manifest["metrics"]["test"]  # the best checkpoint's test metrics
        gate = json.loads((tmp_path / "training_output" / GATE_FILE).read_text())
        assert gate["passed"] is True
        predictions = (
            tmp_path / "training_output" / f"predictions_{version.parent.name}_v1.csv"
        ).read_text()
        assert len(predictions.strip().splitlines()) == 3

    def test_a_failing_gate_keeps_the_run_successful(self, tmp_path):
        code = _train(tmp_path, {"enabled": True, "gate": {"threshold": 1e-12}})
        assert code == 0 and not (tmp_path / "deployments").exists()
        gate = json.loads((tmp_path / "training_output" / GATE_FILE).read_text())
        assert gate["passed"] is False

    def test_disabled_training_deploys_nothing(self, tmp_path):
        assert _train(tmp_path, {"enabled": False}) == 0
        assert not (tmp_path / "deployments").exists()
        assert not (tmp_path / "training_output" / GATE_FILE).exists()
