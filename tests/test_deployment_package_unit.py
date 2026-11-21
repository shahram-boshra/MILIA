"""
Deployment packages built from MILIA checkpoints rebuild the model in any process (F57).

Every target used to save a bare ``state_dict`` and deploy it with ``torch.load(...).eval()``,
which fails on a dict. Unit tests mocked ``torch.load`` to return a module, which hid the bug.
These tests use a real model, a checkpoint written by ``CheckpointManager.save`` and no mocks.
"""

import json

import pytest
import torch
from torch_geometric.data import Batch, Data

from milia_pipeline.exceptions import DeploymentError
from milia_pipeline.models.deployment import DeploymentManager
from milia_pipeline.models.deployment.deployment_strategies import (
    DeploymentConfig,
    LocalDeploymentStrategy,
)
from milia_pipeline.models.deployment.package import (
    MODEL_CARD,
    PACKAGE_CHECKPOINT,
    PACKAGE_FORMAT,
    PACKAGE_MANIFEST,
    is_package,
    load_package,
    package_checkpoint,
)
from milia_pipeline.models.factory.model_factory import ModelFactory
from milia_pipeline.models.post_training import convert_to_pyg
from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import CheckpointManager

# Packaging verifies the package on a SMILES example, so the model takes MILIA's featurization.
IN_CHANNELS = convert_to_pyg("CCO", format="smiles").x.shape[1]
HYPERPARAMETERS = {
    "in_channels": IN_CHANNELS,
    "hidden_channels": 8,
    "out_channels": 1,
    "num_layers": 2,
}


@pytest.fixture
def trained(tmp_path):
    """A MILIA GCN, its checkpoint written by CheckpointManager.save, and a sample batch."""
    torch.manual_seed(0)
    model, info = ModelFactory().create_model_with_info(
        name="GCN", hyperparameters=HYPERPARAMETERS, task_type="graph_regression"
    )
    checkpoint = CheckpointManager(working_root_dir=tmp_path).save(
        tmp_path / "checkpoints" / "best.pt",
        model,
        optimizer=torch.optim.Adam(model.parameters()),
        hyper_parameters={
            "model_name": "GCN",
            "task_type": "graph_regression",
            "hyperparameters": info["hyperparameters_values"],
            "model_info": info,
        },
    )
    graph = Data(
        x=torch.randn(5, IN_CHANNELS), edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    )
    batch = Batch.from_data_list([graph, graph])
    model.eval()
    with torch.no_grad():
        expected = model(batch.x, batch.edge_index, batch=batch.batch)
    return checkpoint, batch, expected


def _predict(model, batch):
    with torch.no_grad():
        return model(batch.x, batch.edge_index, batch=batch.batch)


def test_package_holds_an_inference_checkpoint_and_a_manifest(trained, tmp_path):
    checkpoint, _, _ = trained
    package = package_checkpoint(checkpoint, tmp_path / "package")

    assert is_package(package)
    assert sorted(path.name for path in package.iterdir()) == [
        MODEL_CARD,
        PACKAGE_MANIFEST,
        PACKAGE_CHECKPOINT,
    ]
    manifest = json.loads((package / PACKAGE_MANIFEST).read_text())
    assert manifest["format"] == PACKAGE_FORMAT
    assert (manifest["model_name"], manifest["task_type"]) == ("GCN", "graph_regression")
    stored = torch.load(package / PACKAGE_CHECKPOINT, weights_only=True)
    assert "optimizer_state_dict" not in stored and "model_state_dict" in stored


def test_a_fresh_local_strategy_rebuilds_the_model_from_a_package(trained, tmp_path):
    checkpoint, batch, expected = trained
    LocalDeploymentStrategy(DeploymentConfig(), verbose=False).prepare_model(
        str(checkpoint), tmp_path / "package"
    )

    fresh = LocalDeploymentStrategy(DeploymentConfig(), verbose=False)  # no in-memory module
    fresh.deploy(tmp_path / "package")

    assert fresh.model.training is False
    assert torch.allclose(_predict(fresh.model, batch), expected)


def test_manager_packages_a_checkpoint_path(trained, tmp_path):
    checkpoint, batch, expected = trained
    manager = DeploymentManager(target="local", verbose=False)
    manager.deploy(manager.prepare_model(checkpoint, tmp_path / "package"))
    assert torch.allclose(_predict(manager.strategy.model, batch), expected)


def test_tampered_checkpoint_is_rejected(trained, tmp_path):
    checkpoint, _, _ = trained
    package = package_checkpoint(checkpoint, tmp_path / "package")
    with (package / PACKAGE_CHECKPOINT).open("ab") as stream:
        stream.write(b"\0")
    with pytest.raises(DeploymentError, match="SHA-256"):
        load_package(package)


def test_unknown_manifest_format_is_rejected(trained, tmp_path):
    checkpoint, _, _ = trained
    package = package_checkpoint(checkpoint, tmp_path / "package")
    manifest = json.loads((package / PACKAGE_MANIFEST).read_text())
    manifest["format"] = "milia-deployment/999"
    (package / PACKAGE_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(DeploymentError, match="expected one of"):
        load_package(package)


def test_a_bare_state_dict_cannot_be_packaged(tmp_path):
    bare = tmp_path / "weights.pt"
    torch.save(torch.nn.Linear(4, 1).state_dict(), bare)
    with pytest.raises(DeploymentError, match="missing: model_state_dict"):
        package_checkpoint(bare, tmp_path / "package")


def test_missing_checkpoint_and_missing_manifest_are_reported(tmp_path):
    with pytest.raises(DeploymentError, match="Checkpoint not found"):
        package_checkpoint(tmp_path / "absent.pt", tmp_path / "package")
    with pytest.raises(DeploymentError, match="not a MILIA deployment package"):
        load_package(tmp_path)
