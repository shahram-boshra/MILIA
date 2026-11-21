"""Verified, versioned packages (milia_pipeline.models.deployment.package.publish_package)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from milia_pipeline.exceptions import DeploymentError
from milia_pipeline.models.deployment import package as package_module
from milia_pipeline.models.deployment.package import (
    MODEL_CARD,
    PACKAGE_CHECKPOINT,
    PACKAGE_MANIFEST,
    load_package,
    package_checkpoint,
    publish_package,
    verify_package,
)
from milia_pipeline.models.factory.model_factory import ModelFactory
from milia_pipeline.models.post_training import convert_to_pyg
from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import CheckpointManager


@pytest.fixture
def checkpoint(tmp_path):
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
    return CheckpointManager(working_root_dir=tmp_path).save(
        tmp_path / "run" / "checkpoints" / "best.pt",
        model,
        hyper_parameters={
            "model_name": "GCN",
            "task_type": "graph_regression",
            "hyperparameters": info["hyperparameters_values"],
            "model_info": info,
        },
    )


def _leftovers(output):
    return [p.name for p in output.iterdir() if p.name.startswith(".staging-")]


def test_each_publish_is_a_new_verified_version(checkpoint, tmp_path):
    output = tmp_path / "models"
    first = publish_package(checkpoint, output)
    second = publish_package(checkpoint, output)

    assert first == output / "GCN" / "1" and second == output / "GCN" / "2"
    for version in (first, second):
        assert sorted(p.name for p in version.iterdir()) == sorted(
            [MODEL_CARD, PACKAGE_CHECKPOINT, PACKAGE_MANIFEST]
        )
        verify_package(version)
        load_package(version)
    assert _leftovers(output) == []


def test_the_next_version_follows_the_highest_existing_one(checkpoint, tmp_path):
    model_dir = tmp_path / "models" / "GCN"
    for name in ("1", "3", "notes"):
        (model_dir / name).mkdir(parents=True)
    (model_dir / "4").write_text("a file named like a version is never replaced")

    published = publish_package(checkpoint, tmp_path / "models")
    assert published == model_dir / "5"
    assert (model_dir / "4").read_text().startswith("a file")
    assert list((model_dir / "3").iterdir()) == []


def test_a_package_that_does_not_reproduce_leaves_nothing(checkpoint, tmp_path):
    output = tmp_path / "models"
    with (
        patch("torch.testing.assert_close", side_effect=AssertionError("values differ")),
        pytest.raises(DeploymentError, match="does not reproduce the predictions"),
    ):
        publish_package(checkpoint, output)
    assert not (output / "GCN").exists() and _leftovers(output) == []


def test_package_checkpoint_removes_only_its_own_files_on_a_mismatch(checkpoint, tmp_path):
    target = tmp_path / "pkg"
    target.mkdir()
    (target / "Dockerfile").write_text("FROM scratch\n")
    with (
        patch("torch.testing.assert_close", side_effect=AssertionError("values differ")),
        pytest.raises(DeploymentError, match="does not reproduce"),
    ):
        package_checkpoint(checkpoint, target)
    assert [p.name for p in target.iterdir()] == ["Dockerfile"]


def test_verify_false_skips_the_reload(checkpoint, tmp_path):
    with patch.object(package_module, "_check_reproduces", side_effect=AssertionError("ran")):
        package_checkpoint(checkpoint, tmp_path / "pkg", verify=False)
    verify_package(tmp_path / "pkg")


def test_the_reload_uses_the_recorded_input_example(checkpoint, tmp_path):
    seen = []
    real = package_module._predictions

    def spy(path, smiles):
        seen.append(list(smiles))
        return real(path, smiles)

    with patch.object(package_module, "_predictions", side_effect=spy):
        package_checkpoint(checkpoint, tmp_path / "pkg")
    assert seen == [["CCO"], ["CCO"]]  # no run_info: the fallback example, source then package


def test_a_tampered_published_model_fails_verification(checkpoint, tmp_path):
    published = publish_package(checkpoint, tmp_path / "models")
    with (published / PACKAGE_CHECKPOINT).open("ab") as stream:
        stream.write(b"\0")
    with pytest.raises(DeploymentError, match="does not match the SHA-256"):
        load_package(published)


@pytest.mark.parametrize(
    ("name", "folder"),
    [("GCN", "GCN"), ("my model/v2", "my_model_v2"), ("..", "model"), ("Ens.1", "Ens.1")],
)
def test_model_names_become_one_safe_folder(name, folder):
    assert package_module._folder_name(name) == folder
