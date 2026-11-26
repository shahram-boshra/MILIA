"""``milia --deploy`` and package input to ``--predict`` (milia_pipeline.models.deployment.deploy)."""

from __future__ import annotations

import argparse
import json
import logging
from unittest.mock import patch

import pytest
import torch
from checkpoint_records import IDENTIFIER_DATA_INFO

from milia_pipeline.exceptions import DeploymentError
from milia_pipeline.models.deployment import deploy as deploy_module
from milia_pipeline.models.deployment.deploy import deploy, manifest_summary
from milia_pipeline.models.deployment.package import (
    PACKAGE_CHECKPOINT,
    PACKAGE_MANIFEST,
    model_family,
)
from milia_pipeline.models.deployment.settings import load_deployment_settings
from milia_pipeline.models.factory.model_factory import ModelFactory
from milia_pipeline.models.post_training import convert_to_pyg
from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import CheckpointManager

# The family folder of these tests' checkpoints (no run_info: dataset type and target names unknown)
FAMILY = model_family(
    dataset_type=None, task_type="graph_regression", target_names=None, target_count=2
)["id"]


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
        data_info=IDENTIFIER_DATA_INFO,
    )


def _settings(**section):
    return load_deployment_settings(section)


def test_container_deploys_a_versioned_package_with_a_dockerfile(checkpoint, tmp_path):
    first = deploy(checkpoint, _settings(output_dir=str(tmp_path / "out")), tmp_path)
    second = deploy(checkpoint, _settings(output_dir=str(tmp_path / "out")), tmp_path)

    assert first.version_dir == tmp_path / "out" / FAMILY / "1"
    assert second.version_dir.name == "2" and first.target == "container"
    assert first.files == ("Dockerfile", "MODEL_CARD.md", PACKAGE_MANIFEST, PACKAGE_CHECKPOINT)
    assert "USER root" not in (first.version_dir / "Dockerfile").read_text()


def test_aws_serves_as_root_and_local_needs_no_dockerfile(checkpoint, tmp_path):
    aws = deploy(checkpoint, _settings(target="aws", output_dir="out"), tmp_path)
    local = deploy(checkpoint, _settings(target="local", output_dir="out"), tmp_path)
    assert "USER root" in (aws.version_dir / "Dockerfile").read_text()
    assert "Dockerfile" not in local.files


def test_a_relative_output_dir_is_under_the_working_root(checkpoint, tmp_path):
    result = deploy(checkpoint, _settings(output_dir="deployments"), tmp_path / "root")
    assert result.version_dir == tmp_path / "root" / "deployments" / FAMILY / "1"


def test_deploying_a_package_copies_it_unchanged_as_a_new_version(checkpoint, tmp_path):
    first = deploy(checkpoint, _settings(target="local", output_dir="out"), tmp_path)
    again = deploy(first.version_dir, _settings(output_dir="out"), tmp_path)

    assert again.version_dir.name == "2" and "Dockerfile" in again.files
    for name in (PACKAGE_MANIFEST, PACKAGE_CHECKPOINT, "MODEL_CARD.md"):
        assert (again.version_dir / name).read_bytes() == (first.version_dir / name).read_bytes()


def test_a_tampered_package_is_not_deployed(checkpoint, tmp_path):
    first = deploy(checkpoint, _settings(target="local", output_dir="out"), tmp_path)
    with (first.version_dir / PACKAGE_CHECKPOINT).open("ab") as stream:
        stream.write(b"\0")
    with pytest.raises(DeploymentError, match="does not match the SHA-256"):
        deploy(first.version_dir, _settings(output_dir="out"), tmp_path)
    assert sorted(p.name for p in (tmp_path / "out" / FAMILY).iterdir()) == ["1", "family.json"]


def test_a_failing_target_step_publishes_nothing(checkpoint, tmp_path):
    with (
        patch.object(
            deploy_module, "write_serving_dockerfile", side_effect=DeploymentError("no docker")
        ),
        pytest.raises(DeploymentError, match="no docker"),
    ):
        deploy(checkpoint, _settings(output_dir="out"), tmp_path)
    assert not (tmp_path / "out" / FAMILY).exists()
    assert not [p for p in (tmp_path / "out").iterdir() if p.name.startswith(".staging-")]


def test_onnx_export_is_part_of_the_version(checkpoint, tmp_path):
    pytest.importorskip("onnxruntime", reason="the `onnx` extra is not installed")
    pytest.importorskip("onnxscript", reason="the `onnx` extra is not installed")
    result = deploy(
        checkpoint, _settings(target="local", output_dir="out", onnx={"enabled": True}), tmp_path
    )
    assert "model.onnx" in result.files


def test_manifest_summary_names_model_io_and_metrics(checkpoint, tmp_path):
    result = deploy(checkpoint, _settings(target="local", output_dir="out"), tmp_path)
    lines = manifest_summary(result.manifest)
    assert lines[0].startswith("Model: GCN (graph_regression), package milia-deployment/4")
    assert lines[1].startswith("Inputs: identifier, dataset;")
    assert "output: 2 values" in lines[1]
    assert "monitored metric not recorded" in lines[2]


class TestCli:
    def _args(self, *argv):
        from milia_pipeline.cli_manager import CLIManager

        return CLIManager().parse_args(list(argv))

    def test_deploy_needs_a_model_path(self):
        from milia_pipeline.cli_manager import CLIValidationError

        with pytest.raises(CLIValidationError, match="--model-path is required with --deploy"):
            self._args("--deploy")

    def test_deploy_is_not_combined_with_train(self):
        from milia_pipeline.cli_manager import CLIValidationError

        with pytest.raises(CLIValidationError, match="cannot be combined with --train"):
            self._args("--deploy", "--model-path", "best.pt", "--train")

    def test_deploy_flags_parse(self):
        args = self._args(
            "--deploy",
            "--model-path",
            "best.pt",
            "--deploy-target",
            "aws",
            "--deploy-output-dir",
            "out",
            "--deploy-base-image",
            "img:1",
            "--no-deploy-onnx",
        )
        assert (args.deploy_target, args.deploy_output_dir, args.deploy_base_image) == (
            "aws",
            "out",
            "img:1",
        )
        assert args.deploy_onnx is False
        assert self._args("--deploy", "--model-path", "best.pt").deploy_onnx is None


class TestHandleDeployMode:
    def _run(self, checkpoint, tmp_path, deployment=None, **flags):
        import main

        args = argparse.Namespace(model_path=str(checkpoint), **flags)
        config = {
            "global_paths": {"working_root_dir": str(tmp_path)},
            "models": {"deployment": deployment or {}},
        }
        return main.handle_deploy_mode(args, logging.getLogger("test"), config)

    def test_flags_override_the_section(self, checkpoint, tmp_path):
        code = self._run(
            checkpoint,
            tmp_path,
            {"target": "container", "output_dir": "from_config"},
            deploy_target="local",
            deploy_output_dir="from_flag",
        )
        assert code == 0
        version = tmp_path / "from_flag" / FAMILY / "1"
        assert version.is_dir() and not (version / "Dockerfile").exists()
        assert not (tmp_path / "from_config").exists()

    def test_the_section_gives_the_defaults(self, checkpoint, tmp_path):
        assert self._run(checkpoint, tmp_path, {"output_dir": "from_config"}) == 0
        assert (tmp_path / "from_config" / FAMILY / "1" / "Dockerfile").is_file()

    def test_an_invalid_section_fails(self, checkpoint, tmp_path):
        assert self._run(checkpoint, tmp_path, {"tagret": "aws"}) == 1

    def test_a_missing_model_fails(self, tmp_path):
        assert self._run(tmp_path / "nope.pt", tmp_path) == 1

    def test_a_failed_deployment_fails_and_publishes_nothing(self, checkpoint, tmp_path):
        with patch.object(
            deploy_module, "write_serving_dockerfile", side_effect=DeploymentError("boom")
        ):
            assert self._run(checkpoint, tmp_path, {"output_dir": "out"}) == 1
        assert not (tmp_path / "out" / FAMILY).exists()


def test_predict_refuses_a_tampered_package(checkpoint, tmp_path):
    import main

    published = deploy(checkpoint, _settings(target="local", output_dir="out"), tmp_path)
    with (published.version_dir / PACKAGE_CHECKPOINT).open("ab") as stream:
        stream.write(b"\0")
    molecules = tmp_path / "molecules.csv"
    molecules.write_text("smiles\nCCO\n")
    args = argparse.Namespace(
        model_path=str(published.version_dir),
        test_path=str(molecules),
        preds_path=str(tmp_path / "preds.csv"),
    )
    config = {"global_paths": {"working_root_dir": str(tmp_path)}}
    assert main.handle_predict_mode(args, logging.getLogger("test"), config) == 1
    assert not (tmp_path / "preds.csv").exists()
    assert json.loads((published.version_dir / PACKAGE_MANIFEST).read_text())["model_name"] == "GCN"
