"""The MLflow model registry (deployment/registry.py, mlflow_pyfunc.py); needs the mlflow extra.

The ``models.deployment.registry`` section is tested in test_deployment_settings_unit.py. Every test uses its own SQLite store (a copy of one migrated per session) and artifact folder under
``tmp_path`` and runs with the process working directory in an empty folder, which must stay empty.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from checkpoint_records import IDENTIFIER_DATA_INFO

from milia_pipeline.models.deployment.package import PACKAGE_MANIFEST, publish_package
from milia_pipeline.models.deployment.settings import DeploymentSettings, RegistrySettings

mlflow = pytest.importorskip("mlflow")

REPO = Path(__file__).resolve().parents[1]
# A training commit in the form git records (40 hex digits); repeated, as in the package tests, so it is
# plainly test data rather than a high-entropy string a secret scanner flags
TRAINING_COMMIT = "c" * 40
CALIBRATION = {
    "method": "split_conformal_absolute",
    "alpha": 0.1,
    "n": 10,
    "quantiles": [0.5, 0.25],
}


@pytest.fixture(scope="session")
def migrated_store(tmp_path_factory):
    """One SQLite tracking store with MLflow's schema, created once per session.

    A new store runs every MLflow migration (``SqlAlchemyStore`` → Alembic upgrade): ~40 s of
    SQLite DDL and commits per store on a slow disk (profiled), paid once per store in real use
    but once per test if each test starts from an empty file. Each test gets its own copy instead
    (:func:`_store_copy`), so tests stay isolated; one test still creates a store from nothing.
    """
    path = tmp_path_factory.mktemp("mlflow-template") / "mlflow.db"
    mlflow.MlflowClient(tracking_uri=f"sqlite:///{path}").search_experiments()
    return path


def _store_copy(template, destination):
    """A consistent copy of a SQLite store (the sqlite3 backup API, not a file copy)."""
    import sqlite3

    with sqlite3.connect(template) as source, sqlite3.connect(destination) as target:
        source.backup(target)
    source.close()
    target.close()


@pytest.fixture
def empty_cwd(tmp_path, monkeypatch, migrated_store):
    """The working directory in an empty folder, which must stay empty, and a migrated store at
    the default location (``<working_root_dir>/mlflow.db``, the working root being ``tmp_path``)."""
    _store_copy(migrated_store, tmp_path / "mlflow.db")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    yield cwd
    assert sorted(p.name for p in cwd.iterdir()) == []


def _checkpoint(
    root,
    *,
    task_type="graph_regression",
    out_channels=2,
    names=("homo", "lumo"),
    seed=0,
    data_info=IDENTIFIER_DATA_INFO,
    calibration=None,
    val_loss=1.0,
    structure_example=None,
):
    from milia_pipeline.models.factory.model_factory import ModelFactory
    from milia_pipeline.models.post_training import convert_to_pyg
    from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
        CheckpointManager,
    )

    torch.manual_seed(seed)
    model, info = ModelFactory().create_model_with_info(
        name="GCN",
        hyperparameters={
            "in_channels": convert_to_pyg("CCO", format="smiles").x.shape[1],
            "hidden_channels": 8,
            "out_channels": out_channels,
            "num_layers": 2,
        },
        task_type=task_type,
    )
    path = root / f"seed{seed}" / "best.pt"
    CheckpointManager(working_root_dir=root).save(
        path,
        model,
        epoch=1,
        data_info=data_info,
        metrics_history={"val_loss": [9.0, val_loss]},
        monitored_metric="val_loss",
        monitored_score=val_loss,
        hyper_parameters={
            "model_name": "GCN",
            "task_type": task_type,
            "hyperparameters": info["hyperparameters_values"],
            "model_info": info,
        },
    )
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["run_info"] = {
        "input_example": ["CCO", "CC"],
        "structure_example": structure_example or [],
        "targets": {"names": list(names) if names else None, "count": out_channels},
        "dataset": {"type": "DFT"},
        "git_commit": TRAINING_COMMIT,
    }
    if calibration is not None:
        checkpoint["calibration"] = calibration
    torch.save(checkpoint, path)
    return path


def _settings(tmp_path, **registry):
    return RegistrySettings(enabled=True, **registry)


def _register(tmp_path, version_dir, approval=None, **registry):
    from milia_pipeline.models.deployment.registry import register_package

    return register_package(version_dir, _settings(tmp_path, **registry), tmp_path, approval)


def _client(result):
    return mlflow.MlflowClient(tracking_uri=result.tracking_uri, registry_uri=result.tracking_uri)


def _load(result):
    mlflow.set_registry_uri(result.tracking_uri)
    mlflow.set_tracking_uri(result.tracking_uri)
    return mlflow.pyfunc.load_model(f"models:/{result.model_name}@{result.alias}")


def _packaged(version_dir):
    from milia_pipeline.models.deployment.serving import PackagedPredictor

    return PackagedPredictor.from_package(version_dir)


def test_a_published_package_becomes_version_1_with_the_alias(tmp_path, empty_cwd):
    version = publish_package(_checkpoint(tmp_path), tmp_path / "deployments")
    result = _register(tmp_path, version, {"by": "gate", "metric": "val_loss", "value": 1.0})
    assert (result.version, result.alias, result.reused, result.skipped) == (
        "1",
        "champion",
        False,
        None,
    )
    manifest = json.loads((version / PACKAGE_MANIFEST).read_text())
    family = json.loads((version.parent / "family.json").read_text())["id"]
    assert result.model_name == f"milia.{family}"
    assert result.tracking_uri == f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"
    client = _client(result)
    tags = client.get_model_version(result.model_name, "1").tags
    assert tags["milia.package_sha256"] == manifest["sha256"]
    assert tags["validation_status"] == "approved" and tags["milia.approval"] == "gate"
    assert tags["milia.gate_metric"] == "val_loss" and tags["milia.family"] == family
    run = client.get_run(result.run_id)
    assert run.data.tags["mlflow.source.git.commit"] == manifest["lineage"]["git_commit"]
    assert run.data.metrics["val_loss"] == 1.0
    assert client.get_experiment(run.info.experiment_id).artifact_location == (
        (tmp_path / "mlartifacts").resolve().as_uri()
    )
    # MLflow's URIs are put back: nothing set before, nothing left set
    assert "MLFLOW_TRACKING_URI" not in os.environ and "MLFLOW_REGISTRY_URI" not in os.environ


def test_the_next_package_of_the_family_is_version_2_and_takes_the_alias(tmp_path, empty_cwd):
    out = tmp_path / "deployments"
    first = _register(tmp_path, publish_package(_checkpoint(tmp_path, seed=0), out))
    second = _register(tmp_path, publish_package(_checkpoint(tmp_path, seed=1), out))
    assert (first.model_name, first.version, second.version) == (second.model_name, "1", "2")
    alias = _client(second).get_model_version_by_alias(second.model_name, "champion")
    assert str(alias.version) == "2"


def test_a_package_registered_before_is_not_registered_again(tmp_path, empty_cwd):
    out = tmp_path / "deployments"
    path = _checkpoint(tmp_path, seed=0)
    first = _register(tmp_path, publish_package(path, out))
    _register(tmp_path, publish_package(_checkpoint(tmp_path, seed=1), out))
    again = _register(tmp_path, publish_package(path, out))  # the same checkpoint, version 3 folder
    assert (again.version, again.reused) == (first.version, True)
    client = _client(again)
    assert len(client.search_model_versions(f"name = '{again.model_name}'")) == 2
    assert str(client.get_model_version_by_alias(again.model_name, "champion").version) == "1"


def test_another_family_is_another_registered_model(tmp_path, empty_cwd):
    out = tmp_path / "deployments"
    a = _register(tmp_path, publish_package(_checkpoint(tmp_path), out))
    b = _register(
        tmp_path, publish_package(_checkpoint(tmp_path, out_channels=1, names=("gap",)), out)
    )
    assert a.model_name != b.model_name and a.version == b.version == "1"


def test_the_registered_model_predicts_what_the_package_predicts(tmp_path, empty_cwd):
    import pandas as pd

    version = publish_package(
        _checkpoint(tmp_path, calibration=CALIBRATION), tmp_path / "deployments"
    )
    result = _register(tmp_path, version)
    model = _load(result)
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    frame = model.predict(pd.DataFrame({"smiles": smiles}))
    rows, intervals, _ = _packaged(version).predict_response(smiles)
    assert list(frame.columns) == [
        "homo",
        "homo_lower",
        "homo_upper",
        "lumo",
        "lumo_lower",
        "lumo_upper",
    ]
    np.testing.assert_allclose(frame[["homo", "lumo"]].to_numpy(), np.asarray(rows), atol=1e-6)
    np.testing.assert_allclose(
        frame[["homo_lower", "lumo_lower"]].to_numpy(), np.asarray(intervals["lower"]), atol=1e-6
    )


def test_the_signature_follows_the_package(tmp_path, empty_cwd):
    version = publish_package(_checkpoint(tmp_path), tmp_path / "deployments")
    signature = _load(_register(tmp_path, version)).metadata.signature
    assert signature.inputs.input_names() == ["smiles"]
    assert signature.outputs.input_names() == ["homo", "lumo"]


def test_a_request_outside_the_schema_is_refused_by_mlflow(tmp_path, empty_cwd):
    import pandas as pd
    from mlflow.exceptions import MlflowException

    version = publish_package(_checkpoint(tmp_path), tmp_path / "deployments")
    model = _load(_register(tmp_path, version))
    with pytest.raises(MlflowException, match="smiles"):
        model.predict(pd.DataFrame({"molecule": ["CCO"]}))


def test_a_node_level_model_gives_per_atom_arrays(tmp_path, empty_cwd):
    import pandas as pd

    version = publish_package(
        _checkpoint(tmp_path, task_type="node_regression", out_channels=1, names=("q",)),
        tmp_path / "deployments",
    )
    model = _load(_register(tmp_path, version))
    frame = model.predict(pd.DataFrame({"smiles": ["CCO", "CCCC"]}))
    rows, _, atoms = _packaged(version).predict_response(["CCO", "CCCC"])
    assert list(frame.columns) == ["q", "atomic_numbers"]
    assert [len(v) for v in frame["q"]] == [3, 4]
    for got, want in zip(frame["q"], rows, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want)[:, 0], atol=1e-6)
    assert [list(z) for z in frame["atomic_numbers"]] == atoms


def test_a_structure_model_takes_structures(tmp_path, monkeypatch):
    # Structure inputs are built with the dataset's configuration, read from ./configs (F105)
    monkeypatch.chdir(REPO)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    import pandas as pd
    from test_serve_structures import AMMONIA, STRUCTURE_ONLY, WATER, _data_info
    from test_serve_structures import _checkpoint as structure_checkpoint

    examples = [WATER.to_dict(), AMMONIA.to_dict()]
    version = publish_package(
        structure_checkpoint(tmp_path, _data_info(sources=STRUCTURE_ONLY), examples),
        tmp_path / "deployments",
    )
    model = _load(_register(tmp_path, version))
    signature = model.metadata.signature
    assert "smiles" not in signature.inputs.input_names()
    assert {"atomic_numbers", "coordinates"} <= set(signature.inputs.required_input_names())
    request = pd.DataFrame(
        [{k: e[k] for k in ("atomic_numbers", "coordinates", "charge")} for e in examples]
    )
    frame = model.predict(request)
    want = _packaged(version).predict_instances(examples)
    np.testing.assert_allclose(frame[["a", "b"]].to_numpy(), np.asarray(want), atol=1e-6)


def test_a_package_that_serves_no_instance_kind_is_not_registered(tmp_path, empty_cwd):
    version = publish_package(_checkpoint(tmp_path, data_info=None), tmp_path / "deployments")
    result = _register(tmp_path, version)
    assert result.version is None and "neither SMILES nor structure" in result.skipped
    assert _client(result).search_registered_models() == []


def test_an_unreachable_registry_leaves_the_package_published(tmp_path, empty_cwd):
    from milia_pipeline.models.deployment.registry import RegistryError

    version = publish_package(_checkpoint(tmp_path), tmp_path / "deployments")
    monkey_uri = "http://127.0.0.1:9"
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
    try:
        with pytest.raises(RegistryError, match=str(version)):
            _register(tmp_path, version, tracking_uri=monkey_uri)
    finally:
        del os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]
    assert (version / PACKAGE_MANIFEST).is_file()


def _logged_model_dir(result):
    """The local artifact folder of a version's logged model."""
    from mlflow.artifacts import download_artifacts

    mlflow.set_tracking_uri(result.tracking_uri)
    mlflow.set_registry_uri(result.tracking_uri)
    try:
        return Path(download_artifacts(f"models:/{result.model_name}/{result.version}"))
    finally:
        mlflow.set_tracking_uri(None)
        mlflow.set_registry_uri(None)
        os.environ.pop("MLFLOW_REGISTRY_URI", None)


def test_the_training_commit_tags_the_logged_model_not_the_registering_checkout(
    tmp_path, empty_cwd, monkeypatch
):
    import mlflow.tracking.context.git_context as git_context

    # The registering process is in another checkout
    other = "f" * 40
    monkeypatch.setattr(git_context, "get_git_commit", lambda _path: other)
    monkeypatch.setattr(
        git_context.GitRunContext,
        "_git_info",
        property(lambda self: {"mlflow.source.git.commit": other}),
    )
    version = publish_package(_checkpoint(tmp_path), tmp_path / "deployments")
    result = _register(tmp_path, version)
    commit = json.loads((version / PACKAGE_MANIFEST).read_text())["lineage"]["git_commit"]
    client = _client(result)
    source = client.get_model_version(result.model_name, "1").source  # models:/<logged model id>
    model = client.get_logged_model(source.rsplit("/", 1)[-1])
    assert model.tags["mlflow.source.git.commit"] == commit != other
    assert client.get_run(result.run_id).data.tags["milia.git_commit"] == commit


def test_milia_own_uv_project_is_the_environment_record(tmp_path, empty_cwd):
    from milia_pipeline.models.deployment.registry import milia_uv_project

    project = milia_uv_project()
    if project is None:
        pytest.skip("MILIA is not running from a checkout with uv.lock")
    version = publish_package(_checkpoint(tmp_path), tmp_path / "deployments")
    logged = _logged_model_dir(_register(tmp_path, version))
    assert (logged / "uv.lock").read_bytes() == (project / "uv.lock").read_bytes()
    assert (logged / "requirements.txt").read_text().split()[0].startswith("milia-py==")


def test_an_unrelated_uv_project_in_the_working_directory_is_not_recorded(
    tmp_path, empty_cwd, monkeypatch
):
    from milia_pipeline.models.deployment import registry

    monkeypatch.setattr(registry, "milia_uv_project", lambda: None)
    (empty_cwd / "uv.lock").write_text("version = 1\n")
    (empty_cwd / "pyproject.toml").write_text('[project]\nname = "unrelated"\n')
    version = publish_package(_checkpoint(tmp_path), tmp_path / "deployments")
    logged = _logged_model_dir(_register(tmp_path, version))
    assert not (logged / "uv.lock").exists() and not (logged / "pyproject.toml").exists()
    assert "MLFLOW_UV_AUTO_DETECT" not in os.environ
    for name in ("uv.lock", "pyproject.toml"):  # the empty_cwd check: nothing else was written
        (empty_cwd / name).unlink()


def test_mlflow_tracking_uri_is_used_when_none_is_configured(tmp_path, empty_cwd, monkeypatch):
    # A store created from nothing (MLflow's migrations run here, once)
    uri = f"sqlite:///{tmp_path / 'env.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    version = publish_package(_checkpoint(tmp_path), tmp_path / "deployments")
    assert _register(tmp_path, version).tracking_uri == uri


# --------------------------------------------------------------------------- deploy and the gate


def _deployment(tmp_path, **gate):
    return DeploymentSettings(
        target="local",
        output_dir=str(tmp_path / "deployments"),
        gate=gate or {},
        registry=_settings(tmp_path),
    )


def test_a_passing_gate_registers_the_version_with_its_approval(tmp_path, empty_cwd):
    from milia_pipeline.models.deployment.after_training import deploy_after_training

    decision, result = deploy_after_training(
        _checkpoint(tmp_path), _deployment(tmp_path), tmp_path, tmp_path / "results"
    )
    assert decision.passed and result.registry.version == "1" and result.registry_error is None
    tags = _client(result.registry).get_model_version(result.registry.model_name, "1").tags
    assert tags["milia.approval"] == "gate" and tags["milia.gate_value"] == "1.0"


def test_a_failing_gate_registers_nothing(tmp_path, empty_cwd):
    from milia_pipeline.models.deployment.after_training import deploy_after_training

    decision, result = deploy_after_training(
        _checkpoint(tmp_path, val_loss=5.0),
        _deployment(tmp_path, threshold=1.0),
        tmp_path,
        tmp_path / "results",
    )
    assert not decision.passed and result is None
    store = f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"
    client = mlflow.MlflowClient(tracking_uri=store, registry_uri=store)
    assert client.search_registered_models() == [] and client.search_runs(["0"]) == []


def test_a_failed_registration_is_reported_and_the_version_stays(tmp_path, empty_cwd, monkeypatch):
    from milia_pipeline.models.deployment import registry
    from milia_pipeline.models.deployment.deploy import deploy, registry_summary

    def fail(*_args, **_kwargs):
        raise registry.RegistryError("registering failed (OSError: disk full)")

    monkeypatch.setattr(registry, "register_package", fail)
    result = deploy(_checkpoint(tmp_path), _deployment(tmp_path), tmp_path)
    assert result.registry is None and "disk full" in result.registry_error
    assert (result.version_dir / PACKAGE_MANIFEST).is_file()
    level, lines = registry_summary(result)
    assert level == 40 and str(result.version_dir) in lines[0]


def test_the_deploy_flag_overrides_the_registry_setting():
    from milia_pipeline.cli_manager import CLIManager

    parser = CLIManager().parser
    assert parser.parse_args(["--deploy", "--deploy-register"]).deploy_register is True
    assert parser.parse_args(["--deploy", "--no-deploy-register"]).deploy_register is False
    assert parser.parse_args(["--deploy"]).deploy_register is None
