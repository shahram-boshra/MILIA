"""Where a model's code comes from: ``run_info.model_source`` and ``lineage.model_source``.

A PyG model needs ``torch-geometric``, a plugin shipped with MILIA needs ``milia-py``, a plugin from
an installed distribution needs that distribution, and a directory plugin has nothing to install:
its path and a SHA-256 of its files are recorded. An ensemble records each member.
"""

from __future__ import annotations

import importlib.metadata
import json
from types import SimpleNamespace

import pytest
import torch

from milia_pipeline.models import model_source as source_module
from milia_pipeline.models.model_source import directory_sha256, model_source

PYG = f"torch-geometric=={importlib.metadata.version('torch-geometric')}"


# The shipped example plugin, discovered afresh and removed after the test (as in its e2e test)
from test_model_plugin_e2e import example_models, fresh  # noqa: E402, F401


@pytest.fixture(autouse=True)
def _registry():
    from milia_pipeline.models import list_models

    list_models()


def _milia_pin():
    import milia_pipeline

    # by its distribution name, at the running code's version (not the installed metadata's)
    return f"milia-py=={milia_pipeline.__version__}"


def test_a_pyg_model_needs_torch_geometric():
    source = model_source("GCN")
    assert source["kind"] == "pyg"
    assert source["class"] == "torch_geometric.nn.models.basic_gnn.GCN"
    assert source["requirements"] == [_milia_pin(), PYG]


@pytest.mark.usefixtures("example_models")
def test_a_plugin_shipped_with_milia_needs_milia():
    source = model_source("ExampleGNN")
    assert source["kind"] == "milia" and source["plugin"] == "example_models"
    assert source["requirements"] == [_milia_pin()]


@pytest.mark.usefixtures("example_models")
def test_an_ensemble_records_each_member():
    hyperparameters = {"ensemble_config": {"models": [{"name": "GCN"}, {"name": "ExampleGNN"}]}}
    source = model_source("ensemble", hyperparameters)
    assert source["kind"] == "ensemble"
    assert [(m["model"], m["kind"]) for m in source["members"]] == [
        ("GCN", "pyg"),
        ("ExampleGNN", "milia"),
    ]
    assert source["requirements"] == [_milia_pin(), PYG]


def test_a_model_not_in_the_registry_is_recorded_as_unknown():
    source = model_source("NoSuchModel")
    assert source["kind"] == "unknown" and "not in the model registry" in source["reason"]


def _loader_with(monkeypatch, metadata):
    loader = SimpleNamespace(_plugins={metadata.plugin_name: metadata})
    monkeypatch.setattr(
        "milia_pipeline.models.model_plugin_system.get_plugin_loader", lambda: loader
    )


def test_a_distribution_plugin_is_pinned_by_its_distribution(monkeypatch):
    _loader_with(
        monkeypatch,
        SimpleNamespace(
            plugin_name="installed",
            version="2.0.0",
            entry_point="installed = torch_geometric",
            plugin_path=None,
        ),
    )
    record = source_module._plugin_record("installed")
    assert record["kind"] == "distribution" and record["distribution"] == "torch-geometric"
    assert record["requirement"] == PYG and record["plugin_version"] == "2.0.0"


def test_a_directory_plugin_records_its_path_and_digest(monkeypatch, tmp_path):
    plugin = tmp_path / "my_plugin"
    (plugin / "__pycache__").mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("plugin_name: my_plugin\n")
    (plugin / "model.py").write_text("class M: pass\n")
    (plugin / "__pycache__" / "model.cpython-312.pyc").write_bytes(b"\0")
    _loader_with(
        monkeypatch,
        SimpleNamespace(
            plugin_name="my_plugin", version="0.1.0", entry_point=None, plugin_path=plugin
        ),
    )
    record = source_module._plugin_record("my_plugin")
    assert record["kind"] == "directory" and record["requirement"] is None
    assert record["path"] == str(plugin.resolve())
    assert record["sha256"] == directory_sha256(plugin)


def test_the_directory_digest_follows_the_code_not_the_bytecode(tmp_path):
    (tmp_path / "pkg" / "__pycache__").mkdir(parents=True)
    (tmp_path / "pkg" / "model.py").write_text("x = 1\n")
    before = directory_sha256(tmp_path / "pkg")
    (tmp_path / "pkg" / "__pycache__" / "model.pyc").write_bytes(b"compiled")
    assert directory_sha256(tmp_path / "pkg") == before
    (tmp_path / "pkg" / "model.py").write_text("x = 2\n")
    assert directory_sha256(tmp_path / "pkg") != before


def test_run_info_records_the_model_source():
    from milia_pipeline.models.training.run_info import collect_run_info

    info = collect_run_info(
        {"dataset_type": "DFT"},
        task_type="graph_regression",
        out_channels=2,
        target_names=None,
        data_split=None,
        model_name="GCN",
    )
    assert info["model_source"] == model_source("GCN")
    assert (
        collect_run_info(
            {}, task_type="graph_regression", out_channels=2, target_names=None, data_split=None
        )["model_source"]
        is None
    )


def _checkpoint(root, run_info):
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
            "out_channels": 2,
            "num_layers": 2,
        },
        task_type="graph_regression",
    )
    path = CheckpointManager(working_root_dir=root).save(
        root / "best.pt",
        model,
        hyper_parameters={
            "model_name": "GCN",
            "task_type": "graph_regression",
            "hyperparameters": info["hyperparameters_values"],
            "model_info": info,
        },
    )
    if run_info is not None:
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["run_info"] = run_info
        torch.save(checkpoint, path)
    return path


def test_a_package_records_the_source_and_its_card_says_what_to_install(tmp_path):
    from milia_pipeline.models.deployment.package import (
        MODEL_CARD,
        PACKAGE_MANIFEST,
        package_checkpoint,
    )

    source = model_source("GCN")
    package = package_checkpoint(_checkpoint(tmp_path, {"model_source": source}), tmp_path / "p")
    manifest = json.loads((package / PACKAGE_MANIFEST).read_text())
    assert manifest["lineage"]["model_source"] == source
    card = (package / MODEL_CARD).read_text()
    assert f"- Model code: `GCN` (pyg); install: `{_milia_pin()}`, `{PYG}`" in card


def test_a_package_of_an_older_checkpoint_says_the_source_is_not_recorded(tmp_path):
    from milia_pipeline.models.deployment.package import MODEL_CARD, package_checkpoint

    package = package_checkpoint(_checkpoint(tmp_path, None), tmp_path / "p")
    assert "- Model code: not recorded" in (package / MODEL_CARD).read_text()


def test_milia_is_pinned_by_its_name_at_the_running_version_whatever_the_metadata(monkeypatch):
    """Stale metadata under another name (seen on a host .venv: ``milia 1.2.0`` providing
    ``milia_pipeline`` while the code is 1.21.0) must not name the pin."""
    import milia_pipeline
    from milia_pipeline.models.training.run_info import _versions

    real = importlib.metadata.packages_distributions
    monkeypatch.setattr(
        importlib.metadata,
        "packages_distributions",
        lambda: {**real(), "milia_pipeline": ["milia"]},
    )
    assert model_source("GCN")["requirements"][0] == f"milia-py=={milia_pipeline.__version__}"
    assert _versions()["milia-py"] == milia_pipeline.__version__
