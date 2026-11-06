"""Automatic model plugin discovery when models are built (``ensure_model_plugins_discovered``)."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import uuid

import pytest
import yaml

from milia_pipeline.models import model_plugin_system as mps
from milia_pipeline.models.factory.model_factory import ModelFactory
from milia_pipeline.models.hpo import hpo_manager
from milia_pipeline.models.registry import model_registry
from milia_pipeline.models.utils import config_bridge
from milia_pipeline.models.utils.config_bridge import PluginsConfig

_MODEL_SOURCE = """
import torch
from torch import nn


class AutoNet(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_channels: int = 8, out_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        return self.lin(torch.relu(x))
"""


def _unique(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def _write_plugin(container, plugin_name: str, model_name: str) -> None:
    plugin_dir = container / plugin_name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "model.py").write_text(_MODEL_SOURCE)
    manifest = {
        "schema_version": 2,
        "plugin_name": plugin_name,
        "version": "1.0.0",
        "author": "tests",
        "description": "auto-discovery plugin",
        "license": "MIT",
        "models": [
            {
                "name": model_name,
                "class": "model:AutoNet",
                "category": "basic_gnn",
                "description": "auto-discovery model",
                "supported_tasks": ["graph_regression"],
            }
        ],
    }
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump(manifest))


@pytest.fixture
def fresh(tmp_path):
    """Fresh discovery state and loader; plugin modules and models are removed afterwards."""
    saved_instance = mps.ModelPluginLoader._instance
    saved_module_loader = mps._plugin_loader
    mps.ModelPluginLoader._instance = None
    mps._plugin_loader = None
    mps._reset_model_plugin_discovery()
    registry = model_registry.registry
    before = set(registry.list_available_models())
    try:
        yield
    finally:
        for name in set(registry.list_available_models()) - before:
            registry.unregister_model(name)
        for name, module in list(sys.modules.items()):
            if (getattr(module, "__file__", None) or "").startswith(str(tmp_path)):
                del sys.modules[name]
        mps._reset_model_plugin_discovery()
        mps.ModelPluginLoader._instance = saved_instance
        mps._plugin_loader = saved_module_loader


@pytest.fixture
def configured(tmp_path, monkeypatch):
    """Point the plugins configuration at a temporary container; returns a setter."""
    (tmp_path / "configs").mkdir()
    monkeypatch.chdir(tmp_path)
    container = tmp_path / "plugins"
    container.mkdir()

    def configure(**fields):
        config = PluginsConfig(
            plugin_paths=[str(container)],
            include_builtin=False,
            discover_entry_points=False,
            **fields,
        )
        monkeypatch.setattr(config_bridge, "get_plugins_config", lambda config_path=None: config)
        return container

    return configure


def test_first_factory_discovers_and_registers_once(fresh, configured, monkeypatch):
    container = configured()
    plugin_name, model_name = _unique("auto_plugin_"), _unique("AutoNet")
    _write_plugin(container, plugin_name, model_name)
    calls = []
    real = mps._discover_and_load_configured
    monkeypatch.setattr(
        mps, "_discover_and_load_configured", lambda path: calls.append(path) or real(path)
    )

    factory = ModelFactory()
    ModelFactory()

    assert calls == [None]
    assert factory.registry.has_model(model_name)
    assert mps.ensure_model_plugins_discovered() == [plugin_name]
    assert calls == [None]


@pytest.mark.parametrize("flag", ["enabled", "auto_discover"])
def test_configuration_can_turn_discovery_off(fresh, configured, monkeypatch, flag):
    container = configured(**{flag: False})
    _write_plugin(container, _unique("auto_plugin_"), _unique("AutoNet"))
    monkeypatch.setattr(
        mps.ModelPluginLoader,
        "discover_plugins",
        lambda *a, **k: pytest.fail("discovery must not run"),
    )

    ModelFactory()
    assert mps.ensure_model_plugins_discovered() == []


def test_discovery_error_is_logged_and_the_factory_still_works(fresh, monkeypatch, caplog):
    def broken(config_path):
        raise RuntimeError("broken discovery")

    monkeypatch.setattr(mps, "_discover_and_load_configured", broken)
    with caplog.at_level(logging.ERROR):
        factory = ModelFactory()
    assert factory.registry.has_model("GCN")
    assert "Automatic model plugin discovery failed: broken discovery" in caplog.text
    assert mps.ensure_model_plugins_discovered() == []


def test_plugin_that_fails_to_load_is_left_out(fresh, configured, caplog):
    container = configured()
    good, bad = _unique("auto_good_"), _unique("auto_bad_")
    _write_plugin(container, good, _unique("AutoNet"))
    _write_plugin(container, bad, _unique("AutoNet"))
    (container / bad / "model.py").write_text("raise RuntimeError('import failed')\n")

    with caplog.at_level(logging.ERROR):
        assert mps.ensure_model_plugins_discovered() == [good]
    assert f"Model plugin '{bad}' not loaded" in caplog.text


def test_concurrent_callers_wait_for_the_first(fresh, monkeypatch):
    started, release = threading.Event(), threading.Event()
    calls = []

    def slow(config_path):
        calls.append(config_path)
        started.set()
        release.wait(5)
        return ["slow_plugin"]

    monkeypatch.setattr(mps, "_discover_and_load_configured", slow)
    results = {}
    first = threading.Thread(target=lambda: results.update(a=mps.ensure_model_plugins_discovered()))
    first.start()
    assert started.wait(5)
    second = threading.Thread(
        target=lambda: results.update(b=mps.ensure_model_plugins_discovered())
    )
    second.start()
    second.join(0.2)
    assert second.is_alive()  # waits for the first
    release.set()
    first.join(5)
    second.join(5)

    assert results == {"a": ["slow_plugin"], "b": ["slow_plugin"]} and calls == [None]


def test_reentrant_call_from_plugin_code_returns_nothing(fresh, monkeypatch, caplog):
    inner = []

    def discovering(config_path):
        inner.append(mps.ensure_model_plugins_discovered())
        return ["outer"]

    monkeypatch.setattr(mps, "_discover_and_load_configured", discovering)
    with caplog.at_level(logging.WARNING):
        assert mps.ensure_model_plugins_discovered() == ["outer"]
    assert inner == [[]]
    assert "requested again while it runs" in caplog.text


def test_hpo_search_space_filtering_runs_discovery(fresh, monkeypatch):
    calls = []
    monkeypatch.setattr(mps, "_discover_and_load_configured", lambda path: calls.append(path) or [])
    hpo_manager._ensure_model_plugins_discovered()
    hpo_manager._ensure_model_plugins_discovered()
    assert calls == [None]


def test_discover_and_load_plugins_without_paths_uses_the_configuration(fresh, configured):
    container = configured()
    plugin_name, model_name = _unique("auto_plugin_"), _unique("AutoNet")
    _write_plugin(container, plugin_name, model_name)

    assert mps.discover_and_load_plugins(paths=None) == {plugin_name: True}
    assert model_registry.registry.has_model(model_name)


def test_importing_the_models_package_runs_no_discovery():
    """Discovery happens when models are built, not at import time."""
    code = (
        "import milia_pipeline.models\n"
        "from milia_pipeline.models import model_plugin_system as m\n"
        "assert m._auto_discovery is None, m._auto_discovery\n"
        "assert m.get_plugin_loader()._plugins == {}\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
