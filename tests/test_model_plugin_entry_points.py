"""Model plugins from installed distributions: the ``milia.models`` entry-point group."""

from __future__ import annotations

import importlib.metadata
import logging
import sys
import uuid
from pathlib import Path

import pytest
import yaml

from milia_pipeline.exceptions import PluginModelError
from milia_pipeline.models import model_plugin_system as mps
from milia_pipeline.models.registry import model_registry
from milia_pipeline.models.utils import config_bridge
from milia_pipeline.models.utils.config_bridge import PluginsConfig

_MODEL_SOURCE = """
import torch
from torch import nn


class EntryNet(nn.Module):
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


def _manifest(plugin_name: str, model_name: str, module: str) -> dict:
    return {
        "schema_version": 2,
        "plugin_name": plugin_name,
        "version": "1.0.0",
        "author": "tests",
        "description": "entry-point plugin",
        "license": "MIT",
        "models": [
            {
                "name": model_name,
                "class": f"{module}:EntryNet",
                "category": "basic_gnn",
                "description": "entry-point model",
                "supported_tasks": ["graph_regression"],
            }
        ],
    }


@pytest.fixture
def site(tmp_path, monkeypatch):
    """An importable directory standing in for site-packages; its modules are dropped afterwards."""
    root = tmp_path / "site"
    root.mkdir()
    monkeypatch.syspath_prepend(str(root))
    yield root
    for name, module in list(sys.modules.items()):
        if (getattr(module, "__file__", None) or "").startswith(str(root)):
            del sys.modules[name]


def _install(site: Path, plugin_name: str, model_name: str, init: str = "") -> str:
    """Create an installed plugin package in ``site``; return its import name."""
    package = _unique("ep_pkg_")
    package_dir = site / package
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(init)
    (package_dir / "model.py").write_text(_MODEL_SOURCE)
    manifest = _manifest(plugin_name, model_name, f"{package}.model")
    (package_dir / "plugin.yaml").write_text(yaml.safe_dump(manifest))
    return package


@pytest.fixture
def entry_points(monkeypatch):
    """Replace the installed ``milia.models`` entry points; records each query."""
    installed: list[importlib.metadata.EntryPoint] = []
    queries: list[dict] = []

    def fake_entry_points(**select):
        queries.append(select)
        return [ep for ep in installed if all(getattr(ep, k) == v for k, v in select.items())]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)

    def add(name: str, value: str) -> None:
        installed.append(importlib.metadata.EntryPoint(name, value, mps.ENTRY_POINT_GROUP))

    add.queries = queries
    return add


@pytest.fixture
def loader():
    """A fresh loader singleton; the previous singleton is restored afterwards."""
    saved_instance = mps.ModelPluginLoader._instance
    saved_module_loader = mps._plugin_loader
    mps.ModelPluginLoader._instance = None
    mps._plugin_loader = None
    try:
        yield mps.get_plugin_loader()
    finally:
        mps.ModelPluginLoader._instance = saved_instance
        mps._plugin_loader = saved_module_loader


@pytest.fixture
def registry():
    """The real registry; models registered by a test are removed afterwards."""
    reg = model_registry.registry
    before = set(reg.list_available_models())
    yield reg
    for name in set(reg.list_available_models()) - before:
        reg.unregister_model(name)


def test_entry_point_plugin_is_discovered_loaded_and_registered(
    site, entry_points, loader, registry
):
    plugin_name, model_name = _unique("ep_plugin_"), _unique("EntryNet")
    package = _install(site, plugin_name, model_name)
    entry_points(plugin_name, package)

    assert loader.discover_entry_point_plugins() == [plugin_name]
    metadata = loader._plugins[plugin_name]
    assert metadata.entry_point == f"{plugin_name} = {package}"
    assert metadata.plugin_path.resolve() == (site / package).resolve()

    assert loader.load_plugin(plugin_name) is True
    model_class = registry.get_model(model_name)
    # Absolute class reference: imported by its own name, not under the private namespace.
    assert model_class.__module__ == f"{package}.model"
    assert entry_points.queries == [{"group": "milia.models"}]


def test_entry_point_name_must_match_plugin_name(site, entry_points, loader, caplog):
    package = _install(site, _unique("ep_plugin_"), _unique("EntryNet"))
    entry_points("other_name", package)

    with caplog.at_level(logging.ERROR):
        assert loader.discover_entry_point_plugins() == []
    assert "other_name" not in loader._plugins
    assert "name 'other_name' does not match the manifest's plugin_name" in caplog.text

    with pytest.raises(PluginModelError, match="does not match"):
        loader.discover_entry_point_plugins(validation_level="strict")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("{package}:EntryNet", "must name a package, not an object in it"),
        ("{package}.model", "is not a package"),
        ("{package}.empty", "has no plugin.yaml"),
    ],
    ids=["object", "module", "no-manifest"],
)
def test_entry_point_must_name_a_package_with_a_manifest(
    site, entry_points, loader, caplog, value, message
):
    plugin_name = _unique("ep_plugin_")
    package = _install(site, plugin_name, _unique("EntryNet"))
    (site / package / "empty").mkdir()
    (site / package / "empty" / "__init__.py").write_text("")
    entry_points(plugin_name, value.format(package=package))

    with caplog.at_level(logging.ERROR):
        assert loader.discover_entry_point_plugins() == []
    assert message in caplog.text


def test_failing_entry_point_does_not_stop_the_others(site, entry_points, loader, caplog):
    broken_name, good_name = _unique("ep_broken_"), _unique("ep_good_")
    broken = _install(site, broken_name, _unique("EntryNet"), init="raise RuntimeError('boom')\n")
    good = _install(site, good_name, _unique("EntryNet"))
    entry_points(broken_name, broken)
    entry_points(good_name, good)

    with caplog.at_level(logging.ERROR):
        assert loader.discover_entry_point_plugins() == [good_name]
    assert f"Failed to discover plugin from entry point '{broken_name} = {broken}': boom" in (
        caplog.text
    )


def test_rediscovering_an_entry_point_keeps_its_state(site, entry_points, loader, registry):
    plugin_name = _unique("ep_plugin_")
    entry_points(plugin_name, _install(site, plugin_name, _unique("EntryNet")))

    assert loader.discover_entry_point_plugins() == [plugin_name]
    assert loader.load_plugin(plugin_name) is True
    assert loader.discover_entry_point_plugins() == [plugin_name]
    assert loader._plugins[plugin_name].loaded is True


def test_directory_plugin_keeps_its_name_against_an_entry_point(
    site, tmp_path, entry_points, loader, monkeypatch, caplog
):
    """Configured discovery scans directories first; the entry-point duplicate is rejected."""
    plugin_name = _unique("ep_plugin_")
    container = tmp_path / "dir_plugins"
    (container / "p").mkdir(parents=True)
    (container / "p" / "model.py").write_text(_MODEL_SOURCE)
    manifest = _manifest(plugin_name, _unique("EntryNet"), "model")
    (container / "p" / "plugin.yaml").write_text(yaml.safe_dump(manifest))
    package = _install(site, plugin_name, _unique("EntryNet"))
    entry_points(plugin_name, package)
    monkeypatch.setattr(mps, "_configured_sources", lambda config_path=None: ([container], True))

    with caplog.at_level(logging.ERROR):
        assert mps.discover_plugins() == [plugin_name]
    assert loader._plugins[plugin_name].entry_point is None
    assert (
        f"Duplicate plugin name '{plugin_name}': already discovered at {container / 'p'}, "
        f"ignoring entry point '{plugin_name} = {package}'"
    ) in caplog.text


@pytest.mark.parametrize("flag", [True, False])
def test_discover_entry_points_flag(tmp_path, entry_points, loader, monkeypatch, flag):
    """``discover_entry_points: false`` means no entry-point query at all."""
    (tmp_path / "configs").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        config_bridge,
        "get_plugins_config",
        lambda config_path=None: PluginsConfig(include_builtin=False, discover_entry_points=flag),
    )

    assert mps.discover_plugins() == []
    assert entry_points.queries == ([{"group": "milia.models"}] if flag else [])


def test_explicit_paths_do_not_query_entry_points(tmp_path, entry_points, loader):
    assert mps.discover_plugins([tmp_path]) == []
    assert entry_points.queries == []
