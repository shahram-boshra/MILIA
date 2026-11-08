"""Blocked transformation plugins: their code is never imported and nothing is registered."""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from unittest import mock

import pytest

import main
from milia_pipeline.transformations import plugin_system as ps
from milia_pipeline.transformations.plugin_system import PluginRegistry

MANIFEST = """\
plugin_name: "{plugin}"
version: "1.0.0"
author: "tests"
plugin_type: "user_experimental"
description: "Blocking test plugin"
milia_version: ">=1.2.2"
pyg_version: ">=2.6.0"
python_version: ">=3.10"
license: "MIT"
dependencies: []
transforms:
  - name: "{transform}"
    class_name: "{transform}"
    module_path: "transforms"
    category: "structural"
    description: "Returns the graph unchanged"
"""

MODULE = """\
from pathlib import Path

from milia_pipeline.transformations.custom_transforms import CustomTransformBase, TransformMetadata

Path({sentinel!r}).write_text("imported")


class {transform}(CustomTransformBase):
    def transform(self, data):
        return data

    @classmethod
    def get_metadata(cls):
        return TransformMetadata(
            name="{transform}", version="1.0.0", author="tests", category="structural",
            description="Returns the graph unchanged",
        )
"""


@pytest.fixture
def registry():
    """The plugin registry with its state cleared, restored afterwards."""
    instance = PluginRegistry()
    saved = {
        name: getattr(instance, name).copy()
        for name in ("_plugins", "_enabled_plugins", "_disabled_plugins", "_blocked_plugins")
    }
    saved_paths = list(instance._plugin_paths)
    saved_sys_path = list(sys.path)
    saved_modules = set(sys.modules)
    for name in saved:
        getattr(instance, name).clear()
    instance._plugin_paths.clear()
    try:
        yield instance
    finally:
        ps._import_graph_transforms()
        for metadata in instance._plugins.values():
            for transform_name in metadata.registered_transforms:
                ps.TransformRegistry.unregister(transform_name)
        for name, value in saved.items():
            getattr(instance, name).clear()
            getattr(instance, name).update(value)
        instance._plugin_paths[:] = saved_paths
        sys.path[:] = saved_sys_path
        for name in set(sys.modules) - saved_modules:
            del sys.modules[name]


@pytest.fixture
def plugin(tmp_path):
    """A plugin directory whose module records that it was imported."""
    suffix = uuid.uuid4().hex[:8]
    names = {
        "plugin": f"blocking_{suffix}",
        "transform": f"BlockingProbe{suffix}",
        "sentinel": str(tmp_path / "imported"),
    }
    plugin_dir = tmp_path / "plugins" / names["plugin"]
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(MANIFEST.format(**names))
    (plugin_dir / "transforms.py").write_text(MODULE.format(**names))
    names["root"] = tmp_path / "plugins"
    return names


def _registered(transform_name: str) -> bool:
    ps._import_graph_transforms()
    return ps.TransformRegistry.is_custom_transform(transform_name)


def test_unblocked_plugin_is_imported_and_registered(registry, plugin):
    assert PluginRegistry.discover_plugins([plugin["root"]]) == [plugin["plugin"]]
    assert Path(plugin["sentinel"]).exists()
    assert _registered(plugin["transform"])


def test_blocked_plugin_code_is_never_imported(registry, plugin, caplog):
    PluginRegistry.set_blocked(plugin["plugin"])

    with caplog.at_level(logging.INFO, logger=ps.logger.name):
        assert PluginRegistry.discover_plugins([plugin["root"]]) == []

    assert PluginRegistry.is_blocked(plugin["plugin"])
    assert not Path(plugin["sentinel"]).exists()
    assert plugin["plugin"] not in PluginRegistry.list_plugins()
    assert not _registered(plugin["transform"])
    assert f"Skipping blocked plugin '{plugin['plugin']}'" in caplog.text


def test_blocking_a_registered_plugin_unregisters_it(registry, plugin):
    PluginRegistry.discover_plugins([plugin["root"]])
    assert _registered(plugin["transform"])

    PluginRegistry.set_blocked(plugin["plugin"])

    assert plugin["plugin"] not in PluginRegistry.list_plugins()
    assert not _registered(plugin["transform"])
    assert PluginRegistry.discover_plugins([plugin["root"]]) == []
    assert not _registered(plugin["transform"])


@pytest.mark.parametrize("auto_discover", [True, False])
def test_configured_disabled_plugin_never_runs(registry, plugin, auto_discover):
    """``plugins.disabled_plugins`` blocks the plugin before discovery, whenever it runs."""
    config = {
        "plugins": {
            "enabled": True,
            "plugin_paths": [str(plugin["root"])],
            "auto_discover": auto_discover,
            "disabled_plugins": [plugin["plugin"]],
        }
    }
    with mock.patch("milia_pipeline.config.config_loader.load_config", return_value=config):
        main._discover_and_register_plugins(logging.getLogger(__name__))

    assert PluginRegistry.is_blocked(plugin["plugin"])
    # A later discovery of the same path (for example the CLI's) skips it as well.
    assert PluginRegistry.discover_plugins([plugin["root"]]) == []
    assert not Path(plugin["sentinel"]).exists()
    assert not _registered(plugin["transform"])
