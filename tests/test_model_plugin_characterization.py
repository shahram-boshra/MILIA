"""Characterization tests for the model plugin loader.

These tests pin what the loader does *today* against the real ``ModelRegistry``,
``milia_pipeline.exceptions`` and ``DynamicModelMetadata`` (no mocks of
``milia_pipeline.*``), so that relocating the loader can be shown to preserve
behaviour, and so that each later fix changes a pinned result deliberately.

Tests named ``test_current_*`` pin a known defect: each one is expected to be
flipped, on purpose, by the change that fixes it. Every other test pins
behaviour that is intended to stay.

The loader is imported through ``LOADER_MODULE`` only, so a relocation changes
one line here.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import uuid
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from milia_pipeline import exceptions
from milia_pipeline.models.registry import model_registry

LOADER_MODULE = "milia_pipeline.models.plugins.model_plugin_system"

mps = importlib.import_module(LOADER_MODULE)


# =============================================================================
# Helpers
# =============================================================================

_MODEL_SOURCE = """
import torch
from torch import nn


class {class_name}(nn.Module):
    \"\"\"Minimal regressor used by the characterization tests.\"\"\"

    def __init__(self, in_channels: int = 4, hidden_channels: int = 8, out_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        return self.lin2(torch.relu(self.lin1(x)))
"""


def _valid_model_entry(model_name: str, class_name: str, module_path: str = "model") -> dict:
    """A legacy-format model entry that the current loader registers successfully."""
    return {
        "name": model_name,
        "class_name": class_name,
        "module_path": module_path,
        "category": "basic_gnn",
        "description": "characterization model",
        "supported_tasks": ["graph_regression"],
        "hyperparameters": {
            "hidden_channels": {"type": "integer", "default": 8, "min": 1},
        },
        "min_pyg_version": "2.6.0",
    }


def _write_plugin(
    root: Path,
    dir_name: str,
    plugin_name: str,
    models: list[dict],
    class_names: list[str] | None = None,
    module_file: str = "model.py",
    extra: dict | None = None,
) -> Path:
    """Create ``root/dir_name/{plugin.yaml, model.py}`` and return the plugin directory."""
    plugin_dir = root / dir_name
    plugin_dir.mkdir(parents=True)
    manifest = {
        "plugin_name": plugin_name,
        "version": "1.0.0",
        "author": "tests",
        "description": "characterization plugin",
        "license": "MIT",
        "models": models,
    }
    if extra:
        manifest.update(extra)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    source = "".join(
        textwrap.dedent(_MODEL_SOURCE).format(class_name=name)
        for name in (class_names or [m["class_name"] for m in models])
    )
    (plugin_dir / module_file).write_text(source)
    return plugin_dir


def _unique(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def _cause_chain(exc: BaseException) -> list[BaseException]:
    chain = []
    current: BaseException | None = exc
    while current is not None:
        chain.append(current)
        current = current.__cause__
    return chain


# =============================================================================
# Fixtures
# =============================================================================


# The current loader imports plugin modules by bare name, so tests must own that name.
_BARE_PLUGIN_MODULE = "model"


@pytest.fixture
def import_state(tmp_path):
    """Restore ``sys.path``; drop modules imported from ``tmp_path``; isolate the bare name."""
    saved_path = list(sys.path)
    saved_bare = sys.modules.pop(_BARE_PLUGIN_MODULE, None)
    yield
    sys.path[:] = saved_path
    root = str(tmp_path)
    for name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None) or ""
        if module_file.startswith(root):
            del sys.modules[name]
    if saved_bare is not None:
        sys.modules[_BARE_PLUGIN_MODULE] = saved_bare


@pytest.fixture
def loader(import_state):
    """A fresh loader singleton; the previous singleton is restored afterwards."""
    saved_instance = mps.ModelPluginLoader._instance
    saved_module_loader = mps._plugin_loader
    mps.ModelPluginLoader._instance = None
    mps._plugin_loader = None
    try:
        yield mps.ModelPluginLoader()
    finally:
        mps.ModelPluginLoader._instance = saved_instance
        mps._plugin_loader = saved_module_loader


@pytest.fixture
def registry():
    """The real registry singleton; models registered by a test are removed afterwards."""
    reg = model_registry.registry
    before = set(reg.list_available_models())
    yield reg
    for name in set(reg.list_available_models()) - before:
        reg.unregister_model(name)


# =============================================================================
# Behaviour intended to stay
# =============================================================================


def test_valid_legacy_manifest_discovers_loads_and_registers(tmp_path, loader, registry):
    """A complete legacy manifest goes through discover -> load -> register."""
    plugin_name = _unique("charplugin_")
    model_name = _unique("CharNet")
    _write_plugin(tmp_path, "p", plugin_name, [_valid_model_entry(model_name, "CharNet")])

    assert loader.discover_plugins([tmp_path]) == [plugin_name]
    assert loader.load_plugin(plugin_name) is True

    assert registry.has_model(model_name)
    assert registry.get_model(model_name).__name__ == "CharNet"
    assert registry.list_plugin_models()[plugin_name] == [model_name]
    assert loader.list_plugins(loaded_only=True) == [plugin_name]


def test_plugin_provenance_lives_in_registry_not_metadata(tmp_path, loader, registry):
    """Plugin identity is recorded by the registry; the metadata carries no plugin fields."""
    plugin_name = _unique("charplugin_")
    model_name = _unique("CharNet")
    _write_plugin(tmp_path, "p", plugin_name, [_valid_model_entry(model_name, "CharNet")])
    loader.discover_plugins([tmp_path])
    loader.load_plugin(plugin_name)

    dumped = registry.get_metadata(model_name).model_dump()
    for dropped in ("is_plugin", "plugin_name", "supports_batch", "reference_paper"):
        assert dropped not in dumped
    assert registry.list_plugin_models()[plugin_name] == [model_name]


# =============================================================================
# Current defects (each flipped deliberately by the change that fixes it)
# =============================================================================


def test_current_default_category_fails_registration(tmp_path, loader, registry):
    """Pins current behaviour; changed deliberately later. Default category 'custom' is invalid."""
    plugin_name = _unique("charplugin_")
    model_name = _unique("CharNet")
    entry = _valid_model_entry(model_name, "CharNet")
    del entry["category"]
    _write_plugin(tmp_path, "p", plugin_name, [entry])
    loader.discover_plugins([tmp_path])

    with pytest.raises(exceptions.PluginError) as caught:
        loader.load_plugin(plugin_name)

    errors = [e for e in _cause_chain(caught.value) if isinstance(e, ValidationError)]
    assert errors and {err["loc"] for err in errors[0].errors()} == {("category",)}
    assert not registry.has_model(model_name)


def test_current_missing_min_pyg_version_fails_registration(tmp_path, loader, registry):
    """Pins current behaviour; changed deliberately later. ``min_pyg_version=None`` is rejected."""
    plugin_name = _unique("charplugin_")
    model_name = _unique("CharNet")
    entry = _valid_model_entry(model_name, "CharNet")
    del entry["min_pyg_version"]
    _write_plugin(tmp_path, "p", plugin_name, [entry])
    loader.discover_plugins([tmp_path])

    with pytest.raises(exceptions.PluginError) as caught:
        loader.load_plugin(plugin_name)

    errors = [e for e in _cause_chain(caught.value) if isinstance(e, ValidationError)]
    assert errors and {err["loc"] for err in errors[0].errors()} == {("min_pyg_version",)}
    assert not registry.has_model(model_name)


def test_current_same_module_name_collides_across_plugins(tmp_path, loader, registry):
    """Pins current behaviour; changed deliberately later.

    Two plugins that both use ``module_path: model`` share one ``sys.modules['model']``
    entry, so the second plugin resolves to the first plugin's module.
    """
    first, second = _unique("charplugin_"), _unique("charplugin_")
    first_model, second_model = _unique("AlphaNet"), _unique("BetaNet")
    _write_plugin(tmp_path, "a", first, [_valid_model_entry(first_model, "AlphaNet")])
    _write_plugin(tmp_path, "b", second, [_valid_model_entry(second_model, "BetaNet")])
    loader.discover_plugins([tmp_path])

    assert loader.load_plugin(first) is True
    with pytest.raises(exceptions.PluginError, match="BetaNet"):
        loader.load_plugin(second)

    assert registry.has_model(first_model)
    assert not registry.has_model(second_model)


def test_current_load_inserts_plugin_dir_into_sys_path(tmp_path, loader, registry):
    """Pins current behaviour; changed deliberately later. Loading mutates ``sys.path``."""
    plugin_name = _unique("charplugin_")
    plugin_dir = _write_plugin(
        tmp_path, "p", plugin_name, [_valid_model_entry(_unique("CharNet"), "CharNet")]
    )
    loader.discover_plugins([tmp_path])
    loader.load_plugin(plugin_name)

    assert sys.path[0] == str(plugin_dir)


def test_current_module_path_with_py_suffix_fails(tmp_path, loader, registry):
    """Pins current behaviour; changed deliberately later. ``module_path: model.py`` cannot import."""
    plugin_name = _unique("charplugin_")
    model_name = _unique("CharNet")
    entry = _valid_model_entry(model_name, "CharNet", module_path="model.py")
    _write_plugin(tmp_path, "p", plugin_name, [entry])
    loader.discover_plugins([tmp_path])

    with pytest.raises(exceptions.PluginError) as caught:
        loader.load_plugin(plugin_name)

    assert any(isinstance(e, ModuleNotFoundError) for e in _cause_chain(caught.value))
    assert not registry.has_model(model_name)


def test_current_invalid_plugin_still_discovered_below_strict(tmp_path, loader):
    """Pins current behaviour; changed deliberately later.

    A declaration error (no supported tasks) is only a warning at ``standard``,
    so the plugin is still discovered; ``strict`` skips it.
    """
    plugin_name = _unique("charplugin_")
    entry = _valid_model_entry(_unique("CharNet"), "CharNet")
    entry["supported_tasks"] = []
    _write_plugin(tmp_path, "p", plugin_name, [entry])

    assert loader.discover_plugins([tmp_path], validation_level="standard") == [plugin_name]
    assert any(
        "declares no supported tasks" in msg
        for msg in loader._plugins[plugin_name].validation_errors
    )

    loader._plugins.clear()
    assert loader.discover_plugins([tmp_path], validation_level="strict") == []


def test_current_duplicate_plugin_name_last_wins_silently(tmp_path, loader):
    """Pins current behaviour; changed deliberately later. A duplicate name overwrites silently."""
    plugin_name = _unique("charplugin_")
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    _write_plugin(first_root, "p", plugin_name, [_valid_model_entry(_unique("CharNet"), "CharNet")])
    second_dir = _write_plugin(
        second_root, "p", plugin_name, [_valid_model_entry(_unique("CharNet"), "CharNet")]
    )

    assert loader.discover_plugins([first_root, second_root]) == [plugin_name, plugin_name]
    assert loader._plugins[plugin_name].plugin_path == second_dir


def test_current_loader_lock_is_not_reentrant(loader):
    """Pins current behaviour; changed deliberately later.

    The lock held while plugin code runs is a plain ``Lock``: the same thread cannot
    re-acquire it, so a plugin calling the loader API at import time would deadlock.
    """
    assert loader._lock.acquire(timeout=1)
    try:
        assert loader._lock.acquire(blocking=False) is False
    finally:
        loader._lock.release()


def test_current_pep508_dependency_reported_missing(tmp_path, loader):
    """Pins current behaviour; changed deliberately later.

    Dependencies are checked by importing the string, so an installed
    ``torch>=2.0`` is reported missing.
    """
    plugin_name = _unique("charplugin_")
    _write_plugin(
        tmp_path,
        "p",
        plugin_name,
        [_valid_model_entry(_unique("CharNet"), "CharNet")],
        extra={"dependencies": ["torch>=2.0"]},
    )
    loader.discover_plugins([tmp_path])

    assert (
        "Missing required dependency: torch>=2.0" in loader._plugins[plugin_name].validation_errors
    )


def test_current_factory_cannot_build_registered_plugin_model(tmp_path, loader, registry):
    """Pins current behaviour; changed deliberately later.

    ``ModelFactory`` reads metadata from the PyG introspector, which knows only PyG
    models, so a correctly registered plugin model cannot be built.
    """
    from milia_pipeline.models.factory.model_factory import ModelFactory

    plugin_name = _unique("charplugin_")
    model_name = _unique("CharNet")
    _write_plugin(tmp_path, "p", plugin_name, [_valid_model_entry(model_name, "CharNet")])
    loader.discover_plugins([tmp_path])
    loader.load_plugin(plugin_name)
    assert registry.has_model(model_name)

    with pytest.raises(AttributeError, match="supported_tasks"):
        ModelFactory().create_model(
            name=model_name,
            hyperparameters={"in_channels": 4, "hidden_channels": 8, "out_channels": 1},
            task_type="graph_regression",
        )
