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
import logging
import pickle
import sys
import textwrap
import threading
import types
import uuid
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from milia_pipeline import exceptions
from milia_pipeline.models.registry import model_registry

LOADER_MODULE = "milia_pipeline.models.model_plugin_system"

mps = importlib.import_module(LOADER_MODULE)
PLUGIN_NAMESPACE = mps.PLUGIN_MODULE_NAMESPACE


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


# The loader imports plugin modules under their plugin's package, not by bare name.
# Setting the bare name aside guards against a regression to bare-name imports.
_BARE_PLUGIN_MODULE = "model"


@pytest.fixture
def import_state(tmp_path):
    """Restore ``sys.path``; drop modules and plugin packages from ``tmp_path``."""
    saved_path = list(sys.path)
    saved_bare = sys.modules.pop(_BARE_PLUGIN_MODULE, None)
    yield
    sys.path[:] = saved_path
    root = str(tmp_path)
    for name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None) or ""
        module_dirs = list(getattr(module, "__path__", None) or [])
        if module_file.startswith(root) or any(d.startswith(root) for d in module_dirs):
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


def test_same_module_name_resolves_per_plugin(tmp_path, loader, registry):
    """Two plugins that both use ``module_path: model`` each import their own module."""
    first, second = _unique("charplugin_"), _unique("charplugin_")
    first_model, second_model = _unique("AlphaNet"), _unique("BetaNet")
    _write_plugin(tmp_path, "a", first, [_valid_model_entry(first_model, "AlphaNet")])
    _write_plugin(tmp_path, "b", second, [_valid_model_entry(second_model, "BetaNet")])
    loader.discover_plugins([tmp_path])

    assert loader.load_plugin(first) is True
    assert loader.load_plugin(second) is True

    first_class = registry.get_registration(first_model).model_class
    second_class = registry.get_registration(second_model).model_class
    assert first_class.__name__ == "AlphaNet"
    assert second_class.__name__ == "BetaNet"
    assert first_class.__module__ == f"{PLUGIN_NAMESPACE}.{first}.model"
    assert second_class.__module__ == f"{PLUGIN_NAMESPACE}.{second}.model"


def test_failed_module_execution_leaves_no_module_behind(tmp_path, loader, registry):
    """A plugin module that raises while executing is not left in ``sys.modules``."""
    plugin_name = _unique("charplugin_")
    model_name = _unique("CharNet")
    plugin_dir = _write_plugin(
        tmp_path, "p", plugin_name, [_valid_model_entry(model_name, "CharNet")]
    )
    (plugin_dir / "model.py").write_text("raise RuntimeError('broken plugin module')\n")
    loader.discover_plugins([tmp_path])

    with pytest.raises(exceptions.PluginError, match="broken plugin module"):
        loader.load_plugin(plugin_name)

    assert f"{PLUGIN_NAMESPACE}.{plugin_name}.model" not in sys.modules
    assert not registry.has_model(model_name)


# torch 2.4 unpickles tensor storages with torch.load(weights_only=False) and warns about it;
# the warning concerns torch's tensor format, not the plugin class lookup tested here.
@pytest.mark.filterwarnings(
    "ignore:You are using `torch.load` with `weights_only=False`:FutureWarning"
)
def test_plugin_model_instance_pickles_in_process(tmp_path, loader, registry):
    """Plugin classes have an importable qualified name, so instances pickle."""
    plugin_name = _unique("charplugin_")
    model_name = _unique("CharNet")
    _write_plugin(tmp_path, "p", plugin_name, [_valid_model_entry(model_name, "CharNet")])
    loader.discover_plugins([tmp_path])
    loader.load_plugin(plugin_name)
    model_class = registry.get_registration(model_name).model_class
    model = model_class(in_channels=4)

    restored = pickle.loads(pickle.dumps(model))

    assert pickle.loads(pickle.dumps(model_class)) is model_class
    assert type(restored) is model_class
    assert restored.lin1.in_features == 4


def test_plugin_modules_can_import_each_other_relatively(tmp_path, loader, registry):
    """A plugin's modules import sibling modules with relative imports."""
    plugin_name = _unique("charplugin_")
    model_name = _unique("CharNet")
    plugin_dir = _write_plugin(
        tmp_path, "p", plugin_name, [_valid_model_entry(model_name, "CharNet")]
    )
    (plugin_dir / "helpers.py").write_text("HIDDEN = 5\n")
    model_file = plugin_dir / "model.py"
    model_file.write_text("from .helpers import HIDDEN\n" + model_file.read_text())
    loader.discover_plugins([tmp_path])

    assert loader.load_plugin(plugin_name) is True
    module = sys.modules[f"{PLUGIN_NAMESPACE}.{plugin_name}.model"]
    assert module.HIDDEN == 5


def test_plugin_name_must_be_a_module_name(tmp_path, loader, registry):
    """A plugin name that is not a Python identifier cannot name the plugin's package."""
    plugin_name = _unique("char-plugin-")
    model_name = _unique("CharNet")
    _write_plugin(tmp_path, "p", plugin_name, [_valid_model_entry(model_name, "CharNet")])
    loader.discover_plugins([tmp_path])

    with pytest.raises(exceptions.PluginError, match="cannot be used as a Python module name"):
        loader.load_plugin(plugin_name)

    assert not registry.has_model(model_name)


def test_load_leaves_sys_path_unchanged(tmp_path, loader, registry):
    """Loading a plugin imports its modules without changing ``sys.path``."""
    plugin_name = _unique("charplugin_")
    _write_plugin(tmp_path, "p", plugin_name, [_valid_model_entry(_unique("CharNet"), "CharNet")])
    before = list(sys.path)
    loader.discover_plugins([tmp_path])
    loader.load_plugin(plugin_name)

    assert sys.path == before


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


def test_duplicate_plugin_name_is_rejected_and_first_kept(tmp_path, loader, caplog):
    """A second plugin with an already discovered name is rejected and logged."""
    plugin_name = _unique("charplugin_")
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    first_dir = _write_plugin(
        first_root, "p", plugin_name, [_valid_model_entry(_unique("CharNet"), "CharNet")]
    )
    second_dir = _write_plugin(
        second_root, "p", plugin_name, [_valid_model_entry(_unique("CharNet"), "CharNet")]
    )

    with caplog.at_level(logging.ERROR):
        assert loader.discover_plugins([first_root, second_root]) == [plugin_name]

    assert loader._plugins[plugin_name].plugin_path == first_dir
    assert f"Duplicate plugin name '{plugin_name}'" in caplog.text
    assert str(second_dir) in caplog.text


def test_rediscovering_the_same_directory_keeps_plugin_state(tmp_path, loader):
    """Discovering a directory again is not a duplicate and keeps the loaded state."""
    plugin_name = _unique("charplugin_")
    _write_plugin(tmp_path, "p", plugin_name, [_valid_model_entry(_unique("CharNet"), "CharNet")])
    loader.discover_plugins([tmp_path])
    loader.load_plugin(plugin_name)

    assert loader.discover_plugins([tmp_path]) == [plugin_name]
    assert loader._plugins[plugin_name].loaded is True


def _plugin_with_prelude(tmp_path, prelude: str) -> tuple[str, str]:
    """Write a plugin whose ``model.py`` runs ``prelude`` before the model class."""
    plugin_name = _unique("charplugin_")
    model_name = _unique("CharNet")
    plugin_dir = _write_plugin(
        tmp_path, "p", plugin_name, [_valid_model_entry(model_name, "CharNet")]
    )
    model_file = plugin_dir / "model.py"
    model_file.write_text(textwrap.dedent(prelude) + "\n" + model_file.read_text())
    return plugin_name, model_name


def _load_in_thread(loader, plugin_name: str, timeout: float = 5.0) -> dict:
    """Load a plugin in a worker thread; fail the test if it does not finish in time."""
    outcome: dict = {}

    def target():
        try:
            outcome["result"] = loader.load_plugin(plugin_name)
        except Exception as e:  # reported to the test thread
            outcome["error"] = e

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), f"load_plugin('{plugin_name}') did not finish in {timeout}s"
    return outcome


def test_plugin_code_runs_with_the_loader_lock_free(tmp_path, loader):
    """The loader lock guards state only; it is not held while plugin code runs."""
    plugin_name, _ = _plugin_with_prelude(
        tmp_path,
        f"""
        from {LOADER_MODULE} import get_plugin_loader
        LOCK_HELD = get_plugin_loader()._lock.locked()
        """,
    )
    loader.discover_plugins([tmp_path])

    assert _load_in_thread(loader, plugin_name) == {"result": True}
    assert sys.modules[f"{PLUGIN_NAMESPACE}.{plugin_name}.model"].LOCK_HELD is False


def test_plugin_calling_the_loader_at_import_does_not_deadlock(tmp_path, loader):
    """A plugin module may call the loader API while it is being imported."""
    plugin_name, _ = _plugin_with_prelude(
        tmp_path,
        f"""
        from {LOADER_MODULE} import get_plugin_loader
        PLUGINS_SEEN = get_plugin_loader().list_plugins()
        """,
    )
    loader.discover_plugins([tmp_path])

    assert _load_in_thread(loader, plugin_name) == {"result": True}
    module = sys.modules[f"{PLUGIN_NAMESPACE}.{plugin_name}.model"]
    assert plugin_name in module.PLUGINS_SEEN


def test_plugin_loading_itself_at_import_fails_instead_of_deadlocking(tmp_path, loader, registry):
    """A plugin whose code loads the same plugin gets an error, not a hang."""
    plugin_name, model_name = _plugin_with_prelude(
        tmp_path,
        f"""
        from {LOADER_MODULE} import get_plugin_loader
        get_plugin_loader().load_plugin(__name__.split(".")[-2])
        """,
    )
    loader.discover_plugins([tmp_path])

    outcome = _load_in_thread(loader, plugin_name)

    assert isinstance(outcome["error"], exceptions.PluginError)
    assert "already being loaded by this thread" in str(outcome["error"])
    assert loader._plugins[plugin_name].loaded is False
    assert not registry.has_model(model_name)


@pytest.fixture
def import_counter():
    """A shared counter that plugin modules increment when they are executed."""
    counter = types.SimpleNamespace(count=0)
    sys.modules["_milia_test_import_counter"] = counter
    yield counter
    del sys.modules["_milia_test_import_counter"]


_COUNTING_PRELUDE = """
import sys
import time
sys.modules["_milia_test_import_counter"].count += 1
time.sleep(0.2)  # keep the load in progress while the other threads arrive
"""


def _load_from_threads(loader, plugin_name: str, threads: int = 8) -> list[dict]:
    barrier = threading.Barrier(threads)
    outcomes: list[dict] = [{} for _ in range(threads)]

    def target(index: int):
        barrier.wait()
        try:
            outcomes[index]["result"] = loader.load_plugin(plugin_name)
        except Exception as e:  # reported to the test thread
            outcomes[index]["error"] = e

    workers = [threading.Thread(target=target, args=(i,), daemon=True) for i in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)
    assert not any(worker.is_alive() for worker in workers), "a load_plugin call did not finish"
    return outcomes


def test_concurrent_loads_import_the_plugin_once(tmp_path, loader, registry, import_counter):
    """Eight threads loading one plugin import its module once and all succeed."""
    plugin_name, model_name = _plugin_with_prelude(tmp_path, _COUNTING_PRELUDE)
    loader.discover_plugins([tmp_path])

    outcomes = _load_from_threads(loader, plugin_name)

    assert outcomes == [{"result": True}] * 8
    assert import_counter.count == 1
    assert registry.has_model(model_name)
    assert loader._plugins[plugin_name].loaded is True


def test_concurrent_loads_all_report_a_failed_load(tmp_path, loader, registry, import_counter):
    """Callers waiting on a load that fails get the failure; the plugin stays unloaded.

    A caller that arrives after a failed attempt has finished starts a new attempt, so
    the number of imports is not fixed; every caller still reports the failure.
    """
    plugin_name, model_name = _plugin_with_prelude(
        tmp_path, _COUNTING_PRELUDE + "raise RuntimeError('broken plugin module')\n"
    )
    loader.discover_plugins([tmp_path])

    outcomes = _load_from_threads(loader, plugin_name)

    assert all(isinstance(o.get("error"), exceptions.PluginError) for o in outcomes)
    assert all("broken plugin module" in str(o["error"]) for o in outcomes)
    assert 1 <= import_counter.count < 8  # waiting callers did not import it again
    assert loader._plugins[plugin_name].loaded is False
    assert not registry.has_model(model_name)


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


def test_factory_builds_registered_plugin_model(tmp_path, loader, registry):
    """``ModelFactory`` reads metadata from the registry, so a registered plugin model builds and runs."""
    import torch
    from torch_geometric.data import Batch, Data

    from milia_pipeline.models.factory.model_factory import ModelFactory

    plugin_name = _unique("charplugin_")
    model_name = _unique("CharNet")
    _write_plugin(tmp_path, "p", plugin_name, [_valid_model_entry(model_name, "CharNet")])
    loader.discover_plugins([tmp_path])
    loader.load_plugin(plugin_name)
    assert registry.has_model(model_name)

    model = ModelFactory().create_model(
        name=model_name,
        hyperparameters={"in_channels": 4, "hidden_channels": 16, "out_channels": 1},
        task_type="graph_regression",
    )

    assert model.model.lin1.out_features == 16  # a declared hyperparameter reaches the class
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    batch = Batch.from_data_list(
        [Data(x=torch.randn(3, 4), edge_index=edge_index) for _ in range(2)]
    )
    out = model(batch.x, batch.edge_index, batch=batch.batch)
    assert out.shape == (2, 1)


def test_current_factory_drops_undeclared_plugin_model_channels(tmp_path, loader, registry):
    """Pins current behaviour; changed deliberately later.

    The loader's metadata has no constructor parameters, so the factory passes only the
    manifest's declared hyperparameters; ``in_channels`` and ``out_channels`` fall back to
    the class defaults.
    """
    from milia_pipeline.models.factory.model_factory import ModelFactory

    plugin_name = _unique("charplugin_")
    model_name = _unique("CharNet")
    _write_plugin(tmp_path, "p", plugin_name, [_valid_model_entry(model_name, "CharNet")])
    loader.discover_plugins([tmp_path])
    loader.load_plugin(plugin_name)

    model = ModelFactory().create_model(
        name=model_name,
        hyperparameters={"in_channels": 6, "hidden_channels": 8, "out_channels": 3},
        task_type="graph_regression",
    )

    assert registry.get_metadata(model_name).parameters == {}
    assert model.model.lin1.in_features == 4  # class default, not 6
    assert model.model.out_channels == 1  # class default, not 3


def test_current_discover_and_load_plugins_without_paths_fails(loader):
    """Pins current behaviour; changed deliberately later.

    ``discover_and_load_plugins`` documents ``paths=None`` as "from config", but it
    passes ``None`` to ``ModelPluginLoader.discover_plugins``, which iterates it.
    """
    with pytest.raises(TypeError):
        mps.discover_and_load_plugins(paths=None)
