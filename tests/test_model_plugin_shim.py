"""Tests for the deprecated ``milia_pipeline.models.plugins`` import path.

The model plugin loader lives at ``milia_pipeline.models.model_plugin_system``.
The old package stays importable until MILIA 2.0.0 as a pure alias:

- importing it emits a ``DeprecationWarning`` naming the removal release,
  attributed to the importing module;
- the old submodule path is the *same module object* as the new module, so
  attribute access, assignment and ``patch.object`` through either path reach
  one module;
- the package re-binds the public names to the same objects.

A static check also guarantees that no first-party module imports the old path.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import milia_pipeline
import milia_pipeline.models
from milia_pipeline import exceptions

OLD_PACKAGE = "milia_pipeline.models.plugins"
OLD_SUBMODULE = f"{OLD_PACKAGE}.model_plugin_system"
NEW_MODULE = "milia_pipeline.models.model_plugin_system"

new = importlib.import_module(NEW_MODULE)

_METADATA_NAMES = {"__version__", "__author__"}
# Added after the move; the deprecated package keeps the legacy names only.
_NEW_ONLY_NAMES = {"ensure_model_plugins_discovered"}


@pytest.fixture
def fresh_old_package():
    """Import the old package afresh, then restore the import state exactly."""
    saved = {name: sys.modules.pop(name, None) for name in (OLD_PACKAGE, OLD_SUBMODULE)}
    parent = milia_pipeline.models
    saved_attr = parent.__dict__.get("plugins")
    yield
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    if saved_attr is None:
        parent.__dict__.pop("plugins", None)
    else:
        parent.plugins = saved_attr


def _import_old_package():
    with pytest.warns(DeprecationWarning, match=r"removed in MILIA 2\.0\.0"):
        return importlib.import_module(OLD_PACKAGE)


def test_old_package_import_warns_and_names_the_replacement(fresh_old_package):
    # A plain import statement, as user code writes it: the import system's own
    # frames are skipped, so the warning is attributed to the importing module.
    with pytest.warns(DeprecationWarning) as record:
        import milia_pipeline.models.plugins  # noqa: F401

    messages = [str(w.message) for w in record if issubclass(w.category, DeprecationWarning)]
    assert any("removed in MILIA 2.0.0" in msg and NEW_MODULE in msg for msg in messages), messages
    assert any(Path(w.filename).resolve() == Path(__file__).resolve() for w in record)


def test_old_submodule_is_the_new_module_object(fresh_old_package):
    old = _import_old_package()

    assert importlib.import_module(OLD_SUBMODULE) is new
    assert sys.modules[OLD_SUBMODULE] is sys.modules[NEW_MODULE]
    assert old.model_plugin_system is new


def test_old_package_exposes_every_legacy_name_as_the_same_object(fresh_old_package):
    old = _import_old_package()

    assert len(old.__all__) == 23
    assert set(old.__all__) - _METADATA_NAMES == set(new.__all__) - _NEW_ONLY_NAMES
    assert len(new.__all__) == 21 + len(_NEW_ONLY_NAMES)
    for name in old.__all__:
        assert hasattr(old, name), name
    for name in set(new.__all__) - _NEW_ONLY_NAMES:
        assert getattr(old, name) is getattr(new, name), name


def test_patching_through_old_submodule_reaches_the_new_functions(fresh_old_package):
    _import_old_package()
    old_sub = importlib.import_module(OLD_SUBMODULE)
    declaration = new.ModelDeclaration(
        name="ShimProbe",
        class_name="ShimProbe",
        module_path="unused",
        category="general",
        description="",
        supported_tasks=["graph_regression"],
        hyperparameters={},
        plugin_name="shim_probe",
    )

    with (
        patch.object(old_sub, "MODELS_AVAILABLE", False),
        pytest.raises(exceptions.PluginError, match="ModelRegistry not available"),
    ):
        new.ModelPluginLoader()._register_plugin_model(declaration, None)


def test_one_loader_singleton_serves_both_paths(fresh_old_package):
    old = _import_old_package()

    assert old.get_plugin_loader() is new.get_plugin_loader()
    assert old.get_plugin_loader_instance() is new.get_plugin_loader()


def _resolve_import_from(node: ast.ImportFrom, module_name: str, is_package: bool) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = module_name.split(".")
    if not is_package:
        package_parts = package_parts[:-1]
    base = package_parts[: len(package_parts) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def test_no_first_party_module_imports_the_deprecated_path():
    """Static gate: only the shim itself may refer to the old package."""
    package_root = Path(milia_pipeline.__file__).resolve().parent
    shim = package_root / "models" / "plugins" / "__init__.py"
    offenders = []

    for path in sorted(package_root.rglob("*.py")):
        if path == shim:
            continue
        relative = path.relative_to(package_root.parent).with_suffix("")
        is_package = relative.name == "__init__"
        module_name = ".".join(relative.parts[:-1] if is_package else relative.parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                resolved = _resolve_import_from(node, module_name, is_package)
                targets = [resolved] + [f"{resolved}.{alias.name}" for alias in node.names]
            else:
                continue
            for target in targets:
                if target == OLD_PACKAGE or target.startswith(OLD_PACKAGE + "."):
                    offenders.append(f"{path.relative_to(package_root.parent)}:{node.lineno}")

    assert offenders == []
