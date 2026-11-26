"""Where a model's code comes from, and what must be installed to load it.

A checkpoint holds weights; its model class is code that must be importable where the checkpoint
is loaded. :func:`model_source` records, at training time (``run_info.model_source``, copied to
a package's ``lineage.model_source``), where each class came from:

* ``pyg``: a PyTorch Geometric model (``torch-geometric`` at the running version);
* ``milia``: a model plugin shipped in MILIA's own plugins folder (``milia-py``);
* ``distribution``: a model plugin from an installed distribution (its ``milia.model_plugins``
  entry point): the distribution and its version, as a requirement pin;
* ``directory``: a model plugin loaded from a directory: its path and a SHA-256 of its files,
  since nothing installable names it (the directory must travel with the package);
* ``ensemble``: MILIA's ensemble, with the source of each member.

``requirements`` lists the pins (``name==version``) that install the classes, MILIA's first: the
form of pip requirement specifiers (PEP 508), as in a model's ``requirements.txt``. MILIA is pinned
by its distribution name (``milia-py``, ``MILIA_DISTRIBUTION``) at the version of the code that
ran (``milia_pipeline.__version__``), not by whichever installed metadata provides the
``milia_pipeline`` package: an environment can hold stale metadata under another name, and a pin
by that name could install a different project.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any

__all__ = ["directory_sha256", "model_source"]

_PYG_DISTRIBUTION = "torch-geometric"


def _distribution_of(module: str) -> str | None:
    """The installed distribution providing the top-level package of ``module``."""
    distributions = importlib.metadata.packages_distributions().get(module.split(".")[0]) or []
    return distributions[0] if distributions else None


def _pin(distribution: str | None) -> str | None:
    if distribution is None:
        return None
    try:
        return f"{distribution}=={importlib.metadata.version(distribution)}"
    except importlib.metadata.PackageNotFoundError:
        return None


def directory_sha256(directory: str | Path) -> str:
    """SHA-256 over a directory's files: each relative path and its bytes, in sorted path order
    (``__pycache__`` and ``*.pyc`` left out: they are rebuilt where the code runs)."""
    root = Path(directory)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        digest.update(relative.as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _milia_pin() -> str:
    """MILIA's pin: its distribution name at the running code's version."""
    import milia_pipeline
    from milia_pipeline.models.model_plugin_system import MILIA_DISTRIBUTION

    return f"{MILIA_DISTRIBUTION}=={milia_pipeline.__version__}"


def _plugin_record(plugin_name: str) -> dict[str, Any]:
    from milia_pipeline.models._model_plugin_paths import builtin_model_plugins_directory
    from milia_pipeline.models.model_plugin_system import get_plugin_loader

    metadata = get_plugin_loader()._plugins.get(plugin_name)
    if metadata is None:
        return {"kind": "unknown", "plugin": plugin_name, "reason": "plugin not loaded"}
    record: dict[str, Any] = {"plugin": plugin_name, "plugin_version": metadata.version}
    if metadata.entry_point is not None:
        package = metadata.entry_point.split("=", 1)[1].strip()
        distribution = _distribution_of(package)
        record.update(
            kind="distribution",
            entry_point=metadata.entry_point,
            distribution=distribution,
            requirement=_pin(distribution),
        )
        return record
    path = Path(metadata.plugin_path).resolve() if metadata.plugin_path else None
    if path is not None and path.is_relative_to(builtin_model_plugins_directory()):
        record.update(kind="milia", requirement=_milia_pin())
        return record
    record.update(
        kind="directory",
        path=str(path) if path else None,
        sha256=directory_sha256(path) if path and path.is_dir() else None,
        requirement=None,
    )
    return record


def _class_source(name: str) -> dict[str, Any]:
    from milia_pipeline.models.registry.model_registry import ModelRegistry

    registration = ModelRegistry.get_instance().get_registration(name)
    if registration is None:
        return {"model": name, "kind": "unknown", "reason": "not in the model registry"}
    cls = registration.model_class
    record: dict[str, Any] = {"model": name, "class": f"{cls.__module__}.{cls.__qualname__}"}
    if registration.plugin_name:
        record.update(_plugin_record(registration.plugin_name))
    elif registration.is_builtin:
        record.update(kind="pyg", requirement=_pin(_PYG_DISTRIBUTION))
    else:
        distribution = _distribution_of(cls.__module__)
        record.update(
            kind="distribution", distribution=distribution, requirement=_pin(distribution)
        )
    return record


def model_source(model_name: str, hyperparameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Where the classes of model ``model_name`` come from (see the module docstring).

    Args:
        model_name: The registry name the model was created with (``"ensemble"`` for MILIA's
            ensemble, whose members are read from ``hyperparameters["ensemble_config"]``).
        hyperparameters: The model's hyperparameters, for an ensemble.

    Returns:
        ``{"kind", ..., "requirements"}``; an ensemble has ``members`` (one record each).
    """
    if model_name == "ensemble":
        members = ((hyperparameters or {}).get("ensemble_config") or {}).get("models") or []
        records = [_class_source(str(member.get("name"))) for member in members]
        source: dict[str, Any] = {"model": model_name, "kind": "ensemble", "members": records}
    else:
        records = [_class_source(model_name)]
        source = dict(records[0])
    milia = _milia_pin()
    others = {record.get("requirement") for record in records} - {None, milia}
    source["requirements"] = [milia, *sorted(others, key=str.lower)]
    return source
