"""
Self-describing deployment packages built from MILIA training checkpoints.

A bare ``state_dict`` cannot be served: rebuilding the network needs the model class and the
hyperparameters it was created with. A MILIA v2 checkpoint carries both
(``hyper_parameters``: model name, task type, the complete hyperparameters dict and
``model_info``), and ``ModelLoader``
(``milia_pipeline.models.post_training.inference.model_loader``) rebuilds the model from it.
A package is a directory holding:

* ``model.pt``: the checkpoint reduced to what inference needs (model weights, recreation
  metadata, data compatibility and version information), with no optimizer, scheduler or
  training history;
* ``deployment.json``: the package format, the MILIA version that wrote it, the model name and
  task type, and the SHA-256 of ``model.pt``.

:func:`load_package` checks the format and the digest, then rebuilds the model with
``ModelLoader``. Every read uses ``torch.load(weights_only=True)``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from milia_pipeline.exceptions import DeploymentError
from milia_pipeline.models.checkpoint_io import load_checkpoint_file

__all__ = [
    "PACKAGE_CHECKPOINT",
    "PACKAGE_FORMAT",
    "PACKAGE_MANIFEST",
    "is_package",
    "load_package",
    "package_checkpoint",
    "verify_package",
]

PACKAGE_FORMAT = "milia-deployment/1"
PACKAGE_MANIFEST = "deployment.json"
PACKAGE_CHECKPOINT = "model.pt"

# Checkpoint entries inference needs; training state (optimizer, scheduler, history) is dropped.
_INFERENCE_KEYS = ("model_state_dict", "hyper_parameters", "data_info", "version_info")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_checkpoint(checkpoint_path: str | Path, save_path: str | Path) -> Path:
    """Write a deployment package for a MILIA v2 checkpoint.

    Args:
        checkpoint_path: A checkpoint written by MILIA's trainer (for example
            ``{working_root_dir}/checkpoints/best.pt``).
        save_path: Directory to create the package in.

    Returns:
        The package directory.

    Raises:
        DeploymentError: The checkpoint is missing, unreadable, or lacks the model weights or the
            model name and task type needed to rebuild the model.
    """
    source = Path(checkpoint_path).expanduser()
    if not source.is_file():
        raise DeploymentError(f"Checkpoint not found: {source}")
    try:
        checkpoint = load_checkpoint_file(source, map_location="cpu")
    except Exception as exc:
        raise DeploymentError(f"Cannot read checkpoint {source}: {exc}") from exc

    hyper_parameters = checkpoint.get("hyper_parameters") if isinstance(checkpoint, dict) else None
    missing = [
        name
        for name, present in (
            ("model_state_dict", isinstance(checkpoint, dict) and "model_state_dict" in checkpoint),
            ("hyper_parameters.model_name", bool((hyper_parameters or {}).get("model_name"))),
            ("hyper_parameters.task_type", bool((hyper_parameters or {}).get("task_type"))),
        )
        if not present
    ]
    if missing:
        raise DeploymentError(
            f"{source} is not a MILIA v2 checkpoint that can rebuild its model "
            f"(missing: {', '.join(missing)}). Package a checkpoint written by MILIA's trainer."
        )

    from milia_pipeline import __version__

    package = Path(save_path).expanduser()
    package.mkdir(parents=True, exist_ok=True)
    target = package / PACKAGE_CHECKPOINT
    torch.save({key: checkpoint[key] for key in _INFERENCE_KEYS if key in checkpoint}, target)

    manifest = {
        "format": PACKAGE_FORMAT,
        "milia_version": __version__,
        "checkpoint": PACKAGE_CHECKPOINT,
        "sha256": _sha256(target),
        "model_name": hyper_parameters["model_name"],
        "task_type": hyper_parameters["task_type"],
        "checkpoint_format_version": checkpoint.get("version_info", {}).get(
            "checkpoint_format_version"
        ),
    }
    (package / PACKAGE_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return package


def is_package(path: str | Path) -> bool:
    """Whether ``path`` is a directory holding a deployment manifest."""
    return (Path(path) / PACKAGE_MANIFEST).is_file()


def verify_package(package_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    """Check a package's manifest and checkpoint digest.

    Returns:
        ``(checkpoint_path, manifest)``.

    Raises:
        DeploymentError: The manifest is missing, unreadable or of another format, names a
            checkpoint outside the package, or ``model.pt`` does not match the recorded SHA-256.
    """
    package = Path(package_dir).expanduser()
    manifest_path = package / PACKAGE_MANIFEST
    if not manifest_path.is_file():
        raise DeploymentError(f"No {PACKAGE_MANIFEST} in {package}: not a MILIA deployment package")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise DeploymentError(f"Cannot read {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != PACKAGE_FORMAT:
        found = manifest.get("format") if isinstance(manifest, dict) else manifest
        raise DeploymentError(f"{manifest_path} has format {found!r}; expected {PACKAGE_FORMAT!r}")

    name = str(manifest.get("checkpoint", PACKAGE_CHECKPOINT))
    if Path(name).name != name:
        raise DeploymentError(f"{manifest_path} names a checkpoint outside the package: {name!r}")
    checkpoint = package / name
    if not checkpoint.is_file() or _sha256(checkpoint) != manifest.get("sha256"):
        raise DeploymentError(
            f"{checkpoint} is missing or does not match the SHA-256 in {manifest_path}"
        )
    return checkpoint, manifest


def load_package(
    package_dir: str | Path, device: torch.device | None = None
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model of a deployment package, in eval mode.

    Returns:
        ``(model, model_info)`` as returned by ``ModelLoader.load_from_checkpoint``.

    Raises:
        DeploymentError: As :func:`verify_package`.
    """
    package = Path(package_dir).expanduser()
    checkpoint, _ = verify_package(package)

    from milia_pipeline.models.post_training.inference.model_loader import ModelLoader

    model, model_info = ModelLoader.load_from_checkpoint(
        checkpoint_path=checkpoint, working_root_dir=package, device=device
    )
    model.eval()
    return model, model_info
