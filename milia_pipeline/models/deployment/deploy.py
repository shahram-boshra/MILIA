"""Deploy a trained model: a verified, versioned package plus what its target needs.

:func:`deploy` is what ``milia --deploy`` runs. From a MILIA checkpoint or an existing package
and the ``models.deployment`` settings (:mod:`.settings`) it writes
``<output_dir>/<model_name>/<N>/`` with :func:`~.package.publish_package` and adds, before the
version becomes visible:

- for the ``aws``, ``gcp``, ``azure`` and ``container`` targets, the ``Dockerfile`` that serves the
  package (:func:`~.serving.write_serving_dockerfile`, on ``base_image``); ``local`` needs none;
- with ``onnx.enabled``, ``model.onnx`` exported from the package's model on its input example
  (:func:`~.onnx_export.export_onnx`, which verifies the export).

A failure in any of these steps publishes nothing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from milia_pipeline.models.deployment.deployment_strategies import SERVING_IMAGE_TARGETS
from milia_pipeline.models.deployment.package import (
    PACKAGE_MANIFEST,
    publish_package,
    verify_package,
)
from milia_pipeline.models.deployment.serving import write_serving_dockerfile
from milia_pipeline.models.deployment.settings import DeploymentSettings

logger = logging.getLogger(__name__)

__all__ = ["ONNX_FILE", "DeployResult", "deploy", "manifest_summary"]

ONNX_FILE = "model.onnx"
_FALLBACK_EXAMPLE = ("CCO",)


@dataclass(frozen=True)
class DeployResult:
    """Where a deployment went and what it holds."""

    version_dir: Path
    target: str
    files: tuple[str, ...]
    manifest: dict[str, Any]


def _export_onnx(package_dir: Path) -> None:
    from torch_geometric.data import Batch

    from milia_pipeline.models.deployment.onnx_export import export_onnx
    from milia_pipeline.models.post_training import convert_to_pyg
    from milia_pipeline.models.post_training.inference.predictor import Predictor

    checkpoint, manifest = verify_package(package_dir)
    predictor = Predictor.from_checkpoint(
        checkpoint_path=checkpoint, working_root_dir=package_dir, device=torch.device("cpu")
    )
    model, config = predictor.model.eval(), predictor.structural_features_config
    smiles = list(manifest.get("input_example") or _FALLBACK_EXAMPLE)
    graphs = [convert_to_pyg(s, format="smiles", structural_features_config=config) for s in smiles]
    export_onnx(model, package_dir / ONNX_FILE, Batch.from_data_list(graphs))


def _prepare(settings: DeploymentSettings):
    def prepare(package_dir: Path) -> None:
        if settings.target in SERVING_IMAGE_TARGETS:
            write_serving_dockerfile(
                package_dir,
                base_image=settings.base_image,
                run_as_root=SERVING_IMAGE_TARGETS[settings.target],
            )
        if settings.onnx.enabled:
            _export_onnx(package_dir)

    return prepare


def deploy(
    model_path: str | Path, settings: DeploymentSettings, working_root_dir: str | Path
) -> DeployResult:
    """Publish ``model_path`` (checkpoint or package) for ``settings.target``.

    A relative ``settings.output_dir`` is taken relative to ``working_root_dir``.

    Raises:
        DeploymentError: Packaging or verification failed.
        ExportError: The ONNX export failed or did not reproduce the model.
    """
    output_dir = Path(settings.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path(working_root_dir) / output_dir
    version_dir = publish_package(model_path, output_dir, prepare=_prepare(settings))
    manifest = json.loads((version_dir / PACKAGE_MANIFEST).read_text())
    files = tuple(sorted(p.name for p in version_dir.iterdir()))
    return DeployResult(version_dir, settings.target, files, manifest)


def manifest_summary(manifest: dict[str, Any]) -> list[str]:
    """Lines describing a package manifest, for the console."""
    signature = manifest.get("signature") or {}
    inputs = signature.get("inputs") or {}
    outputs = signature.get("outputs") or {}
    metrics = manifest.get("metrics") or {}
    monitored = metrics.get("monitored") or {}
    names = outputs.get("names")
    return [
        f"Model: {manifest.get('model_name')} ({manifest.get('task_type')}), package "
        f"{manifest.get('format')}, written by MILIA {manifest.get('milia_version')}",
        f"Input: SMILES, {inputs.get('node_features', 'unrecorded')} node features; "
        f"output: {outputs.get('count', 'unrecorded')} values"
        + (f" ({', '.join(names)})" if names else ""),
        f"Checkpoint epoch {metrics.get('epoch')}; "
        + (
            f"{monitored.get('name')} = {monitored.get('value'):.6g}"
            if monitored.get("name") and monitored.get("value") is not None
            else "monitored metric not recorded"
        )
        + ("; test metrics recorded" if metrics.get("test") else "; no test metrics"),
    ]
