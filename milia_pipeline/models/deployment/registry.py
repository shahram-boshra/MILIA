"""Register published deployment packages in an MLflow model registry (the ``mlflow`` extra).

With ``models.deployment.registry.enabled``, every version :func:`~.deploy.deploy` publishes
(after a passing gate, or by ``milia --deploy``) is registered by :func:`register_package`:

* **Registered model:** one per model family (:func:`~.package.manifest_family`: dataset type,
  task, targets, units), named ``milia.<family id>`` unless ``registry.model_name`` names it.
  Architectures compete within it, as versions of one family do in ``<output_dir>/<family>/``.
* **Version:** the package logged as a pyfunc model (:mod:`.mlflow_pyfunc`, Models from Code)
  with its signature, its input example and the package folder as artifact, in one run of the
  experiment ``registry.experiment``. The run records the package's metrics, its MILIA, git and
  model-source lineage as tags, and its pip requirements (``lineage.model_source``); a model
  plugin loaded from a directory travels with it (``code_paths``).
* **Alias:** ``registry.alias`` (``champion``) moves to the new version: the gate, or the person
  running ``milia --deploy``, approved it. The version records how (``milia.approval``).

A package already registered under the model (the same checkpoint SHA-256) is not registered
again; the alias moves to its version. A package the registry's model could not serve (no SMILES
or structure input it accepts, an edge-level output) is not registered; the reason is returned.
Loading a registered model needs MILIA installed where it loads (``env_manager="local"``): torch
and the PyG companions come from per-accelerator indexes that pip requirements cannot name.

The tracking and registry store is ``registry.tracking_uri``, else ``MLFLOW_TRACKING_URI``, else
``sqlite:///<working_root_dir>/mlflow.db`` (MLflow's registry needs a database-backed store). A new
experiment stores its artifacts at ``registry.artifact_location``, else
``<working_root_dir>/mlartifacts``, so nothing is written to the process's working directory.

**Lineage, not the registering process.** The run, the logged model and the version carry the
commit MILIA recorded at training (``lineage.git_commit``) as ``mlflow.source.git.commit``, also
``milia.git_commit``: MLflow would otherwise tag the logged model with the commit of whatever
checkout runs the registration (its git context provider; passed tags take precedence).
"not recorded" is written when training recorded none. MLflow still probes for git through
GitPython; where no ``git`` executable is installed, GitPython prints a notice that its
``GIT_PYTHON_REFRESH=quiet`` silences (MILIA's test image sets it).

**The environment record.** MLflow 3 copies the uv project of the working directory
(``uv.lock``, ``pyproject.toml``) into a logged model. MILIA passes its own project instead when
it runs from a checkout of it (a ``uv.lock`` beside its ``pyproject.toml`` naming ``milia-py``):
the lock records the exact environment, torch's accelerator index included, which pip
requirements cannot. Otherwise auto-detection is off for the call, so an unrelated project in the
working directory is never recorded as the model's environment.

MLflow collects anonymized usage data by default (from 3.2.0); ``MLFLOW_DISABLE_TELEMETRY=true``
or ``DO_NOT_TRACK=true`` turns it off (https://mlflow.org/docs/latest/community/usage-tracking/).
MILIA leaves that choice to the environment.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from milia_pipeline.exceptions import DeploymentError
from milia_pipeline.models.deployment.package import (
    PACKAGE_MANIFEST,
    manifest_family,
    manifest_scores,
)
from milia_pipeline.models.deployment.settings import RegistrySettings

logger = logging.getLogger(__name__)

__all__ = [
    "MODEL_CODE",
    "RegistryError",
    "RegistryResult",
    "register_package",
    "milia_uv_project",
    "registered_model_name",
    "resolve_artifact_location",
    "resolve_tracking_uri",
]

MODEL_CODE = Path(__file__).with_name("mlflow_model.py")  # the Models-from-Code script
TRACKING_DB = "mlflow.db"
ARTIFACT_DIR = "mlartifacts"
SHA_TAG = "milia.package_sha256"


class RegistryError(DeploymentError):
    """Registering a published package failed (the package stays published)."""


@dataclass(frozen=True)
class RegistryResult:
    """What :func:`register_package` did.

    ``version`` is ``None`` when the package was not registered (``skipped`` says why);
    ``reused``: the package was registered before, and the alias moved to that version.
    """

    tracking_uri: str
    model_name: str | None = None
    version: str | None = None
    alias: str | None = None
    run_id: str | None = None
    reused: bool = False
    skipped: str | None = None


def resolve_tracking_uri(settings: RegistrySettings, working_root_dir: str | Path) -> str:
    """``tracking_uri``, else ``MLFLOW_TRACKING_URI``, else SQLite in ``working_root_dir``."""
    if settings.tracking_uri:
        return settings.tracking_uri
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    return f"sqlite:///{(Path(working_root_dir) / TRACKING_DB).resolve()}"


def resolve_artifact_location(settings: RegistrySettings, working_root_dir: str | Path) -> str:
    """``artifact_location``, else ``<working_root_dir>/mlartifacts`` as a file URI."""
    if settings.artifact_location:
        return settings.artifact_location
    return (Path(working_root_dir) / ARTIFACT_DIR).resolve().as_uri()


def registered_model_name(settings: RegistrySettings, manifest: dict[str, Any]) -> str:
    """``model_name``, else ``milia.<family id>``."""
    return settings.model_name or f"milia.{manifest_family(manifest)['id']}"


_SCOPED_VARIABLES = ("MLFLOW_TRACKING_URI", "MLFLOW_REGISTRY_URI", "MLFLOW_UV_AUTO_DETECT")


def milia_uv_project() -> Path | None:
    """MILIA's own uv project when it runs from a checkout of it, else ``None``."""
    import tomllib

    import milia_pipeline

    root = Path(milia_pipeline.__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"
    if not (root / "uv.lock").is_file() or not pyproject.is_file():
        return None
    try:
        name = tomllib.loads(pyproject.read_text()).get("project", {}).get("name")
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return root if name == "milia-py" else None


@contextmanager
def _mlflow_environment(tracking_uri: str, uv_project: Path | None) -> Iterator[None]:
    """MLflow's tracking and registry URIs for the logging calls, and uv auto-detection off
    when MILIA has no project of its own to record; all restored as they were afterwards.

    ``mlflow.set_tracking_uri`` and ``set_registry_uri`` also write ``MLFLOW_TRACKING_URI`` and
    ``MLFLOW_REGISTRY_URI`` (for subprocesses), so those variables record what was set before;
    the URIs and the variables are put back, an unset one unset again. Restoring the value
    ``get_tracking_uri()`` reported instead would set MLflow's default store explicitly and leave
    it in the environment for every later call.
    """
    import mlflow

    previous = {name: os.environ.get(name) for name in _SCOPED_VARIABLES}
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    if uv_project is None:
        os.environ["MLFLOW_UV_AUTO_DETECT"] = "false"
    try:
        yield
    finally:
        mlflow.set_tracking_uri(previous["MLFLOW_TRACKING_URI"])  # None: unset
        mlflow.set_registry_uri(previous["MLFLOW_REGISTRY_URI"])
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _experiment_id(client: Any, name: str, artifact_location: str) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        return client.create_experiment(name, artifact_location=artifact_location)
    if experiment.lifecycle_stage != "active":
        raise RegistryError(f"the MLflow experiment {name!r} is deleted; restore it or rename")
    return experiment.experiment_id


def _code_paths(source: dict[str, Any] | None) -> list[str]:
    """Directory plugins the model's classes come from (``lineage.model_source``)."""
    records = (source or {}).get("members") or ([source] if source else [])
    paths = []
    for record in records:
        if record.get("kind") == "directory":
            path = record.get("path")
            if not path or not Path(path).is_dir():
                raise RegistryError(
                    f"the model's plugin directory {path!r} (lineage.model_source) is not present "
                    "here; register the package where the plugin is"
                )
            paths.append(path)
    return paths


def _requirements(manifest: dict[str, Any]) -> list[str]:
    import mlflow

    source = (manifest.get("lineage") or {}).get("model_source") or {}
    pins = list(source.get("requirements") or [f"milia-py=={manifest.get('milia_version')}"])
    return [*pins, f"mlflow=={mlflow.__version__}"]


def _tags(manifest: dict[str, Any], version_dir: Path, approval: dict[str, Any]) -> dict[str, str]:
    from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT

    lineage = manifest.get("lineage") or {}
    tags = {
        SHA_TAG: str(manifest["sha256"]),
        "milia.family": manifest_family(manifest)["id"],
        "milia.version_dir": str(version_dir),
        "milia.milia_version": str(manifest.get("milia_version")),
        "milia.model_name": str(manifest.get("model_name")),
        "milia.task_type": str(manifest.get("task_type")),
        "milia.approval": str(approval.get("by", "manual")),
    }
    # The training commit, overriding the one MLflow resolves for the registering process
    commit = str(lineage["git_commit"]) if lineage.get("git_commit") else "not recorded"
    tags[MLFLOW_GIT_COMMIT] = tags["milia.git_commit"] = commit
    if lineage.get("training_seed") is not None:
        tags["milia.training_seed"] = str(lineage["training_seed"])
    for key in ("metric", "value"):
        if approval.get(key) is not None:
            tags[f"milia.gate_{key}"] = str(approval[key])
    return tags


def _existing_version(client: Any, name: str, sha256: str) -> Any | None:
    from mlflow.exceptions import MlflowException

    try:
        client.get_registered_model(name)
    except MlflowException:
        return None
    versions = client.search_model_versions(f"name = '{name}' and tag.`{SHA_TAG}` = '{sha256}'")
    return min(versions, key=lambda v: int(v.version)) if versions else None


def _ensure_registered_model(client: Any, name: str, manifest: dict[str, Any]) -> None:
    from mlflow.exceptions import MlflowException

    try:
        client.get_registered_model(name)
        return
    except MlflowException:
        pass
    family = manifest_family(manifest)
    key = family["key"]
    targets = key.get("targets") or {}
    names = targets.get("names") or [f"{targets.get('count')} outputs"]
    client.create_registered_model(
        name,
        tags={"milia.family": family["id"], "milia.dataset_type": str(key.get("dataset_type"))},
        description=(
            f"MILIA model family {family['id']}: {key.get('task_type')} of "
            f"{', '.join(map(str, names))} on {key.get('dataset_type')} data "
            f"(source units: {key.get('source_units')}). Versions are deployment packages."
        ),
    )


def register_package(
    version_dir: str | Path,
    settings: RegistrySettings,
    working_root_dir: str | Path,
    approval: dict[str, Any] | None = None,
) -> RegistryResult:
    """Register the published package ``version_dir`` (see the module docstring).

    Args:
        version_dir: A version folder written by :func:`~.package.publish_package`.
        settings: The ``models.deployment.registry`` section.
        working_root_dir: Where the default tracking store and artifact location go.
        approval: Who approved the version: ``{"by": "gate", "metric": ..., "value": ...}``,
            or ``None`` for ``milia --deploy`` (``{"by": "manual"}``).

    Raises:
        RegistryError: The registry could not be reached or written, or a directory plugin the
            model needs is absent.
    """
    import mlflow
    from mlflow import MlflowClient

    from milia_pipeline.models.deployment.mlflow_pyfunc import (
        PACKAGE_ARTIFACT,
        input_example,
        pyfunc_signature,
    )

    version_dir = Path(version_dir).resolve()
    manifest = json.loads((version_dir / PACKAGE_MANIFEST).read_text())
    tracking_uri = resolve_tracking_uri(settings, working_root_dir)
    try:
        signature = pyfunc_signature(manifest)
    except DeploymentError as exc:
        return RegistryResult(tracking_uri, skipped=str(exc))
    name = registered_model_name(settings, manifest)
    approval = approval or {"by": "manual"}
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    try:
        existing = _existing_version(client, name, str(manifest["sha256"]))
        if existing is not None:
            client.set_registered_model_alias(name, settings.alias, existing.version)
            return RegistryResult(
                tracking_uri, name, str(existing.version), settings.alias, existing.run_id, True
            )
        experiment_id = _experiment_id(
            client, settings.experiment, resolve_artifact_location(settings, working_root_dir)
        )
        tags = _tags(manifest, version_dir, approval)
        run = client.create_run(
            experiment_id,
            tags={**tags, "mlflow.source.name": "milia"},
            run_name=f"{manifest_family(manifest)['id']}-v{version_dir.name}",
        )
        run_id = run.info.run_id
        uv_project = milia_uv_project()
        with _mlflow_environment(tracking_uri, uv_project), mlflow.start_run(run_id=run_id):
            mlflow.log_params(
                {
                    "model_name": manifest.get("model_name"),
                    "task_type": manifest.get("task_type"),
                    "dataset_type": ((manifest.get("lineage") or {}).get("dataset") or {}).get(
                        "type"
                    ),
                    "output_count": ((manifest.get("signature") or {}).get("outputs") or {}).get(
                        "count"
                    ),
                }
            )
            scores = {
                key: value
                for key, value in manifest_scores(manifest.get("metrics")).items()
                if isinstance(value, int | float) and math.isfinite(value)
            }
            if scores:
                mlflow.log_metrics(scores)
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=str(MODEL_CODE),
                artifacts={PACKAGE_ARTIFACT: str(version_dir)},
                signature=signature,
                input_example=input_example(manifest),
                pip_requirements=_requirements(manifest),
                code_paths=_code_paths((manifest.get("lineage") or {}).get("model_source")) or None,
                tags=tags,
                uv_project_path=uv_project,
            )
            _ensure_registered_model(client, name, manifest)
            version = mlflow.register_model(
                info.model_uri, name, tags={**tags, "validation_status": "approved"}
            )
        client.set_registered_model_alias(name, settings.alias, version.version)
    except RegistryError:
        raise
    except Exception as exc:  # MLflow raises MlflowException, but also driver and OS errors
        raise RegistryError(
            f"registering {version_dir} in {tracking_uri} failed ({type(exc).__name__}: {exc}); "
            "the package stays published"
        ) from exc
    return RegistryResult(tracking_uri, name, str(version.version), settings.alias, run_id)
