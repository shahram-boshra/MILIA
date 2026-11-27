"""The ``models.deployment`` configuration section: what MILIA does with a trained model.

::

    models:
      deployment:
        enabled: false          # true: gate, package, verify (export, predict) after standard training
        target: container       # aws, gcp, azure, container, local (DeploymentManager targets)
        base_image: null        # serving image the Dockerfile builds on; null: the published MILIA image
        output_dir: deployments # versioned packages: <output_dir>/<family>/<N>/ (relative: to working_root_dir)
        onnx:
          enabled: false        # also export the packaged model to ONNX
        predict:
          input_path: null      # molecules file to predict with the packaged model
        gate:
          metric: val_loss      # the checkpoint metric that decides whether it is deployed
          mode: min             # min: lower is better; max: higher is better
          threshold: null       # deploy only if the metric is at least this good (null: no threshold)
          compare_to_current: false  # deploy only if better than the newest package of the model family
        registry:               # MLflow model registry (the mlflow extra; .registry)
          enabled: false        # true: register each published version, move the alias to it
          tracking_uri: null    # null: MLFLOW_TRACKING_URI, else sqlite:///<working_root_dir>/mlflow.db
          artifact_location: null  # new experiment's artifact store; null: <working_root_dir>/mlartifacts
          experiment: milia     # the experiment holding one run per registration
          model_name: null      # registered model; null: milia.<family id> (one per model family)
          alias: champion       # the alias moved to each newly registered version

:func:`load_deployment_settings` validates a section: an unknown key is a
:class:`~milia_pipeline.exceptions.ConfigurationError`; so is ``registry.enabled: true`` when MLflow
is not installed (``uv sync --extra mlflow``). The keys of the earlier section, which no
code acted on (``strategy``, ``edge``, ``cloud``, ``federated``, ``optimization``,
``monitoring``), are still accepted: each one set warns with ``FutureWarning`` and a log line, is
ignored, and will be removed in MILIA 2.0.0 (PEP 565: ``FutureWarning`` for settings aimed at the
person running the application; PEP 387: name the removal release).
"""

from __future__ import annotations

import importlib.util
import logging
import posixpath
import re
import warnings
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from milia_pipeline.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = [
    "DEPLOYMENT_CONFIG_REMOVAL_VERSION",
    "DEPLOYMENT_TARGETS",
    "DEPRECATED_DEPLOYMENT_CONFIG_KEYS",
    "DeploymentSettings",
    "GateSettings",
    "OnnxSettings",
    "PredictSettings",
    "RegistrySettings",
    "check_registry_backend",
    "deprecated_deployment_config_keys",
    "load_deployment_settings",
]

DEPLOYMENT_CONFIG_REMOVAL_VERSION = "2.0.0"

# DeploymentManager targets that serve the package (mobile and iot are deprecated targets).
DEPLOYMENT_TARGETS = ("aws", "gcp", "azure", "container", "local")

# Keys of the earlier ``models.deployment`` section; no code acted on any of them.
DEPRECATED_DEPLOYMENT_CONFIG_KEYS: dict[str, str] = {
    "strategy": "it has no effect; choose where the package goes with target",
    "edge": "it has no effect; MILIA builds no on-device model",
    "cloud": "it has no effect; MILIA writes a serving image and provisions no cloud resources",
    "federated": "it has no effect; MILIA has no federated learning",
    "optimization.quantization": ("it has no effect; ModelOptimizer stays available as Python API"),
    "optimization.pruning": "it has no effect; ModelOptimizer stays available as Python API",
    "optimization.distillation": "it has no effect; MILIA has no distillation training",
    "monitoring": "it has no effect; ModelMonitor stays available as Python API",
}


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OnnxSettings(_Section):
    """``models.deployment.onnx``."""

    enabled: bool = False


class PredictSettings(_Section):
    """``models.deployment.predict``."""

    input_path: str | None = None


class GateSettings(_Section):
    """``models.deployment.gate``: whether a checkpoint is good enough to deploy."""

    metric: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    threshold: float | None = None
    compare_to_current: bool = False


# MLflow's naming rules (mlflow/utils/validation.py, 3.18.0), checked when the section is loaded
# rather than after training: an alias holds alphanumerics, underscores and dashes, at most 255
# characters, and is not "latest" (any case) nor a version alias (v1, V2, ...); a registered model
# name is not empty, has no "/" or ":", and does not resolve to another name as a path. MILIA also
# refuses quotes in a name: it looks versions up with MLflow's search filter, which quotes the name.
_ALIAS = re.compile(r"[\w\-]{1,255}")
_VERSION_ALIAS = re.compile(r"[vV]\d+")


class RegistrySettings(_Section):
    """``models.deployment.registry``: the MLflow model registry (:mod:`.registry`)."""

    enabled: bool = False
    tracking_uri: str | None = None
    artifact_location: str | None = None
    experiment: str = Field(default="milia", min_length=1)
    model_name: str | None = None
    alias: str = "champion"

    @field_validator("alias")
    @classmethod
    def _alias(cls, alias: str) -> str:
        if not _ALIAS.fullmatch(alias):
            raise ValueError(
                "an alias holds 1 to 255 alphanumerics, underscores (_) and dashes (-)"
            )
        if alias.lower() == "latest" or _VERSION_ALIAS.fullmatch(alias):
            raise ValueError(f"{alias!r} is reserved by MLflow (latest, v<number>)")
        return alias

    @field_validator("model_name")
    @classmethod
    def _model_name(cls, name: str | None) -> str | None:
        if name is None:
            return None
        normalized = posixpath.normpath(name)
        if (
            not name.strip()
            or any(c in name for c in "/:'\"")
            or normalized != name
            or normalized == "."
            or normalized.startswith("..")
        ):
            raise ValueError(
                "a registered model name is not empty, has no '/', ':' or quotes, and is not a "
                "path that resolves to another name"
            )
        return name


class DeploymentSettings(_Section):
    """The validated ``models.deployment`` section (see the module docstring)."""

    enabled: bool = False
    target: Literal["aws", "gcp", "azure", "container", "local"] = "container"
    base_image: str | None = None
    output_dir: str = Field(default="deployments", min_length=1)
    onnx: OnnxSettings = Field(default_factory=OnnxSettings)
    predict: PredictSettings = Field(default_factory=PredictSettings)
    gate: GateSettings = Field(default_factory=GateSettings)
    registry: RegistrySettings = Field(default_factory=RegistrySettings)


def deprecated_deployment_config_keys(section: Any) -> list[str]:
    """One message per deprecated key set in a ``models.deployment`` mapping, in table order."""
    if not isinstance(section, dict):
        return []
    optimization = section.get("optimization")
    present = set(section)
    if isinstance(optimization, dict):
        present |= {f"optimization.{key}" for key in optimization}
    messages = [
        f"models.deployment.{key} is deprecated and will be removed in "
        f"{DEPLOYMENT_CONFIG_REMOVAL_VERSION}: {reason}"
        for key, reason in DEPRECATED_DEPLOYMENT_CONFIG_KEYS.items()
        if key in present
    ]
    unlisted = (
        sorted(set(optimization) - {"quantization", "pruning", "distillation"})
        if isinstance(optimization, dict)
        else []
    )
    if "optimization" in section and (
        not isinstance(optimization, dict) or unlisted or not optimization
    ):
        messages.append(
            f"models.deployment.optimization is deprecated and will be removed in "
            f"{DEPLOYMENT_CONFIG_REMOVAL_VERSION}: it has no effect"
        )
    return messages


def load_deployment_settings(section: Any, require_backend: bool = True) -> DeploymentSettings:
    """Validate a ``models.deployment`` section.

    Deprecated keys warn (``FutureWarning`` and a log line, one per key) and are ignored.

    Args:
        section: The ``models.deployment`` mapping; ``None`` gives the defaults.
        require_backend: Check that MLflow is installed when the registry is enabled
            (:func:`check_registry_backend`); ``False`` for a caller that overrides the section
            first and checks the result (``milia --deploy``).

    Raises:
        ConfigurationError: The section is not a mapping, has an unknown key, or a value of the
            wrong type; or ``registry.enabled`` is true and MLflow is not installed.
    """
    if section is None:
        return DeploymentSettings()
    if not isinstance(section, dict):
        raise ConfigurationError(
            f"models.deployment must be a mapping, got {type(section).__name__}"
        )
    for message in deprecated_deployment_config_keys(section):
        warnings.warn(message, FutureWarning, stacklevel=2)
        logger.warning(message)
    deprecated_top = {key.split(".", 1)[0] for key in DEPRECATED_DEPLOYMENT_CONFIG_KEYS}
    current = {key: value for key, value in section.items() if key not in deprecated_top}
    try:
        settings = DeploymentSettings.model_validate(current)
    except ValidationError as exc:
        problems = "; ".join(
            f"models.deployment.{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigurationError(f"Invalid models.deployment section: {problems}") from exc
    if require_backend:
        check_registry_backend(settings)
    return settings


def check_registry_backend(settings: DeploymentSettings) -> None:
    """Refuse an enabled registry when MLflow is not installed (checked without importing it).

    Raises:
        ConfigurationError: ``registry.enabled`` and no ``mlflow`` module.
    """
    if settings.registry.enabled and importlib.util.find_spec("mlflow") is None:
        raise ConfigurationError(
            "models.deployment.registry.enabled is true, but MLflow is not installed: install "
            "the mlflow extra (uv sync --extra mlflow), or set it to false"
        )
