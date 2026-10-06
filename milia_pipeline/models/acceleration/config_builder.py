"""Build an :class:`AccelerationManager` from the ``models.acceleration`` configuration (PA-1e / F42).

This is the single place where the YAML acceleration section becomes runtime behaviour. The section is
parsed and validated by the same code as the config loader (``ModelConfig._parse_acceleration_config``,
including the PA-1a runtime-support checks), so every training path accepts and rejects exactly the
same settings.

* ``enabled: false`` (the default) → ``None``: callers keep their current behaviour unchanged.
* Settings the single-process training/HPO paths cannot honour are rejected, never silently ignored.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from milia_pipeline.exceptions import ConfigurationError

if TYPE_CHECKING:
    from milia_pipeline.models.acceleration import AccelerationManager


def _unwired_settings(config: Any) -> list[str]:
    """Enabled settings that MILIA's single-process Trainer paths do not apply."""
    problems: list[str] = []
    if config.distributed.enabled:
        problems.append(
            "distributed.enabled: true — multi-process training needs a launcher (e.g. torchrun) "
            "and an initialized process group; MILIA's training and HPO paths run one process. "
            "Set distributed.enabled: false."
        )
    memory = config.memory
    if memory.gradient_accumulation_steps != 1:
        problems.append(
            f"memory.gradient_accumulation_steps: {memory.gradient_accumulation_steps} is not "
            "applied — gradient accumulation is a Trainer setting (accumulate_grad_batches)."
        )
    if memory.empty_cache_interval != 0 or memory.max_memory_per_gpu is not None:
        problems.append(
            "memory.empty_cache_interval / memory.max_memory_per_gpu are not applied by the "
            "Trainer; leave them at their defaults (0 / null)."
        )
    return problems


def load_acceleration_config(models_config: Mapping[str, Any] | None) -> Any:
    """Parse and validate ``models.acceleration``; return the bridge ``AccelerationConfig``.

    Use it to fail fast once (e.g. before an HPO study) without constructing a manager.

    Raises:
        ConfigurationError: The section is invalid, or enables settings these paths cannot apply.
    """
    accel_dict = dict((models_config or {}).get("acceleration") or {})

    from milia_pipeline.models.utils.config_bridge import ModelConfig

    try:
        config = ModelConfig._parse_acceleration_config(accel_dict)
    except ValueError as e:  # pydantic ValidationError is a ValueError
        raise ConfigurationError(
            f"Invalid models.acceleration configuration: {e}", config_key="models.acceleration"
        ) from e

    if config.enabled:
        problems = _unwired_settings(config)
        if problems:
            raise ConfigurationError(
                "models.acceleration enables settings these training paths cannot apply:\n  - "
                + "\n  - ".join(problems),
                config_key="models.acceleration",
            )
    return config


def build_acceleration(models_config: Mapping[str, Any] | None) -> AccelerationManager | None:
    """Return a new AccelerationManager for an enabled ``models.acceleration``, else ``None``.

    Build one per training run (trial, fold, final fit): the manager owns a GradScaler whose loss
    scale adapts during training and must not carry over between runs.

    Args:
        models_config: The ``models.*`` configuration subtree (as used by main.py and HPO).

    Raises:
        ConfigurationError: The section is invalid, or enables settings these paths cannot apply.
    """
    config = load_acceleration_config(models_config)
    if not config.enabled:
        return None

    from milia_pipeline.models.acceleration import AccelerationManager

    precision = config.memory.mixed_precision
    computation = config.computation
    return AccelerationManager(
        device=config.device.type,
        mixed_precision=precision != "no",
        precision=precision if precision != "no" else "fp16",
        gradient_checkpointing=config.memory.gradient_checkpointing,
        compile_model=computation.compile_model,
        compile_mode=computation.compile_mode,
        compile_dynamic=computation.compile_dynamic,
        cudnn_benchmark=computation.use_cudnn_benchmark,
        verbose=False,
    )
