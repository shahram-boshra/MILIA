"""Deprecated model-selection features and their migration messages.

Model selection has two modes, ``single`` and ``ensemble``. Layer-built custom
architectures (``mode: custom``, ``models.custom_architecture``, CLI ``--mode custom``,
``--custom-architecture``, ``--architecture-config``) still work and are removed in
MILIA 2.0.0: a new architecture is written as a model plugin and used in either mode.

The warnings are ``FutureWarning``: they are meant for the people who run MILIA with a
configuration, not only for developers calling its API (PEP 565), so they are shown by
default. Each is also logged.

TorchScript in the Python API (``ComputationOptimizer(jit_compile=True)`` and its JIT, graph-pass
and fusion methods, ``ModelOptimizer.optimize_for_mobile``) is also removed in 2.0.0. No
configuration or CLI option reaches it, so its warning is a ``DeprecationWarning`` (PEP 565).
"""

from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)

REMOVAL_VERSION = "2.0.0"

CUSTOM_ARCHITECTURE_DEPRECATION = (
    "Custom architectures (mode 'custom', models.custom_architecture, --mode custom, "
    "--custom-architecture, --architecture-config) are deprecated and will be removed in "
    f"MILIA {REMOVAL_VERSION}. Model selection has two modes, 'single' and 'ensemble': write "
    "the architecture as a model plugin (start from milia_pipeline/plugins/models/user_template) "
    "and use it by name in either mode, alone or with PyG and other plugin models."
)

BUILDER_TYPE_DEPRECATION = (
    "builder_type (models.custom_architecture.builder_type, --builder-type) has no effect and "
    f"will be removed in MILIA {REMOVAL_VERSION}. Combine models with an ensemble's strategy "
    "('sequential', 'parallel' or 'hierarchical') instead."
)


TORCHSCRIPT_DEPRECATION = (
    "{api} uses TorchScript, which is deprecated in MILIA and will be removed in "
    f"MILIA {REMOVAL_VERSION}. PyTorch has deprecated TorchScript (torch.jit) in favour of "
    "torch.compile and torch.export. MILIA's graph models cannot be scripted, and a traced graph "
    "model keeps the example batch's sizes. Use torch.compile (compile_model=True) for speed, and "
    "a deployment package (milia_pipeline.models.deployment) to serve a model."
)


def warn_deprecated(message: str, stacklevel: int = 2) -> None:
    """Emit ``message`` as a ``FutureWarning`` and log it.

    Args:
        message: One of this module's messages.
        stacklevel: As for ``warnings.warn``, counted from the caller of this function.
    """
    warnings.warn(message, FutureWarning, stacklevel=stacklevel + 1)
    logger.warning(message)


def warn_api_deprecated(message: str, stacklevel: int = 2) -> None:
    """Emit ``message`` as a ``DeprecationWarning`` and log it.

    For a Python API that no configuration or CLI option reaches: ``DeprecationWarning`` is
    meant for the developers who call the API (PEP 565).

    Args:
        message: One of this module's messages.
        stacklevel: As for ``warnings.warn``, counted from the caller of this function.
    """
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel + 1)
    logger.warning(message)
