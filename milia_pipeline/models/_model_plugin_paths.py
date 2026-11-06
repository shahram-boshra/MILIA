"""Model plugin directories: the built-in container plus the configured user paths.

``plugin_paths`` in the ``models.plugins`` configuration lists **user** paths. Each
is resolved as follows:

1. ``~`` is expanded.
2. An absolute path is used as given.
3. A relative path resolves against the configuration root (the ``configs/``
   directory, or the directory of a single configuration file), not the current
   working directory.
4. If that directory does not exist but the path exists relative to the current
   working directory, the old resolution is used with a ``DeprecationWarning``
   attributed to the configuration root (removal in MILIA 2.0.0).
5. A path that is not an existing directory is logged and skipped.

The built-in container (``milia_pipeline/plugins/models``) is added first when
``include_builtin`` is set. Every path is ``resolve()``d and each directory is
returned once, in order.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

CWD_FALLBACK_REMOVAL_VERSION = "2.0.0"


def builtin_model_plugins_directory() -> Path:
    """The built-in model plugin container, resolved (independent of the working directory)."""
    from milia_pipeline.plugins import get_model_plugins_directory

    return get_model_plugins_directory().resolve()


def _resolve_user_path(raw: str, config_root: Path) -> Path | None:
    """Resolve one ``plugin_paths`` entry; ``None`` if it is not an existing directory."""
    path = Path(raw).expanduser()
    if path.is_absolute():
        candidate = path
    else:
        candidate = config_root / path
        if not candidate.exists():
            legacy = Path.cwd() / path
            if legacy.is_dir():
                message = (
                    f"Model plugin path {raw!r} was found relative to the working directory "
                    f"({legacy.resolve()}), not to the configuration root ({config_root}). "
                    f"Relative plugin_paths resolve against the configuration root; the "
                    f"working-directory fallback will be removed in MILIA "
                    f"{CWD_FALLBACK_REMOVAL_VERSION}. Use a path relative to {config_root} "
                    f"or an absolute path."
                )
                warnings.warn_explicit(
                    message, DeprecationWarning, filename=str(config_root), lineno=0
                )
                logger.warning(message)
                candidate = legacy

    if not candidate.is_dir():
        reason = "is not a directory" if candidate.exists() else "does not exist"
        logger.warning(f"Model plugin path {raw!r} skipped: {candidate} {reason}")
        return None
    return candidate.resolve()


def resolve_model_plugin_paths(
    plugin_paths: Iterable[str], config_root: Path, include_builtin: bool = True
) -> list[Path]:
    """
    The directories to scan for model plugins.

    Args:
        plugin_paths: The configured user paths.
        config_root: The configuration root that relative paths resolve against
            (see ``milia_pipeline.config.get_config_root``).
        include_builtin: Whether to add the built-in container first.

    Returns:
        Existing, resolved directories without duplicates, in order.
    """
    root = Path(config_root).expanduser().resolve()
    candidates: list[Path] = [builtin_model_plugins_directory()] if include_builtin else []
    for raw in plugin_paths:
        resolved = _resolve_user_path(raw, root)
        if resolved is not None:
            candidates.append(resolved)

    return list(dict.fromkeys(candidates))
