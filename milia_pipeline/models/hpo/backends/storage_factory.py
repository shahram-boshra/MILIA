# Location: milia_pipeline/models/hpo/backends/storage_factory.py

"""
Optuna study-storage construction (P2-2a, blueprint S2).

Single construction point for the storage a study runs on, built from ``StudyConfig``:

- ``storage`` (URL string) is returned unchanged — Optuna builds ``RDBStorage`` from it exactly as
  before (``optuna.storages.get_storage``); this legacy path is untouched (Parallel Change, expand).
- ``storage_options`` ``kind="rdb"`` → ``optuna.storages.RDBStorage(url, engine_kwargs=...)`` with the
  URL read from ``url_env`` now (never stored on the config).
- ``storage_options`` ``kind="journal_file"`` →
  ``optuna.storages.JournalStorage(JournalFileBackend(journal_path))`` — several processes on one host;
  Optuna recommends RDB for multi-node because file locks may fail over NFS.
- nothing configured → ``None`` (in-memory).

Credentials never reach logs or exception messages: URLs are rendered with ``redact_url`` (P1-1).
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

from milia_pipeline.exceptions import BackendError, HPOConfigurationError, redact_url

if TYPE_CHECKING:
    from optuna.storages import BaseStorage

    from ..hpo_config import StudyConfig

logger = logging.getLogger(__name__)

__all__ = ["build_storage"]


def build_storage(study: StudyConfig) -> str | BaseStorage | None:
    """Return the storage for ``study``: a URL string (legacy ``storage``), a built Optuna storage
    (``storage_options``) or ``None`` (in-memory).

    Args:
        study: Study configuration (``storage`` and ``storage_options`` are mutually exclusive)

    Returns:
        ``study.storage`` unchanged, an ``optuna.storages.BaseStorage``, or ``None``

    Raises:
        HPOConfigurationError: Unset ``url_env`` variable, or ``engine_kwargs`` rejected by
            ``sqlalchemy.create_engine``
        BackendError: Optuna unavailable, database driver missing, or the storage cannot be opened
    """
    options = study.storage_options
    if options is None:
        return study.storage

    if options.kind == "rdb":
        url = options.resolve_url()
        return _build_rdb_storage(url, options.engine_kwargs)
    if options.kind == "journal_file":
        return _build_journal_file_storage(options.journal_path)
    raise HPOConfigurationError(
        f"Unsupported storage_options kind '{options.kind}'",
        config_key="models.hpo.study.storage_options.kind",
    )


def _optuna_storages() -> Any:
    try:
        import optuna.storages as storages
    except ImportError as e:
        raise BackendError(
            "Optuna is not installed; persistent study storage requires it",
            backend_name="optuna",
            operation="build_storage",
            details="Install with: pip install optuna",
        ) from e
    return storages


def _build_rdb_storage(url: str, engine_kwargs: dict[str, Any] | None) -> BaseStorage:
    storages = _optuna_storages()
    safe_url = redact_url(url)
    # Deep copy: SQLAlchemy may mutate the dict it receives; the frozen config must stay unchanged.
    kwargs = copy.deepcopy(engine_kwargs) if engine_kwargs is not None else None
    try:
        storage = storages.RDBStorage(url, engine_kwargs=kwargs)
    except TypeError as e:
        # sqlalchemy.create_engine: "Invalid argument(s) ... sent to create_engine()"
        raise HPOConfigurationError(
            f"Invalid engine_kwargs for RDB storage {safe_url}",
            config_key="models.hpo.study.storage_options.engine_kwargs",
            details=str(e),
        ) from e
    except ImportError as e:
        raise BackendError(
            f"Database driver for RDB storage {safe_url} is not installed",
            backend_name="optuna",
            operation="build_storage",
            details=str(e),
        ) from e
    except Exception as e:
        raise BackendError(
            f"Failed to open RDB storage {safe_url}: {type(e).__name__}",
            backend_name="optuna",
            operation="build_storage",
            details=str(e),
        ) from e
    logger.info(f"Using RDB study storage {safe_url}")
    return storage


def _build_journal_file_storage(journal_path: str) -> BaseStorage:
    storages = _optuna_storages()
    try:
        storage = storages.JournalStorage(storages.journal.JournalFileBackend(journal_path))
    except OSError as e:
        raise BackendError(
            f"Failed to open journal storage '{journal_path}': {type(e).__name__}",
            backend_name="optuna",
            operation="build_storage",
            details=str(e),
        ) from e
    logger.info(f"Using journal-file study storage '{journal_path}'")
    return storage
