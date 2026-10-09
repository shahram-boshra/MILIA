# Location: milia_pipeline/models/hpo/backends/storage_factory.py

"""
Optuna study-storage construction (P2-2a, blueprint S2).

Single construction point for the storage a study runs on, built from ``StudyConfig``:

- ``storage`` (URL string) is returned unchanged — Optuna builds ``RDBStorage`` from it exactly as
  before (``optuna.storages.get_storage``); this legacy path is untouched (Parallel Change, expand).
- ``storage_options`` ``kind="rdb"`` → ``optuna.storages.RDBStorage(url, engine_kwargs=...)`` with the
  URL read from ``url_env`` or ``url_file`` now (never stored on the config); with ``heartbeat_interval`` also
  ``grace_period`` and, for ``max_retry``, ``RetryHeartbeatStaleTrialCallback`` (P2-2b): trials left
  ``RUNNING`` by a killed worker are failed and, if configured, re-queued a bounded number of times.
- ``storage_options`` ``kind="journal_file"`` →
  ``optuna.storages.JournalStorage(JournalFileBackend(journal_path, lock_obj=...))`` — several processes
  on one host; Optuna recommends RDB for multi-node because file locks may fail over NFS. ``journal_lock``
  (P2-7): absent → Optuna's default ``JournalFileSymlinkLock``; ``"symlink"`` / ``"open"`` →
  ``JournalFileSymlinkLock`` / ``JournalFileOpenLock``, with a warning naming RDB for several hosts.
- nothing configured → ``None`` (in-memory).

Credentials never reach logs or exception messages: URLs are rendered with ``redact_url`` (P1-1).
"""

from __future__ import annotations

import contextlib
import copy
import logging
import warnings
from collections.abc import Iterator
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
        HPOConfigurationError: Unset ``url_env`` variable or unreadable ``url_file``, or ``engine_kwargs`` rejected by
            ``sqlalchemy.create_engine``
        BackendError: Optuna unavailable, database driver missing, or the storage cannot be opened
    """
    options = study.storage_options
    if options is None:
        return study.storage

    if options.kind == "rdb":
        url = options.resolve_url()
        storage = _build_rdb_storage(url, options.engine_kwargs, _heartbeat_kwargs(options))
        if options.heartbeat_interval is not None:
            grace = options.grace_period or 2 * options.heartbeat_interval
            retry = options.max_retry if options.max_retry is not None else "no retry"
            logger.info(
                "Trial heartbeat enabled (experimental in optuna 4.9.0): "
                f"interval={options.heartbeat_interval}s, grace={grace}s, max_retry={retry}"
            )
        return storage
    if options.kind == "journal_file":
        return _build_journal_file_storage(options.journal_path, options.journal_lock)
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


def _heartbeat_kwargs(options: Any) -> dict[str, Any]:
    """``RDBStorage`` heartbeat arguments from ``storage_options`` (P2-2b); empty when disabled.

    ``heartbeat_stale_trial_callback`` is the optuna 4.9.0 name (``failed_trial_callback`` is deprecated
    in 4.9.0, removal 6.0.0). ``RetryHeartbeatStaleTrialCallback(max_retry=None)`` retries without limit,
    so the callback is attached only for an explicit, validated ``max_retry >= 1``.
    """
    if options.heartbeat_interval is None:
        return {}
    kwargs: dict[str, Any] = {
        "heartbeat_interval": options.heartbeat_interval,
        "grace_period": options.grace_period,
    }
    if options.max_retry is not None:
        storages = _optuna_storages()
        with _experimental_api():
            kwargs["heartbeat_stale_trial_callback"] = storages.RetryHeartbeatStaleTrialCallback(
                max_retry=options.max_retry
            )
    return kwargs


@contextlib.contextmanager
def _experimental_api() -> Iterator[None]:
    """Scope-suppress Optuna's ``ExperimentalWarning`` at a deliberate opt-in instantiation (same
    pattern as ``OptunaBackend.create_pruner`` / ``create_sampler``); the INFO log line records the
    experimental status instead."""
    from optuna.exceptions import ExperimentalWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        yield


def _build_rdb_storage(
    url: str, engine_kwargs: dict[str, Any] | None, heartbeat: dict[str, Any]
) -> BaseStorage:
    storages = _optuna_storages()
    safe_url = redact_url(url)
    # Deep copy: SQLAlchemy may mutate the dict it receives; the frozen config must stay unchanged.
    kwargs = copy.deepcopy(engine_kwargs) if engine_kwargs is not None else None
    try:
        with _experimental_api():
            storage = storages.RDBStorage(url, engine_kwargs=kwargs, **heartbeat)
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


def _build_journal_file_storage(journal_path: str, journal_lock: str | None = None) -> BaseStorage:
    storages = _optuna_storages()
    journal = storages.journal
    lock_classes = {"symlink": journal.JournalFileSymlinkLock, "open": journal.JournalFileOpenLock}
    lock_obj = None if journal_lock is None else lock_classes[journal_lock](journal_path)
    if journal_lock is not None:
        # An explicit lock is the NFS setting (F22): Optuna still recommends RDB across hosts
        logger.warning(
            f"Journal lock '{journal_lock}' selected for '{journal_path}': Optuna recommends kind "
            "'rdb' for several hosts, since file locks over NFS may not work correctly"
        )
    try:
        storage = storages.JournalStorage(
            journal.JournalFileBackend(journal_path, lock_obj=lock_obj)
        )
    except OSError as e:
        raise BackendError(
            f"Failed to open journal storage '{journal_path}': {type(e).__name__}",
            backend_name="optuna",
            operation="build_storage",
            details=str(e),
        ) from e
    lock = journal_lock or "symlink (Optuna default)"
    logger.info(f"Using journal-file study storage '{journal_path}' (lock: {lock})")
    return storage
