# Location: milia_pipeline/models/hpo/shared_study.py

"""
Storage requirements for a study shared by several worker processes (P2-3e, blueprint S1 / S3).

A worker (``HPOManager(worker_index=...)``, CLI ``--hpo-worker INDEX``) and the study initializer
(``--hpo-init``) operate on a study that other processes use concurrently. The rules are checked when
the process starts, before any dataset or trial work:

* The storage must be persistent: Optuna's ``InMemoryStorage`` "is not designed to be shared across
  processes" (Optuna distributed-optimization tutorial).
* SQLite is rejected: the Optuna FAQ "would never recommend SQLite3 for parallel optimization"
  (no ``SELECT ... FOR UPDATE``, low concurrency and "database is locked" errors, unsafe on NFS).
* An RDB storage needs a heartbeat (``storage_options.heartbeat_interval``): without it a killed
  worker leaves its trial ``RUNNING`` forever (F5); a plain ``study.storage`` URL cannot carry one.
* ``journal_file`` is accepted: the Optuna tutorial recommends ``JournalStorage`` or ``RDBStorage``
  for several processes on one host (several hosts: P2-7, ``journal_lock``).
* ``constant_liar: false`` with TPE is allowed but logged: TPE then ignores the trials still running
  in other workers and may sample similar parameters (F7).

The rules key on the process being part of a shared study, not on a configured worker count: a worker
knows its own index, never the total, and a declared count could disagree with what was launched.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from milia_pipeline.exceptions import HPOConfigurationError

from .hpo_config import HPOConfig, SamplerType

__all__ = ["validate_shared_study"]

logger = logging.getLogger(__name__)

_STORAGE_KEY = "models.hpo.study.storage_options"


def validate_shared_study(config: HPOConfig, environ: Mapping[str, str] | None = None) -> None:
    """Raise if ``config`` cannot run a study shared by several processes; warn on weak settings.

    Args:
        config: HPO configuration of this process
        environ: Environment for ``storage_options.url_env`` (defaults to ``os.environ``)

    Raises:
        HPOConfigurationError: In-memory storage, a SQLite URL, or an RDB storage without heartbeat.
            Messages never contain the storage URL (it may hold a password).
    """
    study = config.study
    if not study.has_persistent_storage:
        raise HPOConfigurationError(
            "A shared study needs a persistent storage; in-memory storage is private to one process",
            config_key=_STORAGE_KEY,
            details="Set models.hpo.study.storage_options (kind 'rdb' or 'journal_file')",
        )

    options = study.storage_options
    if options is None or options.kind == "rdb":
        try:
            backend_name = make_url(study.resolve_storage_url(environ)).get_backend_name()
        except ArgumentError:
            # surfaced as an HPO configuration error (exit 1 in main); the URL is never echoed (F10)
            raise HPOConfigurationError(
                "The storage URL is not a valid SQLAlchemy URL", config_key=_STORAGE_KEY
            ) from None
        if backend_name == "sqlite":
            raise HPOConfigurationError(
                "SQLite cannot serve a study shared by several processes (Optuna FAQ: never "
                "recommended for parallel optimization)",
                config_key=_STORAGE_KEY,
                details="Use kind 'rdb' with PostgreSQL/MySQL, or kind 'journal_file' on one host",
            )
        if options is None or options.heartbeat_interval is None:
            raise HPOConfigurationError(
                "An RDB storage shared by several processes needs a trial heartbeat, otherwise a "
                "killed worker leaves its trial RUNNING",
                config_key=f"{_STORAGE_KEY}.heartbeat_interval",
                details=(
                    "Use storage_options with kind 'rdb', url_env and heartbeat_interval "
                    "(a plain study.storage URL cannot carry a heartbeat)"
                ),
            )

    sampler = config.sampler
    if sampler.type is SamplerType.TPE and not sampler.constant_liar:
        logger.warning(
            "sampler.constant_liar is false in a shared study: TPE ignores trials still running in "
            "other workers and may sample similar parameters (set constant_liar: true)"
        )
