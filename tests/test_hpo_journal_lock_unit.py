#!/usr/bin/env python3
"""
Contract tests for ``storage_options.journal_lock`` (P2-7, blueprint S1 / F22).

``kind="journal_file"`` builds ``JournalStorage(JournalFileBackend(journal_path, lock_obj=...))``.
Optuna's default lock is ``JournalFileSymlinkLock`` (atomic ``symlink(2)``, NFSv2+);
``JournalFileOpenLock`` (``open(2)`` with ``O_EXCL``, NFSv3+) serves file systems without symbolic
links. ``journal_lock`` selects one explicitly and logs that Optuna recommends RDB across hosts. The
concurrency test runs real ``spawn`` workers on a local file system (NFS itself is not available in CI).
"""

import logging
import multiprocessing
import time

import optuna
import pytest
from optuna.storages.journal import JournalFileOpenLock, JournalFileSymlinkLock
from optuna.trial import TrialState
from pydantic import ValidationError

from milia_pipeline.models.hpo.backends.storage_factory import build_storage
from milia_pipeline.models.hpo.hpo_config import StorageConfig, StudyConfig
from milia_pipeline.models.hpo.seeding import derive_worker_seed

FACTORY_LOGGER = "milia_pipeline.models.hpo.backends.storage_factory"


def _study(path, lock=None) -> StudyConfig:
    return StudyConfig(
        storage_options=StorageConfig(
            kind="journal_file", journal_path=str(path), journal_lock=lock
        )
    )


@pytest.mark.contract
class TestConfig:
    def test_default_is_optuna_default(self):
        assert StorageConfig(kind="journal_file", journal_path="j.log").journal_lock is None

    @pytest.mark.parametrize("lock", ["symlink", "open"])
    def test_accepted_for_journal_file(self, lock):
        assert (
            StorageConfig(kind="journal_file", journal_path="j", journal_lock=lock).journal_lock
            == lock
        )

    @pytest.mark.parametrize("lock", ["flock", "OPEN", "", 1, True])
    def test_rejects_unknown_lock(self, lock):
        with pytest.raises(ValidationError):
            StorageConfig(kind="journal_file", journal_path="j", journal_lock=lock)

    def test_rejected_for_rdb(self):
        with pytest.raises(ValidationError, match="does not accept: journal_lock"):
            StorageConfig(kind="rdb", url_env="X", journal_lock="open")


@pytest.mark.contract
class TestFactory:
    @pytest.mark.parametrize(
        ("lock", "lock_class"),
        [
            (None, JournalFileSymlinkLock),
            ("symlink", JournalFileSymlinkLock),
            ("open", JournalFileOpenLock),
        ],
    )
    def test_lock_object(self, tmp_path, lock, lock_class):
        storage = build_storage(_study(tmp_path / "j.log", lock))
        assert type(storage._backend._lock) is lock_class

    def test_default_lock_logs_no_warning(self, tmp_path, caplog):
        with caplog.at_level(logging.INFO, logger=FACTORY_LOGGER):
            build_storage(_study(tmp_path / "j.log"))
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert "(lock: symlink (Optuna default))" in caplog.text

    @pytest.mark.parametrize("lock", ["symlink", "open"])
    def test_explicit_lock_warns_rdb_across_hosts(self, tmp_path, caplog, lock):
        with caplog.at_level(logging.INFO, logger=FACTORY_LOGGER):
            build_storage(_study(tmp_path / "j.log", lock))
        (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert f"Journal lock '{lock}'" in warning.getMessage()
        assert "kind 'rdb' for several hosts" in warning.getMessage()
        assert f"(lock: {lock})" in caplog.text


def _worker(path: str, lock: str, worker_index: int, budget: int) -> None:
    from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    sampler = optuna.samplers.TPESampler(
        seed=derive_worker_seed(42, worker_index), constant_liar=True, n_startup_trials=3
    )
    study = optuna.load_study(
        study_name="lock", storage=build_storage(_study(path, lock)), sampler=sampler
    )

    def objective(trial):
        x = trial.suggest_float("x", -10.0, 10.0)
        time.sleep(0.05)
        return (x - 2.0) ** 2

    OptunaBackend().optimize(
        study, objective, n_trials=100, n_trials_total=budget, show_progress_bar=False
    )


@pytest.mark.contract
@pytest.mark.parametrize("lock", ["symlink", "open"])
def test_workers_share_journal_with_explicit_lock(tmp_path, lock):
    """3 spawn workers, global budget 12: finished in [12, 14], no duplicates, no lock file left."""
    path = tmp_path / "hpo_journal.log"
    budget, workers = 12, 3
    optuna.create_study(study_name="lock", storage=build_storage(_study(path, lock)))
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_worker, args=(str(path), lock, i, budget)) for i in range(workers)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=300)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
    assert [p.exitcode for p in processes] == [0] * workers
    study = optuna.load_study(study_name="lock", storage=build_storage(_study(path, lock)))
    states = [t.state for t in study.trials]
    finished = states.count(TrialState.COMPLETE) + states.count(TrialState.PRUNED)
    assert budget <= finished <= budget + workers - 1
    assert TrialState.RUNNING not in states
    params = [t.params["x"] for t in study.trials]
    assert len(set(params)) == len(params)
    assert not (tmp_path / "hpo_journal.log.lock").exists()
