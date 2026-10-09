#!/usr/bin/env python3
"""
Integration tests for a study shared through a real PostgreSQL server (P2-5, blueprint S6 / F3, F5).

They run only when ``MILIA_TEST_POSTGRES_URL`` names a reachable database (``postgresql+psycopg://…``)
and the ``hpo-postgres`` extra is installed — the CI job ``test-postgres`` provides a PostgreSQL 18
service container and runs this module 5 times (concurrency tests must not pass by chance). Every test
works on its own uniquely named study and deletes it afterwards.

Covered with real processes (``spawn``) on the real server: workers sharing one study within the global
budget bound (R60) and without duplicated parameters (per-worker seeds, P2-3b); a killed worker's trial
failed by the heartbeat and re-run once (P2-2b); the study lifecycle (``init_study``, shared-study
rules) and URL redaction on an authentication failure (F10).
"""

import importlib.util
import multiprocessing
import os
import time
import uuid

import optuna
import pytest
from optuna.trial import TrialState
from sqlalchemy.engine import make_url

from milia_pipeline.exceptions import BackendError
from milia_pipeline.models.hpo.backends.storage_factory import build_storage
from milia_pipeline.models.hpo.hpo_config import HPOConfig, StorageConfig, StudyConfig
from milia_pipeline.models.hpo.seeding import derive_worker_seed

ENV = "MILIA_TEST_POSTGRES_URL"
# CI sets this so a missing server URL or driver FAILS the job instead of skipping it (no false green)
REQUIRED = os.environ.get("MILIA_REQUIRE_POSTGRES_TESTS") == "1"

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.integration,
    pytest.mark.skipif(
        not REQUIRED and not os.environ.get(ENV), reason=f"{ENV} not set (PostgreSQL integration)"
    ),
    pytest.mark.skipif(
        not REQUIRED and importlib.util.find_spec("psycopg") is None,
        reason="hpo-postgres extra not installed",
    ),
    pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning"),
]


def _study_config(**heartbeat) -> StudyConfig:
    return StudyConfig(storage_options=StorageConfig(kind="rdb", url_env=ENV, **heartbeat))


@pytest.fixture
def study_name():
    """A unique study per test; deleted afterwards so repeated runs never see old trials."""
    name = f"p2-5-{uuid.uuid4().hex[:12]}"
    yield name
    storage = build_storage(_study_config())
    if name in optuna.study.get_all_study_names(storage):
        optuna.delete_study(study_name=name, storage=storage)


def _worker(name: str, worker_index: int, budget: int) -> None:
    """One worker process: seeded TPE (constant liar) on the shared study, global budget."""
    from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    storage = build_storage(_study_config(heartbeat_interval=1))
    sampler = optuna.samplers.TPESampler(
        seed=derive_worker_seed(42, worker_index), constant_liar=True, n_startup_trials=3
    )
    study = optuna.load_study(study_name=name, storage=storage, sampler=sampler)

    def objective(trial):
        x = trial.suggest_float("x", -10.0, 10.0)
        time.sleep(0.2)  # trials of different workers overlap
        return (x - 2.0) ** 2

    OptunaBackend().optimize(study, objective, n_trials=100, n_trials_total=budget)


def _run_processes(target, args_list, timeout=300):
    # "spawn": a forked child of a multi-threaded pytest process can deadlock (R60)
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=target, args=args) for args in args_list]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=timeout)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
    return [process.exitcode for process in processes]


def test_workers_share_one_study_within_budget(study_name):
    budget, workers = 12, 3
    optuna.create_study(study_name=study_name, storage=build_storage(_study_config()))
    exit_codes = _run_processes(_worker, [(study_name, i, budget) for i in range(workers)])
    assert exit_codes == [0] * workers
    study = optuna.load_study(study_name=study_name, storage=build_storage(_study_config()))
    states = [t.state for t in study.trials]
    finished = states.count(TrialState.COMPLETE) + states.count(TrialState.PRUNED)
    assert budget <= finished <= budget + workers - 1  # R60 bound
    assert TrialState.RUNNING not in states and TrialState.WAITING not in states
    params = [t.params["x"] for t in study.trials]
    assert len(set(params)) == len(params)  # per-worker seeds: no repeated configuration


def _hanging_worker(name: str, ready_path: str) -> None:
    """Starts a trial, heart-beats, then hangs until it is killed (SIGKILL: no clean-up runs)."""
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    storage = build_storage(_study_config(heartbeat_interval=1, grace_period=2, max_retry=1))
    study = optuna.load_study(study_name=name, storage=storage)

    def objective(trial):
        trial.suggest_float("x", 0.0, 1.0)
        with open(ready_path, "w", encoding="utf-8") as handle:
            handle.write(str(trial.number))
        time.sleep(600)
        return 0.0

    study.optimize(objective, n_trials=1)


def test_killed_worker_trial_is_failed_and_retried(study_name, tmp_path):
    config = _study_config(heartbeat_interval=1, grace_period=2, max_retry=1)
    optuna.create_study(study_name=study_name, storage=build_storage(config))
    ready = tmp_path / "ready"
    process = multiprocessing.get_context("spawn").Process(
        target=_hanging_worker, args=(study_name, str(ready))
    )
    process.start()
    try:
        deadline = time.monotonic() + 120
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.2)
        assert ready.exists(), "worker never started its trial"
        time.sleep(1.5)  # at least one heartbeat recorded
    finally:
        process.kill()  # SIGKILL
        process.join()
    time.sleep(2 + 1)  # grace_period + margin (PostgreSQL timestamps have microsecond precision)
    live = optuna.load_study(study_name=study_name, storage=build_storage(config))
    live.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=1)
    states = [(t.number, t.state) for t in live.trials]
    assert (0, TrialState.FAIL) in states
    retried = [
        t
        for t in live.trials
        if optuna.storages.RetryHeartbeatStaleTrialCallback.retried_trial_number(t) == 0
    ]
    assert len(retried) == 1


def test_init_study_and_shared_study_rules(study_name):
    from milia_pipeline.models.hpo.backends.optuna_backend import _METRIC_USER_ATTR
    from milia_pipeline.models.hpo.hpo_manager import HPOManager
    from milia_pipeline.models.hpo.shared_study import validate_shared_study

    options = StorageConfig(kind="rdb", url_env=ENV, heartbeat_interval=60)
    config = HPOConfig(
        enabled=True,
        study=StudyConfig(study_name=study_name, metric="val_mae", storage_options=options),
    )
    validate_shared_study(config)  # PostgreSQL + heartbeat: accepted
    HPOManager(config).init_study()
    HPOManager(config).init_study()  # re-running the initializer loads the same study
    stored = optuna.load_study(study_name=study_name, storage=build_storage(config.study))
    assert stored.user_attrs[_METRIC_USER_ATTR] == "val_mae"
    assert len(stored.trials) == 0


def test_authentication_failure_never_exposes_password(monkeypatch):
    secret = "wrong-" + uuid.uuid4().hex  # pragma: allowlist secret
    bad = make_url(os.environ[ENV]).set(password=secret).render_as_string(hide_password=False)
    monkeypatch.setenv(ENV, bad)
    with pytest.raises(BackendError) as excinfo:
        build_storage(_study_config())
    assert secret not in str(excinfo.value)
    assert secret not in repr(vars(excinfo.value))
