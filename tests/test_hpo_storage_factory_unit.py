#!/usr/bin/env python3
"""
Contract tests for ``milia_pipeline.models.hpo.backends.storage_factory`` (P2-2a, blueprint S2).

``build_storage(StudyConfig)`` is the single construction point of the study storage: the legacy
``storage`` URL is returned unchanged; ``storage_options`` builds a real Optuna ``RDBStorage`` or
``JournalStorage``. All storages here are real and live in ``tmp_path``.
"""

import os
import time
import warnings
from unittest.mock import patch

import optuna
import pytest
from optuna.exceptions import ExperimentalWarning
from optuna.trial import TrialState

from milia_pipeline.exceptions import BackendError, HPOConfigurationError, StudyNotFoundError
from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend
from milia_pipeline.models.hpo.backends.storage_factory import build_storage
from milia_pipeline.models.hpo.hpo_config import StorageConfig, StudyConfig

ENV = "MILIA_TEST_HPO_STORAGE_URL"
# SQLite CURRENT_TIMESTAMP is "YYYY-MM-DD HH:MM:SS" (whole seconds), and Optuna stamps heartbeats and
# "now" with the database clock, so measured heartbeat ages are whole-second differences (R57).
SQLITE_TIMESTAMP_RESOLUTION_S = 1


@pytest.fixture(autouse=True)
def _quiet_optuna():
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    yield
    optuna.logging.set_verbosity(verbosity)


def _rdb_study(engine_kwargs=None) -> StudyConfig:
    return StudyConfig(
        storage_options=StorageConfig(kind="rdb", url_env=ENV, engine_kwargs=engine_kwargs)
    )


def _journal_study(path) -> StudyConfig:
    return StudyConfig(storage_options=StorageConfig(kind="journal_file", journal_path=str(path)))


def _run_trials(storage, name: str, n: int) -> None:
    study = optuna.create_study(study_name=name, storage=storage, load_if_exists=True)
    study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=n)


@pytest.mark.contract
class TestLegacyAndInMemory:
    def test_in_memory_is_none(self):
        assert build_storage(StudyConfig()) is None

    def test_storage_url_returned_unchanged(self):
        assert build_storage(StudyConfig(storage="sqlite:///hpo.db")) == "sqlite:///hpo.db"


@pytest.mark.contract
class TestRDBStorage:
    def test_builds_rdb_storage_from_environment(self, tmp_path):
        url = f"sqlite:///{tmp_path / 'hpo.db'}"
        with patch.dict(os.environ, {ENV: url}):
            storage = build_storage(_rdb_study())
        assert isinstance(storage, optuna.storages.RDBStorage)
        assert storage.url == url

    def test_trials_shared_between_independently_built_storages(self, tmp_path):
        """Two builds (as two worker processes would do) see the same persisted study."""
        with patch.dict(os.environ, {ENV: f"sqlite:///{tmp_path / 'hpo.db'}"}):
            _run_trials(build_storage(_rdb_study()), "shared", 2)
            reloaded = optuna.load_study(study_name="shared", storage=build_storage(_rdb_study()))
        assert len(reloaded.trials) == 2

    def test_engine_kwargs_forwarded_and_config_not_mutated(self, tmp_path):
        engine_kwargs = {"connect_args": {"timeout": 30}}
        study = _rdb_study(engine_kwargs)
        with patch.dict(os.environ, {ENV: f"sqlite:///{tmp_path / 'hpo.db'}"}):
            storage = build_storage(study)
        assert storage.engine_kwargs == engine_kwargs
        assert storage.engine_kwargs is not study.storage_options.engine_kwargs
        assert study.storage_options.engine_kwargs == {"connect_args": {"timeout": 30}}

    def test_invalid_engine_kwargs_is_configuration_error(self, tmp_path):
        with (
            patch.dict(os.environ, {ENV: f"sqlite:///{tmp_path / 'hpo.db'}"}),
            pytest.raises(HPOConfigurationError) as exc_info,
        ):
            build_storage(_rdb_study({"not_a_create_engine_argument": 1}))
        assert exc_info.value.config_key == "models.hpo.study.storage_options.engine_kwargs"

    def test_unset_variable_is_configuration_error(self):
        environ = {k: v for k, v in os.environ.items() if k != ENV}
        with patch.dict(os.environ, environ, clear=True), pytest.raises(HPOConfigurationError):
            build_storage(_rdb_study())

    def test_open_failure_never_exposes_password(self, tmp_path):
        """A missing driver / unreachable server surfaces as BackendError without the credential."""
        url = "postgresql+psycopg://milia:s3cr3t-pw@127.0.0.1:1/hpo"  # pragma: allowlist secret
        with patch.dict(os.environ, {ENV: url}), pytest.raises(BackendError) as exc_info:
            build_storage(_rdb_study())
        assert "s3cr3t-pw" not in str(exc_info.value)
        assert "milia:***@127.0.0.1" in str(exc_info.value)


def _heartbeat_study(**heartbeat) -> StudyConfig:
    return StudyConfig(storage_options=StorageConfig(kind="rdb", url_env=ENV, **heartbeat))


@pytest.mark.contract
class TestHeartbeat:
    """P2-2b: heartbeat wiring into RDBStorage and stale-trial recovery."""

    def test_disabled_by_default(self, tmp_path):
        with patch.dict(os.environ, {ENV: f"sqlite:///{tmp_path / 'hpo.db'}"}):
            storage = build_storage(_rdb_study())
        assert storage.heartbeat_interval is None
        assert storage.get_heartbeat_stale_trial_callback() is None

    def test_interval_and_grace_forwarded_without_retry(self, tmp_path):
        with patch.dict(os.environ, {ENV: f"sqlite:///{tmp_path / 'hpo.db'}"}):
            storage = build_storage(_heartbeat_study(heartbeat_interval=60, grace_period=180))
        assert (storage.heartbeat_interval, storage.grace_period) == (60, 180)
        assert storage.get_heartbeat_stale_trial_callback() is None

    def test_max_retry_attaches_bounded_retry_callback(self, tmp_path):
        with patch.dict(os.environ, {ENV: f"sqlite:///{tmp_path / 'hpo.db'}"}):
            storage = build_storage(_heartbeat_study(heartbeat_interval=60, max_retry=3))
        callback = storage.get_heartbeat_stale_trial_callback()
        assert isinstance(callback, optuna.storages.RetryHeartbeatStaleTrialCallback)
        assert callback._max_retry == 3
        assert storage.grace_period is None  # Optuna default: 2 * heartbeat_interval

    def test_no_experimental_warning_escapes(self, tmp_path):
        with (
            patch.dict(os.environ, {ENV: f"sqlite:///{tmp_path / 'hpo.db'}"}),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            build_storage(_heartbeat_study(heartbeat_interval=60, max_retry=1))
        assert not [w for w in caught if issubclass(w.category, ExperimentalWarning)]

    @pytest.mark.integration
    @pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
    @pytest.mark.filterwarnings("ignore:Heartbeat of storage is supposed to be used:UserWarning")
    def test_stale_trial_failed_and_retried(self, tmp_path):
        """A trial whose worker stopped heart-beating is FAILed by the next worker and re-run once."""
        grace_period = 2
        study_config = _heartbeat_study(
            heartbeat_interval=1, grace_period=grace_period, max_retry=1
        )
        with patch.dict(os.environ, {ENV: f"sqlite:///{tmp_path / 'hpo.db'}"}):
            dead_worker = build_storage(study_config)
            study = optuna.create_study(study_name="hb", storage=dead_worker)
            orphan = study.ask()  # the "killed" worker: one heartbeat, then silence
            dead_worker.record_heartbeat(orphan._trial_id)
            # Stale means age > grace. With whole-second stamps the measured age of a real wait d is
            # >= floor(d); waiting grace + resolution makes it >= grace + 1 whatever the sub-second
            # phase (a 2.5 s wait measured 2 s when the heartbeat fell early in a second — R57).
            time.sleep(grace_period + SQLITE_TIMESTAMP_RESOLUTION_S)
            live = optuna.load_study(study_name="hb", storage=build_storage(study_config))
            live.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=1)
        states = [(t.number, t.state) for t in live.trials]
        assert states == [(0, TrialState.FAIL), (1, TrialState.COMPLETE)]
        assert (
            optuna.storages.RetryHeartbeatStaleTrialCallback.retried_trial_number(live.trials[1])
            == 0
        )


@pytest.mark.contract
class TestJournalFileStorage:
    def test_builds_journal_storage(self, tmp_path):
        storage = build_storage(_journal_study(tmp_path / "hpo_journal.log"))
        assert isinstance(storage, optuna.storages.JournalStorage)
        assert (tmp_path / "hpo_journal.log").exists()

    def test_trials_shared_between_independently_built_storages(self, tmp_path):
        path = tmp_path / "hpo_journal.log"
        _run_trials(build_storage(_journal_study(path)), "shared", 2)
        reloaded = optuna.load_study(
            study_name="shared", storage=build_storage(_journal_study(path))
        )
        assert len(reloaded.trials) == 2

    def test_missing_directory_is_backend_error(self, tmp_path):
        with pytest.raises(BackendError, match="journal storage"):
            build_storage(_journal_study(tmp_path / "missing_dir" / "hpo_journal.log"))


@pytest.mark.contract
class TestBackendAcceptsBuiltStorage:
    """``OptunaBackend.create_study`` creates, resumes and guards studies on built storages."""

    def test_create_then_resume_on_journal_storage(self, tmp_path):
        backend = OptunaBackend()
        path = tmp_path / "hpo_journal.log"
        created = backend.create_study(
            "s", "minimize", storage=build_storage(_journal_study(path)), metric="val_loss"
        )
        created.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=2)
        resumed = backend.create_study(
            "s",
            "minimize",
            storage=build_storage(_journal_study(path)),
            metric="val_loss",
            must_exist=True,
        )
        assert len(resumed.trials) == 2

    def test_resume_missing_study_names_storage_type_only(self, tmp_path):
        backend = OptunaBackend()
        with pytest.raises(StudyNotFoundError) as exc_info:
            backend.create_study(
                "absent",
                "minimize",
                storage=build_storage(_journal_study(tmp_path / "hpo_journal.log")),
                must_exist=True,
            )
        assert exc_info.value.storage_url == "<JournalStorage>"
