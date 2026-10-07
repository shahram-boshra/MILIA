#!/usr/bin/env python3
"""
Contract tests for ``milia_pipeline.models.hpo.backends.storage_factory`` (P2-2a, blueprint S2).

``build_storage(StudyConfig)`` is the single construction point of the study storage: the legacy
``storage`` URL is returned unchanged; ``storage_options`` builds a real Optuna ``RDBStorage`` or
``JournalStorage``. All storages here are real and live in ``tmp_path``.
"""

import os
from unittest.mock import patch

import optuna
import pytest

from milia_pipeline.exceptions import BackendError, HPOConfigurationError, StudyNotFoundError
from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend
from milia_pipeline.models.hpo.backends.storage_factory import build_storage
from milia_pipeline.models.hpo.hpo_config import StorageConfig, StudyConfig

ENV = "MILIA_TEST_HPO_STORAGE_URL"


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
