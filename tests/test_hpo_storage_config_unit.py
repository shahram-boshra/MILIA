#!/usr/bin/env python3
"""
Contract tests for HPO study storage selection (P2-1, blueprint S1 / F46).

``StudyConfig.storage_options`` (a ``StorageConfig``) names an environment variable that holds the
storage URL. The URL is read when HPO starts and is never stored on the configuration, so dumps and
saved configs carry only the variable name. ``StudyConfig.storage`` (URL string) is unchanged and the
two are mutually exclusive.
"""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from milia_pipeline.exceptions import HPOConfigurationError
from milia_pipeline.models.hpo.hpo_config import HPOConfig, StorageConfig, StudyConfig

ENV = "MILIA_TEST_HPO_STORAGE_URL"
SECRET_URL = "postgresql+psycopg://milia:s3cr3t-pw@db.example:5432/hpo"  # pragma: allowlist secret


def _storage_options(url_env: str = ENV) -> StorageConfig:
    return StorageConfig(kind="rdb", url_env=url_env)


@pytest.mark.contract
class TestStorageConfig:
    """``StorageConfig`` validation and URL resolution."""

    @pytest.mark.parametrize("name", [ENV, "_PRIVATE", "lower_case_ok", "A1_B2"])
    def test_accepts_portable_variable_names(self, name):
        assert StorageConfig(kind="rdb", url_env=name).url_env == name

    @pytest.mark.parametrize("name", ["", "1LEADING_DIGIT", "HAS-DASH", "HAS SPACE", "A.B", "$ENV"])
    def test_rejects_invalid_variable_names(self, name):
        with pytest.raises(ValidationError, match="url_env must be an environment variable name"):
            StorageConfig(kind="rdb", url_env=name)

    @pytest.mark.parametrize("kind", ["inmemory", "grpc_proxy", "RDB", "journal"])
    def test_rejects_kinds_without_an_implementation(self, kind):
        with pytest.raises(ValidationError):
            StorageConfig(kind=kind, url_env=ENV)

    def test_rejects_unknown_fields(self):
        """Fields of later paces are not silently accepted (F46)."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            StorageConfig(kind="rdb", url_env=ENV, heartbeat_interval=60)

    def test_requires_kind_and_url_env(self):
        with pytest.raises(ValidationError):
            StorageConfig(kind="rdb")
        with pytest.raises(ValidationError):
            StorageConfig(url_env=ENV)

    def test_is_frozen(self):
        config = _storage_options()
        with pytest.raises(ValidationError):
            config.url_env = "OTHER"

    def test_resolve_url_reads_given_mapping(self):
        assert _storage_options().resolve_url({ENV: "sqlite:///hpo.db"}) == "sqlite:///hpo.db"

    def test_resolve_url_reads_process_environment_at_call_time(self):
        config = _storage_options()
        with patch.dict(os.environ, {ENV: "sqlite:///first.db"}):
            assert config.resolve_url() == "sqlite:///first.db"
        with patch.dict(os.environ, {ENV: "sqlite:///second.db"}):
            assert config.resolve_url() == "sqlite:///second.db"

    @pytest.mark.parametrize("environ", [{}, {ENV: ""}])
    def test_unset_or_empty_variable_raises_naming_the_variable(self, environ):
        with pytest.raises(HPOConfigurationError) as exc_info:
            _storage_options().resolve_url(environ)
        assert ENV in str(exc_info.value)
        assert exc_info.value.config_key == "models.hpo.study.storage_options.url_env"


@pytest.mark.contract
class TestStorageKindFields:
    """P2-2a: each kind requires its own fields and rejects the fields of other kinds."""

    def test_journal_file_accepted(self):
        config = StorageConfig(kind="journal_file", journal_path="hpo_journal.log")
        assert config.journal_path == "hpo_journal.log"
        assert config.url_env is None

    def test_rdb_accepts_engine_kwargs(self):
        kwargs = {"pool_pre_ping": True, "connect_args": {"timeout": 30}}
        assert StorageConfig(kind="rdb", url_env=ENV, engine_kwargs=kwargs).engine_kwargs == kwargs

    @pytest.mark.parametrize(
        ("fields", "message"),
        [
            ({"kind": "rdb"}, "requires: url_env"),
            ({"kind": "journal_file"}, "requires: journal_path"),
            (
                {"kind": "rdb", "url_env": ENV, "journal_path": "j.log"},
                "does not accept: journal_path",
            ),
            (
                {"kind": "journal_file", "journal_path": "j.log", "url_env": ENV},
                "does not accept: url_env",
            ),
            (
                {"kind": "journal_file", "journal_path": "j.log", "engine_kwargs": {}},
                "does not accept: engine_kwargs",
            ),
            ({"kind": "journal_file", "journal_path": "  "}, "journal_path cannot be empty"),
        ],
    )
    def test_field_rules_per_kind(self, fields, message):
        with pytest.raises(ValidationError, match=message):
            StorageConfig(**fields)

    def test_journal_file_has_no_url(self):
        options = StorageConfig(kind="journal_file", journal_path="j.log")
        with pytest.raises(HPOConfigurationError, match="has no storage URL"):
            options.resolve_url({})
        study = StudyConfig(storage_options=options)
        assert study.has_persistent_storage is True
        assert study.resolve_storage_url({}) is None


@pytest.mark.contract
class TestStudyConfigStorageOptions:
    """``StudyConfig.storage_options`` beside the unchanged ``storage`` URL."""

    def test_default_is_in_memory(self):
        config = StudyConfig()
        assert config.storage is None
        assert config.storage_options is None
        assert config.has_persistent_storage is False
        assert config.resolve_storage_url() is None

    def test_storage_url_unchanged(self):
        config = StudyConfig(storage="sqlite:///hpo.db")
        assert config.has_persistent_storage is True
        assert config.resolve_storage_url({ENV: "sqlite:///ignored.db"}) == "sqlite:///hpo.db"

    def test_storage_options_resolve_from_environment(self):
        config = StudyConfig(storage_options=_storage_options())
        assert config.has_persistent_storage is True
        assert config.resolve_storage_url({ENV: "sqlite:///hpo.db"}) == "sqlite:///hpo.db"

    def test_storage_and_storage_options_are_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="not both"):
            StudyConfig(storage="sqlite:///hpo.db", storage_options=_storage_options())

    def test_from_dict_parses_nested_storage_options(self):
        config = HPOConfig.from_dict(
            {"study": {"storage_options": {"kind": "rdb", "url_env": ENV}}}
        )
        assert config.study.storage_options == _storage_options()

    def test_from_dict_rejects_invalid_storage_options(self):
        with pytest.raises(ValidationError):
            HPOConfig.from_dict({"study": {"storage_options": {"kind": "rdb", "url_env": "1X"}}})

    def test_resolved_url_never_stored_in_dumps(self):
        """The credential-bearing URL appears in no dump of the configuration (F10, E75)."""
        with patch.dict(os.environ, {ENV: SECRET_URL}):
            config = HPOConfig.from_dict(
                {"study": {"storage_options": {"kind": "rdb", "url_env": ENV}}}
            )
            assert config.study.resolve_storage_url() == SECRET_URL
            for dump in (
                str(config.to_dict()),
                config.model_dump_json(),
                repr(config),
                str(config.study.to_dict()),
            ):
                assert "s3cr3t-pw" not in dump
                assert ENV in dump


@pytest.mark.contract
class TestManagerUsesResolvedUrl:
    """``HPOManager.optimize`` passes the storage built at call time to the optimization run."""

    def test_optimize_passes_built_storage(self):
        """P2-2a: the storage comes from ``build_storage(config.study)`` (URL read from the env there)."""
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        config = HPOConfig(enabled=True, study=StudyConfig(storage_options=_storage_options()))
        with patch("milia_pipeline.models.hpo.hpo_manager.get_backend"):
            manager = HPOManager(config)
        built = object()
        with (
            patch.object(HPOManager, "_run_optimization", return_value={}) as run,
            patch(
                "milia_pipeline.models.hpo.hpo_manager.build_storage", return_value=built
            ) as build,
        ):
            manager.optimize(model_name="GCN", dataset=[])
        build.assert_called_once_with(config.study)
        assert run.call_args.kwargs["storage"] is built

    def test_optimize_legacy_storage_url_unchanged(self):
        """The ``storage`` URL string still reaches the run unchanged (expand phase)."""
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        config = HPOConfig(enabled=True, study=StudyConfig(storage="sqlite:///hpo.db"))
        with patch("milia_pipeline.models.hpo.hpo_manager.get_backend"):
            manager = HPOManager(config)
        with patch.object(HPOManager, "_run_optimization", return_value={}) as run:
            manager.optimize(model_name="GCN", dataset=[])
        assert run.call_args.kwargs["storage"] == "sqlite:///hpo.db"

    def test_optimize_fails_fast_when_variable_unset(self):
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        config = HPOConfig(enabled=True, study=StudyConfig(storage_options=_storage_options()))
        with patch("milia_pipeline.models.hpo.hpo_manager.get_backend"):
            manager = HPOManager(config)
        environ = {k: v for k, v in os.environ.items() if k != ENV}
        with (
            patch.object(HPOManager, "_run_optimization") as run,
            patch.dict(os.environ, environ, clear=True),
            pytest.raises(HPOConfigurationError, match=ENV),
        ):
            manager.optimize(model_name="GCN", dataset=[])
        run.assert_not_called()
