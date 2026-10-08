#!/usr/bin/env python3
"""
Contract tests for ``storage_options.url_file`` (P2-4a, blueprint S1 / S4).

An ``rdb`` storage takes its URL from exactly one of ``url_env`` (environment variable) or ``url_file``
(a file, e.g. a Docker secret mounted at ``/run/secrets/<name>``). Docker documents secrets as the way
to avoid "unintentional information exposure" of passwords passed in environment variables. The file
is read when HPO starts; its content is never stored on the config and never appears in an error.
"""

import optuna
import pytest
from pydantic import ValidationError

from milia_pipeline.exceptions import HPOConfigurationError
from milia_pipeline.models.hpo.backends.storage_factory import build_storage
from milia_pipeline.models.hpo.hpo_config import HPOConfig, StorageConfig, StudyConfig
from milia_pipeline.models.hpo.shared_study import validate_shared_study

SECRET = "s3cr3t-pw"  # pragma: allowlist secret
SERVER_URL = f"postgresql+psycopg://milia:{SECRET}@db.example:5432/hpo"


def _url_file(tmp_path, content, name="hpo_storage_url"):
    path = tmp_path / name
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return str(path)


@pytest.mark.contract
class TestConfig:
    def test_rdb_accepts_url_file(self):
        config = StorageConfig(kind="rdb", url_file="/run/secrets/hpo_storage_url")
        assert config.url_file == "/run/secrets/hpo_storage_url"
        assert config.url_env is None

    @pytest.mark.parametrize(
        ("fields", "message"),
        [
            ({"kind": "rdb"}, "exactly one of: url_env, url_file"),
            (
                {"kind": "rdb", "url_env": "X", "url_file": "/f"},
                "exactly one of: url_env, url_file",
            ),
            ({"kind": "rdb", "url_file": "   "}, "url_file cannot be empty"),
            (
                {"kind": "journal_file", "journal_path": "j.log", "url_file": "/f"},
                "does not accept: url_file",
            ),
        ],
        ids=["none", "both", "blank", "journal"],
    )
    def test_rejected(self, fields, message):
        with pytest.raises(ValidationError, match=message):
            StorageConfig(**fields)

    def test_dump_carries_the_path_not_the_url(self, tmp_path):
        path = _url_file(tmp_path, SERVER_URL)
        config = StorageConfig(kind="rdb", url_file=path)
        assert config.resolve_url() == SERVER_URL
        assert config.model_dump()["url_file"] == path
        assert SECRET not in repr(config.model_dump())


@pytest.mark.contract
class TestResolve:
    @pytest.mark.parametrize("content", [SERVER_URL, SERVER_URL + "\n", f"  {SERVER_URL}\r\n"])
    def test_surrounding_whitespace_removed(self, tmp_path, content):
        config = StorageConfig(kind="rdb", url_file=_url_file(tmp_path, content))
        assert config.resolve_url() == SERVER_URL

    def test_environment_not_consulted(self, tmp_path):
        config = StorageConfig(kind="rdb", url_file=_url_file(tmp_path, SERVER_URL))
        assert config.resolve_url({"MILIA_HPO_STORAGE_URL": "sqlite:///other.db"}) == SERVER_URL

    def test_study_config_resolves_from_file(self, tmp_path):
        options = StorageConfig(kind="rdb", url_file=_url_file(tmp_path, SERVER_URL))
        assert StudyConfig(storage_options=options).resolve_storage_url() == SERVER_URL

    @pytest.mark.parametrize(
        ("content", "message"),
        [(None, "cannot be read"), ("", "is empty"), (" \n", "is empty"), (b"\xff\xfe", "UTF-8")],
        ids=["missing", "empty", "whitespace", "not-utf8"],
    )
    def test_unusable_file_rejected_without_content(self, tmp_path, content, message):
        path = str(tmp_path / "absent") if content is None else _url_file(tmp_path, content)
        with pytest.raises(HPOConfigurationError, match=message) as excinfo:
            StorageConfig(kind="rdb", url_file=path).resolve_url()
        assert path in str(excinfo.value)
        assert excinfo.value.config_key == "models.hpo.study.storage_options.url_file"
        assert excinfo.value.__cause__ is None

    def test_partial_secret_never_echoed(self, tmp_path):
        # A file whose bytes are not UTF-8 after the password: the error must not quote the content
        path = _url_file(tmp_path, SECRET.encode() + b"\xff")
        with pytest.raises(HPOConfigurationError) as excinfo:
            StorageConfig(kind="rdb", url_file=path).resolve_url()
        assert SECRET not in str(excinfo.value)
        assert SECRET not in repr(vars(excinfo.value))


@pytest.mark.contract
class TestConsumers:
    def test_build_storage_opens_rdb_from_file(self, tmp_path):
        url = f"sqlite:///{tmp_path / 'hpo.db'}"
        options = StorageConfig(kind="rdb", url_file=_url_file(tmp_path, url + "\n"))
        storage = build_storage(StudyConfig(storage_options=options))
        assert isinstance(storage, optuna.storages.RDBStorage)
        optuna.create_study(study_name="from_file", storage=storage)
        assert optuna.study.get_all_study_names(storage) == ["from_file"]

    def test_shared_study_reads_the_file(self, tmp_path):
        server = StorageConfig(
            kind="rdb", url_file=_url_file(tmp_path, SERVER_URL), heartbeat_interval=60
        )
        validate_shared_study(HPOConfig(study=StudyConfig(storage_options=server)), {})
        sqlite = StorageConfig(
            kind="rdb",
            url_file=_url_file(tmp_path, "sqlite:///hpo.db", name="sqlite_url"),
            heartbeat_interval=60,
        )
        with pytest.raises(HPOConfigurationError, match="SQLite"):
            validate_shared_study(HPOConfig(study=StudyConfig(storage_options=sqlite)), {})
