"""Model plugin directories: configuration root, user paths and the built-in container."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from milia_pipeline.config import get_config_root
from milia_pipeline.exceptions import ConfigurationError
from milia_pipeline.models import model_plugin_system as mps
from milia_pipeline.models._model_plugin_paths import (
    builtin_model_plugins_directory,
    resolve_model_plugin_paths,
)
from milia_pipeline.models.utils import config_bridge
from milia_pipeline.models.utils.config_bridge import PluginsConfig

REPO_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def config_root(tmp_path):
    root = tmp_path / "project" / "configs"
    root.mkdir(parents=True)
    return root


@pytest.fixture
def elsewhere(tmp_path, monkeypatch):
    """A working directory unrelated to the configuration root."""
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


# --- configuration root ------------------------------------------------------


def test_config_root_of_a_directory_is_the_directory(config_root):
    assert get_config_root(config_root) == config_root.resolve()


def test_config_root_of_a_file_is_its_parent(config_root):
    config_file = config_root / "config.yaml"
    config_file.write_text("models: {}\n")
    assert get_config_root(str(config_file)) == config_root.resolve()


def test_config_root_defaults_to_the_configs_directory(config_root, monkeypatch):
    monkeypatch.chdir(config_root.parent)
    root = get_config_root()
    assert root == config_root.resolve() and root.is_absolute()


def test_config_root_without_a_configs_directory_raises(elsewhere):
    with pytest.raises(ConfigurationError):
        get_config_root()


# --- user paths ----------------------------------------------------------------


def test_absolute_path_is_used_as_given(config_root, tmp_path, elsewhere):
    plugins = tmp_path / "abs_plugins"
    plugins.mkdir()
    assert resolve_model_plugin_paths([str(plugins)], config_root, include_builtin=False) == [
        plugins.resolve()
    ]


def test_home_is_expanded(config_root, tmp_path, monkeypatch, elsewhere):
    home = tmp_path / "home"
    (home / "my_plugins").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    assert resolve_model_plugin_paths(["~/my_plugins"], config_root, include_builtin=False) == [
        (home / "my_plugins").resolve()
    ]


def test_relative_path_resolves_against_the_config_root_not_the_cwd(config_root, elsewhere):
    (config_root.parent / "my_plugins").mkdir()
    (elsewhere / "my_plugins").mkdir()  # must not be chosen
    with_parent = resolve_model_plugin_paths(["../my_plugins"], config_root, include_builtin=False)
    assert with_parent == [(config_root.parent / "my_plugins").resolve()]

    (config_root / "my_plugins").mkdir()
    assert resolve_model_plugin_paths(["my_plugins"], config_root, include_builtin=False) == [
        (config_root / "my_plugins").resolve()
    ]


def test_cwd_relative_fallback_is_deprecated(config_root, elsewhere, caplog):
    (elsewhere / "legacy_plugins").mkdir()
    with caplog.at_level(logging.WARNING), pytest.warns(DeprecationWarning) as record:
        paths = resolve_model_plugin_paths(["legacy_plugins"], config_root, include_builtin=False)

    assert paths == [(elsewhere / "legacy_plugins").resolve()]
    (warning,) = [w for w in record if issubclass(w.category, DeprecationWarning)]
    assert warning.filename == str(config_root.resolve())
    assert "will be removed in MILIA 2.0.0" in str(warning.message)
    assert "relative to the working directory" in caplog.text


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_path_that_is_not_a_directory_is_logged_and_skipped(config_root, elsewhere, caplog, kind):
    if kind == "file":
        (config_root / "plugins").write_text("")
    with caplog.at_level(logging.WARNING):
        assert resolve_model_plugin_paths(["plugins"], config_root, include_builtin=False) == []
    reason = "is not a directory" if kind == "file" else "does not exist"
    assert f"Model plugin path 'plugins' skipped: {config_root.resolve() / 'plugins'} {reason}" in (
        caplog.text
    )


# --- built-in container and order --------------------------------------------------


def test_builtin_container_is_first_and_listed_once(config_root, tmp_path, elsewhere):
    builtin = builtin_model_plugins_directory()
    assert builtin.is_dir() and builtin.as_posix().endswith("milia_pipeline/plugins/models")

    (config_root / "a").mkdir()
    paths = resolve_model_plugin_paths(
        ["a", str(builtin), str(config_root / "a"), "./a"], config_root
    )
    assert paths == [builtin, (config_root / "a").resolve()]


def test_include_builtin_false_leaves_the_builtin_out(config_root, elsewhere):
    assert resolve_model_plugin_paths([], config_root, include_builtin=False) == []
    assert resolve_model_plugin_paths([], config_root) == [builtin_model_plugins_directory()]


# --- the configured paths ------------------------------------------------------------


def test_configured_paths_use_the_config_root(config_root, elsewhere, monkeypatch):
    (config_root / "user_plugins").mkdir()
    seen = []

    def fake_plugins_config(config_path=None):
        seen.append(config_path)
        return PluginsConfig(plugin_paths=["user_plugins"])

    monkeypatch.setattr(config_bridge, "get_plugins_config", fake_plugins_config)
    assert mps._configured_plugin_paths(config_root) == [
        builtin_model_plugins_directory(),
        (config_root / "user_plugins").resolve(),
    ]
    assert seen == [config_root]


def test_configured_paths_fall_back_to_the_builtin_container(monkeypatch, caplog, elsewhere):
    def failing_plugins_config(config_path=None):
        raise ConfigurationError("no configuration")

    monkeypatch.setattr(config_bridge, "get_plugins_config", failing_plugins_config)
    with caplog.at_level(logging.WARNING):
        assert mps._configured_plugin_paths() == [builtin_model_plugins_directory()]
    assert "no configuration; using the built-in container" in caplog.text


def test_shipped_configuration_scans_the_builtin_container_only():
    """The shipped configs/ keeps the effective scan set: the built-in container, once."""
    assert mps._configured_plugin_paths(REPO_CONFIGS) == [builtin_model_plugins_directory()]
