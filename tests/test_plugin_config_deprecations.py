"""Deprecated transformation-plugin configuration keys: each warns when set; none is shipped."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest import mock

import pytest
import yaml

import main
from milia_pipeline.config import (
    DEPRECATED_PLUGIN_CONFIG_KEYS,
    PLUGIN_CONFIG_REMOVAL_VERSION,
    PluginSchemaValidator,
    PluginValidationLevel,
    deprecated_plugin_config_keys,
    get_plugin_config_summary,
)
from milia_pipeline.transformations.plugin_system import PluginRegistry

REPO = Path(__file__).resolve().parents[1]


def _run_main_plugin_setup(plugins_config: dict) -> None:
    with mock.patch(
        "milia_pipeline.config.config_loader.load_config", return_value={"plugins": plugins_config}
    ):
        main._discover_and_register_plugins(logging.getLogger(__name__))


@pytest.mark.parametrize("key", sorted(DEPRECATED_PLUGIN_CONFIG_KEYS))
def test_setting_a_deprecated_key_warns(key, caplog):
    """A FutureWarning (and a log record) names the key and the removal release.

    The warning is emitted even with plugins disabled: the key has no effect either way.
    """
    with pytest.warns(FutureWarning, match=rf"plugins\.{key} is deprecated") as record:
        _run_main_plugin_setup({"enabled": False, key: None})

    (message,) = [str(w.message) for w in record if f"plugins.{key} " in str(w.message)]
    assert PLUGIN_CONFIG_REMOVAL_VERSION in message
    assert DEPRECATED_PLUGIN_CONFIG_KEYS[key] in message
    assert message in caplog.text


def test_supported_keys_do_not_warn(recwarn):
    _run_main_plugin_setup(
        {
            "enabled": False,
            "plugin_paths": [],
            "auto_discover": True,
            "auto_validate": True,
            "disabled_plugins": [],
        }
    )
    assert not [w for w in recwarn if issubclass(w.category, FutureWarning)]


def test_the_shipped_configuration_sets_no_deprecated_key():
    shipped = yaml.safe_load((REPO / "configs" / "plugins.yaml").read_text())["plugins"]
    assert deprecated_plugin_config_keys(shipped) == []


def test_the_paths_alias_still_works(tmp_path):
    """``paths`` warns but keeps working until its removal."""
    calls = []
    with (
        mock.patch.object(
            PluginRegistry,
            "discover_plugins",
            side_effect=lambda paths, **kwargs: calls.append(paths) or [],
        ),
        pytest.warns(FutureWarning, match=r"plugins\.paths is deprecated.*use plugin_paths"),
    ):
        _run_main_plugin_setup({"enabled": True, "paths": [str(tmp_path)]})
    assert calls == [[tmp_path]]


def test_schema_validation_reports_deprecated_keys_and_recommends_none():
    """The validator no longer urges enabling settings that do nothing."""
    validator = PluginSchemaValidator()
    validator.plugin_system_available = True
    config = {
        "enabled": True,
        "plugin_paths": ["plugins"],
        "security_scanning": False,
        "enforce_checksums": False,
        "allow_experimental": True,
    }
    result = validator.validate_plugin_config(config, PluginValidationLevel.STRICT)

    assert result["valid"] is True
    assert result["warnings"] == deprecated_plugin_config_keys(config)
    assert not [w for w in result["warnings"] if "not recommended" in w or "caution" in w]


def test_summary_no_longer_reports_inert_settings():
    summary = get_plugin_config_summary(
        {"enabled": True, "security_scanning": True, "enforce_checksums": True}
    )
    assert "Security Scanning" not in summary
    assert "Enforce Checksums" not in summary
    assert "Trusted Plugins" not in summary
