"""The ``models.deployment`` section (milia_pipeline.models.deployment.settings) and the
deprecations of the earlier deployment schema."""

from __future__ import annotations

import logging
import subprocess
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from milia_pipeline.exceptions import ConfigurationError
from milia_pipeline.models.deployment.settings import (
    DeploymentSettings,
    RegistrySettings,
    deprecated_deployment_config_keys,
    load_deployment_settings,
)

MODELS_YAML = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"

# The section configs/models.yaml shipped before this schema (abridged values, every key).
EARLIER_SECTION = {
    "enabled": False,
    "optimization": {
        "quantization": {"enabled": False, "method": "dynamic"},
        "pruning": {"enabled": False, "amount": 0.3},
        "distillation": {"enabled": False, "temperature": 3.0},
    },
    "strategy": "cloud",
    "edge": {"target_device": "jetson_nano"},
    "cloud": {"provider": None, "auto_scaling": False},
    "federated": {"num_clients": 10},
    "monitoring": {"enabled": True, "drift_detection": {"enabled": True}, "retraining": {}},
}


def test_defaults():
    settings = load_deployment_settings(None)
    assert settings == DeploymentSettings()
    assert (settings.enabled, settings.target, settings.output_dir) == (
        False,
        "container",
        "deployments",
    )
    assert settings.gate.model_dump() == {
        "metric": "val_loss",
        "mode": "min",
        "threshold": None,
        "compare_to_current": False,
    }
    assert not settings.onnx.enabled and settings.predict.input_path is None


def test_the_shipped_section_is_the_new_one_and_disabled():
    section = yaml.safe_load(MODELS_YAML.read_text())["models"]["deployment"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        settings = load_deployment_settings(section)
    assert settings == DeploymentSettings()
    assert deprecated_deployment_config_keys(section) == []


def test_the_earlier_section_warns_once_per_key_and_is_ignored(caplog):
    with (
        caplog.at_level(logging.WARNING, logger="milia_pipeline.models.deployment.settings"),
        pytest.warns(FutureWarning) as recorded,
    ):
        settings = load_deployment_settings(EARLIER_SECTION)

    keys = [str(w.message).split(" is deprecated")[0] for w in recorded]
    assert keys == [
        "models.deployment.strategy",
        "models.deployment.edge",
        "models.deployment.cloud",
        "models.deployment.federated",
        "models.deployment.optimization.quantization",
        "models.deployment.optimization.pruning",
        "models.deployment.optimization.distillation",
        "models.deployment.monitoring",
    ]
    assert all("will be removed in 2.0.0" in str(w.message) for w in recorded)
    assert len([r for r in caplog.records if "is deprecated" in r.getMessage()]) == 8
    assert settings == DeploymentSettings()


def test_an_optimization_block_without_known_keys_still_warns():
    assert deprecated_deployment_config_keys({"optimization": {}}) == [
        "models.deployment.optimization is deprecated and will be removed in 2.0.0: "
        "it has no effect"
    ]


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ({"tagret": "aws"}, r"models\.deployment\.tagret: Extra inputs are not permitted"),
        ({"gate": {"metrc": "val_mae"}}, r"models\.deployment\.gate\.metrc: Extra inputs"),
        ({"target": "edge"}, r"models\.deployment\.target: Input should be"),
        ({"gate": {"mode": "avg"}}, r"models\.deployment\.gate\.mode: Input should be 'min' or"),
        ({"output_dir": ""}, r"models\.deployment\.output_dir"),
        ({"onnx": {"enabled": "maybe"}}, r"models\.deployment\.onnx\.enabled"),
    ],
)
def test_invalid_sections_are_configuration_errors(section, message):
    with pytest.raises(ConfigurationError, match=message):
        load_deployment_settings(section)


def test_a_non_mapping_section_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="must be a mapping, got list"):
        load_deployment_settings(["enabled"])


def test_a_complete_section_loads():
    settings = load_deployment_settings(
        {
            "enabled": True,
            "target": "aws",
            "base_image": "ghcr.io/example/milia:cpu",
            "output_dir": "/srv/models",
            "onnx": {"enabled": True},
            "predict": {"input_path": "molecules.csv"},
            "gate": {
                "metric": "val_r2",
                "mode": "max",
                "threshold": 0.8,
                "compare_to_current": True,
            },
        }
    )
    assert settings.enabled and settings.target == "aws" and settings.onnx.enabled
    assert settings.gate.threshold == 0.8 and settings.gate.mode == "max"


class TestEarlierPythonApi:
    def test_get_deployment_config_is_deprecated(self):
        from milia_pipeline.models.utils import config_bridge

        with (
            patch.object(
                config_bridge, "get_models_config", return_value=config_bridge.ModelConfig()
            ),
            pytest.warns(DeprecationWarning, match="load_deployment_settings"),
        ):
            deployment = config_bridge.get_deployment_config()
        assert isinstance(deployment, config_bridge.DeploymentConfig)

    def test_model_config_deployment_is_deprecated(self):
        from milia_pipeline.models.utils.config_bridge import ModelConfig

        config = ModelConfig()
        with pytest.warns(DeprecationWarning, match="removed in MILIA 2.0.0"):
            config.deployment  # noqa: B018

    def test_is_phase_enabled_does_not_warn(self):
        from milia_pipeline.models.utils.config_bridge import ModelConfig

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert ModelConfig(enabled=True).is_phase_enabled("deployment") is False

    def test_recorded_only_deployment_fields_are_deprecated(self):
        from milia_pipeline.models.deployment.deployment_strategies import DeploymentConfig

        with pytest.warns(
            DeprecationWarning, match=r"DeploymentConfig\.num_instances is deprecated"
        ):
            DeploymentConfig(target="container", num_instances=2)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            DeploymentConfig(target="container", num_instances=1, base_image="img")  # defaults


def test_training_refuses_an_invalid_deployment_section():
    import main

    with patch.object(main, "_run_standard_training") as run:
        code = main.handle_training_mode(
            None, logging.getLogger("test"), None, {"models": {"deployment": {"bogus": 1}}}
        )
    assert code == 1 and not run.called


# The registry section (models.deployment.registry); MLflow's naming rules, checked on load


def test_the_registry_is_off_by_default():
    registry = load_deployment_settings({}).registry
    assert registry == RegistrySettings() and registry.experiment == "milia"
    assert not registry.enabled and registry.alias == "champion" and registry.model_name is None


@pytest.mark.parametrize("alias", ["", "latest", "LATEST", "v3", "V12", "a b", "x/y", "a" * 256])
def test_an_alias_mlflow_refuses_is_refused_when_the_section_loads(alias):
    with pytest.raises(ConfigurationError, match="registry.alias"):
        load_deployment_settings({"registry": {"alias": alias}})


@pytest.mark.parametrize("name", ["", " ", "a/b", "a:b", ".", "../x", "a//b", "a'b", 'a"b'])
def test_a_model_name_mlflow_refuses_is_refused(name):
    with pytest.raises(ConfigurationError, match="registry.model_name"):
        load_deployment_settings({"registry": {"model_name": name}})


def test_an_unknown_registry_key_is_refused():
    with pytest.raises(ConfigurationError, match="registry.stage"):
        load_deployment_settings({"registry": {"stage": "Production"}})


def test_enabling_the_registry_without_mlflow_names_the_extra(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "mlflow" else real(name, *a)
    )
    with pytest.raises(ConfigurationError, match="uv sync --extra mlflow"):
        load_deployment_settings({"registry": {"enabled": True}})


def test_the_registry_off_never_imports_mlflow():
    """Training, packaging and deploying with the registry off import no MLflow module."""
    code = (
        "import sys\n"
        "from milia_pipeline.models.deployment import after_training, deploy, settings\n"
        "settings.load_deployment_settings({'enabled': True})\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'mlflow'))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]"
