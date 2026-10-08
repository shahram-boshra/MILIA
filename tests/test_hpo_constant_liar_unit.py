#!/usr/bin/env python3
"""
Contract tests for the ``constant_liar`` default (P2-3c, blueprint S3 / F7).

TPE's constant liar treats trials still running in other workers as observed, so parallel workers avoid
each other's regions (Optuna recommends it for distributed optimization; default in Optuna 5.0). With no
other running trial — a single process — sampling is unchanged (R7, R64). Every source of the default
(``SamplerConfig``, ``HPOConfig.from_dict``, the config bridge, ``configs/models.yaml``) must agree.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna
import pytest
import yaml

from milia_pipeline.models.hpo.hpo_config import HPOConfig, SamplerConfig
from milia_pipeline.models.utils.config_bridge import HPOSamplerConfigBridge, ModelConfig

MODELS_YAML = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"

# These tests build TPESampler directly; Optuna 4.9.0 flags multivariate / constant_liar as experimental
# (MILIA's own path scope-suppresses this in OptunaBackend.create_sampler).
pytestmark = pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")


@pytest.fixture(autouse=True)
def _quiet_optuna():
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    yield
    optuna.logging.set_verbosity(verbosity)


def _bridge_sampler(sampler: dict) -> HPOSamplerConfigBridge:
    config = ModelConfig.from_dict(
        {
            "enabled": True,
            "selection": {"task_type": "graph_regression", "model_name": "GCN"},
            "hpo": {"enabled": False, "sampler": sampler},
        }
    )
    return config.hpo.sampler


@pytest.mark.contract
class TestDefaultAgreement:
    def test_sampler_config_default(self):
        assert SamplerConfig().constant_liar is True

    def test_hpo_config_from_dict_default(self):
        assert HPOConfig.from_dict({}).sampler.constant_liar is True
        assert HPOConfig.from_dict({"sampler": {"type": "tpe"}}).sampler.constant_liar is True

    def test_bridge_defaults(self):
        assert HPOSamplerConfigBridge().constant_liar is True
        assert _bridge_sampler({"type": "tpe"}).constant_liar is True

    def test_shipped_yaml(self):
        data = yaml.safe_load(MODELS_YAML.read_text())
        assert data["models"]["hpo"]["sampler"]["constant_liar"] is True

    def test_explicit_false_honoured(self):
        assert (
            HPOConfig.from_dict({"sampler": {"constant_liar": False}}).sampler.constant_liar
            is False
        )
        assert _bridge_sampler({"constant_liar": False}).constant_liar is False


def _objective(trial):
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_int("y", 0, 10)
    if trial.number % 11 == 5:  # before any pruning check, so FAIL states always occur
        raise RuntimeError("transient failure")
    for step in range(3):
        trial.report(x * x + y + step, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return x * x + y


def _sequence(seed: int, multivariate: bool, constant_liar: bool):
    sampler = optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=10, multivariate=multivariate, constant_liar=constant_liar
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(sampler=sampler, pruner=pruner)
    study.optimize(_objective, n_trials=40, catch=(RuntimeError,))
    return [(t.state, t.params) for t in study.trials]


@pytest.mark.contract
@pytest.mark.parametrize("multivariate", [True, False])
@pytest.mark.parametrize("seed", [0, 1])
def test_single_process_sampling_unchanged(seed, multivariate):
    """No other trial is running in a sequential run, so the flip cannot change results (R7)."""
    with_liar = _sequence(seed, multivariate, constant_liar=True)
    states = {state for state, _ in with_liar}
    assert {optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL} <= states
    assert with_liar == _sequence(seed, multivariate, constant_liar=False)


def _next_params_with_running_trial(constant_liar: bool) -> dict:
    """Twelve finished trials plus one RUNNING trial (another worker), then sample one more."""
    study = optuna.create_study()
    for i in range(12):
        study.enqueue_trial({"x": -5.0 + i * 10.0 / 11.0})
    study.optimize(lambda t: (t.suggest_float("x", -5.0, 5.0) - 1.0) ** 2, n_trials=12)
    study.sampler = optuna.samplers.TPESampler(
        seed=3, n_startup_trials=10, constant_liar=constant_liar
    )
    study.enqueue_trial({"x": 1.0})
    running = study.ask()  # left RUNNING, as in a parallel worker
    running.suggest_float("x", -5.0, 5.0)
    trial = study.ask()
    return {"x": trial.suggest_float("x", -5.0, 5.0)}


@pytest.mark.contract
def test_running_trial_changes_sampling_only_with_liar():
    """With another trial RUNNING, the constant liar takes it into account (the parallel effect)."""
    assert _next_params_with_running_trial(True) != _next_params_with_running_trial(False)


@pytest.mark.contract
def test_manager_passes_configured_flag():
    from milia_pipeline.models.hpo.hpo_manager import HPOManager

    for configured in (True, False):
        backend = MagicMock()
        backend.get_best_params.return_value = {"lr": 0.01}
        backend.get_best_value.return_value = 0.1  # formatted with :.6f by the manager
        config = HPOConfig(enabled=True, sampler=SamplerConfig(constant_liar=configured))
        with patch("milia_pipeline.models.hpo.hpo_manager.get_backend", return_value=backend):
            manager = HPOManager(config)
        with (
            patch.object(manager, "_filter_search_space_for_model", return_value={}),
            patch.object(manager, "_create_objective", return_value=MagicMock()),
            patch("milia_pipeline.models.hpo.hpo_manager.get_factory", return_value=MagicMock()),
        ):
            manager.optimize(model_name="GCN", dataset=MagicMock())
        assert backend.create_sampler.call_args.kwargs["constant_liar"] is configured
