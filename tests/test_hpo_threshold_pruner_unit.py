#!/usr/bin/env python3
"""
Contract tests for ``pruner.type: threshold`` (Optuna's ``ThresholdPruner``) and the deprecated
``pruner.n_brackets``.

optuna 4.9.0 ``ThresholdPruner(lower=None, upper=None, n_warmup_steps=0, interval_steps=1)`` raises
``TypeError("Either lower or upper must be specified.")`` without a bound and ``ValueError`` when
``lower > upper``; it prunes a trial whose intermediate value falls below ``lower`` or rises above
``upper``. ``HyperbandPruner(min_resource, max_resource, reduction_factor, bootstrap_count)`` has no
brackets argument: the number of brackets follows from those three values.
"""

import math
import warnings

import optuna
import pytest
from optuna.trial import TrialState

from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend
from milia_pipeline.models.hpo.hpo_config import HPOConfig, PrunerConfig, PrunerType


def _pruner_from(config: HPOConfig) -> optuna.pruners.BasePruner:
    """Build the pruner with the arguments HPOManager passes to the backend."""
    pruner = config.pruner
    return OptunaBackend().create_pruner(
        pruner_type=pruner.type.value,
        n_startup_trials=pruner.n_startup_trials,
        n_warmup_steps=pruner.n_warmup_steps,
        interval_steps=pruner.interval_steps,
        percentile=pruner.percentile,
        lower=pruner.lower,
        upper=pruner.upper,
        p_threshold=pruner.p_threshold,
        n_startup_steps=pruner.n_startup_steps,
    )


@pytest.mark.contract
class TestConfig:
    @pytest.mark.parametrize(
        "bounds",
        [
            {"upper": 1.0},
            {"lower": -1.0},
            {"lower": -1.0, "upper": 1.0},
            {"lower": 0.5, "upper": 0.5},
        ],
    )
    def test_accepted_with_a_bound(self, bounds):
        config = HPOConfig.from_dict({"pruner": {"type": "threshold", **bounds}})
        assert config.pruner.type is PrunerType.THRESHOLD
        assert (config.pruner.lower, config.pruner.upper) == (
            bounds.get("lower"),
            bounds.get("upper"),
        )

    @pytest.mark.parametrize(
        ("bounds", "message"),
        [
            ({}, "needs 'lower', 'upper' or both"),
            ({"lower": 2.0, "upper": 1.0}, r"needs lower <= upper \(got lower=2.0, upper=1.0\)"),
            ({"upper": math.inf}, "upper must be a finite number"),
            ({"lower": -math.inf}, "lower must be a finite number"),
            ({"upper": math.nan}, "upper must be a finite number"),
        ],
    )
    def test_rejected(self, bounds, message):
        with pytest.raises(ValueError, match=message):
            HPOConfig.from_dict({"pruner": {"type": "threshold", **bounds}})

    def test_bounds_are_ignored_by_other_pruners(self):
        config = HPOConfig.from_dict({"pruner": {"type": "median", "lower": 2.0, "upper": 1.0}})
        assert config.pruner.type is PrunerType.MEDIAN


@pytest.mark.contract
class TestBackend:
    def test_creates_threshold_pruner_with_bounds_and_steps(self):
        pruner = OptunaBackend().create_pruner(
            "threshold", n_warmup_steps=3, interval_steps=2, lower=-1.0, upper=2.0
        )
        assert isinstance(pruner, optuna.pruners.ThresholdPruner)
        assert (pruner._lower, pruner._upper) == (-1.0, 2.0)
        assert (pruner._n_warmup_steps, pruner._interval_steps) == (3, 2)

    def test_configured_pruner_stops_trials_beyond_the_bound(self):
        config = HPOConfig.from_dict(
            {"pruner": {"type": "threshold", "upper": 1.0, "n_warmup_steps": 0}}
        )
        study = optuna.create_study(pruner=_pruner_from(config))

        def objective(trial):
            value = trial.suggest_categorical("value", [0.5, 2.0])
            trial.report(value, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return value

        study.enqueue_trial({"value": 0.5})
        study.enqueue_trial({"value": 2.0})
        study.optimize(objective, n_trials=2)
        assert [t.state for t in study.trials] == [TrialState.COMPLETE, TrialState.PRUNED]


@pytest.mark.contract
class TestNBracketsDeprecation:
    def test_default_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            config = PrunerConfig(type=PrunerType.HYPERBAND)
        assert config.n_brackets == 4

    def test_setting_it_warns(self):
        with pytest.warns(FutureWarning, match="n_brackets has no effect"):
            HPOConfig.from_dict({"pruner": {"type": "hyperband", "n_brackets": 3}})

    def test_hyperband_pruner_takes_no_brackets_argument(self):
        """The reason it has no effect: Optuna derives the brackets itself."""
        import inspect

        params = inspect.signature(optuna.pruners.HyperbandPruner).parameters
        assert "n_brackets" not in params
        assert {"min_resource", "max_resource", "reduction_factor"} <= set(params)
