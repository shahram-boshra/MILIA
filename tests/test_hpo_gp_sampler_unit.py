#!/usr/bin/env python3
"""
Contract tests for ``sampler.type: gp`` — Optuna's ``GPSampler`` (Gaussian-process Bayesian
optimization).

Optuna recommends it for budgets up to about 500 trials; it handles float, int and categorical
parameters, one or several objectives, and (optuna 4.9.0) conditions on trials still running in other
workers. MILIA passes ``seed`` (per-worker seeds are derived as for every sampler) and
``n_startup_trials`` (random trials before the GP is fitted).
"""

import multiprocessing
import warnings

import optuna
import pytest
from optuna.trial import TrialState

from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend
from milia_pipeline.models.hpo.hpo_config import HPOConfig, SamplerType, StorageConfig, StudyConfig

GP_PARAMS_KEY = (
    "gp:relative_params"  # optuna 4.9.0 _gp/sampler.py: set when the GP chose the params
)


def _quadratic(trial):
    x = trial.suggest_float("x", -5.0, 5.0)
    n = trial.suggest_int("n", 1, 8)
    kind = trial.suggest_categorical("kind", ["a", "b"])
    return (x - 1.5) ** 2 + abs(n - 3) + (0.5 if kind == "b" else 0.0)


def _gp_trials(study):
    return [t for t in study.trials if any(key.startswith(GP_PARAMS_KEY) for key in t.system_attrs)]


@pytest.mark.contract
class TestConfig:
    def test_gp_accepted(self):
        assert HPOConfig.from_dict({"sampler": {"type": "gp"}}).sampler.type is SamplerType.GP

    def test_gp_with_several_objectives(self):
        config = HPOConfig.from_dict(
            {
                "sampler": {"type": "gp"},
                "pruner": {"type": "none"},
                "study": {"objectives": [{"metric": "val_mae"}, {"metric": "n_parameters"}]},
            }
        )
        assert config.study.is_multi_objective


@pytest.mark.contract
class TestBackend:
    def test_creates_gp_sampler_quietly(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", optuna.exceptions.ExperimentalWarning)
            sampler = OptunaBackend().create_sampler("gp", seed=3, n_startup_trials=4)
        assert isinstance(sampler, optuna.samplers.GPSampler)
        assert sampler._n_startup_trials == 4

    def test_gp_samples_after_startup_trials(self):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = OptunaBackend().create_sampler("gp", seed=0, n_startup_trials=4)
        study = optuna.create_study(sampler=sampler)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            study.optimize(_quadratic, n_trials=10)
        gp_numbers = [t.number for t in _gp_trials(study)]
        assert gp_numbers and min(gp_numbers) >= 4
        assert all(t.state == TrialState.COMPLETE for t in study.trials)

    def test_gp_is_reproducible_with_a_seed(self):
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        def run():
            study = optuna.create_study(
                sampler=OptunaBackend().create_sampler("gp", seed=11, n_startup_trials=3)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
                study.optimize(_quadratic, n_trials=6)
            return [t.params for t in study.trials]

        assert run() == run()

    def test_gp_multi_objective(self):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=OptunaBackend().create_sampler("gp", seed=1, n_startup_trials=3),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            study.optimize(
                lambda t: (t.suggest_float("x", 0, 1), (1 - t.suggest_float("y", 0, 1)) ** 2),
                n_trials=7,
            )
        assert len(_gp_trials(study)) == 4
        assert OptunaBackend().get_best_trial_number(study) in {t.number for t in study.best_trials}


def _worker(path: str, index: int) -> None:
    from milia_pipeline.models.hpo.backends.storage_factory import build_storage
    from milia_pipeline.models.hpo.seeding import derive_worker_seed

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
    storage = build_storage(
        StudyConfig(storage_options=StorageConfig(kind="journal_file", journal_path=path))
    )
    sampler = OptunaBackend().create_sampler(
        "gp", seed=derive_worker_seed(5, index), n_startup_trials=3
    )
    study = optuna.load_study(study_name="gp", storage=storage, sampler=sampler)
    study.optimize(_quadratic, n_trials=6)


@pytest.mark.contract
def test_two_workers_share_a_gp_study(tmp_path):
    """Two spawn workers with per-worker seeds on one journal study: every trial completes and no
    parameter set repeats."""
    from milia_pipeline.models.hpo.backends.storage_factory import build_storage

    path = str(tmp_path / "journal.log")
    storage = build_storage(
        StudyConfig(storage_options=StorageConfig(kind="journal_file", journal_path=path))
    )
    optuna.create_study(study_name="gp", storage=storage)
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_worker, args=(path, i)) for i in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=300)
    assert [p.exitcode for p in processes] == [0, 0]
    study = optuna.load_study(study_name="gp", storage=storage)
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    assert len(complete) == 12
    assert len({tuple(sorted(t.params.items())) for t in complete}) == 12
    assert _gp_trials(study)
