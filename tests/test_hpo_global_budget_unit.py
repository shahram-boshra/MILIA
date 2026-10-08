#!/usr/bin/env python3
"""
Contract tests for the study-wide trial budget ``HPOConfig.n_trials_total`` (P2-3a, blueprint S3 / F9).

``n_trials`` is per process (Optuna); ``n_trials_total`` caps finished (COMPLETE + PRUNED) trials of the
whole study via ``optuna.study.MaxTrialsCallback``. The callback runs after each trial, so trials that
are already running in other processes/threads when the budget is reached still finish: the final
count is in ``[n_trials_total, n_trials_total + concurrency - 1]`` (R60). All studies here are real.
"""

import multiprocessing
import time
from unittest.mock import MagicMock, patch

import optuna
import pytest
from pydantic import ValidationError

from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend
from milia_pipeline.models.hpo.hpo_config import HPOConfig

BUDGET_STATES = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)


@pytest.fixture(autouse=True)
def _quiet_optuna():
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    yield
    optuna.logging.set_verbosity(verbosity)


def _finished(study) -> int:
    return len(study.get_trials(deepcopy=False, states=BUDGET_STATES))


def _objective_with_prunes(trial):
    """Every third trial is pruned, the rest complete (both count against the budget)."""
    x = trial.suggest_float("x", 0.0, 1.0)
    if trial.number % 3 == 2:
        raise optuna.TrialPruned()
    return x


@pytest.mark.contract
class TestConfig:
    def test_default_is_no_budget(self):
        assert HPOConfig().n_trials_total is None

    def test_from_dict(self):
        assert HPOConfig.from_dict({"n_trials_total": 40}).n_trials_total == 40

    @pytest.mark.parametrize("value", [0, -1, True, "10", 1.5])
    def test_rejects_non_positive_or_non_int(self, value):
        with pytest.raises(ValidationError):
            HPOConfig(n_trials_total=value)


@pytest.mark.contract
class TestBackendBudget:
    def test_no_budget_runs_n_trials(self):
        backend = OptunaBackend()
        study = optuna.create_study()
        backend.optimize(study, _objective_with_prunes, n_trials=5)
        assert len(study.trials) == 5

    def test_budget_counts_complete_and_pruned(self):
        backend = OptunaBackend()
        study = optuna.create_study()
        backend.optimize(study, _objective_with_prunes, n_trials=50, n_trials_total=7)
        assert _finished(study) == 7
        assert any(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)

    def test_failed_trials_do_not_consume_budget(self):
        def objective(trial):
            x = trial.suggest_float("x", 0.0, 1.0)
            if trial.number < 2:
                raise RuntimeError("transient failure")
            return x

        backend = OptunaBackend()
        study = optuna.create_study()
        backend.optimize(study, objective, n_trials=50, catch=(RuntimeError,), n_trials_total=3)
        states = [t.state for t in study.trials]
        assert states.count(optuna.trial.TrialState.FAIL) == 2
        assert _finished(study) == 3

    def test_budget_is_study_wide_across_runs(self):
        """A resumed run continues towards the same study total."""
        backend = OptunaBackend()
        study = optuna.create_study()
        backend.optimize(study, _objective_with_prunes, n_trials=4)
        backend.optimize(study, _objective_with_prunes, n_trials=50, n_trials_total=6)
        assert _finished(study) == 6

    def test_budget_already_reached_starts_no_trial(self):
        """Without the pre-check Optuna would still run one trial (R60)."""
        backend = OptunaBackend()
        study = optuna.create_study()
        backend.optimize(study, _objective_with_prunes, n_trials=3)
        backend.optimize(study, _objective_with_prunes, n_trials=50, n_trials_total=3)
        assert len(study.trials) == 3

    def test_caller_callbacks_kept_and_not_mutated(self):
        calls = []
        callbacks = [lambda study, trial: calls.append(trial.number)]
        backend = OptunaBackend()
        study = optuna.create_study()
        backend.optimize(
            study, _objective_with_prunes, n_trials=50, callbacks=callbacks, n_trials_total=4
        )
        assert calls == [0, 1, 2, 3]
        assert len(callbacks) == 1


def _budget_worker(path: str, budget: int) -> None:
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(path))
    study = optuna.load_study(study_name="budget", storage=storage)

    def objective(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        time.sleep(0.05)  # overlap between workers
        return x

    OptunaBackend().optimize(study, objective, n_trials=100, n_trials_total=budget)


@pytest.mark.integration
@pytest.mark.slow
def test_budget_across_processes_bounded_by_concurrency(tmp_path):
    """3 worker processes on one journal: finished trials in [budget, budget + workers - 1]."""
    path = str(tmp_path / "budget.log")
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(path))
    optuna.create_study(study_name="budget", storage=storage)
    budget, workers = 10, 3
    # "spawn": forking a multi-threaded process (pytest + torch/OpenMP threads) can deadlock the child;
    # Python >= 3.12 warns about it and 3.14 no longer forks by default on Linux (R60).
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_budget_worker, args=(path, budget)) for _ in range(workers)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=300)  # spawn re-imports milia_pipeline in each child
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
    assert [process.exitcode for process in processes] == [0] * workers
    study = optuna.load_study(study_name="budget", storage=storage)
    assert budget <= _finished(study) <= budget + workers - 1


@pytest.mark.contract
def test_manager_passes_budget_to_backend():
    """``HPOManager`` forwards ``config.n_trials_total`` to ``backend.optimize``."""
    from milia_pipeline.models.hpo.hpo_manager import HPOManager

    backend = MagicMock()
    backend.get_best_params.return_value = {"lr": 0.01}
    backend.get_best_value.return_value = 0.1  # formatted with :.6f by the manager
    with patch("milia_pipeline.models.hpo.hpo_manager.get_backend", return_value=backend):
        manager = HPOManager(HPOConfig(enabled=True, n_trials=5, n_trials_total=20))
    with (
        patch.object(manager, "_filter_search_space_for_model", return_value={}),
        patch.object(manager, "_create_objective", return_value=MagicMock()),
        patch("milia_pipeline.models.hpo.hpo_manager.get_factory", return_value=MagicMock()),
    ):
        manager.optimize(model_name="GCN", dataset=MagicMock())
    kwargs = backend.optimize.call_args.kwargs
    assert kwargs["n_trials"] == 5
    assert kwargs["n_trials_total"] == 20
