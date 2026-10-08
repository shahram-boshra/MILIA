#!/usr/bin/env python3
"""
Contract tests for the configurable progress bar (P2-3d, blueprint S3 / F13).

``OptunaBackend.optimize`` hard-coded ``show_progress_bar=True``: N workers of one study print N
interleaved tqdm bars, and while a bar is shown Optuna routes its log records through tqdm (R67).
``HPOConfig.show_progress_bar``: None (default) = automatic — shown in a single process, hidden in a
worker (``HPOManager(worker_index=...)``); True / False force it.
"""

from unittest.mock import MagicMock, patch

import optuna
import pytest
from pydantic import ValidationError

from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend
from milia_pipeline.models.hpo.hpo_config import HPOConfig

TQDM_BAR = "%|"  # tqdm bar marker, e.g. " 40%|████      | 2/5"


@pytest.mark.contract
class TestConfig:
    def test_default_is_automatic(self):
        assert HPOConfig().show_progress_bar is None

    @pytest.mark.parametrize("value", [True, False])
    def test_explicit_values(self, value):
        assert HPOConfig.from_dict({"show_progress_bar": value}).show_progress_bar is value

    @pytest.mark.parametrize("value", ["yes", 1, 0, "false"])
    def test_rejects_non_bool(self, value):
        with pytest.raises(ValidationError):
            HPOConfig(show_progress_bar=value)


@pytest.mark.contract
class TestBackend:
    @pytest.mark.parametrize(("flag", "drawn"), [(False, False), (True, True)])
    def test_bar_drawn_only_when_enabled(self, capsys, flag, drawn):
        study = optuna.create_study()
        OptunaBackend().optimize(
            study, lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=3, show_progress_bar=flag
        )
        assert (TQDM_BAR in capsys.readouterr().err) is drawn
        assert len(study.trials) == 3

    def test_default_keeps_previous_behaviour(self):
        study = optuna.create_study()
        with patch.object(study, "optimize") as optimize:
            OptunaBackend().optimize(study, lambda t: 0.0, n_trials=1)
        assert optimize.call_args.kwargs["show_progress_bar"] is True


def _bar_passed(show_progress_bar, worker_index):
    from milia_pipeline.models.hpo.hpo_manager import HPOManager

    backend = MagicMock()
    backend.get_best_params.return_value = {"lr": 0.01}
    backend.get_best_value.return_value = 0.1  # formatted with :.6f by the manager
    config = HPOConfig(enabled=True, show_progress_bar=show_progress_bar)
    with patch("milia_pipeline.models.hpo.hpo_manager.get_backend", return_value=backend):
        manager = HPOManager(config, worker_index=worker_index)
    with (
        patch.object(manager, "_filter_search_space_for_model", return_value={}),
        patch.object(manager, "_create_objective", return_value=MagicMock()),
        patch("milia_pipeline.models.hpo.hpo_manager.get_factory", return_value=MagicMock()),
    ):
        manager.optimize(model_name="GCN", dataset=MagicMock())
    return backend.optimize.call_args.kwargs["show_progress_bar"]


@pytest.mark.contract
@pytest.mark.parametrize(
    ("configured", "worker_index", "expected"),
    [
        (None, None, True),  # single process: unchanged behaviour
        (None, 0, False),  # worker: automatic off
        (False, None, False),  # explicit off wins
        (True, 2, True),  # explicit on wins, even in a worker
    ],
)
def test_manager_resolves_flag(configured, worker_index, expected):
    assert _bar_passed(configured, worker_index) is expected
