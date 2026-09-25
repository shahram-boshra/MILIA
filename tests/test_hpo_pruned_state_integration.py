"""Regression test for P1-0 (F17): a pruned HPO trial must persist as ``PRUNED``.

Exception chain under test (real objects, no mocks):
``OptunaPruningCallback.on_epoch_end`` raises ``optuna.TrialPruned`` →
``Trainer._on_epoch_end`` re-raises it → ``Trainer.fit`` must let it leave unwrapped →
``Study.optimize`` records the trial state.

If ``Trainer.fit`` wraps the signal in ``TrainingError``, Optuna records ``FAIL``: TPE then ignores the
trial (its observation states are ``COMPLETE`` and ``PRUNED``) and ``MaxTrialsCallback`` budgets never
count it.

``test_control_trials_complete_without_pruning`` validates the harness itself: it must pass both before
and after the fix, so a failure of ``test_pruned_trial_persists_as_pruned`` can only mean the state
defect, not a broken test setup.
"""

import optuna
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from milia_pipeline.models.hpo.callbacks.optuna_callback import create_hpo_callback
from milia_pipeline.models.training.trainer import Trainer

pytestmark = [pytest.mark.integration, pytest.mark.regression]

_N_TRIALS = 2
_MAX_EPOCHS = 3


class _TinyGraphRegressor(nn.Module):
    """Minimal graph-level regressor: per-node linear layer followed by mean pooling."""

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.lin = nn.Linear(in_channels, 1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        return global_mean_pool(self.lin(x), batch)


def _graphs(n_graphs: int, seed: int) -> list[Data]:
    generator = torch.Generator().manual_seed(seed)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    return [
        Data(
            x=torch.randn(5, 4, generator=generator),
            edge_index=edge_index,
            y=torch.randn(1, 1, generator=generator),
        )
        for _ in range(n_graphs)
    ]


def _objective(trial: optuna.Trial) -> float:
    learning_rate = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
    torch.manual_seed(0)
    model = _TinyGraphRegressor()
    trainer = Trainer(
        model=model,
        train_loader=DataLoader(_graphs(8, seed=0), batch_size=4),
        val_loader=DataLoader(_graphs(4, seed=1), batch_size=4),
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
        loss_fn=nn.MSELoss(),
        device=torch.device("cpu"),
        max_epochs=_MAX_EPOCHS,
        hpo_callback=create_hpo_callback(trial, monitor="val_loss"),
    )
    results = trainer.fit()
    return float(results["best_val_loss"])


def _trial_states(pruner: optuna.pruners.BasePruner) -> list[optuna.trial.TrialState]:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=pruner,
    )
    study.optimize(_objective, n_trials=_N_TRIALS, catch=(Exception,))
    return [trial.state for trial in study.trials]


def test_control_trials_complete_without_pruning() -> None:
    """Harness validity: with pruning disabled every trial must complete."""
    states = _trial_states(optuna.pruners.NopPruner())
    assert states == [optuna.trial.TrialState.COMPLETE] * _N_TRIALS, states


def test_pruned_trial_persists_as_pruned() -> None:
    """A pruner that rejects every reported value must yield PRUNED trials, never FAIL."""
    # ThresholdPruner(upper=-1.0) prunes any value above -1.0; an MSE loss is always >= 0,
    # so every trial is pruned at its first report (step 0, no warm-up).
    states = _trial_states(optuna.pruners.ThresholdPruner(upper=-1.0, n_warmup_steps=0))
    assert optuna.trial.TrialState.FAIL not in states, states
    assert states == [optuna.trial.TrialState.PRUNED] * _N_TRIALS, states
