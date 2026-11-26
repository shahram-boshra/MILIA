"""Seeded training: the same seed, code and data give the same model; the seed is recorded.

``models.training.seed`` seeds weight initialization, the training loader's shuffling (its own
generator) and dropout; ``null`` draws a seed. ``run_info.training_seed`` and a package's
``lineage.training_seed`` record it.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from milia_pipeline.exceptions import ConfigurationError
from milia_pipeline.models.training.loaders import make_loader
from milia_pipeline.models.training.seeding import (
    SEED_LIMIT,
    loader_generator,
    resolve_seed,
    seed_everything,
    seed_worker,
)


@pytest.mark.parametrize("seed", [0, 42, SEED_LIMIT - 1])
def test_a_given_seed_is_used_as_it_is(seed):
    assert resolve_seed(seed) == seed


def test_null_draws_a_seed_in_range():
    seeds = {resolve_seed(None) for _ in range(5)}
    assert all(0 <= s < SEED_LIMIT for s in seeds) and len(seeds) > 1


@pytest.mark.parametrize("seed", [-1, SEED_LIMIT, 1.5, "42", True])
def test_an_invalid_seed_is_refused(seed):
    with pytest.raises(ConfigurationError, match="models.training.seed"):
        resolve_seed(seed)


def _draws():
    return random.random(), float(np.random.rand()), torch.rand(3).tolist()


def test_seed_everything_repeats_python_numpy_and_torch():
    seed_everything(7)
    first = _draws()
    seed_everything(7)
    assert _draws() == first
    seed_everything(8)
    assert _draws() != first


def _graphs(n=20):
    return [
        Data(x=torch.full((2, 1), float(i)), edge_index=torch.zeros(2, 0, dtype=torch.long))
        for i in range(n)
    ]


def _order(seed):
    loader = make_loader(_graphs(), batch_size=4, shuffle=True, generator=loader_generator(seed))
    return [batch.x[::2, 0].tolist() for batch in loader]


def test_the_training_loader_shuffles_the_same_way_for_the_same_seed():
    torch.manual_seed(123)  # the global generator does not decide the order
    first = _order(5)
    torch.manual_seed(456)
    assert _order(5) == first
    assert _order(6) != first


def test_a_seeded_loader_seeds_its_workers():
    loader = make_loader(_graphs(), batch_size=4, shuffle=True, generator=loader_generator(1))
    assert loader.worker_init_fn is seed_worker and loader.generator is not None
    assert make_loader(_graphs(), batch_size=4, shuffle=True).worker_init_fn is None


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)
        self.drop = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch=None):
        from torch_geometric.nn import global_mean_pool

        return global_mean_pool(self.drop(self.lin(x)), batch)


def _train(seed, tmp_path):
    from milia_pipeline.models.training.callbacks import ModelCheckpoint
    from milia_pipeline.models.training.trainer import Trainer

    seed_everything(seed)
    graphs = [
        Data(
            x=torch.randn(3, 1), edge_index=torch.zeros(2, 0, dtype=torch.long), y=torch.randn(1, 1)
        )
        for _ in range(24)
    ]
    model = _Net()
    trainer = Trainer(
        model=model,
        train_loader=make_loader(
            graphs[:16], batch_size=4, shuffle=True, generator=loader_generator(seed)
        ),
        val_loader=make_loader(graphs[16:], batch_size=4, shuffle=False),
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        device=torch.device("cpu"),
        callbacks=[ModelCheckpoint(dirpath=tmp_path / str(seed), verbose=False)],
        model_info={"task_type": "graph_regression"},
        max_epochs=3,
    )
    trainer.fit()
    return {k: v.clone() for k, v in model.state_dict().items()}


def test_the_same_seed_trains_the_same_model(tmp_path):
    first = _train(11, tmp_path / "a")
    again = _train(11, tmp_path / "b")
    other = _train(12, tmp_path / "c")
    assert all(torch.equal(first[k], again[k]) for k in first)
    assert any(not torch.equal(first[k], other[k]) for k in first)


def test_run_info_records_the_training_seed():
    from milia_pipeline.models.training.run_info import collect_run_info

    info = collect_run_info(
        {},
        task_type="graph_regression",
        out_channels=1,
        target_names=None,
        data_split=None,
        training_seed=42,
    )
    assert info["training_seed"] == 42
