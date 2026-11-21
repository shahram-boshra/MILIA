"""Test-after-training scores the best checkpoint (milia_pipeline.models.training.trainer).

``Trainer.fit`` evaluates the test set with the weights of the checkpoint a ``ModelCheckpoint``
saved as best, stores the metrics in that checkpoint, and leaves the final weights in the model.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from milia_pipeline.exceptions import CheckpointError, TrainingError
from milia_pipeline.models.checkpoint_io import load_checkpoint_file
from milia_pipeline.models.training.callbacks import Callback, ModelCheckpoint
from milia_pipeline.models.training.trainer import Trainer


class _Pooled(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, x, edge_index, batch=None):
        return self.lin(global_mean_pool(x, batch))


class _BreakWeightsAfterFirstEpoch(Callback):
    """Makes every epoch after the first worse, so best.pt is epoch 0 and differs from the end."""

    def on_epoch_end(self, trainer, epoch, metrics):
        if epoch == 0:
            with torch.no_grad():
                trainer.unwrapped_model.lin.bias.fill_(1000.0)


def _loader(seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    graphs = []
    for _ in range(8):
        x = torch.randn(3, 4, generator=generator)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        graphs.append(Data(x=x, edge_index=edge_index, y=x.sum().view(1, 1)))
    return DataLoader(graphs, batch_size=4)


def _trainer(tmp_path, callbacks, test_loader=True) -> Trainer:
    torch.manual_seed(0)
    model = _Pooled()
    return Trainer(
        model=model,
        train_loader=_loader(1),
        val_loader=_loader(2),
        test_loader=_loader(3) if test_loader else None,
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-4),
        device=torch.device("cpu"),
        callbacks=callbacks,
        max_epochs=3,
    )


def _checkpoint(tmp_path) -> ModelCheckpoint:
    return ModelCheckpoint(dirpath=tmp_path / "checkpoints", save_top_k=1, verbose=False)


def _test_loss_of(state_dict, trainer: Trainer) -> float:
    model = copy.deepcopy(trainer.unwrapped_model)
    model.load_state_dict(state_dict)
    loss = nn.MSELoss(reduction="sum")
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in trainer.test_loader:
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            total += loss(out, batch.y).item()
            count += batch.y.numel()
    return total / count


def test_fit_tests_the_best_checkpoint_and_keeps_the_final_weights(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    trainer = _trainer(tmp_path, [checkpoint, _BreakWeightsAfterFirstEpoch()])
    results = trainer.fit()

    best = load_checkpoint_file(tmp_path / "checkpoints" / "best.pt", map_location="cpu")
    last = load_checkpoint_file(tmp_path / "checkpoints" / "last.pt", map_location="cpu")
    assert best["epoch"] == 0 and last["epoch"] == 2  # the premise: best is not the end state
    assert results["tested_checkpoint"] == str(tmp_path / "checkpoints" / "best.pt")
    assert results["test_metrics"]["test_loss"] == pytest.approx(
        _test_loss_of(best["model_state_dict"], trainer), rel=1e-5
    )
    assert results["test_metrics"]["test_loss"] != pytest.approx(
        _test_loss_of(last["model_state_dict"], trainer), rel=1e-3
    )
    for name, value in trainer.unwrapped_model.state_dict().items():
        torch.testing.assert_close(value, last["model_state_dict"][name])


def test_fit_stores_the_test_metrics_in_the_best_checkpoint(tmp_path):
    trainer = _trainer(tmp_path, [_checkpoint(tmp_path), _BreakWeightsAfterFirstEpoch()])
    results = trainer.fit()

    best = load_checkpoint_file(tmp_path / "checkpoints" / "best.pt", map_location="cpu")
    assert best["test_metrics"] == results["test_metrics"]
    assert {"model_state_dict", "hyper_parameters", "version_info", "epoch"} <= set(best)
    assert not list((tmp_path / "checkpoints").glob(".*.tmp"))


def test_without_a_checkpoint_callback_fit_tests_the_final_weights(tmp_path):
    trainer = _trainer(tmp_path, [])
    results = trainer.fit()
    assert results["tested_checkpoint"] is None
    assert results["test_metrics"]["test_loss"] == pytest.approx(
        _test_loss_of(trainer.unwrapped_model.state_dict(), trainer), rel=1e-5
    )


def test_fit_with_test_false_skips_the_test_set(tmp_path):
    trainer = _trainer(tmp_path, [_checkpoint(tmp_path)])
    results = trainer.fit(test=False)
    assert results["test_metrics"] == {} and results["tested_checkpoint"] is None
    best = load_checkpoint_file(tmp_path / "checkpoints" / "best.pt", map_location="cpu")
    assert "test_metrics" not in best


def test_test_with_a_checkpoint_path_restores_the_current_weights(tmp_path):
    trainer = _trainer(tmp_path, [_checkpoint(tmp_path), _BreakWeightsAfterFirstEpoch()])
    trainer.fit(test=False)
    before = copy.deepcopy(trainer.unwrapped_model.state_dict())
    best_path = tmp_path / "checkpoints" / "best.pt"

    metrics = trainer.test(ckpt_path=best_path)
    assert metrics == trainer.test(ckpt_path="best")
    best = load_checkpoint_file(best_path, map_location="cpu")
    assert metrics["test_loss"] == pytest.approx(
        _test_loss_of(best["model_state_dict"], trainer), rel=1e-5
    )
    for name, value in trainer.unwrapped_model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_test_best_without_a_best_checkpoint_raises(tmp_path):
    trainer = _trainer(tmp_path, [])
    with pytest.raises(TrainingError, match="needs a checkpoint callback"):
        trainer.test(ckpt_path="best")


def test_test_with_an_unreadable_checkpoint_keeps_the_weights(tmp_path):
    trainer = _trainer(tmp_path, [])
    before = copy.deepcopy(trainer.unwrapped_model.state_dict())
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="Cannot test checkpoint"):
        trainer.test(ckpt_path=broken)
    for name, value in trainer.unwrapped_model.state_dict().items():
        torch.testing.assert_close(value, before[name])
