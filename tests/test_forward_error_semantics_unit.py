"""PA-5 (F44): forward calling-convention fallback and evaluation error semantics.

Fallback happens only for a calling-convention mismatch (``TypeError``); any other exception was raised
by a model that accepted the call and must surface unmasked. Evaluation loops fail like the training
loop instead of skipping batches or returning a fabricated ``val_loss = inf``.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from milia_pipeline.exceptions import TrainingError
from milia_pipeline.models.training.trainer import Trainer

MODEL_INFO = {"task_type": "graph_regression", "out_channels": 1, "uses_edge_features": False}


# --- strategy runner ---------------------------------------------------------------------------
def test_type_error_falls_back_to_next_convention():
    second = Mock(return_value="ok")

    def bad():
        raise TypeError("unexpected keyword")

    assert Trainer._call_forward_strategies([("a", bad), ("b", second)], "ctx") == "ok"
    second.assert_called_once()


def test_non_type_error_propagates_unmasked():
    second = Mock(return_value="ok")

    def failing_inside():
        raise RuntimeError("compile failed inside the model")

    with pytest.raises(RuntimeError, match="compile failed inside the model"):
        Trainer._call_forward_strategies([("a", failing_inside), ("b", second)], "ctx")
    second.assert_not_called()  # no masking by a later convention


def test_all_mismatches_raise_chained_training_error():
    errors = [TypeError("first mismatch"), TypeError("second mismatch")]

    def raiser(err):
        def call():
            raise err

        return call

    with pytest.raises(TrainingError) as exc_info:
        Trainer._call_forward_strategies(
            [("first", raiser(errors[0])), ("second", raiser(errors[1]))], "ctx"
        )
    message = str(exc_info.value)
    assert "(1) first: first mismatch" in message and "(2) second: second mismatch" in message
    assert exc_info.value.__cause__ is errors[0]  # the preferred convention's error is the cause


# --- trainer integration ------------------------------------------------------------------------
def _loader():
    gen = torch.Generator().manual_seed(0)
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    graphs = [
        Data(x=torch.randn(3, 4, generator=gen), edge_index=ei, y=torch.randn(1, 1, generator=gen))
        for _ in range(4)
    ]
    return DataLoader(graphs, batch_size=2)


class FailsInside(nn.Module):
    """Accepts the standard convention, then fails during computation (like a compile error)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, x, edge_index, batch=None):
        raise RuntimeError("backend failed inside forward")


class NeedsBatchObject(nn.Module):
    """Only the batch-object convention fits."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, data):
        from torch_geometric.nn import global_mean_pool

        return global_mean_pool(self.lin(data.x), data.batch)


def _trainer(model, **loaders):
    return Trainer(
        model=model,
        train_loader=_loader(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        device=torch.device("cpu"),
        max_epochs=1,
        model_info=MODEL_INFO,
        **loaders,
    )


def test_inside_model_error_surfaces_from_forward_pass():
    trainer = _trainer(FailsInside())
    with pytest.raises(RuntimeError, match="backend failed inside forward"):
        trainer._forward_pass(next(iter(_loader())))


def test_batch_object_convention_still_found():
    trainer = _trainer(NeedsBatchObject())
    out = trainer._forward_pass(next(iter(_loader())))
    assert out.shape == (2, 1)


def test_validation_fails_with_original_cause():
    trainer = _trainer(FailsInside(), val_loader=_loader())
    with pytest.raises(TrainingError, match="Validation batch failed") as exc_info:
        trainer._validate_epoch()
    assert "backend failed inside forward" in str(exc_info.value.__cause__)


def test_test_loop_fails_with_original_cause():
    trainer = _trainer(FailsInside(), test_loader=_loader())
    with pytest.raises(TrainingError, match="Test batch failed") as exc_info:
        trainer.test()
    assert "backend failed inside forward" in str(exc_info.value.__cause__)
