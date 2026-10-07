"""PA-3: epoch losses are accumulated on device with one host read per epoch.

The old loops summed ``loss.item()`` (a device→host sync per batch) into Python floats (IEEE
doubles). The device accumulator sums in float64, so epoch means are bit-identical.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from milia_pipeline.models.factory.model_factory import GraphLevelModelWrapper
from milia_pipeline.models.training.trainer import Trainer, _DeviceLossSum

MODEL_INFO = {"task_type": "graph_regression", "out_channels": 1, "uses_edge_features": False}


# --- accumulator -----------------------------------------------------------------------------------
def _losses(n: int = 7) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    return [torch.rand((), generator=gen, dtype=torch.float32) * 10 for _ in range(n)]


@pytest.mark.parametrize("scale", [1, 4])
def test_mean_is_bit_identical_to_python_float_sum(scale):
    losses = _losses()
    old = 0.0
    for loss in losses:
        old += loss.item() * scale
    acc = _DeviceLossSum("cpu")
    for loss in losses:
        acc.add(loss, scale=scale)
    assert acc.mean(empty=0.0) == old / len(losses)  # exact equality


@pytest.mark.parametrize("empty", [0.0, float("inf")])
def test_empty_returns_given_value(empty):
    assert _DeviceLossSum("cpu").mean(empty=empty) == empty


def test_accumulator_is_detached_from_graph():
    weight = torch.ones((), requires_grad=True)
    acc = _DeviceLossSum("cpu")
    acc.add(weight * 2.0)
    assert acc._sum.requires_grad is False


@pytest.mark.parametrize(
    ("device", "dtype"), [("cpu", torch.float64), ("cuda", torch.float64), ("mps", torch.float32)]
)
def test_accumulator_dtype_rule(device, dtype):
    assert _DeviceLossSum.accumulator_dtype(device) is dtype


# --- trainer loops: host reads do not scale with the number of batches -----------------------------
class NodeLinear(nn.Module):
    """Node-level model; wrapped below so pooling gets `num_graphs` from the Trainer (PA-1d) and
    reads no tensor values — isolating the loss accumulation under test."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, x, edge_index, batch=None):
        return self.lin(x)


def _loader(n_graphs: int) -> DataLoader:
    gen = torch.Generator().manual_seed(0)
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    graphs = [
        Data(x=torch.randn(3, 4, generator=gen), edge_index=ei, y=torch.randn(1, 1, generator=gen))
        for _ in range(n_graphs)
    ]
    return DataLoader(graphs, batch_size=2)


def _item_calls(n_graphs: int, method: str) -> int:
    torch.manual_seed(0)
    model = GraphLevelModelWrapper(NodeLinear(), "graph_regression")
    trainer = Trainer(
        model=model,
        train_loader=_loader(n_graphs),
        val_loader=_loader(n_graphs),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        device=torch.device("cpu"),
        max_epochs=1,
        model_info=MODEL_INFO,
    )
    calls = 0
    original = torch.Tensor.item

    def counting_item(self):
        nonlocal calls
        calls += 1
        return original(self)

    trainer_logger = logging.getLogger("milia_pipeline.models.training.trainer")
    previous_level = trainer_logger.level
    trainer_logger.setLevel(logging.INFO)  # DEBUG off: the guarded log line must not read
    try:
        with patch.object(torch.Tensor, "item", counting_item):
            getattr(trainer, method)()
    finally:
        trainer_logger.setLevel(previous_level)
    return calls


@pytest.mark.parametrize("method", ["_train_epoch", "_validate_epoch"])
def test_host_reads_do_not_grow_with_batches(method):
    assert _item_calls(4, method) == _item_calls(12, method)  # 2 vs 6 batches
