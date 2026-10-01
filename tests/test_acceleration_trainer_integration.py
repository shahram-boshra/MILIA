"""PA-1b: Trainer ↔ AccelerationManager integration.

Real CPU runs cover the default path, fp32 neutrality and bf16 autocast. The fp16 CUDA GradScaler
protocol (scale → backward → unscale_ before clipping → step → update, once per accumulation
boundary — PyTorch AMP docs) is verified with a recording scaler double, since CI has no GPU.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from milia_pipeline.exceptions import TrainingError
from milia_pipeline.models.acceleration import AccelerationManager
from milia_pipeline.models.acceleration.memory_optimization import MemoryOptimizer
from milia_pipeline.models.training.trainer import Trainer

MODEL_INFO = {"task_type": "graph_regression", "out_channels": 1, "uses_edge_features": False}
N_GRAPHS, BATCH_SIZE = 12, 4  # → 3 training batches per epoch
CPU = torch.device("cpu")


class TinyGCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = GCNConv(4, 8)
        self.head = nn.Linear(8, 1)

    def forward(self, x, edge_index, batch):
        h = torch.relu(self.conv(x, edge_index))
        return self.head(global_mean_pool(h, batch))


def _graphs(seed: int = 0) -> list[Data]:
    gen = torch.Generator().manual_seed(seed)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 1, 2, 3], [1, 2, 3, 4, 0, 0, 1, 2]])
    return [
        Data(
            x=torch.randn(5, 4, generator=gen),
            edge_index=edge_index,
            y=torch.randn(1, 1, generator=gen),
        )
        for _ in range(N_GRAPHS)
    ]


def _trainer(acceleration=None, device=CPU, val=False, **kwargs) -> Trainer:
    torch.manual_seed(0)
    model = TinyGCN()
    loader = DataLoader(_graphs(), batch_size=BATCH_SIZE, shuffle=False)
    return Trainer(
        model=model,
        train_loader=loader,
        val_loader=DataLoader(_graphs(seed=1), batch_size=BATCH_SIZE) if val else None,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        device=device,
        max_epochs=1,
        model_info=MODEL_INFO,
        acceleration=acceleration,
        **kwargs,
    )


def _cpu_manager(**kwargs) -> AccelerationManager:
    return AccelerationManager(device="cpu", verbose=False, **kwargs)


def _fake_acceleration(scaler=None, device=CPU) -> MagicMock:
    accel = MagicMock()
    accel.get_device.return_value = device
    accel.autocast.side_effect = lambda: contextlib.nullcontext()
    accel.get_grad_scaler.return_value = scaler
    return accel


def _recording_scaler() -> MagicMock:
    """Scaler double: scale() is the identity (real backward runs); other calls are recorded."""
    scaler = MagicMock()
    scaler.scale.side_effect = lambda loss: loss
    return scaler


def _head_output_dtypes(trainer: Trainer) -> list[torch.dtype]:
    dtypes: list[torch.dtype] = []
    trainer.model.head.register_forward_hook(lambda _m, _i, out: dtypes.append(out.dtype))
    return dtypes


# --- default path -------------------------------------------------------------------------------
def test_default_has_no_acceleration_and_no_scaler():
    trainer = _trainer()
    assert trainer.acceleration is None
    assert trainer._grad_scaler is None
    assert isinstance(trainer._train_autocast(), contextlib.nullcontext)


def test_fp32_acceleration_is_numerically_identical_to_none():
    plain, accelerated = (
        _trainer(),
        _trainer(acceleration=_cpu_manager(mixed_precision=False), device=None),
    )
    plain._train_epoch()
    accelerated._train_epoch()
    for p, q in zip(plain.model.parameters(), accelerated.model.parameters(), strict=True):
        assert torch.equal(p, q)


# --- CPU bf16 (real autocast) -------------------------------------------------------------------
def test_cpu_bf16_runs_training_forward_in_bfloat16():
    trainer = _trainer(
        acceleration=_cpu_manager(mixed_precision=True, precision="bf16"), device=None
    )
    before = [p.detach().clone() for p in trainer.model.parameters()]
    dtypes = _head_output_dtypes(trainer)
    metrics = trainer._train_epoch()
    assert dtypes and set(dtypes) == {torch.bfloat16}
    assert torch.isfinite(torch.tensor(metrics["train_loss"]))
    assert all(p.dtype == torch.float32 for p in trainer.model.parameters())  # master weights fp32
    assert any(
        not torch.equal(b, p) for b, p in zip(before, trainer.model.parameters(), strict=True)
    )


def test_cpu_bf16_validation_stays_fp32():
    trainer = _trainer(
        acceleration=_cpu_manager(mixed_precision=True, precision="bf16"), device=None, val=True
    )
    dtypes = _head_output_dtypes(trainer)
    trainer._validate_epoch()
    assert dtypes and set(dtypes) == {torch.float32}


# --- GradScaler is fp16-only (MemoryOptimizer) --------------------------------------------------
@pytest.mark.parametrize(("precision", "expects_scaler"), [("fp16", True), ("bf16", False)])
def test_grad_scaler_only_for_fp16(precision, expects_scaler):
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.is_bf16_supported", return_value=True),
    ):
        optimizer = MemoryOptimizer(mixed_precision=True, precision=precision, verbose=False)
    assert (optimizer.get_grad_scaler() is not None) is expects_scaler


# --- fp16 scaler protocol (scaler double) -------------------------------------------------------
def test_scaler_protocol_with_clipping_unscales_before_each_step():
    scaler = _recording_scaler()
    trainer = _trainer(acceleration=_fake_acceleration(scaler), device=None, gradient_clip_val=1.0)
    trainer._train_epoch()
    calls = [
        name for name, *_ in scaler.mock_calls if name in {"scale", "unscale_", "step", "update"}
    ]
    assert calls == ["scale", "unscale_", "step", "update"] * (N_GRAPHS // BATCH_SIZE)
    for _name, args, _kwargs in (c for c in scaler.mock_calls if c[0] in {"unscale_", "step"}):
        assert args == (trainer.optimizer,)


def test_scaler_without_clipping_never_unscales():
    scaler = _recording_scaler()
    trainer = _trainer(acceleration=_fake_acceleration(scaler), device=None)
    trainer._train_epoch()
    assert scaler.unscale_.call_count == 0
    assert scaler.step.call_count == scaler.update.call_count == N_GRAPHS // BATCH_SIZE


def test_scaler_steps_once_per_accumulation_boundary():
    scaler = _recording_scaler()
    trainer = _trainer(
        acceleration=_fake_acceleration(scaler), device=None, accumulate_grad_batches=3
    )
    trainer._train_epoch()
    assert scaler.scale.call_count == N_GRAPHS // BATCH_SIZE  # every batch
    assert scaler.step.call_count == scaler.update.call_count == 1  # one boundary


# --- device resolution --------------------------------------------------------------------------
def test_device_taken_from_acceleration_when_omitted():
    assert _trainer(acceleration=_fake_acceleration(), device=None).device == torch.device("cpu")


def test_conflicting_device_type_fails_fast():
    with pytest.raises(TrainingError, match="conflicts with the acceleration device"):
        _trainer(
            acceleration=_fake_acceleration(device=torch.device("cuda")), device=torch.device("cpu")
        )
