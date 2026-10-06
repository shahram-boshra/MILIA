"""PA-1e (F42): models.acceleration → AccelerationManager, and compile-safe training artifacts.

Compiled-model tests use torch.compile's ``eager`` backend: it produces a real ``OptimizedModule``
wrapper (the object whose ``state_dict`` keys gain an ``_orig_mod.`` prefix) without code generation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from milia_pipeline.exceptions import ConfigurationError
from milia_pipeline.models.acceleration.config_builder import (
    build_acceleration,
    load_acceleration_config,
)
from milia_pipeline.models.factory.model_factory import GraphLevelModelWrapper
from milia_pipeline.models.training.module_utils import unwrap_compiled
from milia_pipeline.models.training.trainer import Trainer

MODEL_INFO = {"task_type": "graph_regression", "out_channels": 1, "uses_edge_features": False}


class NodeLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, x, edge_index, batch=None):
        return self.lin(x)


def _loader():
    gen = torch.Generator().manual_seed(0)
    ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    graphs = [
        Data(x=torch.randn(5, 4, generator=gen), edge_index=ei, y=torch.randn(1, 1, generator=gen))
        for _ in range(8)
    ]
    return DataLoader(graphs, batch_size=4)


def _model():
    torch.manual_seed(0)
    return GraphLevelModelWrapper(NodeLinear(), "graph_regression")


def _cfg(**accel):
    return {"acceleration": accel}


# --- build_acceleration ---------------------------------------------------------------------------
@pytest.mark.parametrize("models_config", [None, {}, _cfg(enabled=False)])
def test_disabled_or_absent_returns_none(models_config):
    assert build_acceleration(models_config) is None


def test_enabled_maps_settings_onto_the_manager():
    manager = build_acceleration(
        _cfg(
            enabled=True,
            device={"type": "cpu"},
            memory={"mixed_precision": "bf16"},
            computation={"compile_model": True, "compile_dynamic": False},
        )
    )
    # Resolve the class from the live module: another test module reloads the acceleration
    # package (importlib.reload), which rebinds new class objects; a name bound at collection
    # would be stale (Python docs: references to old objects are not rebound).
    import milia_pipeline.models.acceleration as acceleration_pkg

    assert isinstance(manager, acceleration_pkg.AccelerationManager)
    assert manager.get_device() == torch.device("cpu")
    assert manager.memory_optimizer.config.mixed_precision is True
    assert manager.memory_optimizer.config.precision == "bf16"
    assert manager.computation_optimizer.config.compile_model is True
    assert manager.computation_optimizer.config.compile_dynamic is False


def test_each_call_returns_a_fresh_manager():
    cfg = _cfg(enabled=True, device={"type": "cpu"})
    assert build_acceleration(cfg) is not build_acceleration(cfg)


def test_invalid_section_raises_configuration_error():
    with pytest.raises(ConfigurationError, match="mixed_precision 'fp8'"):
        build_acceleration(_cfg(enabled=True, memory={"mixed_precision": "fp8"}))


def test_unwired_settings_are_rejected_together():
    with pytest.raises(ConfigurationError) as exc:
        build_acceleration(
            _cfg(
                enabled=True,
                device={"type": "cpu"},
                distributed={"enabled": True, "strategy": "ddp"},
                memory={"gradient_accumulation_steps": 4, "empty_cache_interval": 10},
            )
        )
    text = str(exc.value)
    assert "distributed.enabled" in text
    assert "gradient_accumulation_steps" in text
    assert "empty_cache_interval" in text


def test_unwired_settings_ignored_when_disabled():
    cfg = _cfg(
        enabled=False, distributed={"enabled": True}, memory={"gradient_accumulation_steps": 4}
    )
    assert load_acceleration_config(cfg).enabled is False
    assert build_acceleration(cfg) is None


# --- unwrap_compiled ------------------------------------------------------------------------------
def test_unwrap_plain_module_and_mock_unchanged():
    module, mock = nn.Linear(2, 2), MagicMock()
    assert unwrap_compiled(module) is module
    assert unwrap_compiled(mock) is mock


def test_unwrap_compiled_returns_original():
    module = nn.Linear(2, 2)
    compiled = torch.compile(module, backend="eager")
    assert any(k.startswith("_orig_mod.") for k in compiled.state_dict())
    assert unwrap_compiled(compiled) is module


# --- compiled model: checkpoints and num_graphs ---------------------------------------------------
def _trainer(model):
    return Trainer(
        model=model,
        train_loader=_loader(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        device=torch.device("cpu"),
        max_epochs=1,
        model_info=MODEL_INFO,
    )


def test_compiled_checkpoint_loads_into_plain_model(tmp_path):
    compiled = torch.compile(_model(), backend="eager")
    trainer = _trainer(compiled)
    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path)

    state = torch.load(path, weights_only=False)["model_state_dict"]
    assert not any(k.startswith("_orig_mod.") for k in state)
    plain = _model()
    plain.load_state_dict(state, strict=True)  # loadable without compile

    # ...and back into a compiled trainer
    _trainer(torch.compile(_model(), backend="eager")).load_checkpoint(path)


def test_num_graphs_reaches_compiled_wrapper():
    batch = next(iter(_loader()))
    trainer = _trainer(torch.compile(_model(), backend="eager"))
    assert trainer._num_graphs_kwargs(batch) == {"num_graphs": batch.num_graphs}


# --- YAML → manager → training step ----------------------------------------------------------------
def test_config_built_manager_runs_bf16_training_epoch():
    manager = build_acceleration(
        _cfg(enabled=True, device={"type": "cpu"}, memory={"mixed_precision": "bf16"})
    )
    model = _model()
    train_model = manager.optimize_model(model)
    trainer = Trainer(
        model=train_model,
        train_loader=_loader(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        max_epochs=1,
        model_info=MODEL_INFO,
        acceleration=manager,
    )
    dtypes = []
    model.model.lin.register_forward_hook(lambda _m, _i, out: dtypes.append(out.dtype))
    metrics = trainer._train_epoch()
    assert set(dtypes) == {torch.bfloat16}
    assert torch.isfinite(torch.tensor(metrics["train_loss"]))
