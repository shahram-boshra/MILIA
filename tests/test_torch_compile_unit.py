"""PA-1c: torch.compile configuration and dynamic-shape behaviour.

GNN mini-batches change node/edge counts every batch. PyG's guidance is ``dynamic=True`` (shape-generic
kernels); ``dynamic=False`` always specializes, recompiling per new shape until torch's recompile limit
and then running eager. Recompilation is decided by dynamo (guards), independently of the backend, so
these tests use the ``aot_eager`` backend; the Inductor C++ path is verified at image build
(``docker/verify_build.py --compile``, PA-0b).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError as PydanticValidationError
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GCN

from milia_pipeline.models.acceleration import AccelerationManager
from milia_pipeline.models.acceleration.computation_optimization import ComputationOptimizer
from milia_pipeline.models.factory.model_factory import GraphLevelModelWrapper
from milia_pipeline.models.utils.config_bridge import ComputationConfig, ModelConfig

try:
    from milia_pipeline.exceptions import OptimizationError
except ImportError:  # pragma: no cover - mirrors the module's own fallback
    from milia_pipeline.models.acceleration.computation_optimization import OptimizationError


# --- configuration ------------------------------------------------------------------------------
def test_bridge_compile_dynamic_defaults_true():
    assert ComputationConfig().compile_dynamic is True


@pytest.mark.parametrize(
    "mode", ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]
)
def test_bridge_accepts_torch_compile_modes(mode):
    assert ComputationConfig(compile_mode=mode).compile_mode == mode


def test_bridge_rejects_unknown_compile_mode():
    with pytest.raises(PydanticValidationError, match="Invalid compile_mode 'fast'"):
        ComputationConfig(compile_mode="fast")


def test_yaml_compile_dynamic_is_parsed():
    config = ModelConfig.from_dict(
        {
            "enabled": True,
            "selection": {"task_type": "graph_regression", "model_name": "GCN"},
            "acceleration": {"computation": {"compile_model": True, "compile_dynamic": False}},
        }
    )
    assert config.acceleration.computation.compile_dynamic is False


def test_manager_forwards_compile_dynamic():
    manager = AccelerationManager(
        device="cpu", compile_model=True, compile_dynamic=False, verbose=False
    )
    assert manager.computation_optimizer.config.compile_dynamic is False


# --- explicit failure ---------------------------------------------------------------------------
def test_refuses_silent_eager_fallback():
    import torch._dynamo

    optimizer = ComputationOptimizer(compile_model=True, verbose=False, device=torch.device("cpu"))
    with (
        patch.object(torch._dynamo.config, "suppress_errors", True),
        pytest.raises(OptimizationError, match="suppress_errors"),
    ):
        optimizer.compile_model(torch.nn.Linear(2, 2))


def test_unknown_mode_fails_at_compile_call():
    optimizer = ComputationOptimizer(compile_model=True, verbose=False, device=torch.device("cpu"))
    with pytest.raises(OptimizationError, match="Unrecognized mode"):
        optimizer.compile_model(torch.nn.Linear(2, 2), mode="fast")


# --- dynamic shapes: recompilation behaviour on varying graph sizes ----------------------------
def _batches(nodes_per_graph: tuple[int, ...], n_graphs: int = 4):
    gen = torch.Generator().manual_seed(0)
    for nodes in nodes_per_graph:
        ei = torch.tensor([[i, (i + 1) % nodes] for i in range(nodes)]).t()
        graphs = [
            Data(x=torch.randn(nodes, 4, generator=gen), edge_index=ei) for _ in range(n_graphs)
        ]
        yield next(iter(DataLoader(graphs, batch_size=n_graphs)))


def _compiled_frames(dynamic: bool) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor]]:
    """Run a compiled GCN family model over batches of growing size; record compiled frames."""
    import torch._dynamo
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    torch.manual_seed(0)
    model = GraphLevelModelWrapper(GCN(4, 8, num_layers=2, out_channels=1), "graph_regression")
    model.eval()
    optimizer = ComputationOptimizer(compile_model=True, verbose=False, device=torch.device("cpu"))
    compiled = optimizer.compile_model(model, dynamic=dynamic, backend="aot_eager")

    frames, eager_out, compiled_out = [], [], []
    with torch.no_grad():
        for b in _batches((5, 7, 9)):
            eager_out.append(model(b.x, b.edge_index, batch=b.batch, num_graphs=b.num_graphs))
            compiled_out.append(compiled(b.x, b.edge_index, batch=b.batch, num_graphs=b.num_graphs))
            frames.append(counters["stats"]["unique_graphs"])
    torch._dynamo.reset()
    return frames, eager_out, compiled_out


def test_dynamic_true_does_not_recompile_across_graph_sizes():
    frames, eager_out, compiled_out = _compiled_frames(dynamic=True)
    assert frames[0] > 0  # compiled at the first call
    assert frames[1] == frames[0] and frames[2] == frames[0]  # same kernels for new sizes
    for e, c in zip(eager_out, compiled_out, strict=True):
        torch.testing.assert_close(c, e)


def test_dynamic_false_recompiles_for_new_graph_sizes():
    """Contrast case: specialization recompiles when the batch's graph sizes change."""
    frames, eager_out, compiled_out = _compiled_frames(dynamic=False)
    assert frames[-1] > frames[0]
    for e, c in zip(eager_out, compiled_out, strict=True):
        torch.testing.assert_close(c, e)
