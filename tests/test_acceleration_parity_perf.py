"""PA-4: acceleration parity and CPU timing per GNN family (markers: ``perf``, ``slow``).

* Compile parity — a real Inductor compile (``dynamic=True``, PA-1c) must reproduce the eager validation
  loss on identical weights over variable-size batches, within ``torch.testing.assert_close``'s documented
  float32 defaults. Validation runs in fp32 (PA-1b), so this isolates compile numerics.
* bf16 parity — with a frozen model (``lr=0``) a bf16-autocast training epoch must match the fp32 epoch's
  loss within assert_close's documented bfloat16 defaults: the difference is forward precision only.
* Timing — best-of-N epoch wall-clock (repo convention, ``test_descriptor_performance.py``) is recorded,
  not asserted: compile speed-ups on small CPU graphs are not guaranteed, so a timing assertion would be
  flaky. Values appear in JUnit XML via ``record_property``.
"""

from __future__ import annotations

import time

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GAT, GCN, GIN, GraphSAGE

from milia_pipeline.models.acceleration import AccelerationManager
from milia_pipeline.models.acceleration.computation_optimization import ComputationOptimizer
from milia_pipeline.models.factory.model_factory import GraphLevelModelWrapper
from milia_pipeline.models.training.trainer import Trainer

pytestmark = [pytest.mark.perf, pytest.mark.slow]

MODEL_INFO = {"task_type": "graph_regression", "out_channels": 1, "uses_edge_features": False}
FAMILIES = {"GCN": GCN, "GraphSAGE": GraphSAGE, "GIN": GIN, "GAT": GAT}
# torch.testing.assert_close documented default tolerances
FP32_TOL = {"rtol": 1.3e-6, "atol": 1e-5}
BF16_TOL = {"rtol": 1.6e-2, "atol": 1e-5}
CPU = torch.device("cpu")


def _loader(seed: int) -> DataLoader:
    """Graphs of varying size (5–11 nodes) → batches with varying node counts."""
    gen = torch.Generator().manual_seed(seed)
    graphs = []
    for i in range(24):
        n = 5 + i % 7
        ei = torch.tensor([[j, (j + 1) % n] for j in range(n)]).t()
        graphs.append(
            Data(
                x=torch.randn(n, 8, generator=gen),
                edge_index=ei,
                y=torch.randn(1, 1, generator=gen),
            )
        )
    return DataLoader(graphs, batch_size=6, shuffle=False)


def _model(family: str) -> GraphLevelModelWrapper:
    torch.manual_seed(0)
    return GraphLevelModelWrapper(
        FAMILIES[family](8, 16, num_layers=2, out_channels=1), "graph_regression"
    )


def _trainer(model, *, lr: float = 0.0, acceleration=None) -> Trainer:
    return Trainer(
        model=model,
        train_loader=_loader(0),
        val_loader=_loader(1),
        optimizer=torch.optim.SGD(model.parameters(), lr=lr),
        device=CPU,
        max_epochs=1,
        model_info=MODEL_INFO,
        acceleration=acceleration,
    )


def _best_of(fn, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_compiled_validation_matches_eager(family, record_property):
    import torch._dynamo

    torch._dynamo.reset()
    eager_model = _model(family)
    compiled_model = ComputationOptimizer(
        compile_model=True, verbose=False, device=CPU
    ).compile_model(
        _model(family)  # same seed → identical weights
    )
    eager, compiled = _trainer(eager_model), _trainer(compiled_model)

    eager_loss = eager._validate_epoch()["val_loss"]
    compiled_loss = compiled._validate_epoch()["val_loss"]  # first call compiles (Inductor)
    torch.testing.assert_close(torch.tensor(compiled_loss), torch.tensor(eager_loss), **FP32_TOL)

    record_property(f"{family}_val_epoch_eager_s", _best_of(eager._validate_epoch))
    record_property(f"{family}_val_epoch_compiled_s", _best_of(compiled._validate_epoch))
    torch._dynamo.reset()


@pytest.mark.parametrize("family", list(FAMILIES))
def test_bf16_training_loss_matches_fp32_on_frozen_model(family):
    fp32 = _trainer(_model(family))._train_epoch()["train_loss"]
    manager = AccelerationManager(
        device="cpu", mixed_precision=True, precision="bf16", verbose=False
    )
    model = _model(family)
    bf16 = _trainer(manager.optimize_model(model), acceleration=manager)._train_epoch()[
        "train_loss"
    ]
    torch.testing.assert_close(torch.tensor(bf16), torch.tensor(fp32), **BF16_TOL)
